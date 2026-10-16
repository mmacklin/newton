# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check asset inventory without loading raw third-party models or saved runs."""

import hashlib
import json
import os
from pathlib import Path


def source_files(root):
    """Exclude ignored local test caches, which are not part of the package."""
    for directory, children, names in os.walk(root):
        children[:] = [name for name in children if name not in (".validation", "__pycache__")]
        for name in names:
            yield Path(directory) / name


def main():
    root = Path(__file__).resolve().parent
    inventory = json.loads((root / "assets.json").read_text())
    files = list(source_files(root))
    found = {str(path.relative_to(root)) for path in files if path.suffix == ".npz"}
    if found != set(inventory):
        raise AssertionError(f"Unexpected or missing NPZ assets: {found ^ set(inventory)}")
    for name, expected in inventory.items():
        path = root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected["sha256"]:
            raise AssertionError(f"Asset mismatch: {name}")
    forbidden = {".mp4", ".m4v", ".html", ".pkl", ".pth", ".pt", ".ckpt", ".usdc"}
    for path in files:
        if path.is_file() and path.suffix.lower() in forbidden:
            raise AssertionError(f"Unexpected generated or raw model file: {path}")
    print(f"Verified {len(inventory)} derived assets; no videos or raw models")


if __name__ == "__main__":
    main()
