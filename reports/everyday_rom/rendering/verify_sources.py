# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify the packaged source snapshot without requiring the origin workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify_sources(directory: Path) -> dict:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "source_manifest.json").read_text())
    expected_files = set(manifest["packaged_sha256"])
    actual_files = {path.name for path in directory.glob("*.py")} | {"README.md"}
    if expected_files != actual_files:
        raise ValueError("Source manifest does not cover exactly the packaged Python files and README")
    for name, expected in manifest["packaged_sha256"].items():
        path = (directory / name).resolve()
        if path.parent != directory:
            raise ValueError(f"Source path outside package: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Packaged source hash mismatch: {name}")
    for name, record in manifest["copied_snapshots"].items():
        if not record["adaptations"] and manifest["packaged_sha256"][name] != record["origin_sha256"]:
            raise ValueError(f"Unadapted snapshot differs from origin: {name}")
    return {"verified": True, "files": len(expected_files), "copied_snapshots": len(manifest["copied_snapshots"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify_sources(args.directory), indent=2))


if __name__ == "__main__":
    main()
