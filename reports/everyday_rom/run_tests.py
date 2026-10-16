# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run each research snapshot's CPU tests in an isolated import namespace."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MODAL_PATH_TEST_DEVICE": "cpu",
    }
    directories = [root, *sorted((root / "scenes").iterdir()), root / "rendering"]
    failed = []
    for directory in directories:
        if not directory.is_dir() or not list(directory.glob("test_*.py")):
            continue
        print(f"\n=== {directory.relative_to(root)} ===", flush=True)
        tests = [path.stem for path in sorted(directory.glob("test_*.py"))]
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", *tests],
            cwd=root.parents[1],
            env=env | {"PYTHONPATH": os.pathsep.join((str(directory), str(root.parents[1])))},
            check=False,
        )
        if result.returncode:
            failed.append(str(directory.relative_to(root)))
    if failed:
        raise SystemExit(f"Failed test directories: {', '.join(failed)}")


if __name__ == "__main__":
    main()
