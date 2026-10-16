# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record CPU discovery and CLI-help results without running a GPU trajectory."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import newton


def main():
    here = Path(__file__).resolve().parent
    runtime = Path(newton.__file__).resolve().parents[1]
    manifest = json.loads((here / "sources.json").read_text())
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(runtime),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    jobs = {
        "cpu": ["-m", "unittest", "discover", "-s", str(here), "-p", "test_*.py", "-v"],
        "help": [str(here / (manifest["local_modules"][0] + ".py")), "--help"],
        "refinement-help": [str(here / "refine_render.py"), "--help"],
    }
    results = {}
    for name, args in jobs.items():
        command = [sys.executable, *args]
        completed = subprocess.run(
            command, cwd=runtime, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
        )
        (here / (name + ".log")).write_text(completed.stdout)
        results[name] = {"command": command, "returncode": completed.returncode}
        print(name, completed.returncode, completed.stdout[-6000:], flush=True)
    result = {
        "newton_module": str(Path(newton.__file__).resolve()),
        "runtime_head": subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip(),
        "source_manifest_sha256": hashlib.sha256((here / "sources.json").read_bytes()).hexdigest(),
        "cpu_only": True,
        "fresh_gpu_physics_validated": False,
        "jobs": results,
        "passed": all(row["returncode"] == 0 for row in results.values()),
    }
    (here / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
