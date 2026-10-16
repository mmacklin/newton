# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU-only integrity and local-import checks for this published scene snapshot."""

import ast
import hashlib
import importlib
import json
import unittest
from pathlib import Path

import newton

HERE = Path(__file__).resolve().parent


class PortableSnapshotTests(unittest.TestCase):
    def test_snapshot_hashes_and_local_helpers(self):
        manifest = json.loads((HERE / "sources.json").read_text())
        for name, record in manifest["files"].items():
            path = HERE / name
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["packaged_sha256"])
                if path.suffix == ".py":
                    ast.parse(path.read_text())
        for name in manifest["local_modules"]:
            module = importlib.import_module(name)
            self.assertEqual(Path(module.__file__).resolve().parent, HERE, name)
        self.assertTrue(hasattr(newton, "ModalBasis"))

    def test_no_machine_specific_execution_paths(self):
        manifest = json.loads((HERE / "sources.json").read_text())
        for name in manifest["files"]:
            if name.endswith(".py"):
                self.assertNotIn("/home/", (HERE / name).read_text(), name)


if __name__ == "__main__":
    unittest.main()
