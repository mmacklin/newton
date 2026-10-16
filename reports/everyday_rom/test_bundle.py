# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Publication boundaries and derived-asset contract for the source bundle."""

import unittest
from pathlib import Path

import numpy as np
from check_bundle import main as check_bundle
from check_bundle import source_files


class BundleTests(unittest.TestCase):
    def test_reviewed_inventory(self):
        check_bundle()

    def test_assets_are_numeric_not_pickle_payloads(self):
        for path in source_files(Path(__file__).parent):
            if path.suffix != ".npz":
                continue
            with self.subTest(asset=path.name), np.load(path, allow_pickle=False) as arrays:
                for name in arrays.files:
                    array = arrays[name]
                    self.assertNotEqual(array.dtype.kind, "O", name)
                    if array.dtype.kind in "biufc":
                        self.assertTrue(np.isfinite(array).all(), name)


if __name__ == "__main__":
    unittest.main()
