# SPDX-License-Identifier: Apache-2.0
"""Offline boundary changes preserve the original full-order loading targets."""

import unittest

import numpy as np
from retained_geometry import retained_band_max, retained_constraints


class RetainedConstraintTests(unittest.TestCase):
    def test_retained_metric_excludes_only_one_transition_band(self):
        distance = np.full((7, 51), 0.001)
        distance[6] = 0.01
        distance[5, 0] = 0.1
        distance[4, 0] = 0.006
        self.assertEqual(retained_band_max(distance), 0.006)
        distance[0] = 0.01
        self.assertEqual(retained_band_max(distance), 0.0)

    def test_complete_row_and_existing_targets(self):
        rest = np.arange(15 * 3, dtype=float).reshape(15, 3)
        original = np.array([0, 5, 14])
        targets = rest[original] + 0.1
        fixed, values = retained_constraints(rest, original, targets, [1], columns=5)
        np.testing.assert_array_equal(fixed, [0, 5, 6, 7, 8, 9, 14])
        np.testing.assert_array_equal(values[np.searchsorted(fixed, original)], targets)
        np.testing.assert_array_equal(values[2:6], rest[6:10])

    def test_duplicate_rows_are_idempotent_and_inputs_unchanged(self):
        rest = np.arange(15 * 3, dtype=float).reshape(15, 3)
        original = np.array([0, 14])
        targets = rest[original] + 0.1
        saved = [x.copy() for x in (rest, original, targets)]
        a = retained_constraints(rest, original, targets, [1], columns=5)
        b = retained_constraints(rest, original, targets, [1, 1], columns=5)
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)
        for x, y in zip((rest, original, targets), saved, strict=True):
            np.testing.assert_array_equal(x, y)

    def test_invalid_row_fails(self):
        rest = np.zeros((15, 3))
        for rows in ([], [-1], [3]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                retained_constraints(rest, np.array([0]), rest[:1], rows, columns=5)


if __name__ == "__main__":
    unittest.main()
