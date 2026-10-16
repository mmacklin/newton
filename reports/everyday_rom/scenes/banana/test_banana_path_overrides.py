# SPDX-License-Identifier: Apache-2.0
"""A replacement keyframe cannot change topology or bypass the strain bound."""

import unittest

import numpy as np
from build_banana_path import apply_overrides


class SnapshotOverrideTests(unittest.TestCase):
    def setUp(self):
        self.rest = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
        self.faces = np.array([[0, 1, 2]])
        self.shapes = np.array([self.rest, self.rest])
        self.arrays = {"rest": self.rest.copy(), "faces": self.faces.copy(), "vertices": self.rest.copy()}

    def apply(self):
        return apply_overrides(
            0,
            self.rest,
            self.faces,
            [{"front": 0.35}],
            self.shapes,
            {(0, 0.35): {"arrays": self.arrays, "path": "candidate", "sha256": "test"}},
        )

    def test_replace_only_matching_pose(self):
        self.arrays["vertices"][:, 2] += 0.1
        shapes, used = self.apply()
        np.testing.assert_array_equal(shapes[0], self.rest)
        np.testing.assert_array_equal(shapes[1], self.arrays["vertices"])
        np.testing.assert_array_equal(self.shapes, [self.rest, self.rest])
        self.assertEqual(used[0]["front"], 0.35)

    def test_topology_and_rest_mismatch_rejected(self):
        for name in ("rest", "faces"):
            with self.subTest(name=name):
                original = self.arrays[name].copy()
                self.arrays[name].flat[0] += 1
                with self.assertRaisesRegex(ValueError, "rest mesh or topology"):
                    self.apply()
                self.arrays[name] = original

    def test_invalid_and_overstrained_geometry_rejected(self):
        self.arrays["vertices"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid snapshot"):
            self.apply()
        self.arrays["vertices"] = self.rest * 1.16
        with self.assertRaisesRegex(ValueError, "strain bound"):
            self.apply()


if __name__ == "__main__":
    unittest.main()
