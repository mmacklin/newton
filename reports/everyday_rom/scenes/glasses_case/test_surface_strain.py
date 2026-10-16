# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent principal-stretch regression tests."""

import unittest

import numpy as np
from audit_surface_strain import rest_metric, stretches


class SurfaceStrainTests(unittest.TestCase):
    def test_rigid_motion_and_known_scaling(self):
        points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        inverse, _ = rest_metric(points, faces)
        rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(stretches(points @ rotation + 7, faces, inverse), 1)
        np.testing.assert_allclose(stretches(points * [2, 0.5, 1], faces, inverse), [[0.5, 2]])

    def test_sliver_distortion_not_hidden_by_edge_lengths(self):
        points = np.array([[0.0, 0, 0], [1, 0, 0], [0.5, 0.001, 0]])
        faces = np.array([[0, 1, 2]])
        inverse, _ = rest_metric(points, faces)
        moved = points * [1, 3, 1]
        np.testing.assert_allclose(stretches(moved, faces, inverse), [[1, 3]])
        lengths = np.linalg.norm(points[[1, 2, 0]] - points[[0, 1, 2]], axis=1)
        deformed = np.linalg.norm(moved[[1, 2, 0]] - moved[[0, 1, 2]], axis=1)
        self.assertLess(np.max(np.abs(deformed / lengths - 1)), 0.0001)

    def test_degenerate_not_silently_excluded(self):
        with self.assertRaisesRegex(ValueError, "degenerate"):
            rest_metric(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), np.array([[0, 1, 2]]))


if __name__ == "__main__":
    unittest.main()
