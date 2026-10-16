# SPDX-License-Identifier: Apache-2.0
"""Analytic/finite-difference checks for passive modal keyframe coupling."""

import unittest

import numpy as np
from pod_path_energy import ModalPathEnergy


class PathEnergyTests(unittest.TestCase):
    def test_straight_path_has_no_tangential_penalty(self):
        path = ModalPathEnergy([[0, 0], [1, 0]], [0, 1], 100)
        value = path.evaluate([0.3, 0.2])
        self.assertAlmostEqual(value.energy, 2)
        np.testing.assert_allclose(value.gradient, [0, 20])
        np.testing.assert_allclose(value.hessian, [[0, 0], [0, 100]])
        self.assertAlmostEqual(value.progress, 0.3)

    def test_progress_stiffness_is_schur_complement_not_extra_inertia(self):
        path = ModalPathEnergy([[0, 0], [1, 0]], [0, 1], 100, 25)
        value = path.evaluate([0.3, 0.2])
        self.assertAlmostEqual(value.progress, 0.24)
        np.testing.assert_allclose(value.gradient, [6, 20])
        np.testing.assert_allclose(value.hessian, [[20, 0], [0, 100]])

    def test_gradient_hessian_match_finite_differences(self):
        path = ModalPathEnergy([[0, 0, 0], [0.7, 0.2, -0.1], [1.1, 0.9, 0.1]], [0, 0.4, 1], 73, 0.8)
        q = np.array([0.89, 0.55, 0.13])
        h = 1e-6
        analytic = path.evaluate(q)
        numerical_gradient = []
        numerical_hessian = []
        for axis in np.eye(3):
            plus, minus = path.evaluate(q + h * axis), path.evaluate(q - h * axis)
            numerical_gradient.append((plus.energy - minus.energy) / (2 * h))
            numerical_hessian.append((plus.gradient - minus.gradient) / (2 * h))
        np.testing.assert_allclose(analytic.gradient, numerical_gradient, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(analytic.hessian, np.array(numerical_hessian).T, rtol=1e-8, atol=1e-8)
        self.assertGreaterEqual(np.linalg.eigvalsh(analytic.hessian).min(), -1e-10)

    def test_endpoints_bound_progress_without_clamping_q(self):
        path = ModalPathEnergy([[0, 0], [1, 0]], [0, 1], 10)
        q = np.array([-0.2, 0.1])
        value = path.evaluate(q)
        self.assertEqual(value.progress, 0)
        np.testing.assert_allclose(value.gradient, [-2, 1])
        np.testing.assert_allclose(value.hessian, 10 * np.eye(2))
        np.testing.assert_array_equal(q, [-0.2, 0.1])
        self.assertEqual(path.evaluate([1.2, 0]).progress, 1)

    def test_global_segment_selection_and_zero_energy_on_path(self):
        path = ModalPathEnergy([[0, 0], [1, 0], [1, 1]], [0, 0.5, 1], 10)
        value = path.evaluate([1, 0.7])
        self.assertEqual(value.segment, 1)
        self.assertAlmostEqual(value.progress, 0.85)
        self.assertAlmostEqual(value.energy, 0)

    def test_energy_force_work_agree_within_segment(self):
        path = ModalPathEnergy([[0, 0], [1, 0.2]], [0, 1], 20, 0.5)
        q0, q1 = np.array([0.3, 0.3]), np.array([0.4, 0.35])
        a, b = path.evaluate(q0), path.evaluate(q1)
        self.assertAlmostEqual(b.energy - a.energy, np.dot(0.5 * (a.gradient + b.gradient), q1 - q0))

    def test_reject_invalid_path(self):
        for points, progress in (([[0], [0]], [0, 1]), ([[0], [1]], [0, 0]), ([[0], [np.nan]], [0, 1])):
            with self.assertRaises(ValueError):
                ModalPathEnergy(points, progress, 10)
        with self.assertRaises(ValueError):
            ModalPathEnergy([[0], [1]], [0, 1], -1)


if __name__ == "__main__":
    unittest.main()
