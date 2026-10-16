# SPDX-License-Identifier: Apache-2.0
import unittest

import numpy as np
from ordered_pod import fit_ordered_pod


class OrderedPODTests(unittest.TestCase):
    def data(self):
        rest = np.array([[0, 0, 0], [0, 0, 0.1], [0, 0, 0.2], [0, 0, 0.3]], dtype=float)
        s = np.linspace(0, 1, 13)
        shapes = np.repeat(rest[None], len(s), axis=0)
        shapes[:, 2:, 0] += np.sin(s[:, None]) * [0.1, 0.2]
        shapes[:, 2:, 2] += (np.cos(s[:, None]) - 1) * [0.1, 0.2]
        return rest, shapes, [0.1, 0.2, 0.3, 0.4], s

    def test_mass_orthonormal_and_full_rank_recovery(self):
        fit = fit_ordered_pod(*self.data(), 2, [0, 1])
        gram = np.einsum("n,nic,njc->ij", fit.mass, fit.phi, fit.phi)
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-13)
        self.assertLess(fit.maximum_reconstruction_error, 1e-14)
        np.testing.assert_array_equal(fit.phi[:2], 0)
        np.testing.assert_array_equal(fit.coefficients[0], 0)

    def test_interpolated_shape_matches_linear_vertex_interpolation(self):
        fit = fit_ordered_pod(*self.data(), 2, [0, 1])
        midpoint = (fit.progress[5] + fit.progress[6]) / 2
        actual = fit.recover(fit.interpolate(midpoint))
        expected = (fit.recover(fit.coefficients[5]) + fit.recover(fit.coefficients[6])) / 2
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        np.testing.assert_array_equal(actual[:2], fit.rest[:2])
        with self.assertRaises(ValueError):
            fit.interpolate(1.1)

    def test_rank_truncation_is_measured_not_hidden(self):
        fit = fit_ordered_pod(*self.data(), 1, [0, 1])
        self.assertGreater(fit.maximum_reconstruction_error, 0.01)
        self.assertLess(fit.retained_energy, 1)
        with self.assertRaises(ValueError):
            fit_ordered_pod(*self.data(), 3, [0, 1])

    def test_root_motion_is_rejected_not_projected_away(self):
        rest, shapes, masses, s = self.data()
        shapes[5, 1, 0] += 0.001
        with self.assertRaisesRegex(ValueError, "clamped"):
            fit_ordered_pod(rest, shapes, masses, s, 2, [0, 1])


if __name__ == "__main__":
    unittest.main()
