"""Material-outline regressions, also runnable against the frozen baseline."""

import unittest

import numpy as np
from clamped_geometry import (
    ROOT_WORLD,
    SHELL_CENTER,
    continuity,
    curvature,
    flap_basis,
    keyframe,
    material_orientation,
    root_skin_ids,
    shell_geometry,
)
from test_peel_case import check_mesh


class ScallopedGeometryTests(unittest.TestCase):
    def test_broad_root_concave_shoulders_and_rounded_center_tongue(self):
        mid = keyframe(0)[2]
        width = mid[-1, :, 0]
        s = mid[0, :, 2] / mid[0, -1, 2]
        np.testing.assert_array_equal(width[:2], [0.073, 0.073])
        self.assertGreater(width[s < 0.35].min(), 0.072)
        self.assertGreater(width[(s > 0.2) & (s < 0.74)].min(), 0.0749)
        self.assertLess(width[s > 0.94].max(), 0.027)
        self.assertLess(width[-1], 0.002)
        slope = np.diff(width) / np.diff(s)
        self.assertGreater(np.diff(slope)[(s[1:-1] > 0.86) & (s[1:-1] < 0.92)].min(), 0)
        self.assertLess(np.diff(slope)[s[1:-1] > 0.96].max(), 0)
        np.testing.assert_allclose(mid[:, :, 0], -mid[::-1, :, 0], atol=1e-15)

    def test_material_coordinates_are_identical_in_every_snapshot(self):
        rest, faces, mid = keyframe(0)
        for t in np.linspace(0, 1, 31):
            p, f, m = keyframe(t)
            np.testing.assert_array_equal(p[:, 0], rest[:, 0])
            np.testing.assert_array_equal(f, faces)
            check_mesh(p, f)
            self.assertEqual(material_orientation(p)["material_inverted_cells"], 0)
            np.testing.assert_allclose(
                np.linalg.norm(np.diff(m, axis=1), axis=2), np.linalg.norm(np.diff(mid, axis=1), axis=2), atol=1e-15
            )

    def test_curvature_measures_bending_not_fixed_planform(self):
        self.assertLess(curvature(keyframe(0)[0])["curvature_max_width_degrees"], 1e-6)
        closed = curvature(keyframe(1)[0])
        self.assertLess(closed["curvature_max_width_degrees"], 181)
        self.assertGreater(closed["curvature_min_width_degrees"], 170)

    def test_native_six_mode_basis_matches_geometry_and_fixed_c1(self):
        rest, _faces, _, basis = flap_basis()
        self.assertEqual(basis.mode_count, 6)
        np.testing.assert_array_equal(basis.sample_points[: len(rest)], rest)
        np.testing.assert_array_equal(basis.sample_phi[root_skin_ids()], 0)
        phi = basis.sample_phi[: len(rest)]
        np.testing.assert_array_equal(phi[:, :, 0], 0)
        inverse = np.linalg.pinv(phi.transpose(0, 2, 1).reshape(-1, 6).astype(float))
        shell = shell_geometry()[0][0] + SHELL_CENTER
        for t in np.linspace(0, 1, 41):
            target = keyframe(t)[0]
            q = inverse @ (target - rest).ravel()
            fitted = rest + np.einsum("nmc,m->nc", phi, q)
            self.assertLess(np.linalg.norm(target - fitted, axis=1).max(), 1e-6)
            c1 = continuity(shell, fitted + ROOT_WORLD)
            self.assertLess(c1["root_gap_max_m"], 1e-7)
            self.assertLess(c1["root_tangent_error_max_degrees"], 0.01)
        weights = basis.sample_mass
        np.testing.assert_allclose(weights.sum(), 0.018, rtol=1e-6)
        gram = np.einsum("n,nic,njc->ij", weights, basis.sample_phi, basis.sample_phi, dtype=np.float64)
        np.testing.assert_allclose(gram, np.eye(6), atol=2e-6)


if __name__ == "__main__":
    unittest.main()
