"""Regressions for the user's full-width fixed material-tangent requirement."""

import itertools
import json
import os
import unittest
from pathlib import Path

import numpy as np
from audit_clamped import polygon_crossings, triangle_radius_min, validate_root
from clamped_geometry import (
    NS,
    NX,
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
from peel_case_geometry import surface_strain_metrics

ROOT = Path(os.environ["CASE_HISTORY_ROOT"]) if os.environ.get("CASE_HISTORY_ROOT") else None


class ClampedEvidenceTests(unittest.TestCase):
    def test_arbitrary_modal_coefficients_leave_entire_root_strip_invariant(self):
        rest, _, _, basis = flap_basis()
        np.testing.assert_array_equal(basis.sample_phi[root_skin_ids()], 0.0)
        shell = shell_geometry()[0][0] + SHELL_CENTER
        rng = np.random.default_rng(519)
        for _ in range(20):
            points = rest + np.einsum("nmc,m->nc", basis.sample_phi[: len(rest)], rng.normal(size=basis.mode_count))
            validate_root(continuity(shell, points + ROOT_WORLD))

    def test_all_width_edges_checked_not_only_centerline(self):
        shell = shell_geometry()[0][0] + SHELL_CENTER
        flap = keyframe(0.8)[0] + ROOT_WORLD
        validate_root(continuity(shell, flap))
        broken = flap.copy()
        broken[[1, NX * NS + 1], 1] += 0.001
        with self.assertRaises(AssertionError):
            validate_root(continuity(shell, broken))

    def test_normal_mismatch_rejected_even_with_matching_midline(self):
        shell = shell_geometry()[0][0] + SHELL_CENTER
        flap = keyframe(0.4)[0] + ROOT_WORLD
        flap[0, 2] += 0.0004
        flap[NX * NS, 2] -= 0.0004
        with self.assertRaises(AssertionError):
            validate_root(continuity(shell, flap))

    @unittest.skipUnless(
        ROOT is not None and (ROOT.parent / "accepted/scene.npz").is_file(),
        "Optional historical trajectory; set CASE_HISTORY_ROOT to the c1-revision directory",
    )
    def test_previously_accepted_hinge_is_rejected(self):
        with np.load(ROOT.parent / "accepted/scene.npz") as scene:
            shell = scene["vertices_0"]
            flap = scene["vertices_3"]
        errors = [continuity(a, b, flap_ns=41) for a, b in zip(shell, flap, strict=True)]
        with self.assertRaises(AssertionError):
            validate_root(max(errors, key=lambda x: x["root_tangent_error_max_degrees"]))

    def test_finite_rotation_is_not_material_inversion(self):
        p, f, _ = keyframe(1.0)
        rest = keyframe(0.0)[0]
        self.assertGreater(surface_strain_metrics(rest, p, f)["reversed_surface_triangles"], 0)
        self.assertEqual(material_orientation(p)["material_inverted_cells"], 0)
        inverted = p.copy().reshape(2, NX, NS, 3)[::-1].reshape(-1, 3)
        self.assertGreater(material_orientation(inverted)["material_inverted_cells"], 0)

    def test_dense_keyframe_segment_interpolants_keep_strain_and_root(self):
        rest, f, _, basis = flap_basis()
        shell = shell_geometry()[0][0] + SHELL_CENTER
        phi = basis.sample_phi[: len(rest)]
        inverse = np.linalg.pinv(phi.transpose(0, 2, 1).reshape(-1, basis.mode_count).astype(float))
        path = np.stack([inverse @ (keyframe(t)[0] - rest).ravel() for t in np.linspace(0, 1, 25)])
        for a, b in itertools.pairwise(path):
            for fraction in (0.125, 0.375, 0.625, 0.875):
                points = rest + np.einsum("nmc,m->nc", phi, (1 - fraction) * a + fraction * b)
                validate_root(continuity(shell, points + ROOT_WORLD))
                strain = surface_strain_metrics(rest, points, f)
                self.assertLess(strain["max_abs_principal_strain"], 0.15)
                self.assertLess(strain["max_abs_area_strain"], 0.15)
                self.assertEqual(material_orientation(points)["material_inverted_cells"], 0)

    def test_full_fold_is_distributed(self):
        stats = curvature(keyframe(1.0)[0])
        self.assertGreater(stats["curvature_min_width_degrees"], 170)
        self.assertLess(stats["curvature_peak_six_segment_fraction"], 0.35)

    def test_polygon_self_intersection_negative(self):
        self.assertEqual(polygon_crossings(np.array([[0, 0], [1, 0], [1, 1], [0, 1]])), 0)
        self.assertGreater(polygon_crossings(np.array([[0, 0], [1, 1], [1, 0], [0, 1]])), 0)

    def test_triangle_interior_is_checked_not_only_vertices(self):
        triangles = np.array([[[-2, -2, 0], [2, -2, 0], [0, 2, 0]], [[2, 0, 0], [2, 1, 0], [2, 0, 1]]], float)
        np.testing.assert_allclose(triangle_radius_min(triangles), [0, 2])

    @unittest.skipUnless(
        ROOT is not None and (ROOT / "trial01/metrics.json").is_file(),
        "Optional historical failure evidence; set CASE_HISTORY_ROOT",
    )
    def test_failed_first_candidate_retains_failure(self):
        metrics = json.loads((ROOT / "trial01/metrics.json").read_text())
        self.assertFalse(metrics["test_passed"])
        self.assertGreater(metrics["max_abs_area_strain"], 0.15)


if __name__ == "__main__":
    unittest.main()
