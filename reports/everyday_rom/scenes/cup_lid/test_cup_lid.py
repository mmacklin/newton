# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests and staged-trajectory regression checks."""

import copy
import json
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from baseline import STAGE, digest, supported, write_json
from cup_lid import check_physics, check_trace, strain_operator, surface_strain, world_points

from newton.examples.basic._reduced_elastic_cap import build_cap_rom
from reports.everyday_rom.rendering.render_strain_replay import validate_scene


class TestGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vertices, cls.faces, cls.basis, cls.audit, cls.mid, _, _, cls.sector, cls.depth = build_cap_rom(
            supported.Example.Config()
        )

    def test_closed_connected_lid(self):
        edges, neighbors = Counter(), [set() for _ in self.vertices]
        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                edges[tuple(sorted((u, v)))] += 1
                neighbors[u].add(v)
                neighbors[v].add(u)
        self.assertTrue(all(count == 2 for count in edges.values()))
        seen, todo = set(), [0]
        while todo:
            v = todo.pop()
            if v not in seen:
                seen.add(v)
                todo.extend(neighbors[v] - seen)
        self.assertEqual(len(seen), len(self.vertices))

    def test_mass_and_clamped_annulus(self):
        phi = np.asarray(self.basis.sample_phi)
        p = phi.transpose(0, 2, 1).reshape(-1, 4)
        mass = np.repeat(self.basis.sample_mass, 3)
        np.testing.assert_allclose(p.T @ (mass[:, None] * p), np.diag(self.basis.mode_mass), atol=1e-12, rtol=2e-6)
        np.testing.assert_allclose(phi[: len(self.mid)][(self.sector < 0) | (self.depth == 0)], 0, atol=1e-12)
        self.assertTrue(np.all(np.asarray(self.basis.mode_stiffness) > 0))
        self.assertLess(self.audit["snapshot_max_vertex_error_m"], 0.0015)

    def test_principal_strain_is_rigid_invariant(self):
        inverse, _ = strain_operator(self.vertices, self.faces)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        moved = self.vertices.astype(float) @ rotation.T + [0.02, -0.04, 0.1]
        self.assertLess(surface_strain(moved, self.faces, inverse).max(), 2e-5)
        stretched = self.vertices.astype(float) * 1.2
        np.testing.assert_allclose(surface_strain(stretched, self.faces, inverse), 0.2, atol=2e-5)


class TestTraceGuard(unittest.TestCase):
    def trace(self):
        return {
            "cursor": np.asarray(4),
            "times": np.arange(4) / 240,
            "global_raw": np.full(4, 10),
            "body_raw": np.tile([10, 0], (4, 1)),
            "collisions": np.ones(4),
            "solves": np.ones(4),
            "body_overflow": np.zeros(4),
            "body_q": np.zeros((4, 2, 7)),
            "body_qd": np.zeros((4, 2, 6)),
            "joint_q": np.zeros((4, 18)),
            "joint_qd": np.zeros((4, 16)),
            "solver_metrics": np.zeros((4, 5)),
        }

    def test_valid_trace(self):
        self.assertTrue(check_trace(self.trace(), 4, 4096, 1024, 1 / 240)["passed"])

    def test_each_guard_rejects_corruption(self):
        for name, index, value in (
            ("cursor", (), 3),
            ("collisions", 1, 0),
            ("solves", 2, 2),
            ("global_raw", 0, 4097),
            ("body_raw", (0, 0), 1025),
            ("body_overflow", 3, 1),
            ("times", 1, 0.2),
            ("joint_q", (0, 8), np.nan),
        ):
            with self.subTest(name=name):
                data = self.trace()
                data[name][index] = value
                with self.assertRaises(AssertionError):
                    check_trace(data, 4, 4096, 1024, 1 / 240)


@unittest.skipUnless((STAGE / "integration_manifest.json").exists(), "Run the GPU audit/export first")
class TestStagedBundle(unittest.TestCase):
    def test_source_and_artifact_hashes(self):
        manifest = json.loads((STAGE / "integration_manifest.json").read_text())
        for name, expected in manifest["sha256"].items():
            with self.subTest(artifact=name):
                self.assertEqual(digest(STAGE / name), expected)
        for name in ("baseline.py", "cup_lid.py", "test_cup_lid.py", "README.md"):
            self.assertEqual(digest(STAGE / "source" / name), digest(Path(__file__).with_name(name)))
        baseline = json.loads((STAGE / "baseline/performance.json").read_text())
        self.assertEqual(digest(STAGE / "baseline/baseline.py"), baseline["baseline_source_sha256"])

    def test_existing_gallery_schema_and_exact_recovery(self):
        meta = json.loads((STAGE / "scene.json").read_text())
        arrays = np.load(STAGE / "scene.npz")
        self.assertEqual(validate_scene(meta, arrays), 571)
        basis = np.load(STAGE / "basis.npz")
        saved = np.load(STAGE / "trajectory.npz")
        local = basis["rest_vertices"][None] + np.einsum("vmc,fm->fvc", basis["phi"], saved["modal_q"])
        recovered = world_points(local, saved["body_q"][:, 0])
        np.testing.assert_array_equal(recovered, arrays["vertices_0"])
        self.assertTrue(np.any(np.abs(recovered[240] - recovered[0]) > 0.003))
        # Negative test: visual amplification must fail exact state recovery.
        altered = recovered.copy()
        altered[240] += 0.001
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(altered, arrays["vertices_0"])

    def test_all_substeps_and_negative_control(self):
        for directory, contact in ((STAGE, True), (STAGE / "negative-control", False)):
            with self.subTest(contact=contact):
                audit = np.load(directory / "substep-audit.npz")
                capacity = json.loads((directory / "contact_capacity.json").read_text())
                self.assertEqual(capacity["global_capacity"], 1000)
                self.assertEqual(capacity["body_capacity"], 1024)
                self.assertTrue(
                    check_trace(audit, 2280, capacity["global_capacity"], capacity["body_capacity"], 1 / 240)["passed"]
                )
                metrics = dict(np.load(directory / "substep_metrics.npz"))
                physical = check_physics(metrics, contact)
                self.assertTrue(physical["passed"])
                self.assertEqual(physical["principal_strain_screen"]["midsurface_passed"], not contact)
                self.assertEqual(physical["principal_strain_screen"]["complete_skin_passed"], not contact)
                if contact:
                    altered = copy.deepcopy(metrics)
                    altered["max_deformation_m"][:] = 0
                    with self.assertRaises(AssertionError):
                        check_physics(altered, True)
                    altered = copy.deepcopy(metrics)
                    altered["penetration_m"][1] = 0.01
                    with self.assertRaises(AssertionError):
                        check_physics(altered, True)

    def test_benchmark_is_clean_and_matching(self):
        metrics = json.loads((STAGE / "metrics.json").read_text())
        perf = json.loads((STAGE / "baseline/performance.json").read_text())
        self.assertTrue(metrics["clean_vs_audit_state_check_passed"])
        self.assertFalse(perf["solver_metrics_enabled"])
        self.assertFalse(perf["diagnostic_gpu_work"])
        self.assertEqual(len(perf["trials"]), 3)
        self.assertEqual(len(perf["kernel_profile"]), 6)
        for phase in perf["kernel_profile"]:
            self.assertTrue(phase["matched_state_check_passed"])
            names = " ".join(row["name"] for row in phase["activities"])
            self.assertNotIn("record_collision", names)
            self.assertNotIn("record_solve", names)
            self.assertIn("solve_elastic", names)

    def test_prescribed_straw_is_the_only_motion_source(self):
        data = np.load(STAGE / "substep-audit.npz")
        cfg = supported.Example.Config()
        expected = np.asarray(
            [
                np.asarray(
                    supported.straw_pose(
                        float(t),
                        cfg.lid_z,
                        cfg.straw_half_height,
                        cfg.offset,
                        cfg.stir,
                        float(np.deg2rad(cfg.tilt_degrees)),
                        cfg.stir_radius,
                    )
                )
                for t in data["times"]
            ]
        )
        np.testing.assert_allclose(data["body_q"][:, 1], expected, atol=2e-7)
        fixed = np.array([0, 0, cfg.lid_z, 0, 0, 0, 1])
        np.testing.assert_allclose(data["body_q"][:, 0], np.broadcast_to(fixed, expected.shape), atol=1e-7)
        np.testing.assert_allclose(data["body_qd"][:, 0], 0, atol=1e-7)
        negative = np.load(STAGE / "negative-control/trajectory.npz")
        self.assertLess(np.abs(negative["modal_q"]).max(), 1e-5)

    def test_overview_camera_frames_full_scene(self):
        meta = json.loads((STAGE / "scene.json").read_text())
        arrays = np.load(STAGE / "scene.npz")
        camera = meta["camera"]
        origin = np.asarray(camera["position"])
        forward = np.asarray(camera["target"]) - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        tangent = np.tan(np.deg2rad(camera["fov"]) / 2)
        for item in meta["meshes"]:
            with self.subTest(mesh=item["name"]):
                points = arrays[item["vertices"]].reshape(-1, 3).astype(float) - origin
                depth = points @ forward
                self.assertGreater(depth.min(), 0)
                # The gallery uses a 4:3 image; require a five-percent border.
                self.assertLess(np.max(np.abs(points @ up) / (depth * tangent)), 0.95)
                self.assertLess(np.max(np.abs(points @ right) / (depth * tangent * 4 / 3)), 0.95)

    def test_reported_strain_peak_matches_independent_svd(self):
        basis = np.load(STAGE / "basis.npz")
        trace = np.load(STAGE / "substep-audit.npz")
        metrics = np.load(STAGE / "substep_metrics.npz")
        row = int(metrics["max_midsurface_principal_stretch_deviation"].argmax())
        rest, faces = basis["mid"], basis["mid_faces"]
        deformed = basis["rest_vertices"] + np.einsum("vmc,m->vc", basis["phi"], trace["joint_q"][row, 7:11])
        inverse, _ = strain_operator(rest, faces)
        strain = surface_strain(deformed[: len(rest)], faces, inverse)
        face = faces[int(strain.argmax())]
        reference_edges = (rest[face[1:]] - rest[face[0]]).T
        deformed_edges = (deformed[face[1:]] - deformed[face[0]]).T
        singular = np.linalg.svd(deformed_edges @ np.linalg.pinv(reference_edges), compute_uv=False)[:2]
        self.assertAlmostEqual(float(np.abs(singular - 1).max()), float(strain.max()), places=7)
        self.assertGreater(strain.max(), 0.15)


if __name__ == "__main__":
    program = unittest.main(exit=False, verbosity=2)
    result = program.result
    write_json(
        STAGE / "tests.json",
        {
            "passed": result.wasSuccessful(),
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
        },
    )
    manifest_path = STAGE / "integration_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["contract_tests"] = json.loads((STAGE / "tests.json").read_text())
        manifest["sha256"]["tests.json"] = digest(STAGE / "tests.json")
        write_json(manifest_path, manifest)
    raise SystemExit(not result.wasSuccessful())
