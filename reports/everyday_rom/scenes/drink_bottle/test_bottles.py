# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU geometry tests and audit of saved, real Newton CUDA-graph runs.

CPU: python -m unittest discover -s reports/everyday_rom -p test_bottles.py
GPU: python reports/everyday_rom/bottles.py --slug drink-bottle --validate
     python reports/everyday_rom/bottles.py --slug mustard-bottle --validate
Use the environment and GPU flock from CONTRACT.md for both GPU commands.
"""

import hashlib
import json
import os
import unittest
from pathlib import Path

import numpy as np
from bottle_geometry import (
    SPECS,
    build_bottle_rom,
    build_shell,
    capsule_mesh,
    cavity_volume,
    ovalization_fields,
    signed_volume,
    triangle_reference,
    triangle_stretches,
)
from bottles import transform_points, verify_surface_recovery

import newton


class TestBottleGeometry(unittest.TestCase):
    def test_material_boundaries_follow_complete_rings(self):
        for spec in SPECS.values():
            with self.subTest(slug=spec.slug):
                shell = build_shell(spec)
                vertices, faces, materials = shell["vertices"], shell["faces"], shell["materials"]
                self.assertNotIn(3, materials, "Optional front labels must be absent")
                for lower, upper in ((0, 1), (1, 2)):
                    common = np.intersect1d(faces[materials == lower], faces[materials == upper])
                    self.assertEqual(len(common), 2 * shell["segments"])
                    self.assertEqual(float(np.ptp(vertices[common, 2])), 0.0)

    def test_closed_material_shell_open_cavity(self):
        for spec in SPECS.values():
            with self.subTest(slug=spec.slug):
                shell, _, audit = build_bottle_rom(spec)
                v, f = shell["vertices"], shell["faces"]
                edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
                _, counts = np.unique(edges, axis=0, return_counts=True)
                self.assertTrue(np.all(counts == 2))
                oriented = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
                self.assertEqual(len(np.unique(oriented, axis=0)), len(oriented))
                tri = v[f]
                area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
                self.assertGreater(float(area2.min()), 1e-11)
                self.assertGreater(signed_volume(v, f), 0)
                self.assertGreater(cavity_volume(v, shell), 200e-6)
                top_faces = f[np.all(v[f, 2] > v[:, 2].max() - 1e-7, axis=1)]
                self.assertEqual(len(top_faces), 2 * shell["segments"])
                self.assertGreater(np.linalg.norm(v[top_faces, :2].mean(axis=1), axis=1).min(), 0.002)
                self.assertGreater(audit["wall_mass_kg"], 0.015)

    def test_modes_are_local_and_mass_orthogonal(self):
        for spec in SPECS.values():
            with self.subTest(slug=spec.slug):
                shell, basis, _ = build_bottle_rom(spec)
                v, phi = shell["vertices"], basis.sample_phi
                fixed = (v[:, 2] <= spec.fixed_z) | (v[:, 2] >= spec.neck_z)
                self.assertEqual(float(np.abs(phi[fixed]).max()), 0.0)
                self.assertTrue(np.isfinite(phi).all())
                self.assertEqual(basis.mode_count, 4)
                gram = np.einsum("vmi,v,vni->mn", phi.astype(float), basis.sample_mass, phi.astype(float))
                np.testing.assert_allclose(gram, np.diag(basis.mode_mass), atol=1e-9, rtol=2e-6)
                n = shell["outer_count"]
                np.testing.assert_allclose(phi[:n], phi[n : 2 * n], atol=1e-7)
                self.assertTrue(np.all(basis.mode_stiffness > 0))
                self.assertTrue(np.all(basis.mode_damping > 0))

    def test_compensated_ring_bending_at_six_mm(self):
        for spec in SPECS.values():
            with self.subTest(slug=spec.slug):
                shell = build_shell(spec)
                field = ovalization_fields(shell, spec)[:, 0]
                vertices, faces = shell["vertices"], shell["faces"]
                reference = triangle_reference(vertices, faces)
                stretch = triangle_stretches(vertices + 0.006 * field, faces, reference)
                self.assertLess(float(np.max(abs(stretch - 1))), 0.075)
                n = shell["outer_count"]
                mid = 0.5 * (vertices[:n] + vertices[n : 2 * n]).reshape(-1, shell["segments"], 3)
                displacement = field[:n].reshape(-1, shell["segments"], 3)
                edge = np.roll(mid, -1, axis=1) - mid
                derivative = np.roll(displacement, -1, axis=1) - displacement
                # Only the in-plane ring displacement is projected; axial warping
                # contributes finite second-order stretch, measured above.
                first_order = np.einsum("rsi,rsi->rs", edge[:, :, :2], derivative[:, :, :2])
                self.assertLess(float(np.abs(first_order).max()), 1e-9)

    def test_triangle_principal_stretch_and_rigid_invariance(self):
        vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.3, 1.0, 0.0], [1.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        reference = triangle_reference(vertices, faces)
        rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        np.testing.assert_allclose(triangle_stretches(vertices @ rotation.T + 4, faces, reference), 1, atol=1e-14)
        stretched = vertices * [0.88, 1.12, 1.0]
        np.testing.assert_allclose(triangle_stretches(stretched, faces, reference), [[0.88, 1.12]] * 2, atol=1e-14)

    def test_floating_frame_mass_and_coupling_are_positive(self):
        for spec in SPECS.values():
            with self.subTest(slug=spec.slug):
                shell, basis, audit = build_bottle_rom(spec)
                mass = basis.sample_mass.astype(float)
                vertices = shell["vertices"].astype(float)
                center = np.average(vertices, weights=mass, axis=0)
                np.testing.assert_allclose(center, audit["center_of_mass_local_m"], atol=1e-10)
                self.assertAlmostEqual(float(mass.sum()), audit["wall_mass_kg"], places=8)
                relative = vertices - center
                rigid = np.zeros((len(vertices), 6, 3))
                rigid[:, :3] = np.eye(3)
                for i in range(3):
                    rigid[:, 3 + i] = np.cross(np.eye(3)[i], relative)
                fields = np.concatenate([rigid, basis.sample_phi], axis=1)
                gram = np.einsum("vmi,v,vni->mn", fields, mass, fields)
                np.testing.assert_allclose(gram[3:6, 3:6], audit["inertia_about_com_kg_m2"], atol=1e-10)
                normalized = gram / np.sqrt(np.outer(gram.diagonal(), gram.diagonal()))
                self.assertGreater(float(np.linalg.eigvalsh(normalized).min()), 0.05)

    def test_capsule_matches_analytic_contact(self):
        v, f = capsule_mesh(0.012, 0.020)
        p = v.astype(float).copy()
        p[:, 2] -= np.clip(p[:, 2], -0.020, 0.020)
        np.testing.assert_allclose(np.linalg.norm(p, axis=1), 0.012, atol=4e-9)
        self.assertGreater(signed_volume(v, f), 0)


class TestBottleExports(unittest.TestCase):
    def test_cuda_trajectories_and_surface_recovery(self):
        root = Path(os.environ.get("BOTTLE_TEST_ROOT", "output"))
        if not all((root / slug / "metrics.json").exists() for slug in SPECS):
            self.skipTest("Run both --validate CUDA demos first")
        for slug in SPECS:
            with self.subTest(slug=slug):
                output = root / slug
                meta = json.loads((output / "scene.json").read_text())
                metrics = json.loads((output / "metrics.json").read_text())
                for name, digest in metrics["source_sha256"].items():
                    self.assertEqual(hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest(), digest)
                if "render_only_refresh" in metrics:
                    self.assertTrue(metrics["render_only_refresh"]["physics_files_unchanged"])
                    for name, digest in metrics["render_only_refresh"]["physics_sha256"].items():
                        self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
                self.assertEqual(metrics["test_status"], "passed")
                self.assertTrue(metrics["cuda_graph"])
                self.assertEqual(metrics["elastic_body_count"], 1)
                self.assertTrue(metrics["test_passed"])
                self.assertTrue(metrics["negative_control_passed"])
                self.assertTrue(metrics["floating_frame_dynamic"])
                self.assertEqual(metrics["negative_control"]["test_status"], "passed")
                self.assertLess(metrics["negative_control"]["hold_deformation_ratio_to_contact"], 0.10)
                self.assertGreater(min(metrics["negative_control"]["contact_excess_indentation_per_side_m"]), 0.0035)
                self.assertGreater(min(metrics["hold_mean_indentation_per_side_m"]), 0.0035)
                self.assertLess(metrics["max_penetration_m"], 0.001)
                self.assertLess(metrics["release_max_deformation_m"], 0.0003)
                self.assertLess(metrics["max_edge_length_change"], 0.12)
                self.assertLess(metrics["max_principal_stretch_deviation"], 0.12)
                self.assertLess(metrics["max_surface_area_fraction_above_10pct"], 0.01)
                self.assertGreater(metrics["frame_vertical_range_m"], 0.045)
                self.assertGreaterEqual(metrics["airborne_drop_frames"], 3)
                self.assertGreater(metrics["ground_contact_frames"], 60)
                self.assertLess(metrics["settled_deformation_peak_to_peak_m"], 0.00015)
                self.assertLess(metrics["settled_max_frame_speed_m_s"], 0.025)
                self.assertTrue(
                    (
                        Path(os.environ.get("BOTTLE_PRIOR_ROOT", str(root)))
                        / slug
                        / "baseline-clamped"
                        / "trajectory.npz"
                    ).exists()
                )
                self.assertLess(verify_surface_recovery(output), 3e-8)
                self.assertLess(verify_surface_recovery(output / "negative-control"), 3e-8)
                for name, digest in metrics["render_artifact_sha256"].items():
                    self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
                with (
                    np.load(output / "scene.npz") as scene,
                    np.load(output / "trajectory.npz") as trajectory,
                    np.load(output / "basis.npz") as basis,
                    np.load(output / "surface_map.npz") as mapping,
                ):
                    np.testing.assert_allclose(scene["times"], trajectory["times"])
                    np.testing.assert_allclose(np.diff(scene["times"]), 1 / 60, atol=1e-12)
                    self.assertGreaterEqual(len(scene["times"]), 331)
                    self.assertLessEqual(len(scene["times"]), 481)
                    for mesh in meta["meshes"]:
                        v, f = scene[mesh["vertices"]], scene[mesh["faces"]]
                        self.assertTrue(np.isfinite(v).all())
                        self.assertEqual(v.shape[-1], 3)
                        self.assertEqual(f.shape[1], 3)
                        self.assertTrue(np.issubdtype(f.dtype, np.integer))
                        self.assertLess(int(f.max()), v.shape[-2])
                    body = int(trajectory["bottle_body"])
                    pose = trajectory["body_q"][:, body]
                    self.assertGreater(float(trajectory["body_mass"][body]), 0.015)
                    self.assertFalse(int(trajectory["body_flags"][body]) & int(newton.BodyFlags.KINEMATIC))
                    self.assertGreater(float(np.ptp(pose[:, 2])), 0.045)
                    self.assertGreater(float(np.max(np.abs(trajectory["body_qd"][:, body, 2]))), 0.2)
                    reference = triangle_reference(basis["rest_vertices"], basis["faces"])
                    peak_strain, peak_fraction = 0.0, 0.0
                    for q in trajectory["modal_q"]:
                        local = basis["rest_vertices"] + np.einsum("vmc,m->vc", basis["phi"], q)
                        stretches = triangle_stretches(local, basis["faces"], reference)
                        strain = np.max(abs(stretches - 1), axis=1)
                        peak_strain = max(peak_strain, float(strain.max()))
                        fraction = reference[1][strain > 0.10].sum() / reference[1].sum()
                        peak_fraction = max(peak_fraction, float(fraction))
                    self.assertLess(peak_strain, 0.12)
                    self.assertLess(peak_fraction, 0.01)
                    self.assertAlmostEqual(peak_strain, metrics["max_principal_stretch_deviation"], places=7)
                    self.assertAlmostEqual(peak_fraction, metrics["max_surface_area_fraction_above_10pct"], places=7)
                    for frame in [0, 160, len(scene["times"]) - 1]:
                        recovered = basis["rest_vertices"] + np.einsum(
                            "vmc,m->vc", basis["phi"], trajectory["modal_q"][frame]
                        )
                        recovered = transform_points(recovered, pose[frame])
                        for i in range(len(mapping.files)):
                            np.testing.assert_allclose(
                                scene[f"vertices_{i}"][frame], recovered[mapping[f"mesh_{i}_source_vertex"]], atol=3e-8
                            )
                    recovered_faces = np.concatenate(
                        [mapping[f"mesh_{i}_source_vertex"][scene[f"faces_{i}"]] for i in range(len(mapping.files))]
                    )
                    self.assertEqual(len(recovered_faces), len(basis["faces"]))
                    np.testing.assert_array_equal(np.unique(recovered_faces, axis=0), np.unique(basis["faces"], axis=0))
                    self.assertEqual(len(mapping.files), 3)
                    self.assertNotIn("printed-panel", [mesh["name"] for mesh in meta["meshes"]])

    def test_rejected_runs_fail_principal_stretch_gate(self):
        for slug in SPECS:
            output = Path(os.environ.get("BOTTLE_PRIOR_ROOT", "output")) / slug / "rejected-large-strain"
            if not (output / "trajectory.npz").exists():
                self.skipTest("Rejected-run audit artifacts are not available")
            with (
                self.subTest(slug=slug),
                np.load(output / "basis.npz") as basis,
                np.load(output / "trajectory.npz") as trajectory,
            ):
                local = basis["rest_vertices"] + np.einsum("vmc,m->vc", basis["phi"], trajectory["modal_q"][150])
                reference = triangle_reference(basis["rest_vertices"], basis["faces"])
                stretch = triangle_stretches(local, basis["faces"], reference)
                strain = np.max(abs(stretch - 1), axis=1)
                self.assertGreater(float(strain.max()), 0.25)
                self.assertGreater(float(reference[1][strain > 0.10].sum() / reference[1].sum()), 0.20)


if __name__ == "__main__":
    unittest.main()
