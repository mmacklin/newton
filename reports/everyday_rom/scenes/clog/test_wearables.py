# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU geometry/basis tests and exact surface recovery checks for wearables.

Run this file for CPU tests. Run each positive and --no-contact experiment
under a separate shared GPU flock, then run --validate-pair on the output.
"""

import argparse
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

import cap
import clog
import numpy as np
from wearable_geometry import (
    capsule_mesh,
    orthogonal_basis,
    transform_points,
    triangle_principal_stretches,
    triangle_reference,
)


def verify_export(output):
    """Reconstruct every exported elastic vertex from the original Newton state."""
    output = Path(output)
    meta = json.loads((output / "scene.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    for name, expected in metrics.get("source_sha256", {}).items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected, name
    worst = 0.0
    checked = 0
    minimum_render_area = float("inf")
    with (
        np.load(output / "scene.npz", allow_pickle=False) as scene,
        np.load(output / "state.npz", allow_pickle=False) as state,
        np.load(output / "basis.npz", allow_pickle=False) as basis,
    ):
        np.testing.assert_array_equal(scene["times"], state["times"])
        np.testing.assert_allclose(np.diff(scene["times"]), 1 / meta["fps"], atol=1e-12)
        q_start, body = int(state["modal_q_start"]), int(state["elastic_body"])
        m = basis["phi"].shape[1]
        for item in meta["meshes"]:
            points, faces = scene[item["vertices"]], scene[item["faces"]]
            assert points.dtype.kind == "f" and np.isfinite(points).all()
            assert faces.dtype.kind in "iu" and faces.ndim == 2 and faces.shape[1] == 3
            assert faces.min() >= 0 and faces.max() < points.shape[-2]
            if points.ndim == 3:
                assert len(points) == len(scene["times"])
            rest = points[0] if points.ndim == 3 else points
            triangle_reference(rest, faces)
            triangles = rest[faces].astype(np.float64)
            area = (
                np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
                / 2
            )
            minimum_render_area = min(minimum_render_area, float(area.min()))
            if "basis_vertex_start" not in item:
                continue
            start = item["basis_vertex_start"]
            ids = slice(start, start + item["basis_vertex_count"])
            rest, phi = basis["rest_vertices"][ids], basis["phi"][ids]
            for frame, q in enumerate(state["joint_q"]):
                local = rest + np.einsum("nmc,m->nc", phi, q[q_start : q_start + m])
                recovered = transform_points(local, state["body_q"][frame, body]).astype(np.float32)
                worst = max(worst, float(np.abs(points[frame] - recovered).max()))
                np.testing.assert_allclose(points[frame], recovered, atol=2e-7, rtol=0)
                checked += len(recovered)
        renderer = Path(__file__).with_name("render_scene.py")
        if renderer.exists():
            spec = importlib.util.spec_from_file_location("everyday_render_schema", renderer)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.validate_scene(meta, scene)
    return {
        "passed": True,
        "max_absolute_error_m": worst,
        "vertex_frames_checked": checked,
        "all_render_meshes_nondegenerate": True,
        "render_meshes_checked": len(meta["meshes"]),
        "minimum_render_reference_triangle_area_m2": minimum_render_area,
        "method": "All elastic component vertices reconstructed from saved body_q, joint_q, rest vertices, and exact runtime phi; no display amplification",
    }


def validate_pair(output):
    """Compare independently run contact-on/off experiments, then certify export."""
    output = Path(output)
    positive = json.loads((output / "metrics.json").read_text())
    negative = json.loads((output / "negative-control" / "metrics.json").read_text())
    assert positive["test_passed"] and negative["test_passed"]
    assert positive["graph_enabled"] and negative["graph_enabled"], "CUDA-graph evidence is required"
    assert positive["config"]["contact_enabled"] and not negative["config"]["contact_enabled"]
    expected_config = {**positive["config"], "contact_enabled": False}
    assert negative["config"] == expected_config
    assert positive["duration_s"] == negative["duration_s"]
    assert positive["indentation_m"] == negative["indentation_m"]
    assert positive["schedule_s"] == negative["schedule_s"]
    ratio = negative["hold_max_deformation_m"] / max(positive["hold_max_deformation_m"], 1e-12)
    assert ratio < 0.4, ratio
    assert negative["max_displacement_m"] < positive["bounds"]["minimum_deformation_m"]
    positive["negative_control_passed"] = True
    positive["negative_control"] = {
        "test_status": "passed",
        "max_deformation_m": negative["max_displacement_m"],
        "initial_drop_max_deformation_m": negative["initial_drop_max_deformation_m"],
        "hold_max_deformation_m": negative["hold_max_deformation_m"],
        "hold_deformation_ratio": ratio,
        "all_trajectory_peak_ratio": negative["max_displacement_m"] / positive["max_displacement_m"],
        "comparison_scope": "Pressing hold interval, separated from the identical gravity-driven initial drop transient",
        "positive_deformation_assertion_rejects_control": True,
        "driving_interaction": "Disable pressing-pad contact only; identical free fall, physical ground/support contact, mass, modes, damping, pad path and solver",
    }
    positive["surface_recovery"] = verify_export(output)
    negative["surface_recovery"] = verify_export(output / "negative-control")
    with (
        np.load(output / "state.npz") as loaded,
        np.load(output / "negative-control" / "state.npz") as control,
        np.load(output / "basis.npz") as basis,
        np.load(output / "negative-control" / "basis.npz") as control_basis,
    ):
        for key in basis.files:
            np.testing.assert_array_equal(basis[key], control_basis[key])
        pads = loaded["press_bodies"]
        np.testing.assert_array_equal(loaded["body_q"][:, pads], control["body_q"][:, pads])
        np.testing.assert_array_equal(loaded["body_q"][0], control["body_q"][0])
        before_press = loaded["times"] < positive["schedule_s"][0]
        np.testing.assert_allclose(loaded["body_q"][before_press], control["body_q"][before_press], atol=2e-5, rtol=0)
        np.testing.assert_allclose(loaded["joint_q"][before_press], control["joint_q"][before_press], atol=2e-5, rtol=0)
        positive["negative_control"]["matching_pre_press_drop_verified"] = True
        positive["negative_control"]["pre_press_max_pose_difference"] = float(
            np.abs(loaded["body_q"][before_press] - control["body_q"][before_press]).max()
        )
        positive["negative_control"]["pre_press_comparison_tolerance"] = 2e-5
        body = int(loaded["elastic_body"])
        poses = loaded["body_q"][:, body]
        frame_motion = float(np.linalg.norm(poses[:, :3] - poses[0, :3], axis=1).max())
        assert frame_motion > 0.008, frame_motion
        positive["max_rigid_frame_displacement_m"] = frame_motion
        assert positive["floating_frame_dynamic"] and positive["body_inverse_mass_kg_inv"] > 0
        start = int(loaded["modal_q_start"])
        count = basis["phi"].shape[1]
        error_q = loaded["joint_q"][-1, start : start + count] - control["joint_q"][-1, start : start + count]
        difference = np.einsum("nmc,m->nc", basis["phi"], error_q)
        recovery_error = float(np.linalg.norm(difference, axis=1).max())
        positive["final_error_vs_unloaded_control_m"] = recovery_error
        recovery_ratio = positive["final_displacement_m"] / positive["max_displacement_m"]
        assert recovery_ratio < 0.2, recovery_ratio
        positive["recovered_peak_deformation_fraction"] = 1 - recovery_ratio
        positive["gravity_equilibrium_note"] = (
            "Both bodies are free; pressing can change their final floor/support contacts. Final modal differences are reported, not assumed zero under different final gravity/contact loads. Recovery is tested against the loaded peak."
        )
        frames = loaded["joint_q"][:, start : start + count]
        rest, faces = basis["rest_vertices"], basis["faces"]
        tri = rest[faces]
        original = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        original_area = np.linalg.norm(original, axis=1)
        inverse_reference, quality = triangle_reference(rest, faces)
        maximum_stretch, minimum_stretch = 1.0, 1.0
        maximum_angle, minimum_area = 0.0, 1.0
        for amplitudes in frames:
            vertices = rest + np.einsum("nmc,m->nc", basis["phi"], amplitudes)
            tri = vertices[faces]
            normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            area = np.linalg.norm(normal, axis=1)
            cosine = np.einsum("nc,nc->n", normal, original) / (area * original_area)
            maximum_angle = max(maximum_angle, float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))).max()))
            minimum_area = min(minimum_area, float((area / original_area).min()))
            stretches = triangle_principal_stretches(vertices, faces, inverse_reference)
            maximum_stretch = max(maximum_stretch, float(stretches.max()))
            minimum_stretch = min(minimum_stretch, float(stretches.min()))
        assert maximum_angle < 45, maximum_angle
        assert minimum_area > 0.50, minimum_area
        principal_strain = max(maximum_stretch - 1, 1 - minimum_stretch)
        principal_bound = positive["bounds"].get("principal_strain", positive["bounds"]["edge_strain"])
        assert principal_strain < principal_bound, (minimum_stretch, maximum_stretch, principal_bound)
        positive["max_principal_stretch"] = maximum_stretch
        positive["min_principal_stretch"] = minimum_stretch
        positive["max_principal_strain"] = principal_strain
        positive["minimum_rest_triangle_quality"] = float(quality.min())
        positive["surface_strain_coverage"] = "Every triangle of every elastic material component; no exclusions"
        positive["max_face_normal_rotation_deg"] = maximum_angle
        positive["min_triangle_area_ratio"] = minimum_area
        positive["surface_validity_passed"] = True
        positive["surface_validity_bounds"] = {
            "max_face_rotation_deg": 45,
            "min_triangle_area_ratio": 0.5,
            "max_principal_strain": principal_bound,
        }
    (output / "metrics.json").write_text(json.dumps(positive, indent=2) + "\n")
    (output / "negative-control" / "metrics.json").write_text(json.dumps(negative, indent=2) + "\n")
    scene = json.loads((output / "scene.json").read_text())
    scientific_status = positive.get("scientific_status", "prototype")
    scene["status"] = "validated floating-body prototype" if scientific_status == "prototype" else scientific_status
    (output / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    return positive


class WearableGeometryTests(unittest.TestCase):
    def test_support_capsule_closed_surface_and_poles(self):
        radius, half_height = 0.0325, 0.075
        vertices, faces = capsule_mesh(radius, half_height)
        _, quality = triangle_reference(vertices, faces)
        self.assertGreater(quality.min(), 0.04)
        edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, 2)
        self.assertEqual(len(vertices) - len(unique) + len(faces), 2)
        np.testing.assert_array_equal(np.unique(faces), np.arange(len(vertices)))
        axis = np.zeros_like(vertices)
        axis[:, 1] = np.clip(vertices[:, 1], -half_height, half_height)
        np.testing.assert_allclose(np.linalg.norm(vertices - axis, axis=1), radius, atol=1e-14)
        triangles = vertices[faces]
        normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        center = triangles.mean(axis=1)
        center[:, 1] -= np.clip(center[:, 1], -half_height, half_height)
        self.assertTrue(np.all(np.einsum("fc,fc->f", normal, center) > 0))
        self.assertEqual(np.count_nonzero(np.isclose(vertices[:, 1], half_height + radius)), 1)
        self.assertEqual(np.count_nonzero(np.isclose(vertices[:, 1], -half_height - radius)), 1)

    def test_principal_stretch_detects_sliver_strain(self):
        vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0.5, 0.0001, 0]])
        faces = np.array([[0, 1, 2]])
        inverse, quality = triangle_reference(vertices, faces)
        moved = vertices * [1, 3, 1]
        stretches = triangle_principal_stretches(moved, faces, inverse)
        np.testing.assert_allclose(stretches[0], [3, 1], atol=1e-12)
        self.assertLess(quality[0], 0.001)
        edges = np.array([[0, 1], [1, 2], [2, 0]])
        ratio = np.linalg.norm(moved[edges[:, 1]] - moved[edges[:, 0]], axis=1) / np.linalg.norm(
            vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1
        )
        self.assertLess(np.abs(ratio - 1).max(), 1e-6)

    def test_geometry_and_basis(self):
        for make in (clog.make_spec, cap.make_spec):
            spec = make()
            with self.subTest(slug=spec["slug"]):
                vertices, faces, mass, region = spec["assembly"].finish()
                self.assertTrue(np.isfinite(vertices).all())
                self.assertTrue((mass > 0).all())
                self.assertGreater(len(faces), 5000)
                triangles = vertices[faces]
                area = np.linalg.norm(
                    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
                )
                self.assertGreater(area.min(), 1e-10)
                _, quality = triangle_reference(vertices, faces)
                self.assertGreater(quality.min(), 0.01)
                self.assertGreater(vertices[:, 2].min(), -0.001)
                raw = spec["raw_modes"] if "raw_modes" in spec else spec["modes"](vertices, region)
                basis, audit = orthogonal_basis(vertices, raw, mass, spec["frequencies"], 0.08, floating=True)
                self.assertLess(audit["max_mass_orthogonality_error"], 1e-6)
                self.assertTrue((basis.mode_mass > 0).all())
                self.assertTrue((basis.mode_stiffness > 0).all())
                self.assertTrue(audit["rigid_fields_removed"])
                self.assertLess(audit["max_modal_translation_coupling_kg"], 1e-7)
                if spec["slug"] == "clog":
                    self.assertGreater(basis.mode_count, 8)
                    self.assertGreater(np.ptp(basis.sample_phi[region == 0, 0, 2]), 0.2)
                gram = np.einsum("n,nic,njc->ij", mass, basis.sample_phi.astype(float), basis.sample_phi.astype(float))
                np.testing.assert_allclose(gram, np.diag(basis.mode_mass), rtol=2e-6, atol=1e-9)
                center = np.sum(mass[:, None] * vertices, axis=0) / mass.sum()
                torque = np.sum(mass[:, None, None] * np.cross((vertices - center)[:, None], basis.sample_phi), axis=0)
                np.testing.assert_allclose(torque, 0, atol=1e-8)

    def test_clog_perforations_are_topological_holes(self):
        v, f = clog.perforated_upper()
        edges = np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, 2)
        # An annulus with 13 additional through-holes, thickened to a closed
        # surface, has genus 14 and Euler characteristic 2 - 2 * 14 = -26.
        self.assertEqual(len(v) - len(unique) + len(f), -26)

    def test_clog_offset_skin_retriangulation(self):
        old_vertices, old_faces = clog.perforated_upper(repair=False)
        vertices, faces = clog.perforated_upper()
        np.testing.assert_array_equal(vertices, old_vertices)
        self.assertEqual(faces.shape, old_faces.shape)
        np.testing.assert_array_equal(np.unique(faces), np.arange(len(vertices)))
        half = len(vertices) // 2
        walls = np.any(old_faces < half, axis=1) & np.any(old_faces >= half, axis=1)
        np.testing.assert_array_equal(faces[walls], old_faces[walls])
        _, old_quality = triangle_reference(old_vertices, old_faces)
        _, quality = triangle_reference(vertices, faces)
        self.assertLess(old_quality.min(), 0.01, "Negative control must reproduce the old slivers")
        self.assertGreater(quality.min(), 0.01)
        areas = []
        for triangles in (old_faces, faces):
            points = vertices[triangles]
            areas.append(
                np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1).sum()
            )
        self.assertLess(abs(areas[1] / areas[0] - 1), 0.001)

    def test_cap_bill_snapshot_root_is_fixed_and_tip_is_flexible(self):
        a = np.linspace(-1.24, 1.24, 31)
        root = cap.brim_point(np.zeros_like(a), a)
        tip = cap.brim_point(np.ones_like(a), a)
        np.testing.assert_allclose(cap.modes(root, np.ones(len(root)))[:, :3], 0, atol=2e-6)
        self.assertGreater(np.linalg.norm(cap.modes(tip, np.ones(len(tip)))[:, 0], axis=1).min(), 0.99)

    def test_cap_bill_coordinate_round_trip(self):
        depth, angle = np.meshgrid(np.linspace(0, 1, 33), np.linspace(-1.24, 1.24, 65))
        points = cap.brim_point(depth.ravel(), angle.ravel())
        shape = cap.modes(points, np.ones(len(points)))[:, 0, 2]
        np.testing.assert_allclose(shape, depth.ravel() ** 2, atol=1e-6, rtol=0)

    def test_whole_crown_pod(self):
        spec = cap.make_spec()
        audit = spec["pod_audit"]
        self.assertEqual(audit["snapshot_count"], 30)
        self.assertEqual(audit["retained_modes"], 12)
        self.assertGreater(audit["retained_mass_weighted_energy"], 0.95)
        self.assertLess(audit["snapshot_weighted_rms_error_m"], 0.0025)
        self.assertLess(audit["snapshot_max_edge_strain"], audit["training_pose_strain_bound"])

    def test_cap_projected_membrane_stiffness(self):
        spec = cap.make_spec()
        matrices = spec["stiffness_matrices"]
        stiffness = matrices["membrane"] + matrices["bending"]
        rotation = matrices["rotation"]
        mass = rotation.T @ matrices["mass"] @ rotation
        projected = rotation.T @ stiffness @ rotation
        expected = np.diag(mass) * (2 * np.pi * np.asarray(spec["frequencies"])) ** 2
        np.testing.assert_allclose(projected, np.diag(expected), rtol=1e-8, atol=1e-7)
        self.assertGreater(np.linalg.eigvalsh(stiffness).min(), 0)
        self.assertGreater(np.linalg.eigvalsh(matrices["membrane"]).min(), -1e-8)
        self.assertGreater(np.linalg.eigvalsh(matrices["bending"]).min(), -1e-8)

    def test_cap_bounded_validation_metadata(self):
        spec = cap.make_spec()
        self.assertEqual(spec["status"], "validated bounded whole-crown contact/release/drop prototype")
        self.assertIn("5.37 mm", spec["description"])
        self.assertIn("Large floppy folds remain unvalidated", spec["description"])
        self.assertIn("not a woven-fabric model", spec["model"])
        self.assertNotIn("WIP", spec["model"])
        self.assertNotIn("not yet validated", spec["pod_audit"]["status"])
        self.assertEqual(spec["bounds"]["edge_strain"], 0.14)
        self.assertEqual(spec["bounds"]["principal_strain"], 0.14)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--validate-pair", type=Path)
    parser.add_argument("--cap-basis-audit", type=Path)
    args, remaining = parser.parse_known_args()
    if args.cap_basis_audit:
        spec = cap.make_spec()
        vertices, faces, mass, region = spec["assembly"].finish()
        args.cap_basis_audit.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.cap_basis_audit / "whole-crown-pod.npz",
            rest_vertices=vertices,
            faces=faces,
            sample_mass=mass,
            region=region,
            raw_modes=spec["raw_modes"],
            snapshots=spec["snapshots"],
        )
        (args.cap_basis_audit / "basis-audit.json").write_text(json.dumps(spec["pod_audit"], indent=2) + "\n")
        print(json.dumps(spec["pod_audit"], indent=2))
    elif args.validate_pair:
        print(json.dumps(validate_pair(args.validate_pair), indent=2))
    elif args.verify:
        print(json.dumps(verify_export(args.verify), indent=2))
    else:
        unittest.main(argv=[__file__, *remaining])
