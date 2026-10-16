# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU geometry checks and opt-in, flock-coordinated real CUDA tests.

Run this file with --simulation straw or --simulation spatula for the complete
390-frame contact and negative-control runs. GPU locking is then automatic.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import spatula
import straw
from utensil_geometry import (
    capsule_mesh,
    capsule_sdf,
    compare_negative_control,
    diagonalize_basis,
    lathe,
    run_cli,
    transform_vertices,
    verify_export,
)


def assert_closed(test, vertices, faces):
    test.assertTrue(np.isfinite(vertices).all())
    test.assertEqual(faces.shape[1], 3)
    triangles = vertices[faces].astype(float)
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    test.assertGreater(float(area.min()), 1e-12)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    unique, count = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    test.assertTrue((count == 2).all(), "Mesh is not a closed two-manifold")
    for a, b in unique:
        test.assertEqual(int(np.sum((edges[:, 0] == a) & (edges[:, 1] == b))), 1)
    volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    test.assertGreater(volume, 0)


class GeometryTests(unittest.TestCase):
    def test_straw_floating_basis_preserves_space_and_stiffness(self):
        cfg = straw.Config()
        vertices, _faces, basis, _audit = straw.build_straw(cfg)
        com = np.average(vertices.astype(float), weights=basis.sample_mass, axis=0)
        projected, audit = straw.project_rigid_fields(cfg, basis, com)
        reference, _ = project_straw_rigid_fields(cfg, basis, com)
        np.testing.assert_array_equal(projected.sample_phi, reference.sample_phi)
        np.testing.assert_array_equal(projected.sample_psi, reference.sample_psi)
        self.assertLess(audit["combined_span_max_error"], 2e-7)
        self.assertLess(audit["stiffness_transform_relative_error"], 2e-6)
        self.assertLess(audit["normalized_rigid_modal_mass_coupling"], 1e-6)
        np.testing.assert_array_equal(projected.sample_mass, basis.sample_mass)
        np.testing.assert_array_equal(projected.sample_points, basis.sample_points)
        anchor = np.array([0.0, 0.0, 0.020])
        phi, psi = straw.projected_root_samples(anchor, com, audit)
        index = projected.add_sample(anchor, phi=phi, psi=psi)
        np.testing.assert_allclose(projected.evaluate(anchor)[0], phi, atol=2e-8)
        np.testing.assert_allclose(projected.evaluate(anchor)[1], psi, atol=2e-6)
        self.assertEqual(projected.sample_mass[index], 0)
        self.assertGreater(np.linalg.norm(phi), 0.1)
        self.assertGreater(np.linalg.norm(psi), 1)

    def test_control_status_requires_both_assertion_suites(self):
        root = Path(__file__).resolve().parent
        root.mkdir(parents=True, exist_ok=True)
        for primary_passed, control_passed in ((False, False), (True, False), (False, True), (True, True)):
            with (
                self.subTest(primary=primary_passed, control=control_passed),
                tempfile.TemporaryDirectory(dir=root) as temp,
            ):
                output = Path(temp)
                (output / "negative-control").mkdir()
                for primary, passed in ((True, primary_passed), (False, control_passed)):
                    path = output if primary else output / "negative-control"
                    metrics = {
                        "slug": "spatula",
                        "config": {"contact_enabled": primary},
                        "frames": 1,
                        "dynamic_release": True,
                        "driven_body_indices": [0],
                        "release_time_s": 3.2,
                        "test_passed": passed,
                        "test_status": "passed" if passed else "not_run",
                        "max_bowl_contacts": int(primary),
                        "max_contacts": int(primary),
                        "max_deformation_m": 0.02 if primary else 0.002,
                        "loading_max_deformation_m": 0.02 if primary else 0.002,
                    }
                    (path / "metrics.json").write_text(json.dumps(metrics))
                    np.savez(path / "state.npz", body_q=np.zeros((1, 1, 7)), times=[0.0])
                (output / "scene.json").write_text(json.dumps({"status": "WIP"}))
                expected = primary_passed and control_passed
                if expected:
                    compare_negative_control(output)
                else:
                    with self.assertRaises(AssertionError):
                        compare_negative_control(output)
                result = json.loads((output / "metrics.json").read_text())
                self.assertEqual(result["test_passed"], expected)
                self.assertEqual(result["negative_control_passed"], expected)
                self.assertEqual(result["negative_control"]["bowl_contacts"], 0)
                self.assertEqual(
                    json.loads((output / "scene.json").read_text())["status"],
                    "validated prototype" if expected else "WIP",
                )

    def test_validation_rejects_skipped_assertions_before_warp_init(self):
        with (
            patch.object(sys, "argv", ["straw.py", "--no-test", "--validate"]),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            patch("utensil_geometry.wp.init", side_effect=AssertionError("GPU initialization must not run")) as init,
            self.assertRaises(SystemExit) as error,
        ):
            run_cli(straw.Example, straw.Config, "straw")
        self.assertEqual(error.exception.code, 2)
        self.assertIn("--no-test cannot be combined with --validate", stderr.getvalue())
        init.assert_not_called()

    def test_straw_is_hollow_closed_wall(self):
        cfg = straw.Config()
        v, f, basis, audit = straw.build_straw(cfg)
        assert_closed(self, v, f)
        s = straw.straw_sections(cfg)
        centers, tangent, _normal, _curvature = straw.centerline(cfg, s)
        sections = v.reshape(2, len(s), 40, 3)
        offset = sections - centers[None, :, None]
        radius = np.linalg.norm(offset, axis=-1)
        self.assertGreater(radius.min(), 0.0029)
        self.assertLess(radius.max(), 0.0046)
        self.assertGreater(np.ptp(radius[:, -1]), 0.001)
        np.testing.assert_allclose(np.einsum("snkc,nc->snk", offset, tangent), 0, atol=2e-8)
        tip_direction = sections[:, -1].mean(axis=(0, 1)) - sections[:, -2].mean(axis=(0, 1))
        tip_direction /= np.linalg.norm(tip_direction)
        self.assertAlmostEqual(float(np.rad2deg(np.arccos(tip_direction[2]))), 30, places=3)
        self.assertGreater(float(centers[-1, 0]), 0.019)
        # Both mouth boundaries are concentric, open, and perpendicular to the tip.
        for ring in sections[:, -1]:
            np.testing.assert_allclose(ring.mean(axis=0), centers[-1], atol=3e-8)
        self.assertEqual(basis.mode_count, 4)
        np.testing.assert_allclose(basis.sample_phi[v[:, 2] <= cfg.clamp_length], 0, atol=1e-6)
        self.check_basis(basis, audit)

    def test_curved_straw_ritz_kinematics(self):
        cfg = straw.Config()
        s = np.array([0.075, 0.13, 0.17, 0.19, 0.21])
        h = 1e-6
        derivative = (straw.centerline_modes(cfg, s + h) - straw.centerline_modes(cfg, s - h)) / (2 * h)
        omega = straw.rotation_modes(cfg, s)[0]
        tangent = straw.centerline(cfg, s)[1]
        np.testing.assert_allclose(derivative, np.cross(omega, tangent[:, None]), rtol=1e-8, atol=1e-7)
        np.testing.assert_allclose(np.einsum("nmc,nc->nm", derivative, tangent), 0, atol=1e-7)
        np.testing.assert_array_equal(straw.centerline_modes(cfg, [0, cfg.clamp_length]), np.zeros((2, 4, 3)))

    def test_straight_straw_baseline(self):
        cfg = straw.Config(tip_angle_degrees=0)
        vertices, faces, basis, audit = straw.build_straw(cfg)
        assert_closed(self, vertices, faces)
        np.testing.assert_allclose(
            audit["frequency_hz"],
            [3.46300377014036, 3.46300377014042, 33.8321019642960, 33.8321019642961],
            rtol=1e-7,
        )
        np.testing.assert_allclose(vertices.reshape(2, 49, 40, 3).mean(axis=(0, 2))[:, :2], 0, atol=1e-9)
        self.check_basis(basis, audit)

    def test_blade_is_tapered_closed_and_clamped(self):
        cfg = spatula.Config()
        v, f, basis, audit = spatula.build_blade(cfg)
        assert_closed(self, v, f)
        self.assertGreater(np.ptp(v[:, 1]), 0.060)
        self.assertLess(np.ptp(v[v[:, 0] == 0, 1]), 0.020)
        np.testing.assert_allclose(basis.sample_phi[v[:, 0] == 0], 0, atol=1e-6)
        self.assertEqual(basis.mode_count, 5)
        np.testing.assert_allclose(
            audit["frequency_hz"],
            [10.0623626790017, 24.6666554103554, 68.2281307768374, 75.5720799329823, 131.914359625332],
            rtol=1e-7,
        )
        self.check_basis(basis, audit)

    def check_basis(self, basis, audit):
        p = basis.sample_phi.transpose(0, 2, 1).reshape(-1, basis.mode_count).astype(float)
        gram = p.T @ (np.repeat(basis.sample_mass, 3)[:, None] * p)
        np.testing.assert_allclose(gram, np.diag(basis.mode_mass), atol=2e-9, rtol=1e-5)
        self.assertTrue((basis.mode_stiffness > 0).all())
        self.assertTrue((basis.mode_damping > 0).all())
        self.assertLess(audit["mass_orthogonality_relative_error"], 1e-10)

    def test_visuals_and_exact_capsule(self):
        a, b, radius = np.array([0.0, 0, 0]), np.array([0.08, 0.03, 0.02]), 0.007
        v, f = capsule_mesh(a, b, radius)
        assert_closed(self, v, f)
        np.testing.assert_allclose(capsule_sdf(v, a, b, radius), 0, atol=1e-8)
        v, f = lathe([(0.004, 0), (0.008, 0), (0.008, 0.006), (0.004, 0.006)], 32)
        assert_closed(self, v, f)
        assert_closed(self, *spatula.handle_mesh())

    def test_bowl_is_a_closed_concave_collision_shell(self):
        vertices, faces, profile = spatula.bowl_geometry(32)
        assert_closed(self, vertices, faces)
        np.testing.assert_allclose(spatula.bowl_distance(vertices, profile), 0, atol=2e-8)
        samples = np.array([[0.155, 0.025, 0.06], [0.205, 0.025, 0.10], [0.247, 0.025, 0.098], [0.180, 0.025, 0.015]])
        distance = spatula.bowl_distance(samples, profile)
        self.assertTrue((distance[:2] > 0).all(), "Bowl cavity must be empty")
        self.assertTrue((distance[2:] < 0).all(), "Rim and base must be solid")

    def test_spatula_release_changes_only_the_actuator(self):
        cfg = spatula.Config()
        before = np.asarray(spatula.grip_position(cfg.release_time, -1.0, cfg.release_time))
        left = np.asarray(spatula.grip_position(cfg.release_time + 0.5, -1.0, cfg.release_time))
        right = np.asarray(spatula.grip_position(cfg.release_time + 0.5, 1.0, cfg.release_time))
        self.assertGreater(left[2] - before[2], 0.08)
        self.assertAlmostEqual(float(right[1] - left[1]), 0.04, places=6)
        self.assertGreater(cfg.handle_mass, 0.02)
        self.assertEqual(cfg.substeps, 4)


def check_straw_camera_and_mouth(output):
    """Check the actual world export, including the tilted open mouth and framing."""
    metadata = json.loads((output / "scene.json").read_text())
    with (
        np.load(output / "scene.npz", allow_pickle=False) as scene,
        np.load(output / "rest_centerline.npz", allow_pickle=False) as centerline,
        np.load(output / "basis.npz", allow_pickle=False) as basis,
    ):
        n = len(centerline["arclength"])
        rest = basis["rest_vertices"].reshape(2, n, 40, 3)
        rest_centers = rest.mean(axis=(0, 2), dtype=np.float64)
        np.testing.assert_allclose(rest_centers, centerline["centers"], atol=3e-8)
        tip = rest_centers[-1] - rest_centers[-2]
        measured_angle = float(np.rad2deg(np.arccos(tip[2] / np.linalg.norm(tip))))
        np.testing.assert_allclose(measured_angle, metadata["rest_tip_angle_deg"], atol=0.001)
        vertices = scene["vertices_0"]
        sections = vertices.reshape(len(vertices), 2, n, 40, 3)
        mouth = sections[:, 1, -1]
        mouth_centers = mouth.mean(axis=1, dtype=np.float64)
        mouth_radius = np.linalg.norm(mouth - mouth_centers[:, None], axis=2)
        assert mouth_radius.min() > 0.0031, "Mouth opening collapsed"
        camera = metadata["camera"]
        origin = np.asarray(camera["position"])
        forward = np.asarray(camera["target"]) - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        scale = np.tan(np.deg2rad(camera["fov"]) / 2)

        def project(points):
            relative = np.asarray(points) - origin
            depth = relative @ forward
            assert depth.min() > 0, "Geometry behind camera"
            return np.stack((relative @ right, relative @ up), axis=-1) / (scale * depth[..., None])

        max_frame_fraction = 0.0
        for item in metadata["meshes"]:
            screen = project(scene[item["vertices"]])
            max_frame_fraction = max(max_frame_fraction, float((np.abs(screen) / [4 / 3, 1]).max()))
        assert max_frame_fraction < 0.9, max_frame_fraction
        screen = project(mouth)
        screen -= screen.mean(axis=1, keepdims=True)
        covariance = np.einsum("fki,fkj->fij", screen, screen) / screen.shape[1]
        # A sampled ellipse has variance r^2/2; use the short principal axis.
        minor_diameter_px = np.sqrt(2 * np.linalg.eigvalsh(covariance)[:, 0]) * 720
        minimum_pixels = 4.0 if metadata["rest_tip_angle_deg"] else 1.0
        visible_window = np.ones(len(vertices), dtype=bool)
        if (output / "mechanism.npz").exists():
            with np.load(output / "mechanism.npz") as mechanism:
                visible_window = mechanism["support_enabled"].astype(bool)
        assert minor_diameter_px[visible_window].min() > minimum_pixels, minor_diameter_px[visible_window].min()
        return {
            "rest_tip_angle_deg_measured_from_mesh": measured_angle,
            "minimum_mouth_clear_radius_m": float(mouth_radius.min()),
            "minimum_projected_mouth_diameter_px_at_720p": float(minor_diameter_px[visible_window].min()),
            "mouth_visibility_window": "Supported loading/unloading; a freely falling mouth may turn edge-on",
            "maximum_camera_frame_fraction_4_by_3": max_frame_fraction,
        }


def check_outputs(slug, straight=False, output=None):
    if output is None:
        output = straw.Example.default_output if slug == "straw" else Path("output") / slug
    if straight:
        output /= "straight-baseline"
    result = compare_negative_control(output)
    assert result["test_passed"] and result["negative_control_passed"], result
    assert result["graph_enabled"] and result["negative_control"]["deformation_ratio"] < 0.25
    verify_export(output)
    # Reuse the coordinator's exact schema validator if present, without RTX.
    path = Path(__file__).with_name("render_scene.py")
    if path.exists():
        spec = importlib.util.spec_from_file_location("everyday_scene_validator", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metadata = json.loads((output / "scene.json").read_text())
        with np.load(output / "scene.npz", allow_pickle=False) as arrays:
            module.validate_scene(metadata, arrays)
        assert 0 <= metadata["poster_frame"] < result["frames"]
    if slug == "straw":
        result["export_geometry_checks"] = check_straw_camera_and_mouth(output)
        if result.get("dynamic_release"):
            result["export_geometry_checks"].update(check_dynamic_straw_outputs(output))
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    elif slug == "spatula":
        result["export_geometry_checks"] = check_spatula_export(output)
        check_spatula_export(output / "negative-control")
        for filename, keys in (
            (
                "basis.npz",
                ("rest_vertices", "faces", "phi", "sample_mass", "mode_mass", "mode_stiffness", "mode_damping"),
            ),
            (
                "mechanism.npz",
                (
                    "body_mass",
                    "body_inertia",
                    "body_com",
                    "body_flags",
                    "grip_joint_enabled",
                    "bowl_vertices",
                    "bowl_faces",
                ),
            ),
        ):
            with np.load(output / filename) as positive, np.load(output / "negative-control" / filename) as negative:
                for key in keys:
                    np.testing.assert_array_equal(positive[key], negative[key])
        with np.load(output / "state.npz") as positive, np.load(output / "negative-control" / "state.npz") as negative:
            np.testing.assert_array_equal(positive["body_q"][0], negative["body_q"][0])
            np.testing.assert_array_equal(positive["joint_q"][0], negative["joint_q"][0])
        result["negative_control"]["basis_mass_inertia_and_initial_state_identical"] = True
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{slug}: saved surface recovery and renderer schema passed")


def check_dynamic_straw_outputs(output):
    """Independent recovery of a floating material root and finite-mass drop."""
    with (
        np.load(output / "state.npz") as state,
        np.load(output / "basis.npz") as basis,
        np.load(output / "mechanism.npz") as mechanism,
    ):
        body = int(state["elastic_body"])
        joint = int(mechanism["support_joint"])
        parent = int(mechanism["joint_parent"][joint])
        anchor = mechanism["joint_child_xform"][joint, :3]
        parent_anchor = mechanism["joint_parent_xform"][joint, :3]
        start = int(state["modal_q_start"])
        q = state["joint_q"][:, start : start + 4]
        held = mechanism["support_enabled"].astype(bool)
        assert held.any() and (~held).any()
        np.testing.assert_array_equal(held, state["times"] <= 3.8 + 1e-6)
        assert mechanism["body_mass"][body] > 0
        assert not int(mechanism["body_flags"][body]) & int(straw.newton.BodyFlags.KINEMATIC)
        weights, vertices = basis["sample_mass"].astype(float), basis["rest_vertices"].astype(float)
        mass = weights.sum()
        com = np.average(vertices, weights=weights, axis=0)
        r = vertices - com
        inertia = np.eye(3) * np.sum(weights * np.sum(r * r, axis=1)) - (r * weights[:, None]).T @ r
        np.testing.assert_allclose(mechanism["body_mass"][body], mass, rtol=1e-6)
        np.testing.assert_allclose(mechanism["body_com"][body], com, atol=1e-8)
        np.testing.assert_allclose(mechanism["body_inertia"][body], inertia, rtol=1e-5, atol=1e-11)
        root_error, root_angle, centers = [], [], []
        for i, poses in enumerate(state["body_q"]):
            centers.append(transform_vertices(com[None], poses[body])[0])
            if not held[i]:
                continue
            position = transform_vertices((anchor + q[i] @ mechanism["root_phi"])[None], poses[body])[0]
            expected = transform_vertices(parent_anchor[None], poses[parent])[0]
            root_error.append(float(np.linalg.norm(position - expected)))
            theta = q[i] @ mechanism["root_psi"]
            angle = np.linalg.norm(theta)
            rotation = (
                straw.wp.quat_from_axis_angle(straw.wp.vec3(*(theta / angle)), float(angle))
                if angle > 1e-10
                else straw.wp.quat_identity()
            )
            error = (
                straw.wp.quat_inverse(straw.wp.quat(*poses[parent, 3:])) * straw.wp.quat(*poses[body, 3:]) * rotation
            )
            root_angle.append(float(2 * np.arctan2(np.linalg.norm(np.asarray(error)[:3]), abs(error[3]))))
        centers = np.asarray(centers)
        drop = float(centers[np.flatnonzero(~held)[0], 2] - centers[-1, 2])
        assert max(root_error) < 0.0005 and max(root_angle) < 0.005
        assert drop > 0.07, drop
        if (output / "negative-control" / "mechanism.npz").exists():
            for filename, keys in (
                ("basis.npz", ("rest_vertices", "phi", "sample_mass", "mode_mass", "mode_stiffness", "mode_damping")),
                ("mechanism.npz", ("body_mass", "body_com", "body_inertia", "root_phi", "root_psi", "support_enabled")),
            ):
                with np.load(output / filename) as a, np.load(output / "negative-control" / filename) as b:
                    for key in keys:
                        np.testing.assert_array_equal(a[key], b[key])
            with np.load(output / "negative-control" / "state.npz") as control:
                for key in ("body_q", "body_qd", "joint_q", "joint_qd"):
                    np.testing.assert_array_equal(state[key][0], control[key][0])
        return {
            "reconstructed_root_joint_error_m": max(root_error),
            "reconstructed_root_joint_angle_error_rad": max(root_angle),
            "free_body_com_drop_m": drop,
            "dynamic_mass_inertia_and_transformed_root_verified": True,
            "matched_control_first_saved_state_verified": (output / "negative-control" / "mechanism.npz").exists(),
        }


def check_spatula_export(output):
    """Audit the dynamic mechanism independently from its original saved states."""
    metadata = json.loads((output / "scene.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    with (
        np.load(output / "state.npz", allow_pickle=False) as state,
        np.load(output / "mechanism.npz", allow_pickle=False) as mechanism,
        np.load(output / "scene.npz", allow_pickle=False) as scene,
    ):
        blade, handle = metrics["blade_body_index"], metrics["handle_body_index"]
        for body in (blade, handle):
            assert mechanism["body_mass"][body] > 0.02
            assert not mechanism["body_flags"][body] & int(spatula.newton.BodyFlags.KINEMATIC)
            assert np.linalg.eigvalsh(mechanism["body_inertia"][body]).min() > 0
        root = int(mechanism["root_joint"])
        assert mechanism["joint_parent"][root] == handle and mechanism["joint_child"][root] == blade
        poses, times = state["body_q"], state["times"]
        np.testing.assert_array_equal(mechanism["grip_joint_enabled"], times <= metrics["release_time_s"] + 1e-8)
        handle_mesh = next(part for part in metadata["meshes"] if part["name"] == "rigid_handle")
        exported = scene[handle_mesh["vertices"]]
        recovered = np.stack([transform_vertices(mechanism["handle_rest_vertices"], p[handle]) for p in poses])
        handle_error = float(np.abs(recovered - exported).max())
        assert handle_error < 5e-8, handle_error
        bowl_mesh = next(part for part in metadata["meshes"] if part["name"] == "ceramic_bowl")
        np.testing.assert_array_equal(scene[bowl_mesh["vertices"]], mechanism["bowl_vertices"])
        np.testing.assert_array_equal(scene[bowl_mesh["faces"]], mechanism["bowl_faces"])
        assert bool(mechanism["bowl_collision_enabled"]) == metrics["config"]["contact_enabled"]
        # Finite body motion must not be a duplicate of the prescribed jaw path.
        separation = np.linalg.norm(poses[-1, handle, :3] - poses[-1, mechanism["driven_bodies"][0], :3])
        assert separation > 0.05, separation
        origin = np.asarray(metadata["camera"]["position"])
        forward = np.asarray(metadata["camera"]["target"]) - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        scale = np.tan(np.deg2rad(metadata["camera"]["fov"]) / 2)
        frame_fraction = 0.0
        for part in metadata["meshes"]:
            relative = scene[part["vertices"]] - origin
            depth = relative @ forward
            assert depth.min() > 0
            screen = np.stack((relative @ right, relative @ up), axis=-1) / (scale * depth[..., None])
            frame_fraction = max(frame_fraction, float((np.abs(screen) / [4 / 3, 1]).max()))
        assert frame_fraction < 0.92, frame_fraction
        return {
            "rigid_handle_recovery_max_error_m": handle_error,
            "bowl_render_mesh_identical_to_collision_mesh": True,
            "dynamic_body_masses_and_inertias_positive": True,
            "release_flags_verified_from_original_state": True,
            "final_handle_to_grip_separation_m": float(separation),
            "maximum_camera_frame_fraction_4_by_3": frame_fraction,
        }


def project_straw_rigid_fields(cfg, basis, com):
    """Floating-frame gauge probe: project fields, never zero coupling integrals."""
    x = basis.sample_points.astype(float)
    weights = np.repeat(basis.sample_mass.astype(float), 3)
    phi = basis.sample_phi.transpose(0, 2, 1).reshape(-1, 4).astype(float)
    rigid = np.zeros((len(x), 3, 6))
    rigid[:, :, :3] = np.eye(3)
    rigid[:, :, 3:] = np.stack([np.cross(axis, x - com) for axis in np.eye(3)], axis=-1)
    rigid = rigid.reshape(-1, 6)
    frame_mass = rigid.T @ (weights[:, None] * rigid)
    gauge = np.linalg.solve(frame_mass, rigid.T @ (weights[:, None] * phi))
    projected = phi - rigid @ gauge
    reduced_mass = projected.T @ (weights[:, None] * projected)
    result, _audit = diagonalize_basis(
        x,
        projected.reshape(len(x), 3, 4).transpose(0, 2, 1),
        basis.sample_mass,
        np.diag(basis.mode_stiffness),
        cfg,
        "diagnostic_floating_curved_straw",
    )
    new_phi = result.sample_phi.transpose(0, 2, 1).reshape(-1, 4).astype(float)
    transform = np.linalg.solve(reduced_mass, projected.T @ (weights[:, None] * new_phi))
    stiffness = transform.T @ np.diag(basis.mode_stiffness) @ transform
    cross = rigid.T @ (weights[:, None] * new_phi)
    normalized = cross / np.sqrt(np.diag(frame_mass)[:, None] * result.mode_mass[None, :])
    # Recover actual Kirchhoff section rotations, including the root's new gauge rotation.
    sections = straw.straw_sections(cfg)
    centers = np.tile(np.repeat(straw.centerline(cfg, sections)[0], 40, axis=0), (2, 1))
    omega = np.tile(np.repeat(straw.rotation_modes(cfg, sections)[0], 40, axis=0), (2, 1, 1))
    center_modes = np.tile(np.repeat(straw.centerline_modes(cfg, sections), 40, axis=0), (2, 1, 1))
    raw = (center_modes + np.cross(omega, (x - centers)[:, None])).transpose(0, 2, 1).reshape(-1, 4)
    old_transform = np.linalg.solve(raw.T @ (weights[:, None] * raw), raw.T @ (weights[:, None] * phi))
    old_psi = np.einsum("nmc,mk->nkc", omega, old_transform)
    result.sample_psi[:] = np.einsum("nmc,mk->nkc", old_psi - gauge[3:].T[None], transform)
    # This exact damping congruence is generally dense; matched GPU probes use zero damping in BOTH bases.
    damping = transform.T @ np.diag(basis.mode_damping) @ transform
    audit = {
        "combined_span_max_error": float(np.abs(new_phi - projected @ transform).max()),
        "stiffness_transform_relative_error": float(
            np.linalg.norm(stiffness - np.diag(result.mode_stiffness)) / np.linalg.norm(stiffness)
        ),
        "normalized_rigid_modal_mass_coupling": float(np.abs(normalized).max()),
        "dense_transformed_damping_off_diagonal_fraction": float(
            np.linalg.norm(damping - np.diag(np.diag(damping))) / np.linalg.norm(damping)
        ),
        "root_phi_is_nonzero": bool(np.linalg.norm(result.sample_phi[x[:, 2] < cfg.clamp_length]) > 0),
        "root_psi_is_nonzero": bool(np.linalg.norm(result.sample_psi[x[:, 2] < cfg.clamp_length]) > 0),
        "full_stiffness_congruence_applied": True,
        "gauge_coefficients": gauge.tolist(),
        "modal_transform": transform.tolist(),
    }
    return result, audit


def straw_checkpoint(output):
    """Cache one pre-failure state so the reproducer does not need the full clip."""
    path = output / "checkpoint.npz"
    if not path.exists():
        with np.load(output.parent / "release-trial-02" / "state.npz") as state:
            np.savez_compressed(
                path,
                **{
                    key: state[key][250]
                    for key in ("body_q", "body_qd", "joint_q", "joint_qd", "times", "device_clock_s")
                },
            )
    with np.load(path) as checkpoint:
        return {key: checkpoint[key].copy() for key in checkpoint.files}


def diagnose_free_straw(case, cfg, basis, mass, com, inertia, output):
    """Separate rigid angular integration from the reduced inertial solve."""
    wp, newton = straw.wp, straw.newton
    wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", "/tmp/newton-everyday-utensils-warp")
    wp.init()
    wp.set_device("cuda:0")
    source = output / "source"
    source.mkdir(exist_ok=True)
    for name in ("straw.py", "spatula.py", "utensil_geometry.py", "test_utensils.py"):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    runtime = Path(newton.__file__).resolve().parent
    runtime_sources = [
        runtime / "_src/solvers/vbd/solver_vbd.py",
        runtime / "_src/solvers/vbd/reduced_elastic_kernels.py",
    ]
    provenance = {
        "runtime_module": str(runtime),
        "runtime_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in runtime_sources},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.glob("*.py")},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    state = straw_checkpoint(output)
    pose = state["body_q"][0].copy()
    velocity = state["body_qd"][0].copy()
    q, qd = state["joint_q"][:11].copy(), state["joint_qd"][:10].copy()
    np.savez_compressed(
        output / "reproducer-inputs.npz",
        body_q=pose,
        body_qd=velocity,
        joint_q=q,
        joint_qd=qd,
        rest_vertices=basis.sample_points,
        phi=basis.sample_phi,
        sample_mass=basis.sample_mass,
        mode_mass=basis.mode_mass,
        mode_stiffness=basis.mode_stiffness,
        mode_damping=basis.mode_damping,
        body_mass=mass,
        body_com=com,
        body_inertia=inertia,
    )
    original_world = transform_vertices(basis.sample_points, pose)
    projection_audit = None
    gravity = (0, 0, -9.81)
    if case in ("elastic-boost-zero", "elastic-boost-z"):
        gravity = (0, 0, 0)
        velocity[:3] = (0, 0, -1.0 if case == "elastic-boost-z" else 0.0)
        qd[:3] = velocity[:3]
    if case in ("elastic-rest-free", "elastic-projected-free"):
        if case == "elastic-projected-free":
            basis, projection_audit = project_straw_rigid_fields(cfg, basis, com)
        basis.mode_damping[:] = 0
        q[7:], qd[6:] = 0, 0
    if case == "elastic-com-free":
        # A change of reference origin, not a change in the physical body.
        pose[:3] += np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(*com)))
        q[:7] = pose
        basis = newton.ModalBasis(
            sample_points=basis.sample_points.astype(float) - com,
            sample_phi=basis.sample_phi,
            sample_psi=basis.sample_psi,
            sample_mass=basis.sample_mass,
            mode_mass=basis.mode_mass,
            mode_stiffness=basis.mode_stiffness,
            mode_damping=basis.mode_damping,
        )
        com = np.zeros(3)
    origin_change_error = float(np.abs(transform_vertices(basis.sample_points, pose) - original_world).max())
    assert origin_change_error < 2e-8
    np.savez_compressed(
        output / f"{case}-inputs.npz",
        body_q=pose,
        body_qd=velocity,
        joint_q=q,
        joint_qd=qd,
        rest_vertices=basis.sample_points,
        phi=basis.sample_phi,
        psi=basis.sample_psi,
        sample_mass=basis.sample_mass,
        mode_mass=basis.mode_mass,
        mode_stiffness=basis.mode_stiffness,
        mode_damping=basis.mode_damping,
        body_mass=mass,
        body_com=com,
        body_inertia=inertia,
        gravity=gravity,
    )
    builder = newton.ModelBuilder(gravity=gravity, up_axis="Z")
    options = {"xform": wp.transform(*pose), "mass": mass, "com": wp.vec3(*com), "inertia": wp.mat33(inertia)}
    elastic = case != "rigid-free"
    if elastic:
        builder.add_body_elastic(**options, modal_basis=basis, lock_inertia=True)
    else:
        builder.add_body(**options, lock_inertia=True)
    builder.color()
    model = builder.finalize()
    states = [model.state(), model.state()]
    for state in states:
        state.body_qd.assign(velocity[None])
        if elastic:
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
    solver = newton.solvers.SolverVBD(
        model, iterations=cfg.iterations, elastic_body_relaxation=cfg.elastic_relaxation, rigid_compliant_alm=False
    )
    diagnostic_override = None
    if case == "elastic-cross-mass-off":
        solver.elastic_implicit_mass_coupling.assign([False])
        diagnostic_override = "Implicit cross-mass Hessian disabled in memory; physically incomplete diagnostic only"
    elif case == "elastic-centrifugal-off":
        model.elastic_mode_coupling_centrifugal.zero_()
        diagnostic_override = "Centrifugal projection disabled in memory; physically incomplete diagnostic only"
    pipeline = newton.CollisionPipeline(model)
    contacts, control = pipeline.contacts(), model.control()
    solver.reset(states[0], flags=0)
    with wp.ScopedCapture() as capture:
        for _ in range(4):
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 1 / 240)
            states.reverse()
    records, poses, velocities, modes, mode_velocities = [], [], [], [], []
    for frame in range(75):
        wp.capture_launch(capture.graph)
        wp.synchronize()
        pose, velocity = states[0].body_q.numpy()[0], states[0].body_qd.numpy()[0]
        amplitude = states[0].joint_q.numpy()[7:] if elastic else np.zeros(4)
        local_omega = np.asarray(wp.quat_rotate_inv(wp.quat(*pose[3:]), wp.vec3(*velocity[3:])), dtype=float)
        record = {
            "time_s": (frame + 1) / 60,
            "angular_speed_rad_s": float(np.linalg.norm(velocity[3:].astype(float))),
            "rotational_energy_J": float(0.5 * local_omega @ inertia @ local_omega),
            "max_deformation_m": float(
                np.linalg.norm(
                    np.einsum("nmc,m->nc", basis.sample_phi.astype(float), amplitude.astype(float)), axis=1
                ).max()
            ),
            "contacts": int(contacts.rigid_contact_count.numpy()[0]),
        }
        records.append(record)
        poses.append(pose)
        velocities.append(velocity)
        modes.append(amplitude)
        mode_velocities.append(states[0].joint_qd.numpy()[6:] if elastic else np.zeros(4))
        if record["angular_speed_rad_s"] > 30 or record["max_deformation_m"] > 0.05:
            break
    np.savez_compressed(
        output / f"{case}-state.npz",
        body_q=poses,
        body_qd=velocities,
        modes=modes,
        mode_velocities=mode_velocities,
        times=[r["time_s"] for r in records],
    )
    return {
        "graph_enabled": True,
        "substeps": 4,
        "frames": records,
        "stable": len(records) == 75,
        "maximum_angular_speed_rad_s": max(r["angular_speed_rad_s"] for r in records),
        "maximum_deformation_m": max(r["max_deformation_m"] for r in records),
        "actual_implicit_mass_coupling": bool(solver.elastic_implicit_mass_coupling.numpy()[0]) if elastic else None,
        "coordinate_origin_world_geometry_error_m": origin_change_error,
        "diagnostic_override": diagnostic_override,
        "projection_audit": projection_audit,
        "matched_rest_zero_damping_probe": case in ("elastic-rest-free", "elastic-projected-free"),
        "gravity_m_s2": gravity,
    }


def compare_straw_boosts():
    """Compare intrinsic motion under a uniform inertial-frame velocity boost."""
    output = Path("output/straw/debug")
    with (
        np.load(output / "elastic-boost-zero-inputs.npz") as a,
        np.load(output / "elastic-boost-z-inputs.npz") as b,
    ):
        for key in a.files:
            if key not in ("body_qd", "joint_qd"):
                np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a["body_qd"][3:], b["body_qd"][3:])
        np.testing.assert_array_equal(a["joint_qd"][3:], b["joint_qd"][3:])
        np.testing.assert_array_equal(a["body_qd"][:3], [0, 0, 0])
        np.testing.assert_array_equal(b["body_qd"][:3], [0, 0, -1])
        phi = a["phi"].astype(float)
    with (
        np.load(output / "elastic-boost-zero-state.npz") as a,
        np.load(output / "elastic-boost-z-state.npz") as b,
    ):
        count = min(len(a["times"]), len(b["times"]))
        np.testing.assert_array_equal(a["times"][:count], b["times"][:count])
        mode_difference = b["modes"][:count].astype(float) - a["modes"][:count]
        deformation_error = np.linalg.norm(np.einsum("nmc,fm->fnc", phi, mode_difference), axis=2).max(axis=1)
        omega_error = np.linalg.norm(b["body_qd"][:count, 3:].astype(float) - a["body_qd"][:count, 3:], axis=1)
        translation_error = np.linalg.norm(
            b["body_q"][:count, :3].astype(float) - a["body_q"][:count, :3] - a["times"][:count, None] * [0, 0, -1],
            axis=1,
        )
        records = [
            {
                "time_s": float(a["times"][i]),
                "modal_surface_difference_m": float(deformation_error[i]),
                "angular_velocity_difference_rad_s": float(omega_error[i]),
                "boost_corrected_origin_difference_m": float(translation_error[i]),
            }
            for i in range(count)
        ]
    result = {
        "matched_inputs_except_uniform_translation": True,
        "gravity_m_s2": [0, 0, 0],
        "contacts_and_support_joints": 0,
        "runtime_overrides": None,
        "first_frame_surface_difference_m": records[0]["modal_surface_difference_m"],
        "first_frame_angular_velocity_difference_rad_s": records[0]["angular_velocity_difference_rad_s"],
        "galilean_check_passed": bool(
            records[0]["modal_surface_difference_m"] < 1e-5 and records[0]["angular_velocity_difference_rad_s"] < 1e-3
        ),
        "first_frame_tolerances": {"surface_m": 1e-5, "angular_velocity_rad_s": 1e-3},
        "records": records,
    }
    (output / "galilean-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def diagnose_straw(case):
    """CPU inertia audit or one lock-coordinated replay of a pre-failure state."""
    output = Path("output/straw/debug")
    cfg = straw.Config(floating_basis=False)
    vertices, _faces, basis, _audit = straw.build_straw(cfg)
    x, weights, phi = vertices.astype(float), np.asarray(basis.sample_mass, dtype=float), basis.sample_phi.astype(float)
    mass = weights.sum()
    com = (x * weights[:, None]).sum(axis=0) / mass
    r = x - com
    inertia = np.eye(3) * np.sum(weights * np.sum(r * r, axis=1)) - (r * weights[:, None]).T @ r
    jacobian = np.zeros((len(x), 3, 10))
    jacobian[:, :, :3] = np.eye(3)
    jacobian[:, :, 3:6] = np.stack([np.cross(axis, r) for axis in np.eye(3)], axis=-1)
    jacobian[:, :, 6:] = phi.transpose(0, 2, 1)
    generalized = np.einsum("n,nci,ncj->ij", weights, jacobian, jacobian)
    eig = np.linalg.eigvalsh(generalized)
    scale = np.sqrt(np.diag(generalized))
    result = {
        "case": case,
        "mass_kg": mass,
        "com_m": com.tolist(),
        "inertia_kg_m2": inertia.tolist(),
        "inertia_eigenvalues": np.linalg.eigvalsh(inertia).tolist(),
        "coupled_mass_eigenvalues": eig.tolist(),
        "coupled_mass_condition_number": float(eig[-1] / eig[0]),
        "normalized_mass_eigenvalues": np.linalg.eigvalsh(generalized / np.outer(scale, scale)).tolist(),
    }
    if case in (
        "rigid-free",
        "elastic-free",
        "elastic-com-free",
        "elastic-cross-mass-off",
        "elastic-centrifugal-off",
        "elastic-rest-free",
        "elastic-projected-free",
        "elastic-boost-zero",
        "elastic-boost-z",
    ):
        result.update(diagnose_free_straw(case, cfg, basis, mass, com, inertia, output))
    elif case == "mass":
        result["modal_mass_relative_error"] = float(
            np.linalg.norm(generalized[6:, 6:] - np.diag(basis.mode_mass)) / np.linalg.norm(basis.mode_mass)
        )
        assert eig.min() > 0 and result["modal_mass_relative_error"] < 1e-6
    else:
        if case == "floor-off":
            cfg.diagnostic_floor_enabled = False
        elif case == "frictionless":
            cfg.friction = 0.0
        straw.wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", "/tmp/newton-everyday-utensils-warp")
        straw.wp.init()
        straw.wp.set_device("cuda:0")
        example = straw.Example(cfg=cfg)
        result["model_com_error_m"] = float(np.linalg.norm(example.model.body_com.numpy()[0] - com))
        result["model_inertia_relative_error"] = float(
            np.linalg.norm(example.model.body_inertia.numpy()[0] - inertia) / np.linalg.norm(inertia)
        )
        np.testing.assert_allclose(example.model.body_mass.numpy()[0], mass, rtol=1e-6)
        np.testing.assert_allclose(example.model.body_com.numpy()[0], com, rtol=1e-5, atol=1e-9)
        np.testing.assert_allclose(example.model.body_inertia.numpy()[0], inertia, rtol=1e-5, atol=1e-12)
        model = example.model
        linear = model.elastic_mode_coupling_linear.numpy().astype(float).T
        angular = model.elastic_mode_coupling_angular.numpy().astype(float).T
        cross = np.vstack((linear, -(angular + np.cross(model.body_com.numpy()[0], linear.T).T)))
        runtime_mass = np.zeros_like(generalized)
        runtime_mass[:3, :3] = float(model.body_mass.numpy()[0]) * np.eye(3)
        runtime_mass[3:6, 3:6] = model.body_inertia.numpy()[0]
        runtime_mass[:6, 6:], runtime_mass[6:, :6] = cross, cross.T
        runtime_mass[6:, 6:] = np.diag(model.elastic_mode_mass.numpy())
        runtime_scale = np.sqrt(np.diag(runtime_mass))
        result.update(
            actual_implicit_mass_coupling=bool(example.solver.elastic_implicit_mass_coupling.numpy()[0]),
            runtime_mass_eigenvalues=np.linalg.eigvalsh(runtime_mass).tolist(),
            runtime_normalized_mass_eigenvalues=np.linalg.eigvalsh(
                runtime_mass / np.outer(runtime_scale, runtime_scale)
            ).tolist(),
            runtime_raw_spd_cutoff=1e-8 * max(float(np.diag(runtime_mass).max()), 1.0),
        )
        if case == "mass-model":
            output.mkdir(exist_ok=True)
            (output / f"{case}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            return result
        state = straw_checkpoint(output)
        for target in (example.state_0, example.state_1):
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                getattr(target, name).assign(state[name])
        example.clock.assign([state["device_clock_s"]])
        example.sim_time = float(state["times"])
        example.solver.reset(example.state_0, flags=0)
        result["frames"] = []
        for _ in range(75):
            example.step()
            record = dict(example.records[-1])
            pose = example.saved_body_q[-1][0]
            axis = np.asarray(straw.wp.quat_rotate(straw.wp.quat(*pose[3:]), straw.wp.vec3(0, 0, 1)))
            record["frame_tilt_deg"] = float(np.rad2deg(np.arccos(np.clip(axis[2], -1, 1))))
            result["frames"].append(record)
            if record["max_deformation_m"] > 0.05 or record["frame_angular_speed_rad_s"] > 30:
                break
        result["stable"] = len(result["frames"]) == 75
        result["maximum_deformation_m"] = max(r["max_deformation_m"] for r in result["frames"])
        result["maximum_angular_speed_rad_s"] = max(r["frame_angular_speed_rad_s"] for r in result["frames"])
    output.mkdir(exist_ok=True)
    (output / f"{case}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", choices=("straw", "spatula"))
    parser.add_argument("--check-outputs", choices=("straw", "spatula"))
    parser.add_argument("--output", type=Path, help="Optional simulation/check artifact directory")
    parser.add_argument(
        "--compare-boosts", action="store_true", help="CPU audit of the separately locked zero-gravity boost pair"
    )
    parser.add_argument("--straight", action="store_true", help="Validate the optional straight-straw baseline")
    parser.add_argument(
        "--diagnose-straw",
        choices=(
            "mass",
            "mass-model",
            "replay",
            "floor-off",
            "frictionless",
            "rigid-free",
            "elastic-free",
            "elastic-com-free",
            "elastic-cross-mass-off",
            "elastic-centrifugal-off",
            "elastic-rest-free",
            "elastic-projected-free",
            "elastic-boost-zero",
            "elastic-boost-z",
        ),
    )
    parser.add_argument(
        "--assert-stable", action="store_true", help="Fail if the diagnostic trajectory violates its stability bounds"
    )
    args, rest = parser.parse_known_args()
    if args.straight and (args.simulation or args.check_outputs) != "straw":
        parser.error("--straight requires --simulation straw or --check-outputs straw")
    if args.compare_boosts:
        diagnostic = compare_straw_boosts()
        if args.assert_stable:
            assert diagnostic["galilean_check_passed"], (
                "Uniform velocity boost changes intrinsic reduced-elastic dynamics"
            )
    elif args.diagnose_straw:
        diagnostic = diagnose_straw(args.diagnose_straw)
        if args.assert_stable:
            assert diagnostic.get("stable", False), "Free-body stability regression: " + str(args.diagnose_straw)
    elif args.simulation:
        source = Path(__file__).resolve().parent
        env = dict(
            os.environ,
            PYTHONPATH=str(Path(straw.newton.__file__).resolve().parents[1]),
            OPENBLAS_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
            WARP_CACHE_PATH="/tmp/newton-everyday-utensils-warp",
        )
        for options in ([], ["--no-contact"]):
            subprocess.run(
                [
                    "flock",
                    "/tmp/newton-everyday-rom-gpu.lock",
                    "timeout",
                    "5m",
                    sys.executable,
                    str(source / (args.simulation + ".py")),
                    *(["--straight"] if args.straight else []),
                    *(["--output", str(args.output)] if args.output else []),
                    *options,
                ],
                env=env,
                check=True,
            )
        check_outputs(args.simulation, args.straight, args.output)
    elif args.check_outputs:
        check_outputs(args.check_outputs, args.straight, args.output)
    else:
        unittest.main(argv=[sys.argv[0], *rest])
