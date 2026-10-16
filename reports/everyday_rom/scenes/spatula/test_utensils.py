# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU geometry checks and opt-in, flock-coordinated real CUDA tests.

Run this file with --simulation straw or --simulation spatula for the complete
390-frame contact and negative-control runs. GPU locking is then automatic.
"""

import argparse
import importlib.util
import io
import json
import os
import subprocess
import sys
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
        assert minor_diameter_px.min() > minimum_pixels, minor_diameter_px.min()
        return {
            "rest_tip_angle_deg_measured_from_mesh": measured_angle,
            "minimum_mouth_clear_radius_m": float(mouth_radius.min()),
            "minimum_projected_mouth_diameter_px_at_720p": float(minor_diameter_px.min()),
            "maximum_camera_frame_fraction_4_by_3": max_frame_fraction,
        }


def check_outputs(slug, straight=False, output=None):
    if output is None:
        output = Path("output") / slug
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", choices=("straw", "spatula"))
    parser.add_argument("--check-outputs", choices=("straw", "spatula"))
    parser.add_argument("--straight", action="store_true", help="Validate the optional straight-straw baseline")
    args, rest = parser.parse_known_args()
    if args.straight and (args.simulation or args.check_outputs) != "straw":
        parser.error("--straight requires --simulation straw or --check-outputs straw")
    if args.simulation:
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
                    *options,
                ],
                env=env,
                check=True,
            )
        check_outputs(args.simulation, args.straight)
    elif args.check_outputs:
        check_outputs(args.check_outputs, args.straight)
    else:
        unittest.main(argv=[sys.argv[0], *rest])
