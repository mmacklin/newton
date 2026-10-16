# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU geometry/basis checks and independent exported-state recovery checks."""

import argparse
import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from brush_glove_geometry import (
    CapturedExample,
    Config,
    brush_handle,
    brush_head_rom,
    brush_tuft,
    capsule_y,
    coupled_mass_audit,
    ellipsoid,
    glove_rom,
    gravity_equilibrium,
    record_contact_peak,
    sampled_mass_properties,
    transform_points,
)
from toothbrush import Example

import newton
import newton.solvers


def check_mesh(test, p, f):
    test.assertTrue(np.isfinite(p).all())
    test.assertEqual(f.shape[1], 3)
    test.assertGreaterEqual(int(f.min()), 0)
    test.assertLess(int(f.max()), len(p))
    triangle = p[f]
    area2 = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1)
    test.assertGreater(float(area2.min()), 1e-15)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, return_counts=True, axis=0)
    test.assertTrue(np.all(counts == 2), "Mesh must be closed and manifold")
    volume = np.sum(np.einsum("ij,ij->i", triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2]))) / 6
    test.assertGreater(volume, 0, "Mesh winding must face outward")


class GeometryTests(unittest.TestCase):
    def test_contact_peak_catches_intermediate_overflow(self):
        with wp.ScopedDevice("cpu"):
            peak = wp.zeros(1, dtype=int)
            count = wp.zeros(1, dtype=int)
            for contacts in [100, 3826, 0]:
                count.assign(np.array([contacts], dtype=np.int32))
                wp.launch(record_contact_peak, 1, inputs=[count, peak])
            self.assertEqual(int(peak.numpy()[0]), 3826)
            example = CapturedExample()
            example.contact_peak = peak
            example.contacts = SimpleNamespace(rigid_contact_max=1000)
            with self.assertRaisesRegex(AssertionError, "Contact buffer overflow"):
                example.update_metrics()

    def test_dynamic_brush_topology_and_control(self):
        def skip_solver(example):
            example.state_0 = SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: np.asarray(example.builder.body_q)))

        with wp.ScopedDevice("cpu"), patch.object(Example, "finish", skip_solver):
            driven = Example(cfg=Config(contact_enabled=True))
            control = Example(cfg=Config(contact_enabled=False))
        self.assertEqual(len(driven.root_joints), 1)
        self.assertEqual(len(driven.elastic), 1)
        self.assertEqual(len(driven.head_parts), 22)
        self.assertEqual(driven.elastic[0]["basis"].mode_count, 40)
        builder = driven.builder
        for body in [driven.handle, *[item["body"] for item in driven.elastic]]:
            self.assertEqual(builder.body_flags[body], int(newton.BodyFlags.DYNAMIC))
            self.assertGreater(builder.body_mass[body], 0)
            self.assertTrue(np.all(np.linalg.eigvalsh(np.asarray(builder.body_inertia[body]).reshape(3, 3)) > 0))
        for joint, item in zip(driven.root_joints, driven.elastic, strict=True):
            self.assertEqual(builder.joint_parent[joint], driven.handle)
            self.assertEqual(builder.joint_child[joint], item["body"])
            self.assertEqual(builder.joint_type[joint], newton.JointType.FIXED)
            np.testing.assert_allclose(np.asarray(builder.joint_X_p[joint])[:3], item["root"], atol=1e-8)
            np.testing.assert_array_equal(item["source_phi"][: driven.head_parts[2]["start"]], 0)
        self.assertFalse(builder.joint_enabled[driven.grasp_joint])
        for body in driven.mechanism_bodies.numpy():
            self.assertEqual(builder.body_flags[body], int(newton.BodyFlags.KINEMATIC))
        additional = set(control.builder.shape_collision_filter_pairs) - set(builder.shape_collision_filter_pairs)
        expected = {tuple(sorted((item["shape"], control.tool_shape))) for item in control.elastic}
        self.assertEqual(additional, expected)
        np.testing.assert_allclose(builder.gravity, [0, 0, -9.81])

    def test_physical_head_mass_and_independent_fields(self):
        _, _, _, plate, plastic, root = brush_handle(split_head=True)
        p, f, basis, parts = brush_head_rom(plastic, plate, root)
        check_mesh(self, p, f)
        plate_count = parts[2]["start"]
        np.testing.assert_allclose(basis.sample_mass[:plate_count].sum(), 0.002, rtol=1e-6)
        np.testing.assert_array_equal(basis.sample_phi[:plate_count], 0)
        for j, part in enumerate(parts[2:]):
            row, col = divmod(j, 4)
            height = 0.0118 + (0.0004 if row in (0, 4) else 0.0) + (0.0003 if col in (0, 3) else 0)
            _, _, original = brush_tuft(height)
            start, end = part["start"], part["start"] + part["count"]
            phi = basis.sample_phi[start:end]
            np.testing.assert_array_equal(phi[:, 2 * j : 2 * j + 2], original.sample_phi)
            np.testing.assert_array_equal(phi[:, : 2 * j], 0)
            np.testing.assert_array_equal(phi[:, 2 * j + 2 :], 0)
            for key in ("mode_mass", "mode_stiffness", "mode_damping"):
                np.testing.assert_array_equal(getattr(basis, key)[2 * j : 2 * j + 2], getattr(original, key))
        mass, com, inertia = sampled_mass_properties(p, basis.sample_mass)
        audit = coupled_mass_audit(
            mass, com, inertia, basis.mode_mass, basis.mode_coupling_linear, basis.mode_coupling_angular
        )
        self.assertTrue(audit["runtime_spd_passed"], audit)
        self.assertLess(audit["normalized_condition_number"], 10)
        self.assertEqual(audit["dimension"], 46)
        without_plate = basis.sample_mass.copy()
        without_plate[:plate_count] = 0
        mass, com, inertia = sampled_mass_properties(p, without_plate)
        control = coupled_mass_audit(
            mass, com, inertia, basis.mode_mass, basis.mode_coupling_linear, basis.mode_coupling_angular
        )
        self.assertFalse(control["runtime_spd_passed"], "The physical plate mass must resolve the cutoff")

    def test_actual_head_implicit_coupling_enabled(self):
        def construct_solver_only(example):
            example.builder.color()
            example.model = example.builder.finalize(device="cpu")
            example.state_0 = example.model.state()
            example.solver = newton.solvers.SolverVBD(
                example.model,
                iterations=example.cfg.iterations,
                rigid_compliant_alm=False,
                rigid_contact_hard=False,
                **example.solver_options(),
            )

        with wp.ScopedDevice("cpu"), patch.object(Example, "finish", construct_solver_only):
            example = Example()
        audit = example.mass_audit()
        self.assertEqual(audit["actual_implicit_mass_coupling"], [True])
        self.assertTrue(audit["inertia_matches_samples_exact_float32"])
        self.assertFalse(audit["inertia_regularization_applied"])

    def test_brush_beam(self):
        p, f, basis = brush_tuft()
        check_mesh(self, p, f)
        self.assertEqual(basis.mode_count, 2)
        np.testing.assert_allclose(basis.sample_phi[p[:, 2] == 0], 0, atol=1e-8)
        self.assertTrue(np.all(basis.mode_stiffness > 0))
        self.assertTrue(np.all(basis.mode_mass > 0))
        gram = np.einsum(
            "v,vmc,vnc->mn",
            basis.sample_mass.astype(float),
            basis.sample_phi.astype(float),
            basis.sample_phi.astype(float),
        )
        np.testing.assert_allclose(gram, np.diag(basis.mode_mass), rtol=1e-5, atol=1e-10)

    def test_rigid_geometry(self):
        for p, f in [*brush_handle(), ellipsoid((0.02, 0.02, 0.02)), capsule_y(0.006, 0.018)]:
            check_mesh(self, p, f)

    def test_validation_requires_both_tests(self):
        example = CapturedExample()
        for positive, negative in [(False, False), (True, False), (False, True)]:
            with self.assertRaises(AssertionError):
                example.export(None, "validated prototype", test_passed=positive, negative_control_passed=negative)

    def test_glove_basis(self):
        p, f, basis = glove_rom()
        check_mesh(self, p, f)
        self.assertEqual(basis.mode_count, 9)
        np.testing.assert_allclose(basis.sample_phi[p[:, 1] < -0.055], 0, atol=1e-8)
        gram = np.einsum(
            "v,vmc,vnc->mn",
            basis.sample_mass.astype(float),
            basis.sample_phi.astype(float),
            basis.sample_phi.astype(float),
        )
        np.testing.assert_allclose(gram, np.diag(basis.mode_mass), rtol=1e-5, atol=1e-9)
        self.assertGreater(np.ptp(p[:, 1]), 0.24)
        self.assertGreater(np.ptp(p[:, 0]), 0.13)

    def test_glove_gravity_equilibrium(self):
        p, _, basis = glove_rom()
        q = gravity_equilibrium(basis, (0, 0, -9.81))
        gravity_force = basis.mode_coupling_linear.astype(float) @ np.array([0, 0, -9.81])
        np.testing.assert_allclose(basis.mode_stiffness * q, gravity_force, rtol=1e-6, atol=1e-9)
        disp = np.einsum("vmc,m->vc", basis.sample_phi, q)
        sag = float(np.linalg.norm(disp, axis=1).max())
        self.assertGreater(sag, 0.002)
        self.assertLess(sag, 0.004)
        np.testing.assert_allclose(disp[p[:, 1] < -0.055], 0, atol=1e-10)


def verify_export(directory):
    directory = Path(directory)
    scene = json.loads((directory / "scene.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    with (
        np.load(directory / "scene.npz", allow_pickle=False) as arrays,
        np.load(directory / "basis.npz", allow_pickle=False) as basis,
        np.load(directory / "state.npz", allow_pickle=False) as state,
        np.load(directory / "negative_state.npz", allow_pickle=False) as negative,
    ):
        np.testing.assert_allclose(arrays["times"], state["times"], atol=1e-12)
        assert len(arrays["times"]) == metrics["frames"]
        assert np.all(np.diff(arrays["times"]) > 0)
        max_error = 0.0
        max_surface_step = 0.0
        max_second_difference = 0.0
        max_edge_change = 0.0
        max_control_drift = 0.0
        max_held_control_drift = 0.0
        final_matched_difference = 0.0
        for i, mesh in enumerate(scene["meshes"]):
            vertices, faces = arrays[mesh["vertices"]], arrays[mesh["faces"]]
            assert vertices.dtype != object and faces.dtype.kind in "iu"
            assert np.isfinite(vertices).all()
            body = mesh["body_index"]
            rest = basis[f"rest_vertices_{i}"]
            if f"phi_{i}" in basis:
                phi, start = basis[f"phi_{i}"], mesh["q_start"]
                np.testing.assert_allclose(phi, basis[f"source_phi_{i}"], atol=1e-6)
                assert basis[f"sample_mass_{i}"].shape == (len(rest),)
                all_disp = np.einsum("vmc,fm->fvc", phi, state["joint_q"][:, start : start + phi.shape[1]])
                equilibrium = basis[f"equilibrium_q_{i}"]
                np.testing.assert_allclose(
                    basis[f"mode_stiffness_{i}"] * basis[f"analytic_equilibrium_q_{i}"],
                    basis[f"mode_coupling_linear_{i}"] @ np.asarray(metrics["gravity_m_s2"]),
                    rtol=1e-6,
                    atol=1e-9,
                )
                control_q = negative["joint_q"][:, start : start + phi.shape[1]]
                control_drift = np.einsum("vmc,fm->fvc", phi, control_q - equilibrium[None])
                max_control_drift = max(max_control_drift, float(np.linalg.norm(control_drift, axis=2).max()))
                held = (state["times"] > 2.5) & (state["times"] < 4.2)
                max_held_control_drift = max(
                    max_held_control_drift, float(np.linalg.norm(control_drift[held], axis=2).max())
                )
                matched = np.einsum(
                    "vmc,m->vc", phi, state["joint_q"][-1, start : start + phi.shape[1]] - control_q[-1]
                )
                final_matched_difference = max(final_matched_difference, float(np.linalg.norm(matched, axis=1).max()))
                max_surface_step = max(max_surface_step, float(np.linalg.norm(np.diff(all_disp, axis=0), axis=2).max()))
                max_second_difference = max(
                    max_second_difference, float(np.linalg.norm(np.diff(all_disp, n=2, axis=0), axis=2).max())
                )
                edges = np.unique(
                    np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0
                )
                rest_edge = rest[edges[:, 1]] - rest[edges[:, 0]]
                peak = int(np.argmax(np.linalg.norm(all_disp, axis=2).max(axis=1)))
                edge = rest_edge + all_disp[peak, edges[:, 1]] - all_disp[peak, edges[:, 0]]
                change = np.abs(np.linalg.norm(edge, axis=1) / np.linalg.norm(rest_edge, axis=1) - 1)
                max_edge_change = max(max_edge_change, float(change.max()))
            for frame in range(len(arrays["times"])):
                local = rest
                if f"phi_{i}" in basis:
                    phi, start = basis[f"phi_{i}"], mesh["q_start"]
                    local = rest + np.einsum("vmc,m->vc", phi, state["joint_q"][frame, start : start + phi.shape[1]])
                expected = local if body < 0 else transform_points(local, state["body_q"][frame, body])
                got = vertices if vertices.ndim == 2 else vertices[frame]
                max_error = max(max_error, float(np.max(np.abs(got - expected))))
        assert max_error < 1e-7, max_error
        assert metrics["graph_enabled"]
        assert metrics["test_passed"] and metrics["negative_control_passed"]
        dynamic = scene.get("dynamic_assembly", False)
        if dynamic:
            dynamic_result = verify_dynamic_assembly(scene, metrics, basis, state, negative)
            assert max_held_control_drift < 0.0001, max_held_control_drift
        else:
            dynamic_result = {}
            assert metrics["negative_control"]["deformation_ratio"] < 0.1
            assert max_control_drift < 1e-5, max_control_drift
        assert final_matched_difference < 0.0002, final_matched_difference
        assert max_surface_step < 0.001, max_surface_step
        assert max_second_difference < 0.001, max_second_difference
        result = {
            "export_recovery_max_error_m": max_error,
            "frames": len(arrays["times"]),
            "meshes": len(scene["meshes"]),
            "max_surface_step_m": max_surface_step,
            "max_surface_second_difference_m": max_second_difference,
            "max_edge_length_change_at_peak": max_edge_change,
            "negative_control_drift_from_gravity_equilibrium_m": max_control_drift,
            "held_no_tool_contact_displacement_m": max_held_control_drift,
            "final_contact_vs_control_displacement_m": final_matched_difference,
            "status": "passed",
            **dynamic_result,
        }
        source = directory / "source"
        source.mkdir(exist_ok=True)
        hashes = {}
        for name in scene["source_files"]:
            path = source / name
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if "source_sha256" in metrics:
            assert hashes == metrics["source_sha256"], "Source snapshot changed after simulation export"
        result["source_sha256"] = hashes
        (directory / "export_verification.json").write_text(json.dumps(result, indent=2) + "\n")
        metrics["export_verification"] = result
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return result


def verify_dynamic_assembly(scene, metrics, basis, state, negative):
    """Independently recover constraints and free flight from original model/state."""
    handle = metrics["handle_body_index"]
    by_body = {mesh["body_index"]: mesh for mesh in scene["meshes"] if mesh["q_start"] is not None}
    elastic = list(by_body.values())
    body_indices = [handle, *[mesh["body_index"] for mesh in elastic]]
    assert np.all(basis["model_body_flags"][body_indices] == int(newton.BodyFlags.DYNAMIC))
    assert np.all(basis["model_body_mass"][body_indices] > 0)
    if scene.get("single_physical_head", False):
        assert len(elastic) == 1 and len(body_indices) == 2
        assert len(metrics["root_joints"]) == 1
        assert metrics["mass_audit"]["actual_implicit_mass_coupling"] == [True]
        np.testing.assert_array_equal(basis["solver_elastic_implicit_mass_coupling"], [True])
        pieces = [(i, mesh) for i, mesh in enumerate(scene["meshes"]) if mesh["q_start"] is not None]
        points = np.concatenate([basis[f"rest_vertices_{i}"] for i, _ in pieces])
        weights = np.concatenate([basis[f"sample_mass_{i}"] for i, _ in pieces])
        mass, com, inertia = sampled_mass_properties(points, weights)
        head = elastic[0]["body_index"]
        np.testing.assert_array_equal(basis["model_body_inertia"][head], inertia.astype(np.float32))
        np.testing.assert_array_equal(basis["model_body_com"][head], com.astype(np.float32))
        assert basis["model_body_mass"][head] == np.float32(mass)
        phi = np.concatenate([basis[f"phi_{i}"] for i, _ in pieces]).astype(float)
        angular = np.einsum("v,vmc->mc", weights.astype(float), np.cross(phi, points[:, None, :]))
        linear = np.einsum("v,vmc->mc", weights.astype(float), phi)
        i = pieces[0][0]
        np.testing.assert_allclose(linear, basis[f"mode_coupling_linear_{i}"], atol=1e-11, rtol=1e-6)
        np.testing.assert_allclose(angular, basis[f"mode_coupling_angular_{i}"], atol=1e-13, rtol=1e-6)
        audit = coupled_mass_audit(mass, com, inertia, basis[f"mode_mass_{i}"], linear, angular)
        assert audit["runtime_spd_passed"] and audit["normalized_condition_number"] < 10, audit
    else:
        inertia = basis["model_body_inertia"][body_indices[1:]]
        assert np.min(np.linalg.eigvalsh(inertia)) >= 1e-10
        assert np.max(np.linalg.eigvalsh(inertia)) < 1e-8, "Unexpected automatic inertia inflation"
    max_error = 0.0
    for joint, mesh in zip(metrics["root_joints"], elastic, strict=True):
        child = mesh["body_index"]
        assert basis["model_joint_type"][joint] == newton.JointType.FIXED
        assert basis["model_joint_parent"][joint] == handle
        assert basis["model_joint_child"][joint] == child
        assert np.all(state["joint_enabled"][:, joint])
        for poses in state["body_q"]:
            parent = transform_points(basis["model_joint_X_p"][joint, :3][None], poses[handle])[0]
            child_root = transform_points(basis["model_joint_X_c"][joint, :3][None], poses[child])[0]
            max_error = max(max_error, float(np.linalg.norm(parent - child_root)))
    assert max_error < 0.0005, max_error
    np.testing.assert_allclose(max_error, metrics["max_root_anchor_error_m"], atol=1e-8)
    grasp = metrics["grasp_joint_index"]
    assert not np.any(state["joint_enabled"][state["times"] > 5.22, grasp])
    assert np.all(state["joint_enabled"][(state["times"] > 0.37) & (state["times"] < 5.19), grasp])
    flying = (state["times"] > 5.22) & (state["times"] < 5.32)
    acceleration = float(np.mean(np.diff(state["body_qd"][flying, handle, 2])) * 60)
    assert -10.2 < acceleration < -9.4, acceleration
    assert metrics["handle_lift_m"] > 0.1
    assert metrics["contact_window_deformation_m"] > 0.001
    ratio = metrics["negative_control"]["contact_window_deformation_m"] / metrics["contact_window_deformation_m"]
    assert ratio < 0.1, ratio
    kinematic = basis["model_body_flags"] == int(newton.BodyFlags.KINEMATIC)
    np.testing.assert_array_equal(state["body_q"][:, kinematic], negative["body_q"][:, kinematic])
    np.testing.assert_array_equal(state["joint_enabled"], negative["joint_enabled"])
    return {
        "recovered_root_anchor_error_m": max_error,
        "recovered_freefall_acceleration_m_s2": acceleration,
        "matched_contact_window_deformation_ratio": ratio,
        "dynamic_body_count": len(body_indices),
        "fixed_root_joint_count": len(elastic),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", nargs="*")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    for directory in args.exports or []:
        print(json.dumps(verify_export(directory), indent=2))
