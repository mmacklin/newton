# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU construction and algebra only; these are not trajectory acceptance tests."""

import ast
import hashlib
import inspect
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp
from brush_glove_geometry import Config, brush_handle, brush_head_rom, sampled_mass_properties, surface_mass
from fused_toothbrush import Example, detailed_meshes, fused_mass_properties, virtual_handle
from toothbrush import Example as FiniteNeckExample

import newton
import newton.solvers

HERE = Path(__file__).resolve().parent


def skew(vector):
    """Cross-product matrix for an independently formed velocity pullback."""
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def require_digest(array, expected):
    """Reject shape, dtype, or bit changes, including a single zero-Phi leak."""
    if list(array.shape) != expected["shape"] or str(array.dtype) != expected["dtype"]:
        raise AssertionError("Shape or dtype differs")
    if hashlib.sha256(array.tobytes()).hexdigest() != expected["sha256"]:
        raise AssertionError("Array bits differ")


class FusedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """No solver step, capture, or physical preroll is allowed in this suite."""
        with (
            wp.ScopedDevice("cpu"),
            patch.object(newton.solvers.SolverVBD, "step", side_effect=AssertionError("No simulation authorized")),
            patch.object(Example, "simulate", side_effect=AssertionError("No simulation authorized")),
        ):
            cls.example = Example(construction_only=True)
            cls.control = Example(cfg=Config(iterations=16, contact_enabled=False), construction_only=True)
        cls.reference = json.loads((HERE / "fused_reference.json").read_text())

    def test_packaged_provenance_and_local_public_dependencies(self):
        """Every shipped adapter dependency is locally present and hash bound."""
        manifest = json.loads((HERE / "fused_sources.json").read_text())
        for name, expected in manifest["files"].items():
            with self.subTest(file=name):
                source = (HERE / name).read_bytes()
                self.assertEqual(len(source), expected["bytes"])
                self.assertEqual(hashlib.sha256(source).hexdigest(), expected["packaged_sha256"])
                if name.endswith(".py"):
                    self.assertNotIn(b"/" + b"home/", source)
        self.assertEqual(manifest["new_binary_assets"], [])

    def test_finalized_model_matches_sealed_approved_construction(self):
        """All 65 selected model arrays match the independent sealed reference."""
        for name, expected in self.reference["model"].items():
            with self.subTest(array=name):
                require_digest(getattr(self.example.model, name).numpy(), expected)

    def test_all_meshes_and_detailed_embedding_are_exact(self):
        """Keep all 30 material meshes, 20 tufts, and the approved Loop surface."""
        self.assertEqual(len(self.example.meshes), 30)
        self.assertEqual(len(self.example.head_parts) - 2, 20)
        for index, expected in self.reference["coarse_meshes"].items():
            mesh = self.example.meshes[int(index)]
            with self.subTest(mesh=expected["name"]):
                self.assertEqual(mesh["name"], expected["name"])
                self.assertEqual(mesh["body"], expected["body"])
                require_digest(mesh["p"], expected["points"])
                require_digest(mesh["f"], expected["faces"])
        for index, mesh in detailed_meshes(self.example).items():
            expected = self.reference["detailed_meshes"][str(index)]
            require_digest(mesh["points"], expected["points"])
            require_digest(mesh["faces"], expected["faces"])
            require_digest(mesh["weights"], self.reference["embedding"][str(index)])
            self.assertGreaterEqual(float(mesh["weights"].min()), 0)
            np.testing.assert_allclose(mesh["weights"].sum(axis=1), 1, rtol=0, atol=1e-14)

    def test_topology_settings_and_generic_dispatch(self):
        """Only the redundant neck frame disappears, not modes or physical DOFs."""
        example = self.example
        self.assertEqual(example.model.body_count, 5)
        self.assertEqual(example.model.elastic_body_count, 1)
        self.assertEqual(example.builder.fusion_basis.mode_count, 40)
        self.assertEqual(example.root_joints, [])
        self.assertEqual(example.handle, example.head_body)
        self.assertEqual(example.model.joint_q.shape, (75,))
        self.assertEqual(example.model.joint_qd.shape, (70,))
        self.assertEqual(example.cfg.substeps, 4)
        self.assertEqual(example.cfg.iterations, 16)
        self.assertEqual(example.cfg.elastic_relaxation, 0.4)
        self.assertEqual(example.cfg.contact_ke, 3000)
        self.assertEqual(example.cfg.contact_kd, 0)
        self.assertEqual(example.cfg.friction, 0.2)
        self.assertEqual(example.cfg.contact_gap, 0.00015)
        self.assertIs(type(example.solver), newton.solvers.SolverVBD)
        self.assertFalse(any(key.startswith("_experimental") for key in vars(example.model)))
        self.assertFalse(any(key.startswith("_experimental") for key in vars(example.solver)))
        audit = example.mass_audit()
        self.assertTrue(audit["runtime_spd_passed"])
        self.assertTrue(audit["cholesky_passed"])
        self.assertFalse(audit["inertia_regularization_applied"])
        self.assertEqual(audit["actual_implicit_mass_coupling"], [True])
        np.testing.assert_array_equal(example.builder.gravity, np.array([0, 0, -9.81], np.float32))

    def test_attachments_and_grasp_have_exact_zero_modes(self):
        """Appended samples carry no modal mass, displacement, or rotation."""
        example = self.example
        basis = example.builder.fusion_basis
        start, end = example.builder.attachment_sample_start, example.builder.attachment_sample_end
        self.assertEqual(start, len(example.elastic[0]["p"]))
        indices = example.builder.attachment_sample_indices
        self.assertEqual(len(indices), 442 + 722 + 2040 + 1)
        self.assertGreater(end, start)
        np.testing.assert_array_equal(basis.sample_phi[indices], 0)
        np.testing.assert_array_equal(basis.sample_psi[indices], 0)
        np.testing.assert_array_equal(basis.sample_phi[start:end], 0)
        np.testing.assert_array_equal(basis.sample_psi[start:end], 0)
        np.testing.assert_array_equal(basis.sample_mass[start:end], 0)
        np.testing.assert_array_equal(example.model.elastic_endpoint_phi.numpy(), 0)
        np.testing.assert_array_equal(example.model.elastic_endpoint_psi.numpy(), 0)
        np.testing.assert_array_equal(example.model.body_flags.numpy()[1:], int(newton.BodyFlags.KINEMATIC))
        np.testing.assert_array_equal(example.mechanism_bodies.numpy(), [1, 2, 3, 4])

    def test_negative_contact_arm_changes_only_original_pair(self):
        """The matched negative arm removes tool/head contacts, retaining floor."""
        positive = set(self.example.builder.shape_collision_filter_pairs)
        negative = set(self.control.builder.shape_collision_filter_pairs)
        self.assertEqual(positive - negative, set())
        self.assertEqual(
            negative - positive, {tuple(sorted((self.example.elastic[0]["shape"], self.example.tool_shape)))}
        )
        for name in (
            "body_mass",
            "body_inertia",
            "body_com",
            "elastic_mode_mass",
            "elastic_mode_stiffness",
            "elastic_mode_damping",
        ):
            np.testing.assert_array_equal(
                getattr(self.example.model, name).numpy(), getattr(self.control.model, name).numpy()
            )

    def test_locked_mass_energy_and_contact_virtual_work(self):
        """Independently pull back the original 52-DOF mass onto 46 fused DOFs."""
        handle, _, _, plate, plastic, root = brush_handle(split_head=True)
        points, _, basis, _ = brush_head_rom(plastic, plate, root)
        properties = [
            sampled_mass_properties(handle[0], surface_mass(*handle, 0.018)),
            sampled_mass_properties(points, basis.sample_mass),
        ]
        mass = np.asarray([p[0] for p in properties], np.float32).astype(float)
        centers = np.asarray([p[1] for p in properties], np.float32).astype(float)
        moments = np.asarray([p[2] for p in properties], np.float32).astype(float)
        centers[0] -= root.astype(float)
        total, center, inertia = fused_mass_properties(handle, points, basis, root)
        linear, angular = basis.mode_coupling_linear.astype(float), basis.mode_coupling_angular.astype(float)
        old, fused, mapping = np.zeros((52, 52)), np.zeros((46, 46)), np.zeros((52, 46))
        for i in range(2):
            start = 6 * i
            old[start : start + 3, start : start + 3] = mass[i] * np.eye(3)
            old[start + 3 : start + 6, start + 3 : start + 6] = moments[i]
            mapping[start : start + 3, :3] = np.eye(3)
            mapping[start : start + 3, 3:6] = -skew(centers[i] - center)
            mapping[start + 3 : start + 6, 3:6] = np.eye(3)
        mapping[12:, 6:] = np.eye(40)
        coupling = np.vstack([linear.T, -(angular + np.cross(centers[1], linear)).T])
        old[6:12, 12:], old[12:, 6:12] = coupling, coupling.T
        old[12:, 12:] = np.diag(basis.mode_mass)
        fused[:3, :3], fused[3:6, 3:6] = total * np.eye(3), inertia
        coupling = np.vstack([linear.T, -(angular + np.cross(center, linear)).T])
        fused[:6, 6:], fused[6:, :6] = coupling, coupling.T
        fused[6:, 6:] = np.diag(basis.mode_mass)
        pulled = mapping.T @ old @ mapping
        relative = np.linalg.norm(pulled - fused) / np.linalg.norm(fused)
        self.assertLess(relative, 1e-12)
        rng = np.random.default_rng(84)
        for _ in range(24):
            velocity = rng.normal(size=46)
            old_velocity = mapping @ velocity
            np.testing.assert_allclose(
                old_velocity @ old @ old_velocity, velocity @ fused @ velocity, rtol=1e-12, atol=1e-15
            )
            force = rng.normal(size=3)
            point = rng.normal(size=3) * 0.01
            jacobian = np.zeros((3, 52))
            jacobian[:, :3] = np.eye(3)
            jacobian[:, 3:6] = -skew(point - centers[0])
            direct = np.zeros((3, 46))
            direct[:, :3], direct[:, 3:6] = np.eye(3), -skew(point - center)
            np.testing.assert_allclose(jacobian @ mapping, direct, rtol=0, atol=1e-15)
            np.testing.assert_allclose(
                force @ jacobian @ old_velocity, (direct.T @ force) @ velocity, rtol=1e-12, atol=1e-15
            )
        wrong = mapping.copy()
        wrong[:3, 3:6] = -skew(centers[0] + root - center)
        self.assertGreater(np.linalg.norm(wrong.T @ old @ wrong - fused) / np.linalg.norm(fused), 1e-3)

    def test_reference_rejects_single_bit_mutations(self):
        """Independent hashes reject missing zero-Phi seeding and mass drift."""
        for name in ("elastic_endpoint_phi", "body_inertia", "elastic_mode_mass"):
            array = getattr(self.example.model, name).numpy().copy()
            array.flat[0] = np.nextafter(array.flat[0], np.float32(np.inf))
            with self.subTest(array=name), self.assertRaisesRegex(AssertionError, "bits differ"):
                require_digest(array, self.reference["model"][name])

    def test_virtual_handle_diagnostics_do_not_modify_state(self):
        """Recover velocities at the old handle COM without touching native state."""
        example = self.example
        pose = example.state_0.body_q.numpy()[0]
        speed = np.array([0.1, 0.2, 0.3, 2, 3, 4], np.float32)
        old_pose, old_speed = pose.copy(), speed.copy()
        result, velocity = virtual_handle(
            pose, speed, example.builder.fusion_root, example.handle_com, example.model.body_com.numpy()[0]
        )
        np.testing.assert_array_equal(pose, old_pose)
        np.testing.assert_array_equal(speed, old_speed)
        np.testing.assert_allclose(result[:3], pose[:3] - example.builder.fusion_root, atol=1e-8)
        expected = speed[:3] + np.cross(
            speed[3:], example.handle_com - example.builder.fusion_root - example.model.body_com.numpy()[0]
        )
        np.testing.assert_allclose(velocity[:3], expected, atol=1e-7)

    def test_original_physics_methods_and_gates_are_shared(self):
        """Only the explicit body-count structural gate differs between examples."""
        for name in ("simulate", "solver_options", "test_final", "validate_negative_control", "update_metrics"):
            self.assertIs(getattr(Example, name), getattr(FiniteNeckExample, name))
        self.assertEqual(FiniteNeckExample.expected_dynamic_body_count, 2)
        self.assertEqual(Example.expected_dynamic_body_count, 1)
        for name in Example.source_files:
            if not name.endswith(".py"):
                continue
            tree = ast.parse((HERE / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("newton._src"))
                if isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.startswith("newton._src") for alias in node.names))
        self.assertNotIn("_experimental", inspect.getsource(Example))


if __name__ == "__main__":
    unittest.main()
