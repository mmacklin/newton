# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU scrub regressions; optionally audit an existing GPU export.

Run with the shared uv runtime and this worktree on PYTHONPATH. Pass
--exports DIRECTORY to check actual solved handle/tuft motion, not just the
carriage command. No simulation, GPU work, or artifact writes occur here.
"""

import argparse
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from brush_glove_geometry import Config
from toothbrush import Example, drive_mechanism, mechanism_position, nominal_handle

import newton
import newton.solvers

REFERENCE = Path(__file__).resolve().parent / "frozen"
PHYSICS_KEYS = ("mode_mass", "mode_stiffness", "mode_damping", "mode_coupling_linear", "mode_coupling_angular")
MODEL_KEYS = ("body_mass", "body_com", "body_inertia", "body_flags", "joint_parent", "joint_child", "joint_type")


@wp.kernel
def sample_motion(times: wp.array[float], initial_z: float, motion: wp.array2d[wp.vec3]):
    i = wp.tid()
    motion[i, 0] = nominal_handle(times[i], initial_z)
    for part in range(4):
        motion[i, part + 1] = mechanism_position(times[i], initial_z, part)


def motion_samples(times, initial_z):
    with wp.ScopedDevice("cpu"):
        times = wp.array(np.asarray(times, dtype=np.float32), dtype=float)
        result = wp.empty((len(times), 5), dtype=wp.vec3)
        wp.launch(sample_motion, len(times), inputs=[times, initial_z, result])
        return result.numpy()


def finish_without_simulation(example):
    example.state_0 = SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: np.asarray(example.builder.body_q)))


def capsule_distance(points, center, radius, half_length):
    relative = np.asarray(points) - center
    relative = relative.copy()
    relative[..., 1] -= np.clip(relative[..., 1], -half_length, half_length)
    return np.linalg.norm(relative, axis=-1) - radius


def scrub_strokes(times, positions):
    """Require three resolved lateral strokes [m], without prescribing amplitude."""
    times, positions = np.asarray(times), np.asarray(positions)
    assert positions.shape == (len(times), 3) and np.isfinite(positions).all()
    assert np.all(np.diff(times) > 0)
    strokes = []
    for start, end, direction in ((2.55, 3.0, 1), (3.15, 3.75, -1), (3.85, 4.45, 1)):
        selected = positions[(times >= start) & (times <= end), 1]
        assert len(selected) >= 10, "Insufficient samples in a scrub stroke"
        stroke = float(selected[-1] - selected[0])
        assert direction * stroke > 0.001, f"Missing actual lateral scrub/reversal: {stroke:g} m"
        strokes.append(stroke)
    return strokes


def compare_export_physics(scene, basis):
    """Compare saved head/handle physics against the frozen accepted 40-mode head."""
    original_scene = json.loads((REFERENCE / "scene.json").read_text())
    original_meshes = {mesh["name"]: i for i, mesh in enumerate(original_scene["meshes"])}
    with np.load(REFERENCE / "basis.npz", allow_pickle=False) as original:
        for key in MODEL_KEYS:
            np.testing.assert_array_equal(basis[f"model_{key}"], original[f"model_{key}"], err_msg=key)
        np.testing.assert_array_equal(basis["solver_elastic_implicit_mass_coupling"], [True])
        for i, mesh in enumerate(scene["meshes"]):
            if mesh["q_start"] is None:
                continue
            j = original_meshes[mesh["name"]]
            for key in ("rest_vertices", "faces", "phi", "source_phi", "sample_mass", *PHYSICS_KEYS):
                np.testing.assert_array_equal(
                    basis[f"{key}_{i}"], original[f"{key}_{j}"], err_msg=f"{mesh['name']} {key}"
                )


class ScrubCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with wp.ScopedDevice("cpu"), patch.object(Example, "finish", finish_without_simulation):
            cls.driven = Example()
            cls.control = Example(cfg=Config(iterations=24, contact_enabled=False))
        cls.times = np.arange(0, 1801, dtype=np.float64) / 240
        cls.motion = motion_samples(cls.times, cls.driven.initial_z)

    def _check_sweeping_visible_capsule(self):
        """Sweep the unchanged visible collision capsule across the bristle field."""
        tool = self.motion[:, 1]
        scan = (self.times >= 2.3) & (self.times <= 4.5)
        self.assertGreater(float(np.ptp(tool[scan, 0])), 0.005)
        self.assertTrue(np.all(np.diff(tool[scan, 0]) <= 0))
        self.assertLess(float(np.max(np.linalg.norm(np.diff(tool, axis=0), axis=1))), 0.0015)
        example = self.driven
        self.assertAlmostEqual(example.radius, 0.0025)
        self.assertAlmostEqual(example.tool_half_length, 0.035)
        self.assertEqual(example.builder.shape_type[example.tool_shape], newton.GeoType.CAPSULE)
        xform = example.builder.shape_transform[example.tool_shape]
        axis = np.asarray(wp.quat_rotate(wp.transform_get_rotation(xform), wp.vec3(0, 0, 1)))
        np.testing.assert_allclose(np.abs(axis), [0, 1, 0], atol=2e-7)
        visual = next(mesh for mesh in example.meshes if mesh["name"] == "rounded_contact_tool")
        self.assertEqual(visual["body"], example.actuator)
        points, faces = visual["p"], visual["f"]
        distance = capsule_distance(points, np.zeros(3), example.radius, example.tool_half_length)
        # The loft stops 0.015 rad before each pole and caps that ring with a disk.
        pole_inset = example.radius * (1 - np.cos(0.015))
        poles = np.linalg.norm(points[:, [0, 2]], axis=1) < 1e-12
        self.assertEqual(int(poles.sum()), 2)
        np.testing.assert_allclose(distance[~poles], 0, atol=1e-8)
        np.testing.assert_allclose(distance[poles], -pole_inset, atol=2e-9)
        self.assertLessEqual(float(np.abs(distance).max()), pole_inset + 2e-9)
        np.testing.assert_array_equal(np.unique(faces), np.arange(len(points)))
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, 2)
        triangle = points[faces]
        areas = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1)
        self.assertGreater(float(areas.min()), 1e-15)

    def test_pickup_precedes_approach_and_scrub_reverses(self):
        """Lift before approaching, reverse tangential travel, and withdraw before release."""
        self._check_sweeping_visible_capsule()
        self._check_motion_detector_rejects_stationary_and_monotonic_handles()
        handle, carriage = self.motion[:, 0], self.motion[:, 2]
        np.testing.assert_array_equal(handle[self.times <= 1.5, :2], 0)
        np.testing.assert_allclose(handle[self.times <= 0.5, 2], self.driven.initial_z, atol=1e-8)
        self.assertGreater(float(handle[np.searchsorted(self.times, 1.4), 2] - handle[0, 2]), 0.10)
        before_release = self.times <= 5.2
        np.testing.assert_allclose(
            carriage[before_release] - handle[before_release],
            np.broadcast_to([0, -0.105, 0.010], (before_release.sum(), 3)),
            atol=2e-8,
        )
        scrub_strokes(self.times, carriage)
        scrubbing = (self.times >= 2.5) & (self.times <= 4.5)
        self.assertLess(float(np.ptp(handle[scrubbing, 0])), 1e-7)
        self.assertLess(float(np.ptp(handle[scrubbing, 2])), 1e-7)
        self.assertGreater(
            float(handle[np.searchsorted(self.times, 4.5), 0] - handle[np.searchsorted(self.times, 4.9), 0]), 0.01
        )

    def test_driver_does_not_prescribe_dynamic_object(self):
        """Write only tool/jaw poses and release the real grasp joint at 5.2 s."""
        self._check_dynamic_topology_and_matched_control_filter()
        self._check_tool_contact_metrics_filter()
        example = self.driven
        dynamic = [example.handle, example.head_body]
        with wp.ScopedDevice("cpu"):
            poses = np.asarray(example.builder.body_q, dtype=np.float32)
            velocities = np.full((len(poses), 6), 0.123, dtype=np.float32)
            for t, grasp in (
                (0.0, False),
                (0.34, False),
                (0.36, True),
                (3.5, True),
                (5.19, True),
                (5.21, False),
                (7.5, False),
            ):
                states = [
                    (wp.array(poses, dtype=wp.transform), wp.array(velocities, dtype=wp.spatial_vector))
                    for _ in range(2)
                ]
                enabled = wp.array(example.builder.joint_enabled, dtype=wp.bool)
                before = enabled.numpy().copy()
                wp.launch(
                    drive_mechanism,
                    4,
                    inputs=[
                        wp.array([t], dtype=float),
                        example.dt,
                        example.initial_z,
                        example.mechanism_bodies,
                        example.grasp_joint,
                        enabled,
                        *states[0],
                        *states[1],
                    ],
                )
                expected = motion_samples([t], example.initial_z)[0, 1:]
                for body_q, body_qd in states:
                    np.testing.assert_array_equal(body_q.numpy()[dynamic], poses[dynamic])
                    np.testing.assert_array_equal(body_qd.numpy()[dynamic], velocities[dynamic])
                    np.testing.assert_allclose(
                        body_q.numpy()[example.mechanism_bodies.numpy(), :3], expected, atol=1e-8
                    )
                    next_position = motion_samples([t + example.dt], example.initial_z)[0, 1]
                    np.testing.assert_allclose(
                        body_qd.numpy()[example.actuator, :3], (next_position - expected[0]) / example.dt, atol=2e-6
                    )
                    np.testing.assert_array_equal(body_qd.numpy()[example.actuator, 3:], 0)
                before[example.grasp_joint] = grasp
                np.testing.assert_array_equal(enabled.numpy(), before)

    def test_contact_exposes_tips_not_plate_or_roots(self):
        """Expose several tuft tips while retaining root/plate clearance and unload space."""
        example = self.driven
        item = example.elastic[0]
        plate_count = example.head_parts[2]["start"]
        local = item["p"] + item["root"]
        rotation = np.eye(3)
        for t in (2.55, 3.0, 3.75, 4.45):
            handle, tool = motion_samples([t], example.initial_z)[0, :2]
            distance = capsule_distance(
                (local + handle - tool) @ rotation, np.zeros(3), example.radius, example.tool_half_length
            )
            self.assertGreater(float(distance[:plate_count].min()), example.cfg.contact_gap)
            exposed = 0
            for part in example.head_parts[2:]:
                exposed += float(distance[part["tip_indices"]].min()) < example.cfg.contact_gap
                points = local[part["start"] : part["start"] + part["count"]]
                roots = points[np.isclose(points[:, 2], points[:, 2].min(), atol=1e-7)]
                self.assertGreater(
                    float(
                        capsule_distance(
                            (roots + handle - tool) @ rotation, np.zeros(3), example.radius, example.tool_half_length
                        ).min()
                    ),
                    example.cfg.contact_gap,
                )
            self.assertGreaterEqual(exposed, 5)
        for t in (0, 0.35, 0.5, 1.4, 1.5, 4.9, 5.19):
            handle, tool = motion_samples([t], example.initial_z)[0, :2]
            self.assertGreater(
                float(
                    capsule_distance(
                        (local + handle - tool) @ rotation, np.zeros(3), example.radius, example.tool_half_length
                    ).min()
                ),
                0.005,
            )

    def test_original_forty_mode_physics_is_unchanged(self):
        """Retain the frozen mass, stiffness, damping, geometry, and physical inertia."""
        example = self.driven
        original_scene = json.loads((REFERENCE / "scene.json").read_text())
        meshes = {mesh["name"]: i for i, mesh in enumerate(original_scene["meshes"])}
        basis = example.elastic[0]["basis"]
        self.assertEqual(basis.mode_count, 40)
        with np.load(REFERENCE / "basis.npz", allow_pickle=False) as original:
            for key in MODEL_KEYS:
                expected = original[f"model_{key}"]
                np.testing.assert_array_equal(
                    np.asarray(getattr(example.builder, key)).astype(expected.dtype).reshape(expected.shape),
                    expected,
                    err_msg=key,
                )
            for mesh in example.meshes:
                if mesh["elastic"] is None:
                    continue
                i = meshes[mesh["name"]]
                selection = slice(mesh["elastic_vertex_start"], mesh["elastic_vertex_end"])
                for key, value in (
                    ("rest_vertices", mesh["p"]),
                    ("faces", mesh["f"]),
                    ("source_phi", basis.sample_phi[selection]),
                    ("sample_mass", basis.sample_mass[selection]),
                ):
                    np.testing.assert_array_equal(value, original[f"{key}_{i}"], err_msg=f"{mesh['name']} {key}")
                for key in PHYSICS_KEYS:
                    np.testing.assert_array_equal(getattr(basis, key), original[f"{key}_{i}"], err_msg=key)
        self._check_real_cpu_solver_enables_implicit_mass_coupling()

    def _check_dynamic_topology_and_matched_control_filter(self):
        """Keep two dynamic bodies, the fixed zero-Phi root, and floor-active controls."""
        driven, control = self.driven, self.control
        builder = driven.builder
        self.assertEqual(len(driven.elastic), 1)
        self.assertEqual(len(driven.root_joints), 1)
        self.assertEqual(
            np.flatnonzero(np.asarray(builder.body_flags) == int(newton.BodyFlags.DYNAMIC)).tolist(),
            [driven.handle, driven.head_body],
        )
        joint = driven.root_joints[0]
        self.assertEqual(builder.joint_type[joint], newton.JointType.FIXED)
        self.assertEqual(builder.joint_parent[joint], driven.handle)
        self.assertEqual(builder.joint_child[joint], driven.head_body)
        np.testing.assert_allclose(np.asarray(builder.joint_X_p[joint])[:3], driven.elastic[0]["root"], atol=1e-8)
        np.testing.assert_array_equal(np.asarray(builder.joint_X_c[joint]), [0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(driven.elastic[0]["source_phi"][: driven.head_parts[2]["start"]], 0)
        self.assertEqual(builder.joint_parent[driven.grasp_joint], driven.gripper[0])
        self.assertEqual(builder.joint_child[driven.grasp_joint], driven.handle)
        self.assertFalse(builder.joint_enabled[driven.grasp_joint])
        extra = set(control.builder.shape_collision_filter_pairs) - set(builder.shape_collision_filter_pairs)
        self.assertEqual(extra, {tuple(sorted((driven.elastic[0]["shape"], driven.tool_shape)))})
        self.assertFalse(set(builder.shape_collision_filter_pairs) - set(control.builder.shape_collision_filter_pairs))
        self.assertFalse(any(driven.floor_shape in pair for pair in extra))
        for key in (*MODEL_KEYS, "shape_flags", "shape_material_ke", "shape_material_kd", "shape_material_mu"):
            np.testing.assert_array_equal(getattr(builder, key), getattr(control.builder, key), err_msg=key)
        np.testing.assert_allclose(builder.gravity, [0, 0, -9.81])
        self.assertGreater(driven.cfg.friction, 0)
        self.assertEqual(driven.cfg.substeps, 4)
        self.assertEqual(driven.cfg.iterations, 24)

    def _check_tool_contact_metrics_filter(self):
        """Count only tool/head pairs, including reversed ordering and unused capacity."""
        example = self.driven

        def array(value):
            return SimpleNamespace(numpy=lambda: np.asarray(value))

        tool, head, handle, floor = (
            example.tool_shape,
            example.elastic[0]["shape"],
            example.handle_collision,
            example.floor_shape,
        )
        zero = np.zeros_like(example.elastic[0]["p"])
        harness = SimpleNamespace(
            handle=example.handle,
            handle_com=example.handle_com,
            meshes=example.meshes,
            elastic=[{**example.elastic[0], "equilibrium_disp": zero}],
            head_parts=example.head_parts,
            tool_shape=tool,
            floor_shape=floor,
            grasp_joint=example.grasp_joint,
            state_0=SimpleNamespace(
                body_q=array(example.builder.body_q), body_qd=array(np.zeros((len(example.builder.body_q), 6)))
            ),
            model=SimpleNamespace(joint_enabled=array(example.builder.joint_enabled)),
        )
        for pairs, expected in (
            ([(tool, head), (head, tool), (tool, handle), (handle, tool), (head, floor)], (2, 2, 1)),
            ([(tool, handle), (handle, tool)], (0, 2, 0)),
            ([], (0, 0, 0)),
        ):
            capacity = np.asarray([*pairs, (tool, head)], dtype=np.int32)
            harness.contacts = SimpleNamespace(
                rigid_contact_shape0=array(capacity[:, 0]), rigid_contact_shape1=array(capacity[:, 1])
            )
            frame = {"contacts": len(pairs)}
            Example.object_metrics(harness, frame, [zero])
            self.assertEqual(frame["tool_contacts"], expected[0])
            self.assertEqual((frame["tool_contacts"], frame["other_tool_contacts"], frame["floor_contacts"]), expected)

    def _check_real_cpu_solver_enables_implicit_mass_coupling(self):
        """Verify the unchanged sampled inertia clears the real solver SPD gate."""
        with wp.ScopedDevice("cpu"):
            self.driven.builder.color()
            model = self.driven.builder.finalize(device="cpu")
            solver = newton.solvers.SolverVBD(
                model,
                iterations=24,
                rigid_compliant_alm=False,
                rigid_contact_hard=False,
                **self.driven.solver_options(),
            )
        np.testing.assert_array_equal(solver.elastic_implicit_mass_coupling.numpy(), [True])
        self.assertEqual(model.elastic_body_count, 1)
        self.assertEqual(model.elastic_max_mode_count, 40)

    def _check_motion_detector_rejects_stationary_and_monotonic_handles(self):
        """Reject tool-only sweeping and one-way translations as handle scrubbing."""
        scrub_strokes(self.times, self.motion[:, 2])
        for positions in (
            np.zeros((len(self.times), 3)),
            np.column_stack((self.times * 0, self.times * 0.01, self.times * 0)),
        ):
            with self.assertRaisesRegex(AssertionError, "Missing actual lateral"):
                scrub_strokes(self.times, positions)


def verify_export(directory):
    """Audit solved motion and modal recovery from an existing, read-only export."""
    directory = Path(directory)
    scene = json.loads((directory / "scene.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    meshes = {mesh["name"]: (i, mesh) for i, mesh in enumerate(scene["meshes"])}
    with (
        np.load(directory / "state.npz", allow_pickle=False) as state,
        np.load(directory / "basis.npz", allow_pickle=False) as basis,
    ):
        compare_export_physics(scene, basis)
        times, body_q = state["times"], state["body_q"]
        assert len(times) >= 420 and np.isfinite(body_q).all()
        np.testing.assert_allclose(np.diff(times), 1 / 60, atol=1e-10)
        tool_id = meshes["rounded_contact_tool"][1]["body_index"]
        handle_id = meshes["ivory_handle"][1]["body_index"]
        tool = body_q[:, tool_id]
        scan = (times >= 2.3) & (times <= 4.5)
        assert np.ptp(tool[scan, 0]) > 0.005, "Missing cross-field capsule sweep"
        np.testing.assert_array_equal(tool[:, 3:], np.broadcast_to(tool[0, 3:], tool[:, 3:].shape))
        handle = body_q[:, handle_id, :3]
        strokes = scrub_strokes(times, handle)
        assert np.ptp(handle[times < 0.35, :2], axis=0).max() < 0.0002, "Handle moves before pickup"
        assert np.max(np.linalg.norm(handle[times < 1.5, :2] - handle[0, :2], axis=1)) < 0.0005, (
            "Handle approaches before lift"
        )
        assert handle[(times > 1.4) & (times < 1.5), 2].min() - handle[0, 2] > 0.10
        records = metrics["records"]
        np.testing.assert_allclose([r["time"] for r in records], times, atol=1e-10)
        np.testing.assert_array_equal([r["handle_position_m"] for r in records], handle)
        scrub = (times > 2.5 + 1e-9) & (times < 4.5 - 1e-9)
        recovered_tips, max_deformation = [], 0.0
        for name, (i, mesh) in meshes.items():
            if mesh["q_start"] is None:
                continue
            phi, start = basis[f"phi_{i}"], mesh["q_start"]
            q = state["joint_q"][:, start : start + phi.shape[1]] - basis[f"equilibrium_q_{i}"]
            displacement = np.einsum("vmc,fm->fvc", phi, q, optimize=True)
            max_deformation = max(max_deformation, float(np.linalg.norm(displacement, axis=2).max()))
            if name.startswith("tuft_"):
                rest = basis[f"rest_vertices_{i}"]
                tips = rest[:, 2] > rest[:, 2].max() - 1e-6
                recovered_tips.append(displacement[:, tips, 1].mean(axis=1))
        tips = np.asarray(recovered_tips).T
        assert tips.shape == (len(times), 20)
        np.testing.assert_allclose(tips, [r["tip_y_displacement_m"] for r in records], atol=2e-8)
        reversing = int(np.count_nonzero((tips[scrub].max(axis=0) > 0.00025) & (tips[scrub].min(axis=0) < -0.00025)))
        peak_to_peak = float(np.ptp(tips[scrub], axis=0).max())
        assert reversing >= 5, f"Only {reversing} tufts reverse above +/-0.25 mm"
        assert peak_to_peak > 0.001
        assert reversing == metrics["scrub_tuft_reversals"]
        np.testing.assert_allclose(peak_to_peak, metrics["scrub_tangential_peak_to_peak_m"], atol=2e-8)
        np.testing.assert_allclose(max_deformation, metrics["max_deformation_m"], atol=2e-8)
        assert metrics["graph_enabled"] and scene["parameters"]["substeps"] == 4
        assert scene["parameters"]["iterations"] == 24
        assert metrics["mass_audit"]["actual_implicit_mass_coupling"] == [True]
        result = {
            "directory": str(directory),
            "actual_handle_strokes_m": strokes,
            "scrub_tuft_reversals": reversing,
            "scrub_tangential_peak_to_peak_m": peak_to_peak,
            "recovered_max_displacement_m": max_deformation,
            "unchanged_40_mode_physics": True,
            "negative_control_available": (directory / "negative_state.npz").is_file(),
            "negative_control_passed": metrics["negative_control_passed"],
        }
        print(json.dumps(result, indent=2))
        assert max_deformation < 0.006, f"Physical deformation gate failed: {max_deformation * 1000:.3f} mm >= 6 mm"
        if result["negative_control_available"]:
            with np.load(directory / "negative_state.npz", allow_pickle=False) as negative:
                np.testing.assert_array_equal(negative["times"], times)
                np.testing.assert_array_equal(negative["body_q"][:, tool_id], tool)
                for name in ("gripper_part_1", "gripper_part_2", "gripper_part_3"):
                    body = meshes[name][1]["body_index"]
                    np.testing.assert_array_equal(negative["body_q"][:, body], body_q[:, body])
                scrub_strokes(times, negative["body_q"][:, handle_id, :3])
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, help="Read-only audit of an existing simulation export")
    args, unittest_args = parser.parse_known_args()
    suite = unittest.main(argv=[__file__, *unittest_args], exit=False)
    if not suite.result.wasSuccessful():
        raise SystemExit(1)
    if args.exports:
        verify_export(args.exports)
