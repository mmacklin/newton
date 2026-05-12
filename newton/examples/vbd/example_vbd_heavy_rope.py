# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""AVBD 3D Heavy Rope scene recreated with Newton VBD."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples

REFERENCE_DT = 1.0 / 60.0
REFERENCE_ITERATIONS = 10
REFERENCE_ALPHA = 0.99
REFERENCE_BETA_LINEAR = 10000.0
REFERENCE_BETA_ANGULAR = 100.0
REFERENCE_GAMMA = 0.999
REFERENCE_GRAVITY = -10.0
REFERENCE_PENALTY_MIN = 1.0
REFERENCE_PENALTY_MAX = 1.0e10
REFERENCE_COLLISION_MARGIN = 0.01
REFERENCE_CONTACT_MATCH_POS_THRESHOLD = 0.05
REFERENCE_CONTACT_MATCH_NORMAL_DOT_THRESHOLD = 0.9
REFERENCE_STICK_THRESH = 1.0e-5
REFERENCE_FRICTION_EPSILON = 0.0
REFERENCE_STICK_FREEZE_TRANSLATION_EPS = 0.0
REFERENCE_STICK_FREEZE_ANGULAR_EPS = 0.0

TITLE = "Heavy Rope"
DESCRIPTION = "Rope with a large heavy terminal block."
CAMERA_TARGET = (11.0, 0.0, 5.0)
CAMERA_DISTANCE = 42.0
CAMERA_PITCH = -24.0
CAMERA_YAW = 135.0
CAMERA_FOV = 55.0
POSITION_LIMIT = 500.0
SPEED_LIMIT = 500.0
MIN_Z_LIMIT = -40.0
REPORT_FRAMES = 180
STATUS_NOTE = "Reference parameters reproduced."

PYRAMID_PLANE_ABS_Y_LIMIT = 0.10
PYRAMID_PLANE_SPAN_LIMIT = 0.16
PYRAMID_ROW_PLANE_SPAN_LIMIT = 0.12
STACK_CENTER_RADIUS_LIMIT = 0.15
STACK_XY_SPREAD_LIMIT = 0.08
STACK_LINE_FIT_LIMIT = 0.06
STACK_TILT_LIMIT = 0.02


def _vec(values: tuple[float, float, float]) -> wp.vec3:
    return wp.vec3(float(values[0]), float(values[1]), float(values[2]))


def _xform(
    position: tuple[float, float, float],
    rotation: wp.quat | None = None,
) -> wp.transform:
    if rotation is None:
        rotation = wp.quat_identity()
    return wp.transform(p=_vec(position), q=rotation)


def _quat_axis_angle(axis: tuple[float, float, float], angle: float) -> wp.quat:
    return wp.quat_from_axis_angle(_vec(axis), float(angle))


def _rotate_y(angle: float, values: tuple[float, float, float]) -> tuple[float, float, float]:
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = values
    return (c * x + s * z, y, -s * x + c * z)


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _camera_front(yaw: float, pitch: float) -> tuple[float, float, float]:
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    cos_pitch = math.cos(pitch_rad)
    return (
        math.cos(yaw_rad) * cos_pitch,
        math.sin(yaw_rad) * cos_pitch,
        math.sin(pitch_rad),
    )


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _fit_line_metrics(points: np.ndarray) -> tuple[float, float, float]:
    centered = points - np.mean(points, axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    distances = np.linalg.norm(np.cross(centered, axis), axis=1)
    tilt_xy_per_z = float(np.linalg.norm(axis[0:2]) / max(abs(axis[2]), 1.0e-8))
    return float(np.max(distances)), float(np.sqrt(np.mean(distances * distances))), tilt_xy_per_z


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.title = TITLE

        self.fps = 60
        self.frame_dt = REFERENCE_DT
        self.sim_dt = REFERENCE_DT
        self.sim_substeps = 1
        self.sim_time = 0.0
        self.solver_iterations = REFERENCE_ITERATIONS

        self.initial_body_velocities: dict[int, tuple[float, float, float]] = {}
        self.body_labels: dict[str, int] = {}
        self.body_sizes: dict[int, tuple[float, float, float]] = {}
        self.initial_body_positions: dict[int, tuple[float, float, float]] = {}
        self.dynamic_body_indices: list[int] = []
        self.soft_joint_indices: list[int] = []
        self.breakable_joint_indices: list[int] = []
        self.broken_joint_count = 0
        self.rigid_joint_linear_ke = REFERENCE_PENALTY_MAX
        self.rigid_joint_angular_ke = REFERENCE_PENALTY_MAX
        self.rigid_joint_linear_kd = 0.0
        self.rigid_joint_angular_kd = 0.0
        self.rigid_avbd_contact_alpha = REFERENCE_ALPHA
        self.rigid_contact_stick_freeze_translation_eps = REFERENCE_STICK_FREEZE_TRANSLATION_EPS
        self.rigid_contact_stick_freeze_angular_eps = REFERENCE_STICK_FREEZE_ANGULAR_EPS
        self.rigid_contact_k_start = REFERENCE_PENALTY_MIN
        self.rigid_body_contact_buffer_size = 512
        self.rigid_body_serial_reverse = False
        self.soft_joint_hard = True
        self.rigid_contact_history = True
        self.contact_matching = "latest"

        builder = newton.ModelBuilder(gravity=REFERENCE_GRAVITY)
        builder.rigid_gap = REFERENCE_COLLISION_MARGIN
        builder.default_shape_cfg.density = 1.0
        builder.default_shape_cfg.ke = REFERENCE_PENALTY_MAX
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.mu = 0.5
        builder.default_shape_cfg.margin = REFERENCE_COLLISION_MARGIN * 0.5
        builder.default_shape_cfg.gap = REFERENCE_COLLISION_MARGIN

        self._build_scene(builder)
        builder.color()
        body_order = (
            range(builder.body_count - 1, -1, -1) if self.rigid_body_serial_reverse else range(builder.body_count)
        )
        builder.body_color_groups = [np.array([body], dtype=int) for body in body_order]
        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        rigid_contact_max = max(4096, self.model.shape_count * 64)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            contact_matching=self.contact_matching,
            contact_matching_pos_threshold=REFERENCE_CONTACT_MATCH_POS_THRESHOLD,
            contact_matching_normal_dot_threshold=REFERENCE_CONTACT_MATCH_NORMAL_DOT_THRESHOLD,
            rigid_contact_max=rigid_contact_max,
        )

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.solver_iterations,
            friction_epsilon=REFERENCE_FRICTION_EPSILON,
            rigid_avbd_alpha=REFERENCE_ALPHA,
            rigid_avbd_joint_alpha=REFERENCE_ALPHA,
            rigid_avbd_contact_alpha=self.rigid_avbd_contact_alpha,
            rigid_avbd_linear_beta=REFERENCE_BETA_LINEAR,
            rigid_avbd_angular_beta=REFERENCE_BETA_ANGULAR,
            rigid_avbd_gamma=REFERENCE_GAMMA,
            rigid_contact_hard=True,
            rigid_contact_history=self.rigid_contact_history,
            rigid_contact_stick_motion_eps=REFERENCE_STICK_THRESH,
            rigid_contact_stick_freeze_translation_eps=self.rigid_contact_stick_freeze_translation_eps,
            rigid_contact_stick_freeze_angular_eps=self.rigid_contact_stick_freeze_angular_eps,
            rigid_contact_k_start=self.rigid_contact_k_start,
            rigid_joint_linear_k_start=REFERENCE_PENALTY_MIN,
            rigid_joint_angular_k_start=REFERENCE_PENALTY_MIN,
            rigid_joint_linear_ke=self.rigid_joint_linear_ke,
            rigid_joint_angular_ke=self.rigid_joint_angular_ke,
            rigid_joint_linear_kd=self.rigid_joint_linear_kd,
            rigid_joint_angular_kd=self.rigid_joint_angular_kd,
            rigid_body_contact_buffer_size=self.rigid_body_contact_buffer_size,
        )

        if not self.soft_joint_hard:
            for joint_index in self.soft_joint_indices:
                self.solver.set_joint_constraint_mode(joint_index, hard=False)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self._assign_initial_velocities()

        self.contacts = self.collision_pipeline.contacts()
        self.viewer.set_model(self.model)
        front = _camera_front(CAMERA_YAW, CAMERA_PITCH)
        camera_pos = (
            CAMERA_TARGET[0] - front[0] * CAMERA_DISTANCE,
            CAMERA_TARGET[1] - front[1] * CAMERA_DISTANCE,
            CAMERA_TARGET[2] - front[2] * CAMERA_DISTANCE,
        )
        self.viewer.set_camera(
            pos=_vec(camera_pos),
            pitch=CAMERA_PITCH,
            yaw=CAMERA_YAW,
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = CAMERA_FOV
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(_vec(CAMERA_TARGET))

        self.max_speed = 0.0
        self.final_speed = 0.0
        self.max_angular_speed = 0.0
        self.final_angular_speed = 0.0
        self.max_abs_position = 0.0
        self.final_max_abs_position = 0.0
        self.min_z = float("inf")
        self.max_z = float("-inf")
        self.final_min_z = float("inf")
        self.final_max_z = float("-inf")
        self.final_xy_radius = 0.0
        self.max_contacts = 0
        self.nan_detected = False
        self.extra_metrics: dict[str, float | int | bool] = {}

    def _cfg(self, density: float, friction: float) -> newton.ModelBuilder.ShapeConfig:
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = density
        cfg.ke = REFERENCE_PENALTY_MAX
        cfg.kd = 0.0
        cfg.mu = friction
        cfg.margin = REFERENCE_COLLISION_MARGIN * 0.5
        cfg.gap = REFERENCE_COLLISION_MARGIN
        return cfg

    def add_box(
        self,
        builder: newton.ModelBuilder,
        size: tuple[float, float, float],
        density: float,
        friction: float,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: wp.quat | None = None,
        label: str | None = None,
    ) -> tuple[int, int]:
        body = builder.add_body(
            xform=_xform(position, rotation),
            is_kinematic=density <= 0.0,
            label=label,
        )
        shape = builder.add_shape_box(
            body,
            hx=0.5 * size[0],
            hy=0.5 * size[1],
            hz=0.5 * size[2],
            cfg=self._cfg(density, friction),
            label=f"{label}_shape" if label else None,
        )
        if density > 0.0:
            self.dynamic_body_indices.append(body)
        self.body_sizes[body] = size
        self.initial_body_positions[body] = position
        if label is not None:
            self.body_labels[label] = body
        if velocity != (0.0, 0.0, 0.0):
            self.initial_body_velocities[body] = velocity
            builder.body_qd[body] = wp.spatial_vector(velocity[0], velocity[1], velocity[2], 0.0, 0.0, 0.0)
        return body, shape

    def add_ball_joint(
        self,
        builder: newton.ModelBuilder,
        parent: int,
        child: int,
        parent_anchor: tuple[float, float, float],
        child_anchor: tuple[float, float, float],
        label: str,
        collision_filter_parent: bool = True,
    ) -> int:
        return builder.add_joint_ball(
            parent=parent,
            child=child,
            parent_xform=_xform(parent_anchor),
            child_xform=_xform(child_anchor),
            collision_filter_parent=collision_filter_parent,
            label=label,
        )

    def add_fixed_joint(
        self,
        builder: newton.ModelBuilder,
        parent: int,
        child: int,
        parent_anchor: tuple[float, float, float],
        child_anchor: tuple[float, float, float],
        label: str,
        soft: bool = False,
        collision_filter_parent: bool = True,
    ) -> int:
        joint = builder.add_joint_fixed(
            parent=parent,
            child=child,
            parent_xform=_xform(parent_anchor),
            child_xform=_xform(child_anchor),
            collision_filter_parent=collision_filter_parent,
            label=label,
        )
        if soft:
            self.soft_joint_indices.append(joint)
        return joint

    def add_cable_joint(
        self,
        builder: newton.ModelBuilder,
        parent: int,
        child: int,
        parent_anchor: tuple[float, float, float],
        child_anchor: tuple[float, float, float],
        stiffness: float,
        label: str,
        collision_filter_parent: bool = True,
    ) -> int:
        return builder.add_joint_cable(
            parent=parent,
            child=child,
            parent_xform=_xform(parent_anchor),
            child_xform=_xform(child_anchor),
            stretch_stiffness=stiffness,
            stretch_damping=0.0,
            bend_stiffness=0.0,
            bend_damping=0.0,
            collision_filter_parent=collision_filter_parent,
            label=label,
        )

    def body_index(self, label: str) -> int:
        try:
            return self.body_labels[label]
        except KeyError as exc:
            raise AssertionError(f"{self.title}: missing body label {label}") from exc

    def body_indices_with_prefix(self, prefix: str) -> list[int]:
        return [body for label, body in self.body_labels.items() if label.startswith(prefix)]

    def body_position(self, label: str) -> np.ndarray:
        return self.state_0.body_q.numpy()[self.body_index(label), 0:3]

    def body_linear_velocity(self, label: str) -> np.ndarray:
        return self.state_0.body_qd.numpy()[self.body_index(label), 0:3]

    def body_linear_speed(self, label: str) -> float:
        return float(np.linalg.norm(self.state_0.body_qd.numpy()[self.body_index(label), 0:3]))

    def body_angular_speed(self, label: str) -> float:
        return float(np.linalg.norm(self.state_0.body_qd.numpy()[self.body_index(label), 3:6]))

    def body_local_axis_world(self, label: str, axis: tuple[float, float, float]) -> np.ndarray:
        q = self.state_0.body_q.numpy()[self.body_index(label), 3:7]
        return _quat_wxyz_to_matrix(q) @ np.asarray(axis, dtype=np.float64)

    def assert_final_speed_below(self, limit: float) -> None:
        if self.final_speed > limit:
            raise AssertionError(f"{self.title}: final speed {self.final_speed:.3f} exceeds {limit:.3f}")

    def assert_final_angular_speed_below(self, limit: float) -> None:
        if self.final_angular_speed > limit:
            raise AssertionError(
                f"{self.title}: final angular speed {self.final_angular_speed:.3f} exceeds {limit:.3f}"
            )

    def assert_final_xy_radius_below(self, limit: float) -> None:
        if self.final_xy_radius > limit:
            raise AssertionError(f"{self.title}: final xy radius {self.final_xy_radius:.3f} exceeds {limit:.3f}")

    def _build_scene(self, builder: newton.ModelBuilder) -> None:
        n = 20
        large_size = 5.0
        self.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, -20.0), label="ground")
        prev = -1
        for i in range(n):
            is_last = i == n - 1
            size = (large_size, large_size, large_size) if is_last else (1.0, 0.5, 0.5)
            x = float(i) + (large_size * 0.5 if is_last else 0.0)
            body, _shape = self.add_box(
                builder,
                size,
                0.0 if i == 0 else 1.0,
                0.5,
                (x, 0.0, 10.0),
                label=f"heavy_rope_{i}",
            )
            if prev >= 0:
                child_anchor = (-large_size * 0.5, 0.0, 0.0) if is_last else (-0.5, 0.0, 0.0)
                self.add_ball_joint(builder, prev, body, (0.5, 0.0, 0.0), child_anchor, f"heavy_rope_joint_{i}")
            prev = body

    def _validate_scene(self) -> None:
        pass

    def _update_scene_alignment_metrics(self, body_q: np.ndarray) -> None:
        del body_q

    def _assign_initial_velocities(self) -> None:
        if not self.initial_body_velocities:
            return
        for state in (self.state_0, self.state_1):
            qd = state.body_qd.numpy()
            for body, velocity in self.initial_body_velocities.items():
                qd[body, 0:3] = np.asarray(velocity, dtype=np.float32)
                qd[body, 3:6] = 0.0
            state.body_qd.assign(qd)

    def _break_overloaded_joints(self) -> None:
        if not self.breakable_joint_indices:
            return

        lambda_ang = self.solver.joint_lambda_ang.numpy()
        enabled = self.model.joint_enabled.numpy()
        changed = False
        for joint_index in self.breakable_joint_indices:
            if enabled[joint_index] and np.linalg.norm(lambda_ang[joint_index]) > 90.0:
                enabled[joint_index] = False
                self.broken_joint_count += 1
                changed = True
        if changed:
            self.model.joint_enabled.assign(enabled)

    def _update_metrics(self) -> None:
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(body_qd)):
            self.nan_detected = True
            return

        if body_q.size:
            pos = body_q[:, 0:3]
            self.max_abs_position = max(self.max_abs_position, float(np.max(np.abs(pos))))
            self.final_max_abs_position = float(np.max(np.abs(pos)))
            self.final_min_z = float(np.min(pos[:, 2]))
            self.final_max_z = float(np.max(pos[:, 2]))
            self.final_xy_radius = float(np.max(np.linalg.norm(pos[:, 0:2], axis=1)))
            self.min_z = min(self.min_z, self.final_min_z)
            self.max_z = max(self.max_z, self.final_max_z)
        if body_qd.size:
            self.final_speed = float(np.max(np.linalg.norm(body_qd[:, 0:3], axis=1)))
            self.final_angular_speed = float(np.max(np.linalg.norm(body_qd[:, 3:6], axis=1)))
            self.max_speed = max(self.max_speed, self.final_speed)
            self.max_angular_speed = max(self.max_angular_speed, self.final_angular_speed)
        if self.contacts.rigid_contact_count is not None:
            self.max_contacts = max(self.max_contacts, int(self.contacts.rigid_contact_count.numpy()[0]))
        self._update_scene_alignment_metrics(body_q)

    def simulate(self) -> None:
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self._break_overloaded_joints()
        self._update_metrics()

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        if self.nan_detected:
            raise AssertionError(f"{self.title}: non-finite state detected")
        if self.max_abs_position > POSITION_LIMIT:
            raise AssertionError(
                f"{self.title}: max |position| {self.max_abs_position:.3f} exceeds {POSITION_LIMIT:.3f}"
            )
        if self.max_speed > SPEED_LIMIT:
            raise AssertionError(f"{self.title}: max speed {self.max_speed:.3f} exceeds {SPEED_LIMIT:.3f}")
        if self.min_z < MIN_Z_LIMIT:
            raise AssertionError(f"{self.title}: min z {self.min_z:.3f} below {MIN_Z_LIMIT:.3f}")
        self._validate_scene()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=REPORT_FRAMES)
        return parser

    @property
    def metric_summary(self) -> dict[str, float | int | bool]:
        return {
            "max_speed": self.max_speed,
            "final_speed": self.final_speed,
            "max_angular_speed": self.max_angular_speed,
            "final_angular_speed": self.final_angular_speed,
            "max_abs_position": self.max_abs_position,
            "final_max_abs_position": self.final_max_abs_position,
            "min_z": self.min_z,
            "max_z": self.max_z,
            "final_min_z": self.final_min_z,
            "final_max_z": self.final_max_z,
            "final_xy_radius": self.final_xy_radius,
            "max_contacts": self.max_contacts,
            "broken_joint_count": self.broken_joint_count,
            "nan_detected": self.nan_detected,
            **self.extra_metrics,
        }


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
