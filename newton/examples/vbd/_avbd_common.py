# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""AVBD 3D reference-scene recreations for Newton VBD examples."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples

REFERENCE_COMMIT = "7701bd427d55ca5d03ea1fdf331912ded9169f4b"
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

REPORT_FRAMES = 180
PYRAMID_PLANE_ABS_Y_LIMIT = 0.10
PYRAMID_PLANE_SPAN_LIMIT = 0.16
PYRAMID_ROW_PLANE_SPAN_LIMIT = 0.12
STACK_CENTER_RADIUS_LIMIT = 0.15
STACK_XY_SPREAD_LIMIT = 0.08
STACK_LINE_FIT_LIMIT = 0.06
STACK_TILT_LIMIT = 0.02


@dataclass(frozen=True)
class SceneInfo:
    """Metadata for one AVBD reference scene."""

    title: str
    description: str
    build: Callable[[AvbdSceneExample, newton.ModelBuilder], None]
    camera_target: tuple[float, float, float]
    camera_distance: float
    camera_pitch: float
    camera_yaw: float
    camera_fov: float = 55.0
    validate: Callable[[AvbdSceneExample], None] | None = None
    position_limit: float = 500.0
    speed_limit: float = 500.0
    min_z_limit: float = -80.0
    report_frames: int = REPORT_FRAMES
    status_note: str = "Reference parameters reproduced."


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


class AvbdSceneExample:
    """Base class used by thin per-scene AVBD reference examples."""

    scene_name = "ground"

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.scene = SCENES[self.scene_name]

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

        self.scene.build(self, builder)
        builder.color()
        # The AVBD demo updates rigid bodies serially. Joint-only coloring puts
        # unjointed contact stacks into one parallel color, which changes the
        # contact solve to a Jacobi update and destabilizes the reference scenes.
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
        front = _camera_front(self.scene.camera_yaw, self.scene.camera_pitch)
        camera_pos = (
            self.scene.camera_target[0] - front[0] * self.scene.camera_distance,
            self.scene.camera_target[1] - front[1] * self.scene.camera_distance,
            self.scene.camera_target[2] - front[2] * self.scene.camera_distance,
        )
        self.viewer.set_camera(
            pos=_vec(camera_pos),
            pitch=self.scene.camera_pitch,
            yaw=self.scene.camera_yaw,
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = self.scene.camera_fov
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(_vec(self.scene.camera_target))

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
            raise AssertionError(f"{self.scene.title}: missing body label {label}") from exc

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
            raise AssertionError(f"{self.scene.title}: final speed {self.final_speed:.3f} exceeds {limit:.3f}")

    def assert_final_angular_speed_below(self, limit: float) -> None:
        if self.final_angular_speed > limit:
            raise AssertionError(
                f"{self.scene.title}: final angular speed {self.final_angular_speed:.3f} exceeds {limit:.3f}"
            )

    def assert_final_xy_radius_below(self, limit: float) -> None:
        if self.final_xy_radius > limit:
            raise AssertionError(f"{self.scene.title}: final xy radius {self.final_xy_radius:.3f} exceeds {limit:.3f}")

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

    def _update_scene_alignment_metrics(self, body_q: np.ndarray) -> None:
        if self.scene_name == "pyramid":
            brick_indices = self.body_indices_with_prefix("brick_")
            brick_pos = body_q[brick_indices, 0:3]
            abs_y = float(np.max(np.abs(brick_pos[:, 1])))
            y_span = float(np.max(brick_pos[:, 1]) - np.min(brick_pos[:, 1]))
            row_y_span = 0.0
            for row in range(16):
                row_indices = [self.body_index(f"brick_{row}_{x}") for x in range(16 - row)]
                row_y = body_q[row_indices, 1]
                row_y_span = max(row_y_span, float(np.max(row_y) - np.min(row_y)))
            self.extra_metrics["pyramid_observed_abs_y"] = max(
                abs_y, float(self.extra_metrics.get("pyramid_observed_abs_y", 0.0))
            )
            self.extra_metrics["pyramid_observed_lateral_span"] = max(
                y_span, float(self.extra_metrics.get("pyramid_observed_lateral_span", 0.0))
            )
            self.extra_metrics["pyramid_observed_row_lateral_span"] = max(
                row_y_span, float(self.extra_metrics.get("pyramid_observed_row_lateral_span", 0.0))
            )
        elif self.scene_name == "stack":
            stack_indices = [self.body_index(f"stack_{i}") for i in range(10)]
            stack_pos = body_q[stack_indices, 0:3]
            xy = stack_pos[:, 0:2]
            xy_center = np.mean(xy, axis=0)
            xy_radius = float(np.max(np.linalg.norm(xy, axis=1)))
            xy_spread = float(np.max(np.linalg.norm(xy - xy_center, axis=1)))
            line_fit, _line_rms, tilt = _fit_line_metrics(stack_pos)
            self.extra_metrics["stack_observed_xy_radius"] = max(
                xy_radius, float(self.extra_metrics.get("stack_observed_xy_radius", 0.0))
            )
            self.extra_metrics["stack_observed_xy_spread"] = max(
                xy_spread, float(self.extra_metrics.get("stack_observed_xy_spread", 0.0))
            )
            self.extra_metrics["stack_observed_line_fit"] = max(
                line_fit, float(self.extra_metrics.get("stack_observed_line_fit", 0.0))
            )
            self.extra_metrics["stack_observed_tilt_xy_per_z"] = max(
                tilt, float(self.extra_metrics.get("stack_observed_tilt_xy_per_z", 0.0))
            )

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
            raise AssertionError(f"{self.scene.title}: non-finite state detected")
        if self.max_abs_position > self.scene.position_limit:
            raise AssertionError(
                f"{self.scene.title}: max |position| {self.max_abs_position:.3f} exceeds "
                f"{self.scene.position_limit:.3f}"
            )
        if self.max_speed > self.scene.speed_limit:
            raise AssertionError(
                f"{self.scene.title}: max speed {self.max_speed:.3f} exceeds {self.scene.speed_limit:.3f}"
            )
        if self.min_z < self.scene.min_z_limit:
            raise AssertionError(f"{self.scene.title}: min z {self.min_z:.3f} below {self.scene.min_z_limit:.3f}")
        if self.scene.validate is not None:
            self.scene.validate(self)

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


def build_ground(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    example.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, 0.0), label="ground")
    example.add_box(builder, (1.0, 1.0, 1.0), 1.0, 0.5, (0.0, 0.0, 4.0), label="box")


def build_dynamic_friction(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    example.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, 0.0), label="ground")
    for x in range(11):
        friction = 5.0 - (x / 10.0 * 5.0)
        example.add_box(
            builder,
            (1.0, 1.0, 0.5),
            1.0,
            friction,
            (0.0, -30.0 + x * 2.0, 0.75),
            velocity=(10.0, 0.0, 0.0),
            label=f"slider_{x}",
        )


def build_static_friction(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    example.rigid_avbd_contact_alpha = 0.9

    example.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, 0.0), label="ground")

    angle = math.radians(30.0)
    ramp_quat = _quat_axis_angle((0.0, 1.0, 0.0), angle)
    ramp_position = (0.0, 0.0, 3.0)
    example.add_box(builder, (40.0, 24.0, 1.0), 0.0, 1.0, ramp_position, rotation=ramp_quat, label="ramp")

    ramp_tangent = _rotate_y(angle, (1.0, 0.0, 0.0))
    ramp_normal = _rotate_y(angle, (0.0, 0.0, 1.0))
    for i in range(11):
        friction = i / 10.0 * 0.25 + 0.25
        y = -10.0 + i * 2.0
        pos = _add(_add(_add(ramp_position, _mul(ramp_tangent, -12.0)), (0.0, y, 0.0)), _mul(ramp_normal, 1.05))
        example.add_box(builder, (1.0, 1.0, 1.0), 1.0, friction, pos, label=f"ramp_box_{i}")


def build_pyramid(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    example.rigid_body_serial_reverse = True
    example.rigid_avbd_contact_alpha = 0.85
    example.rigid_contact_k_start = 1000.0

    size = 16
    example.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, -0.5), label="ground")
    for y in range(size):
        for x in range(size - y):
            example.add_box(
                builder,
                (1.0, 0.5, 0.5),
                1.0,
                0.5,
                (x * 1.01 + y * 0.5 - size / 2.0, 0.0, y * 0.85 + 0.5),
                label=f"brick_{y}_{x}",
            )


def build_rope(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    example.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, -20.0), label="ground")
    prev = -1
    for i in range(20):
        body, _shape = example.add_box(
            builder,
            (1.0, 0.5, 0.5),
            0.0 if i == 0 else 1.0,
            0.5,
            (float(i), 0.0, 10.0),
            label=f"rope_{i}",
        )
        if prev >= 0:
            example.add_ball_joint(builder, prev, body, (0.5, 0.0, 0.0), (-0.5, 0.0, 0.0), f"rope_joint_{i}")
        prev = body


def build_heavy_rope(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    n = 20
    large_size = 5.0
    example.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, -20.0), label="ground")
    prev = -1
    for i in range(n):
        is_last = i == n - 1
        size = (large_size, large_size, large_size) if is_last else (1.0, 0.5, 0.5)
        x = float(i) + (large_size * 0.5 if is_last else 0.0)
        body, _shape = example.add_box(
            builder,
            size,
            0.0 if i == 0 else 1.0,
            0.5,
            (x, 0.0, 10.0),
            label=f"heavy_rope_{i}",
        )
        if prev >= 0:
            child_anchor = (-large_size * 0.5, 0.0, 0.0) if is_last else (-0.5, 0.0, 0.0)
            example.add_ball_joint(builder, prev, body, (0.5, 0.0, 0.0), child_anchor, f"heavy_rope_joint_{i}")
        prev = body


def build_spring(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    example.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, 0.0), label="ground")
    anchor, _anchor_shape = example.add_box(builder, (1.0, 1.0, 1.0), 0.0, 0.5, (0.0, 0.0, 14.0), label="anchor")
    block, _block_shape = example.add_box(builder, (2.0, 2.0, 2.0), 1.0, 0.5, (0.0, 0.0, 8.0), label="block")
    example.add_cable_joint(
        builder,
        anchor,
        block,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 4.0),
        100.0,
        "spring_joint",
        collision_filter_parent=False,
    )


def build_springs_ratio(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    n = 8
    example.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, -10.0), label="ground")
    prev = -1
    for i in range(n):
        x = (i - (n - 1) * 0.5) * 3.0
        body, _shape = example.add_box(
            builder,
            (1.0, 0.75, 0.75),
            0.0 if i == 0 or i == n - 1 else 1.0,
            0.5,
            (x, 0.0, 12.0),
            label=f"spring_ratio_{i}",
        )
        if prev >= 0:
            stiffness = 10.0 if i % 2 == 0 else 10000.0
            example.add_cable_joint(
                builder,
                prev,
                body,
                (0.5, 0.0, 0.0),
                (-2.5, 0.0, 0.0),
                stiffness,
                f"spring_ratio_joint_{i}",
            )
        prev = body


def build_stack(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    example.contact_matching = "disabled"
    example.rigid_contact_history = False
    example.rigid_avbd_contact_alpha = 0.9
    example.rigid_contact_k_start = 1000.0
    example.rigid_contact_stick_freeze_translation_eps = 5.0e-2
    example.rigid_contact_stick_freeze_angular_eps = 1.0e-1

    example.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, 0.0), label="ground")
    for i in range(10):
        example.add_box(builder, (1.0, 1.0, 1.0), 1.0, 0.5, (0.0, 0.0, i * 1.5 + 1.0), label=f"stack_{i}")


def build_stack_ratio(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    ground_thickness = 1.0
    example.add_box(builder, (100.0, 100.0, ground_thickness), 0.0, 0.5, (0.0, 0.0, 0.0), label="ground")
    top_z = ground_thickness * 0.5
    size = 1.0
    for i in range(4):
        half = size * 0.5
        center_z = top_z + half
        example.add_box(builder, (size, size, size), 1.0, 0.5, (0.0, 0.0, center_z), label=f"ratio_stack_{i}")
        top_z = center_z + half
        size *= 2.0


def build_soft_body(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    example.solver_iterations = 40
    example.soft_joint_hard = False
    example.rigid_joint_linear_ke = 1000.0
    example.rigid_joint_angular_ke = 250.0
    example.rigid_joint_linear_kd = 1.0
    example.rigid_joint_angular_kd = 1.0
    example.rigid_body_contact_buffer_size = 1024
    example.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, 0.0), label="ground")

    width = 4
    depth = 4
    height = 4
    stacks = 3
    size = 0.8
    half = size * 0.5
    base_z = 8.0
    stack_gap = 2.0

    for stack in range(stacks):
        bodies: list[list[list[int]]] = [[[0 for _z in range(height)] for _y in range(depth)] for _x in range(width)]
        shapes: list[list[list[int]]] = [[[0 for _z in range(height)] for _y in range(depth)] for _x in range(width)]
        stack_z = stack * (height * size + stack_gap)
        for x in range(width):
            for y in range(depth):
                for z in range(height):
                    px = (x - (width - 1) * 0.5) * size
                    py = (y - (depth - 1) * 0.5) * size
                    pz = base_z + stack_z + z * size
                    body, shape = example.add_box(
                        builder,
                        (size, size, size),
                        1.0,
                        0.5,
                        (px, py, pz),
                        label=f"soft_{stack}_{x}_{y}_{z}",
                    )
                    bodies[x][y][z] = body
                    shapes[x][y][z] = shape

        for x in range(1, width):
            for y in range(depth):
                for z in range(height):
                    example.add_fixed_joint(
                        builder,
                        bodies[x - 1][y][z],
                        bodies[x][y][z],
                        (half, 0.0, 0.0),
                        (-half, 0.0, 0.0),
                        f"soft_x_{stack}_{x}_{y}_{z}",
                        soft=True,
                    )
        for x in range(width):
            for y in range(1, depth):
                for z in range(height):
                    example.add_fixed_joint(
                        builder,
                        bodies[x][y - 1][z],
                        bodies[x][y][z],
                        (0.0, half, 0.0),
                        (0.0, -half, 0.0),
                        f"soft_y_{stack}_{x}_{y}_{z}",
                        soft=True,
                    )
        for x in range(width):
            for y in range(depth):
                for z in range(1, height):
                    example.add_fixed_joint(
                        builder,
                        bodies[x][y][z - 1],
                        bodies[x][y][z],
                        (0.0, 0.0, half),
                        (0.0, 0.0, -half),
                        f"soft_z_{stack}_{x}_{y}_{z}",
                        soft=True,
                    )

        for x in range(1, width):
            for y in range(depth):
                for z in range(1, height):
                    builder.add_shape_collision_filter_pair(shapes[x - 1][y][z - 1], shapes[x][y][z])
                    builder.add_shape_collision_filter_pair(shapes[x][y][z - 1], shapes[x - 1][y][z])
        for x in range(width):
            for y in range(1, depth):
                for z in range(1, height):
                    builder.add_shape_collision_filter_pair(shapes[x][y - 1][z - 1], shapes[x][y][z])
                    builder.add_shape_collision_filter_pair(shapes[x][y][z - 1], shapes[x][y - 1][z])
        for x in range(1, width):
            for y in range(1, depth):
                for z in range(height):
                    builder.add_shape_collision_filter_pair(shapes[x - 1][y - 1][z], shapes[x][y][z])
                    builder.add_shape_collision_filter_pair(shapes[x][y - 1][z], shapes[x - 1][y][z])


def build_bridge(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    n = 40
    plank_length = 1.0
    plank_width = 4.0
    plank_height = 0.5
    half_length = plank_length * 0.5
    half_width = plank_width * 0.5

    example.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, 0.0), label="ground")
    prev = -1
    for i in range(n):
        body, _shape = example.add_box(
            builder,
            (plank_length, plank_width, plank_height),
            0.0 if i == 0 or i == n - 1 else 1.0,
            0.5,
            (float(i) - n / 2.0, 0.0, 10.0),
            label=f"plank_{i}",
        )
        if prev >= 0:
            example.add_ball_joint(
                builder,
                prev,
                body,
                (half_length, half_width, 0.0),
                (-half_length, half_width, 0.0),
                f"bridge_joint_a_{i}",
            )
            example.add_ball_joint(
                builder,
                prev,
                body,
                (half_length, -half_width, 0.0),
                (-half_length, -half_width, 0.0),
                f"bridge_joint_b_{i}",
            )
        prev = body

    for x in range(n // 4):
        for y in range(n // 8):
            example.add_box(
                builder,
                (1.0, 1.0, 1.0),
                1.0,
                0.5,
                (float(x) - n / 8.0, 0.0, float(y) + 12.0),
                label=f"bridge_load_{x}_{y}",
            )


def build_breakable(example: AvbdSceneExample, builder: newton.ModelBuilder) -> None:
    example.rigid_contact_stick_freeze_translation_eps = 1.0e-3
    example.rigid_contact_stick_freeze_angular_eps = 5.0e-3

    n = 10
    m = 5
    example.add_box(builder, (100.0, 100.0, 1.0), 0.0, 0.5, (0.0, 0.0, 0.0), label="ground")
    prev = -1
    for i in range(n + 1):
        body, _shape = example.add_box(
            builder,
            (1.0, 1.0, 0.5),
            1.0,
            0.5,
            (float(i) - n / 2.0, 0.0, 6.0),
            label=f"breakable_beam_{i}",
        )
        if prev >= 0:
            joint = example.add_fixed_joint(
                builder,
                prev,
                body,
                (0.5, 0.0, 0.0),
                (-0.5, 0.0, 0.0),
                f"breakable_joint_{i}",
            )
            example.breakable_joint_indices.append(joint)
        prev = body

    example.add_box(builder, (1.0, 1.0, 5.0), 0.0, 0.5, (-n / 2.0, 0.0, 2.5), label="left_support")
    example.add_box(builder, (1.0, 1.0, 5.0), 0.0, 0.5, (n / 2.0, 0.0, 2.5), label="right_support")
    for i in range(m):
        example.add_box(builder, (2.0, 1.0, 1.0), 1.0, 0.5, (0.0, 0.0, i * 2.0 + 8.0), label=f"falling_load_{i}")


def validate_static_friction(example: AvbdSceneExample) -> None:
    ramp_tangent = np.asarray(_rotate_y(math.radians(30.0), (1.0, 0.0, 0.0)), dtype=np.float64)
    ramp_normal = np.asarray(_rotate_y(math.radians(30.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    low_friction_min_slide = float("inf")
    low_friction_min_downhill_speed = float("inf")
    max_tilt = 0.0
    max_speed = 0.0
    max_angular_speed = 0.0

    for i in range(2):
        label = f"ramp_box_{i}"
        body = example.body_index(label)
        initial_pos = np.asarray(example.initial_body_positions[body], dtype=np.float64)
        pos = example.body_position(label)
        slide = float(np.dot(pos - initial_pos, ramp_tangent))
        downhill_speed = float(np.dot(example.body_linear_velocity(label), ramp_tangent))
        low_friction_min_slide = min(low_friction_min_slide, slide)
        low_friction_min_downhill_speed = min(low_friction_min_downhill_speed, downhill_speed)

    for i in (9, 10):
        label = f"ramp_box_{i}"
        box_normal = example.body_local_axis_world(label, (0.0, 0.0, 1.0))
        dot = float(np.clip(np.dot(box_normal, ramp_normal), -1.0, 1.0))
        tilt = math.degrees(math.acos(dot))
        max_tilt = max(max_tilt, tilt)
        max_speed = max(max_speed, example.body_linear_speed(label))
        max_angular_speed = max(max_angular_speed, example.body_angular_speed(label))

    example.extra_metrics["low_friction_min_slide"] = low_friction_min_slide
    example.extra_metrics["low_friction_min_downhill_speed"] = low_friction_min_downhill_speed
    example.extra_metrics["high_friction_tilt_deg"] = max_tilt
    example.extra_metrics["high_friction_speed"] = max_speed
    example.extra_metrics["high_friction_angular_speed"] = max_angular_speed

    if low_friction_min_slide < 1.5:
        raise AssertionError(
            f"{example.scene.title}: low-friction ramp boxes slid only {low_friction_min_slide:.3f} m; "
            "expected visible downhill slip"
        )
    if low_friction_min_downhill_speed < 0.8:
        raise AssertionError(
            f"{example.scene.title}: low-friction ramp boxes downhill speed only "
            f"{low_friction_min_downhill_speed:.3f} m/s; expected continued sliding"
        )
    if max_tilt > 25.0:
        raise AssertionError(
            f"{example.scene.title}: high-friction ramp boxes tilted {max_tilt:.2f} deg; expected no tumbling"
        )
    if max_speed > 0.75:
        raise AssertionError(
            f"{example.scene.title}: high-friction ramp boxes moving at {max_speed:.3f} m/s; expected sticking"
        )
    if max_angular_speed > 0.75:
        raise AssertionError(
            f"{example.scene.title}: high-friction ramp boxes angular speed {max_angular_speed:.3f} rad/s; "
            "expected no tumbling"
        )


def validate_pyramid(example: AvbdSceneExample) -> None:
    body_q = example.state_0.body_q.numpy()
    brick_indices = example.body_indices_with_prefix("brick_")
    brick_pos = body_q[brick_indices, 0:3]
    initial_pos = np.asarray([example.initial_body_positions[body] for body in brick_indices], dtype=np.float64)
    displacement = np.linalg.norm(brick_pos - initial_pos, axis=1)
    max_brick_displacement = float(np.max(displacement))
    mean_brick_displacement = float(np.mean(displacement))
    brick_final_max_z = float(np.max(brick_pos[:, 2]))
    brick_lateral_span = float(np.max(brick_pos[:, 1]) - np.min(brick_pos[:, 1]))
    brick_abs_y = float(np.max(np.abs(brick_pos[:, 1])))
    max_row_lateral_span = 0.0
    for row in range(16):
        row_indices = [example.body_index(f"brick_{row}_{x}") for x in range(16 - row)]
        row_y = body_q[row_indices, 1]
        max_row_lateral_span = max(max_row_lateral_span, float(np.max(row_y) - np.min(row_y)))
    observed_abs_y = max(brick_abs_y, float(example.extra_metrics.get("pyramid_observed_abs_y", 0.0)))
    observed_lateral_span = max(
        brick_lateral_span, float(example.extra_metrics.get("pyramid_observed_lateral_span", 0.0))
    )
    observed_row_lateral_span = max(
        max_row_lateral_span, float(example.extra_metrics.get("pyramid_observed_row_lateral_span", 0.0))
    )

    example.extra_metrics["pyramid_max_brick_displacement"] = max_brick_displacement
    example.extra_metrics["pyramid_mean_brick_displacement"] = mean_brick_displacement
    example.extra_metrics["pyramid_final_max_z"] = brick_final_max_z
    example.extra_metrics["pyramid_lateral_span"] = brick_lateral_span
    example.extra_metrics["pyramid_abs_y"] = brick_abs_y
    example.extra_metrics["pyramid_row_lateral_span"] = max_row_lateral_span
    example.extra_metrics["pyramid_observed_abs_y"] = observed_abs_y
    example.extra_metrics["pyramid_observed_lateral_span"] = observed_lateral_span
    example.extra_metrics["pyramid_observed_row_lateral_span"] = observed_row_lateral_span

    example.assert_final_speed_below(5.0)
    example.assert_final_angular_speed_below(8.0)
    if observed_abs_y > PYRAMID_PLANE_ABS_Y_LIMIT:
        raise AssertionError(
            f"{example.scene.title}: brick centers left the source y=0 plane by {observed_abs_y:.3f} m; "
            f"expected <= {PYRAMID_PLANE_ABS_Y_LIMIT:.3f} m"
        )
    if observed_lateral_span > PYRAMID_PLANE_SPAN_LIMIT:
        raise AssertionError(
            f"{example.scene.title}: brick y-span reached {observed_lateral_span:.3f} m; "
            f"expected <= {PYRAMID_PLANE_SPAN_LIMIT:.3f} m"
        )
    if observed_row_lateral_span > PYRAMID_ROW_PLANE_SPAN_LIMIT:
        raise AssertionError(
            f"{example.scene.title}: one pyramid row spread {observed_row_lateral_span:.3f} m out of plane; "
            f"expected <= {PYRAMID_ROW_PLANE_SPAN_LIMIT:.3f} m"
        )
    if max_brick_displacement > 8.0:
        raise AssertionError(
            f"{example.scene.title}: brick displacement {max_brick_displacement:.3f} m exceeds 8.000; "
            "pyramid did not stay compact"
        )
    if brick_final_max_z < 6.0:
        raise AssertionError(
            f"{example.scene.title}: tallest brick center is {brick_final_max_z:.3f} m; "
            "expected the pile to retain visible height"
        )
    if brick_lateral_span > 4.0:
        raise AssertionError(
            f"{example.scene.title}: brick lateral span {brick_lateral_span:.3f} m exceeds 4.000; "
            "pyramid scattered sideways"
        )
    if example.final_min_z < -2.0:
        raise AssertionError(f"{example.scene.title}: final min z {example.final_min_z:.3f} below -2.000")
    if example.final_max_z > 12.0:
        raise AssertionError(
            f"{example.scene.title}: final max z {example.final_max_z:.3f} exceeds 12.000; pyramid launched upward"
        )


def validate_stack_ratio(example: AvbdSceneExample) -> None:
    example.assert_final_speed_below(2.0)
    example.assert_final_angular_speed_below(3.0)
    example.assert_final_xy_radius_below(2.5)
    if example.final_max_z > 13.0:
        raise AssertionError(
            f"{example.scene.title}: final max z {example.final_max_z:.3f} exceeds 13.000; stack launched upward"
        )


def validate_stack(example: AvbdSceneExample) -> None:
    body_q = example.state_0.body_q.numpy()
    stack_indices = [example.body_index(f"stack_{i}") for i in range(10)]
    stack_pos = body_q[stack_indices, 0:3]
    xy_radius = float(np.max(np.linalg.norm(stack_pos[:, 0:2], axis=1)))
    xy_center = np.mean(stack_pos[:, 0:2], axis=0)
    xy_spread = float(np.max(np.linalg.norm(stack_pos[:, 0:2] - xy_center, axis=1)))
    line_fit, line_fit_rms, tilt_xy_per_z = _fit_line_metrics(stack_pos)
    observed_xy_radius = max(xy_radius, float(example.extra_metrics.get("stack_observed_xy_radius", 0.0)))
    observed_xy_spread = max(xy_spread, float(example.extra_metrics.get("stack_observed_xy_spread", 0.0)))
    observed_line_fit = max(line_fit, float(example.extra_metrics.get("stack_observed_line_fit", 0.0)))
    observed_tilt = max(tilt_xy_per_z, float(example.extra_metrics.get("stack_observed_tilt_xy_per_z", 0.0)))
    z_spacing = np.diff(stack_pos[:, 2])
    min_z_spacing = float(np.min(z_spacing))
    max_z_spacing = float(np.max(z_spacing))
    top_z = float(stack_pos[-1, 2])

    example.extra_metrics["stack_xy_radius"] = xy_radius
    example.extra_metrics["stack_xy_spread"] = xy_spread
    example.extra_metrics["stack_line_fit"] = line_fit
    example.extra_metrics["stack_line_fit_rms"] = line_fit_rms
    example.extra_metrics["stack_tilt_xy_per_z"] = tilt_xy_per_z
    example.extra_metrics["stack_observed_xy_radius"] = observed_xy_radius
    example.extra_metrics["stack_observed_xy_spread"] = observed_xy_spread
    example.extra_metrics["stack_observed_line_fit"] = observed_line_fit
    example.extra_metrics["stack_observed_tilt_xy_per_z"] = observed_tilt
    example.extra_metrics["stack_min_z_spacing"] = min_z_spacing
    example.extra_metrics["stack_max_z_spacing"] = max_z_spacing
    example.extra_metrics["stack_top_z"] = top_z

    example.assert_final_speed_below(0.25)
    example.assert_final_angular_speed_below(0.5)
    if observed_xy_radius > STACK_CENTER_RADIUS_LIMIT:
        raise AssertionError(
            f"{example.scene.title}: stack center radius reached {observed_xy_radius:.3f} m; "
            f"expected <= {STACK_CENTER_RADIUS_LIMIT:.3f} m"
        )
    if observed_xy_spread > STACK_XY_SPREAD_LIMIT:
        raise AssertionError(
            f"{example.scene.title}: stack xy spread reached {observed_xy_spread:.3f} m; "
            f"expected <= {STACK_XY_SPREAD_LIMIT:.3f} m"
        )
    if observed_line_fit > STACK_LINE_FIT_LIMIT:
        raise AssertionError(
            f"{example.scene.title}: stack centers deviated {observed_line_fit:.3f} m from a fitted line; "
            f"expected <= {STACK_LINE_FIT_LIMIT:.3f} m"
        )
    if observed_tilt > STACK_TILT_LIMIT:
        raise AssertionError(
            f"{example.scene.title}: stack fitted line tilt reached {observed_tilt:.3f} xy/z; "
            f"expected <= {STACK_TILT_LIMIT:.3f}"
        )
    if min_z_spacing < 0.85 or max_z_spacing > 1.15:
        raise AssertionError(
            f"{example.scene.title}: vertical spacing range {min_z_spacing:.3f}-{max_z_spacing:.3f} m; "
            "expected separated unit cubes"
        )
    if top_z < 9.5:
        raise AssertionError(f"{example.scene.title}: top cube center {top_z:.3f} m below expected stack height")
    if example.final_max_z > 16.0:
        raise AssertionError(
            f"{example.scene.title}: final max z {example.final_max_z:.3f} exceeds 16.000; stack expanded upward"
        )


def validate_soft_body(example: AvbdSceneExample) -> None:
    body_q = example.state_0.body_q.numpy()
    min_stack_z_extent = float("inf")
    min_neighbor_distance = float("inf")
    for stack in range(3):
        stack_indices = example.body_indices_with_prefix(f"soft_{stack}_")
        stack_pos = body_q[stack_indices, 0:3]
        z_extent = float(np.max(stack_pos[:, 2]) - np.min(stack_pos[:, 2]))
        min_stack_z_extent = min(min_stack_z_extent, z_extent)
        positions_by_label = {
            label: body_q[body, 0:3]
            for label, body in example.body_labels.items()
            if label.startswith(f"soft_{stack}_")
        }
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    p = positions_by_label[f"soft_{stack}_{x}_{y}_{z}"]
                    for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                        nx = x + dx
                        ny = y + dy
                        nz = z + dz
                        if nx < 4 and ny < 4 and nz < 4:
                            q = positions_by_label[f"soft_{stack}_{nx}_{ny}_{nz}"]
                            min_neighbor_distance = min(min_neighbor_distance, float(np.linalg.norm(q - p)))

    example.extra_metrics["soft_body_min_stack_z_extent"] = min_stack_z_extent
    example.extra_metrics["soft_body_min_neighbor_distance"] = min_neighbor_distance

    example.assert_final_speed_below(1.0)
    example.assert_final_angular_speed_below(2.0)
    if min_stack_z_extent < 2.0:
        raise AssertionError(
            f"{example.scene.title}: lattice z extent collapsed to {min_stack_z_extent:.3f} m; "
            "expected each 4x4x4 block to retain most of its height"
        )
    if min_neighbor_distance < 0.55:
        raise AssertionError(
            f"{example.scene.title}: neighboring lattice bodies are {min_neighbor_distance:.3f} m apart; "
            "expected soft joints to prevent self-collapse"
        )
    if example.final_max_z > 22.0:
        raise AssertionError(
            f"{example.scene.title}: final max z {example.final_max_z:.3f} exceeds 22.000; lattice is expanding"
        )


def validate_bridge(example: AvbdSceneExample) -> None:
    example.assert_final_speed_below(12.0)
    example.assert_final_angular_speed_below(16.0)
    if example.final_max_abs_position > 24.0:
        raise AssertionError(
            f"{example.scene.title}: final max |position| {example.final_max_abs_position:.3f} exceeds 24.000"
        )


def validate_breakable(example: AvbdSceneExample) -> None:
    example.assert_final_speed_below(10.0)
    example.assert_final_angular_speed_below(16.0)
    if example.final_max_abs_position > 18.0:
        raise AssertionError(
            f"{example.scene.title}: final max |position| {example.final_max_abs_position:.3f} exceeds 18.000"
        )


SCENES: dict[str, SceneInfo] = {
    "ground": SceneInfo(
        "Ground",
        "Single dynamic cube falling onto a static box ground.",
        build_ground,
        (0.0, 0.0, 2.0),
        14.0,
        -20.0,
        135.0,
    ),
    "dynamic_friction": SceneInfo(
        "Dynamic Friction",
        "Eleven sliding boxes with friction decreasing from 5 to 0.",
        build_dynamic_friction,
        (15.0, -20.0, 1.2),
        42.0,
        -25.0,
        120.0,
        position_limit=800.0,
    ),
    "static_friction": SceneInfo(
        "Static Friction",
        "Boxes with friction 0.25 to 0.5 resting on a 30 degree ramp.",
        build_static_friction,
        (-5.0, 0.0, 5.0),
        36.0,
        -25.0,
        135.0,
        validate=validate_static_friction,
    ),
    "pyramid": SceneInfo(
        "Pyramid",
        "Sixteen-row brick pyramid settling under gravity.",
        build_pyramid,
        (0.0, 0.0, 6.0),
        30.0,
        -22.0,
        135.0,
        validate=validate_pyramid,
        speed_limit=100.0,
    ),
    "rope": SceneInfo(
        "Rope",
        "Twenty rigid box links connected by hard ball joints.",
        build_rope,
        (9.5, 0.0, 5.0),
        34.0,
        -25.0,
        135.0,
        min_z_limit=-40.0,
    ),
    "heavy_rope": SceneInfo(
        "Heavy Rope",
        "Rope with a large heavy terminal block.",
        build_heavy_rope,
        (11.0, 0.0, 5.0),
        42.0,
        -24.0,
        135.0,
        min_z_limit=-40.0,
    ),
    "spring": SceneInfo(
        "Spring",
        "Block suspended from a static anchor by a soft stretch constraint.",
        build_spring,
        (0.0, 0.0, 8.0),
        24.0,
        -18.0,
        135.0,
        status_note="Uses a Newton cable stretch constraint to approximate the AVBD rigid spring.",
    ),
    "springs_ratio": SceneInfo(
        "Spring Ratio",
        "Alternating soft and stiff stretch constraints between eight links.",
        build_springs_ratio,
        (0.0, 0.0, 11.0),
        34.0,
        -18.0,
        115.0,
        status_note="Uses cable stretch constraints with the source stiffness ratios.",
    ),
    "stack": SceneInfo(
        "Stack",
        "Ten unit cubes settling into a vertical stack.",
        build_stack,
        (0.0, 0.0, 7.0),
        24.0,
        -22.0,
        135.0,
        validate=validate_stack,
        speed_limit=1000.0,
    ),
    "stack_ratio": SceneInfo(
        "Stack Ratio",
        "Four cubes with side lengths 1, 2, 4, and 8 stacked by increasing mass ratio.",
        build_stack_ratio,
        (0.0, 0.0, 5.0),
        30.0,
        -22.0,
        135.0,
        validate=validate_stack_ratio,
        speed_limit=10.0,
    ),
    "soft_body": SceneInfo(
        "Soft Body",
        "Three 4x4x4 lattices of rigid cubes joined by finite-stiffness soft fixed joints.",
        build_soft_body,
        (0.0, 0.0, 14.0),
        34.0,
        -22.0,
        135.0,
        validate=validate_soft_body,
        report_frames=120,
        speed_limit=20.0,
        status_note="Uses source soft fixed-joint stiffness with 40 Newton VBD iterations to prevent lattice collapse.",
    ),
    "bridge": SceneInfo(
        "Bridge",
        "Forty planks coupled by paired hard ball joints and loaded by falling cubes.",
        build_bridge,
        (0.0, 0.0, 10.0),
        50.0,
        -18.0,
        90.0,
        validate=validate_bridge,
        position_limit=25.0,
        report_frames=150,
        speed_limit=25.0,
    ),
    "breakable": SceneInfo(
        "Breakable",
        "Beam of hard fixed joints with runtime joint disabling above the source break force.",
        build_breakable,
        (0.0, 0.0, 7.0),
        28.0,
        -22.0,
        135.0,
        validate=validate_breakable,
        speed_limit=15.0,
        status_note="Fracture disables overloaded joints; collision filters remain static after breakage.",
    ),
}


def main(example_cls: type[AvbdSceneExample]) -> None:
    parser = example_cls.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(example_cls(viewer, args), args)
