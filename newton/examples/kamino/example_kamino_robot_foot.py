# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example for a linkage-actuated robot foot.
#
# Command: python -m newton.examples kamino_robot_foot --device cpu
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples


_PITCH_AMPLITUDE = math.radians(50.0)
_ROLL_AMPLITUDE = math.radians(40.0)
_PITCH_FREQUENCY = 2.55
_ROLL_FREQUENCY = 3.15

_ANKLE_POS = np.array([0.0, 0.0, 0.55], dtype=np.float32)
_PITCH_MOTOR_POS = np.array([-0.30, 0.0, 1.50], dtype=np.float32)
_ROLL_MOTOR_POS = np.array([0.24, 0.0, 1.24], dtype=np.float32)
_PITCH_HORN_LENGTH = 0.24
_ROLL_HORN_HALF_WIDTH = 0.30
_FOOT_REAR_PIVOT = np.array([-0.42, 0.0, -0.06], dtype=np.float32)
_FOOT_LEFT_PIVOT = np.array([0.0, 0.30, -0.14], dtype=np.float32)
_FOOT_RIGHT_PIVOT = np.array([0.0, -0.30, -0.14], dtype=np.float32)

_LEG_COLOR = wp.vec3(0.42, 0.45, 0.48)
_FOOT_COLOR = wp.vec3(0.12, 0.30, 0.42)
_SOLE_COLOR = wp.vec3(0.08, 0.08, 0.08)
_PITCH_COLOR = wp.vec3(0.86, 0.36, 0.20)
_ROLL_COLOR = wp.vec3(0.22, 0.55, 0.72)
_ROD_COLOR = wp.vec3(0.80, 0.68, 0.28)
_JOINT_COLOR = wp.vec3(0.10, 0.10, 0.12)

_CAMERA_VIEWS = {
    "side": (wp.vec3(0.0, -3.0, 0.95), -4.0, 90.0, 42.0),
    "front": (wp.vec3(3.0, 0.0, 0.95), -4.0, -180.0, 42.0),
    "perspective": (wp.vec3(2.1, -2.6, 1.6), -18.0, 135.0, 58.0),
    "three-quarter": (wp.vec3(2.1, -2.6, 1.6), -18.0, 135.0, 58.0),
}


def _as_vec3(v: np.ndarray) -> wp.vec3:
    return wp.vec3(float(v[0]), float(v[1]), float(v[2]))


def _set_camera(viewer: newton.viewer.ViewerBase, camera_view: str) -> None:
    camera_pos, pitch, yaw, fov = _CAMERA_VIEWS[camera_view]
    if hasattr(viewer, "set_camera"):
        viewer.set_camera(camera_pos, pitch, yaw)
    if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
        viewer.camera.fov = fov


def _visual_shape_cfg() -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0
    cfg.collision_group = 0
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    return cfg


def _box_inertia(mass: float, hx: float, hy: float, hz: float) -> wp.mat33:
    x = 2.0 * hx
    y = 2.0 * hy
    z = 2.0 * hz
    return wp.mat33(
        (mass * (y * y + z * z) / 12.0, 0.0, 0.0),
        (0.0, mass * (x * x + z * z) / 12.0, 0.0),
        (0.0, 0.0, mass * (x * x + y * y) / 12.0),
    )


def _quat_between(a: np.ndarray, b: np.ndarray) -> wp.quat:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1.0e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return wp.quat_from_axis_angle(_as_vec3(axis.astype(np.float32)), math.pi)

    cross = np.cross(a, b)
    scale = math.sqrt((1.0 + dot) * 2.0)
    inv_scale = 1.0 / scale
    return wp.quat(
        float(cross[0] * inv_scale),
        float(cross[1] * inv_scale),
        float(cross[2] * inv_scale),
        float(0.5 * scale),
    )


def _motor_axis_config(axis: newton.Axis, lower: float, upper: float) -> newton.ModelBuilder.JointDofConfig:
    return newton.ModelBuilder.JointDofConfig(
        axis=axis,
        limit_lower=lower,
        limit_upper=upper,
        limit_ke=250.0,
        limit_kd=15.0,
        target_ke=700.0,
        target_kd=45.0,
        armature=0.02,
        friction=0.005,
        effort_limit=700.0,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )


def _passive_axis(axis: newton.Axis, lower: float = -math.pi, upper: float = math.pi) -> newton.ModelBuilder.JointDofConfig:
    return newton.ModelBuilder.JointDofConfig(
        axis=axis,
        limit_lower=lower,
        limit_upper=upper,
        limit_ke=50.0,
        limit_kd=3.0,
        armature=0.002,
        friction=0.002,
        actuator_mode=newton.JointTargetMode.NONE,
    )


def _universal_axes() -> list[newton.ModelBuilder.JointDofConfig]:
    return [
        _passive_axis(newton.Axis.X, -1.4, 1.4),
        _passive_axis(newton.Axis.Y, -1.4, 1.4),
    ]


def _add_rod_universal(
    builder: newton.ModelBuilder,
    parent: int,
    child: int,
    parent_pos: wp.vec3,
    child_pos: wp.vec3,
    rod_q: wp.quat,
    label: str,
) -> int:
    return builder.add_joint_d6(
        parent=parent,
        child=child,
        angular_axes=_universal_axes(),
        parent_xform=wp.transform(p=parent_pos, q=rod_q),
        child_xform=wp.transform(p=child_pos, q=wp.quat_identity()),
        label=label,
    )


def _add_rod(
    builder: newton.ModelBuilder,
    label: str,
    top: np.ndarray,
    bottom: np.ndarray,
    shape_cfg: newton.ModelBuilder.ShapeConfig,
    radius: float = 0.018,
) -> tuple[int, float, wp.quat]:
    vector = bottom - top
    length = float(np.linalg.norm(vector))
    center = 0.5 * (top + bottom)
    rod_q = _quat_between(np.array([0.0, 0.0, 1.0], dtype=np.float32), vector)
    body = builder.add_link(
        xform=wp.transform(p=_as_vec3(center), q=rod_q),
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_box_inertia(0.08, radius, radius, 0.5 * length),
        mass=0.08,
        label=label,
        lock_inertia=True,
    )
    builder.add_shape_box(
        body,
        hx=radius,
        hy=radius,
        hz=0.5 * length,
        cfg=shape_cfg,
        color=_ROD_COLOR,
        label=f"{label}_bar",
    )
    return body, length, rod_q


def _add_robot_foot(builder: newton.ModelBuilder) -> dict[str, str]:
    shape_cfg = _visual_shape_cfg()

    leg = builder.add_link(
        xform=wp.transform(p=_as_vec3(_ANKLE_POS), q=wp.quat_identity()),
        com=wp.vec3(0.0, 0.0, 0.55),
        inertia=_box_inertia(4.0, 0.08, 0.08, 0.65),
        mass=4.0,
        label="lower_leg",
        lock_inertia=True,
    )
    foot = builder.add_link(
        xform=wp.transform(p=_as_vec3(_ANKLE_POS), q=wp.quat_identity()),
        com=wp.vec3(0.08, 0.0, -0.16),
        inertia=_box_inertia(1.4, 0.42, 0.30, 0.06),
        mass=1.4,
        label="foot_platform",
        lock_inertia=True,
    )
    roll_frame = builder.add_link(
        xform=wp.transform(p=_as_vec3(_ANKLE_POS), q=wp.quat_identity()),
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_box_inertia(0.35, 0.05, 0.36, 0.04),
        mass=0.35,
        label="ankle_roll_yoke",
        lock_inertia=True,
    )
    pitch_horn = builder.add_link(
        xform=wp.transform(p=_as_vec3(_PITCH_MOTOR_POS), q=wp.quat_identity()),
        com=wp.vec3(0.5 * _PITCH_HORN_LENGTH, 0.0, 0.0),
        inertia=_box_inertia(0.18, 0.5 * _PITCH_HORN_LENGTH, 0.025, 0.025),
        mass=0.18,
        label="pitch_motor_horn",
        lock_inertia=True,
    )
    roll_horn = builder.add_link(
        xform=wp.transform(p=_as_vec3(_ROLL_MOTOR_POS), q=wp.quat_identity()),
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_box_inertia(0.22, 0.025, _ROLL_HORN_HALF_WIDTH, 0.025),
        mass=0.22,
        label="roll_motor_crossbar",
        lock_inertia=True,
    )

    pitch_horn_tip = _PITCH_MOTOR_POS + np.array([_PITCH_HORN_LENGTH, 0.0, 0.0], dtype=np.float32)
    roll_left_tip = _ROLL_MOTOR_POS + np.array([0.0, _ROLL_HORN_HALF_WIDTH, 0.0], dtype=np.float32)
    roll_right_tip = _ROLL_MOTOR_POS + np.array([0.0, -_ROLL_HORN_HALF_WIDTH, 0.0], dtype=np.float32)
    pitch_foot_anchor = _ANKLE_POS + _FOOT_REAR_PIVOT
    left_foot_anchor = _ANKLE_POS + _FOOT_LEFT_PIVOT
    right_foot_anchor = _ANKLE_POS + _FOOT_RIGHT_PIVOT

    pitch_rod, pitch_rod_length, pitch_rod_q = _add_rod(
        builder, "pitch_pushrod", pitch_horn_tip, pitch_foot_anchor, shape_cfg
    )
    left_rod, left_rod_length, left_rod_q = _add_rod(builder, "roll_left_pushrod", roll_left_tip, left_foot_anchor, shape_cfg)
    right_rod, right_rod_length, right_rod_q = _add_rod(
        builder, "roll_right_pushrod", roll_right_tip, right_foot_anchor, shape_cfg
    )

    builder.add_shape_box(
        leg,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.55), q=wp.quat_identity()),
        hx=0.07,
        hy=0.07,
        hz=0.65,
        cfg=shape_cfg,
        color=_LEG_COLOR,
        label="lower_leg_column",
    )
    builder.add_shape_box(
        leg,
        xform=wp.transform(p=wp.vec3(-0.30, 0.0, 0.95), q=wp.quat_identity()),
        hx=0.14,
        hy=0.11,
        hz=0.08,
        cfg=shape_cfg,
        color=_PITCH_COLOR,
        label="pitch_motor_case",
    )
    builder.add_shape_box(
        leg,
        xform=wp.transform(p=wp.vec3(0.24, 0.0, 0.69), q=wp.quat_identity()),
        hx=0.12,
        hy=0.12,
        hz=0.08,
        cfg=shape_cfg,
        color=_ROLL_COLOR,
        label="roll_motor_case",
    )

    builder.add_shape_box(
        foot,
        xform=wp.transform(p=wp.vec3(0.08, 0.0, -0.16), q=wp.quat_identity()),
        hx=0.42,
        hy=0.28,
        hz=0.045,
        cfg=shape_cfg,
        color=_FOOT_COLOR,
        label="foot_main_plate",
    )
    builder.add_shape_box(
        foot,
        xform=wp.transform(p=wp.vec3(0.12, 0.0, -0.22), q=wp.quat_identity()),
        hx=0.46,
        hy=0.30,
        hz=0.018,
        cfg=shape_cfg,
        color=_SOLE_COLOR,
        label="foot_sole",
    )
    builder.add_shape_sphere(
        foot,
        xform=wp.transform(p=_as_vec3(_FOOT_REAR_PIVOT), q=wp.quat_identity()),
        radius=0.035,
        cfg=shape_cfg,
        color=_JOINT_COLOR,
        label="heel_pitch_pivot",
    )
    builder.add_shape_box(
        roll_frame,
        hx=0.045,
        hy=0.36,
        hz=0.035,
        cfg=shape_cfg,
        color=_JOINT_COLOR,
        label="ankle_roll_yoke_crossmember",
    )
    for side, y in (("left", 0.30), ("right", -0.30)):
        builder.add_shape_box(
            roll_frame,
            xform=wp.transform(p=wp.vec3(0.0, y, -0.08), q=wp.quat_identity()),
            hx=0.035,
            hy=0.025,
            hz=0.12,
            cfg=shape_cfg,
            color=_JOINT_COLOR,
            label=f"{side}_roll_yoke_cheek",
        )
    for label, pivot in (
        ("left_roll_pivot", _FOOT_LEFT_PIVOT),
        ("right_roll_pivot", _FOOT_RIGHT_PIVOT),
    ):
        builder.add_shape_sphere(
            roll_frame,
            xform=wp.transform(p=_as_vec3(pivot), q=wp.quat_identity()),
            radius=0.035,
            cfg=shape_cfg,
            color=_JOINT_COLOR,
            label=label,
        )

    builder.add_shape_box(
        pitch_horn,
        xform=wp.transform(p=wp.vec3(0.5 * _PITCH_HORN_LENGTH, 0.0, 0.0), q=wp.quat_identity()),
        hx=0.5 * _PITCH_HORN_LENGTH,
        hy=0.025,
        hz=0.025,
        cfg=shape_cfg,
        color=_PITCH_COLOR,
        label="pitch_servo_horn",
    )
    builder.add_shape_box(
        roll_horn,
        hx=0.025,
        hy=_ROLL_HORN_HALF_WIDTH,
        hz=0.025,
        cfg=shape_cfg,
        color=_ROLL_COLOR,
        label="roll_servo_crossbar",
    )
    for body, label, local_tip in (
        (pitch_horn, "pitch_horn_tip", wp.vec3(_PITCH_HORN_LENGTH, 0.0, 0.0)),
        (roll_horn, "roll_left_horn_tip", wp.vec3(0.0, _ROLL_HORN_HALF_WIDTH, 0.0)),
        (roll_horn, "roll_right_horn_tip", wp.vec3(0.0, -_ROLL_HORN_HALF_WIDTH, 0.0)),
    ):
        builder.add_shape_sphere(
            body,
            xform=wp.transform(p=local_tip, q=wp.quat_identity()),
            radius=0.03,
            cfg=shape_cfg,
            color=_JOINT_COLOR,
            label=label,
        )

    tree_joints = []
    tree_joints.append(
        builder.add_joint_fixed(
            parent=-1,
            child=leg,
            parent_xform=wp.transform(p=_as_vec3(_ANKLE_POS), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            label="world_to_lower_leg",
        )
    )
    tree_joints.append(
        builder.add_joint_revolute(
            parent=leg,
            child=roll_frame,
            axis=_passive_axis(newton.Axis.X, -0.9, 0.9),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            label="ankle_dual_axis_gimbal_roll",
        )
    )
    tree_joints.append(
        builder.add_joint_revolute(
            parent=roll_frame,
            child=foot,
            axis=_passive_axis(newton.Axis.Y, -1.1, 1.1),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            label="ankle_dual_axis_gimbal_pitch",
        )
    )
    pitch_motor_joint = builder.add_joint_revolute(
        parent=leg,
        child=pitch_horn,
        axis=_motor_axis_config(newton.Axis.Y, -math.radians(60.0), math.radians(60.0)),
        parent_xform=wp.transform(p=_as_vec3(_PITCH_MOTOR_POS - _ANKLE_POS), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        label="pitch_motor_reducer",
    )
    tree_joints.append(pitch_motor_joint)
    tree_joints.append(
        _add_rod_universal(
            builder,
            pitch_horn,
            pitch_rod,
            wp.vec3(_PITCH_HORN_LENGTH, 0.0, 0.0),
            wp.vec3(0.0, 0.0, -0.5 * pitch_rod_length),
            pitch_rod_q,
            "pitch_horn_to_pushrod_universal",
        )
    )

    roll_motor_joint = builder.add_joint_revolute(
        parent=leg,
        child=roll_horn,
        axis=_motor_axis_config(newton.Axis.X, -math.radians(50.0), math.radians(50.0)),
        parent_xform=wp.transform(p=_as_vec3(_ROLL_MOTOR_POS - _ANKLE_POS), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        label="roll_motor_reducer",
    )
    tree_joints.append(roll_motor_joint)
    for rod, rod_length, rod_q, side, horn_tip in (
        (left_rod, left_rod_length, left_rod_q, "left", wp.vec3(0.0, _ROLL_HORN_HALF_WIDTH, 0.0)),
        (right_rod, right_rod_length, right_rod_q, "right", wp.vec3(0.0, -_ROLL_HORN_HALF_WIDTH, 0.0)),
    ):
        tree_joints.append(
            _add_rod_universal(
                builder,
                roll_horn,
                rod,
                horn_tip,
                wp.vec3(0.0, 0.0, -0.5 * rod_length),
                rod_q,
                f"roll_horn_to_{side}_pushrod_universal",
            )
        )

    builder.add_articulation(tree_joints, label="linkage_actuated_foot")

    _add_rod_universal(
        builder,
        foot,
        pitch_rod,
        _as_vec3(_FOOT_REAR_PIVOT),
        wp.vec3(0.0, 0.0, 0.5 * pitch_rod_length),
        pitch_rod_q,
        "pitch_pushrod_to_foot_universal",
    )
    _add_rod_universal(
        builder,
        roll_frame,
        left_rod,
        _as_vec3(_FOOT_LEFT_PIVOT),
        wp.vec3(0.0, 0.0, 0.5 * left_rod_length),
        left_rod_q,
        "roll_left_pushrod_to_gimbal_universal",
    )
    _add_rod_universal(
        builder,
        roll_frame,
        right_rod,
        _as_vec3(_FOOT_RIGHT_PIVOT),
        wp.vec3(0.0, 0.0, 0.5 * right_rod_length),
        right_rod_q,
        "roll_right_pushrod_to_gimbal_universal",
    )

    return {
        "pitch": "pitch_motor_reducer",
        "roll": "roll_motor_reducer",
    }


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = 0.004
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.sim_time = 0.0
        self.viewer = viewer

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        motor_labels = _add_robot_foot(builder)

        self.model = builder.finalize(skip_validation_joints=True)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.motor_target_q_indices = self._find_motor_target_indices(motor_labels)
        self.motor_qd_indices = self._find_motor_qd_indices(motor_labels)
        self.foot_body = self._find_body("foot_platform")

        solver_config = newton.solvers.SolverKamino.Config.from_model(self.model)
        solver_config.use_collision_detector = False
        solver_config.use_fk_solver = False
        solver_config.angular_velocity_damping = 0.04
        solver_config.dynamics.preconditioning = True
        solver_config.padmm.primal_tolerance = 5.0e-5
        solver_config.padmm.dual_tolerance = 5.0e-5
        solver_config.padmm.compl_tolerance = 5.0e-5
        solver_config.padmm.max_iterations = 120
        solver_config.padmm.rho_0 = 0.1
        solver_config.padmm.use_acceleration = True
        solver_config.padmm.warmstart_mode = "containers"

        self.solver = newton.solvers.SolverKamino(model=self.model, config=solver_config)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.target_q = self.model.joint_target_q.numpy()
        self.target_qd = self.model.joint_target_qd.numpy()
        self.max_pitch_target = 0.0
        self.max_roll_target = 0.0

        self.viewer.set_model(self.model)
        camera_view = getattr(args, "camera_view", "perspective")
        _set_camera(self.viewer, camera_view)

        self._update_motor_targets(0.0)
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.solver.reset(self.state_0)
        self.capture()

    def _find_body(self, name: str) -> int:
        for body_id, label in enumerate(self.model.body_label):
            if label.rsplit("/", 1)[-1] == name:
                return body_id
        raise RuntimeError(f"Missing body in constructed scene: {name}")

    def _find_joint(self, name: str) -> int:
        for joint_id, label in enumerate(self.model.joint_label):
            if label.rsplit("/", 1)[-1] == name:
                return joint_id
        raise RuntimeError(f"Missing joint in constructed scene: {name}")

    def _find_motor_target_indices(self, motor_labels: dict[str, str]) -> dict[str, int]:
        target_starts = self.model.joint_target_q_start.numpy()
        return {name: int(target_starts[self._find_joint(label)]) for name, label in motor_labels.items()}

    def _find_motor_qd_indices(self, motor_labels: dict[str, str]) -> dict[str, int]:
        qd_starts = self.model.joint_qd_start.numpy()
        return {name: int(qd_starts[self._find_joint(label)]) for name, label in motor_labels.items()}

    def _update_motor_targets(self, sim_time: float) -> None:
        pitch_phase = 2.0 * math.pi * _PITCH_FREQUENCY * sim_time
        roll_phase = 2.0 * math.pi * _ROLL_FREQUENCY * sim_time
        pitch = _PITCH_AMPLITUDE * math.sin(pitch_phase)
        roll = _ROLL_AMPLITUDE * math.sin(roll_phase)
        pitch_vel = _PITCH_AMPLITUDE * 2.0 * math.pi * _PITCH_FREQUENCY * math.cos(pitch_phase)
        roll_vel = _ROLL_AMPLITUDE * 2.0 * math.pi * _ROLL_FREQUENCY * math.cos(roll_phase)

        self.target_q[self.motor_target_q_indices["pitch"]] = pitch
        self.target_q[self.motor_target_q_indices["roll"]] = roll
        self.target_qd[self.motor_qd_indices["pitch"]] = pitch_vel
        self.target_qd[self.motor_qd_indices["roll"]] = roll_vel
        self.control.joint_target_q.assign(self.target_q)
        self.control.joint_target_qd.assign(self.target_qd)
        self.max_pitch_target = max(self.max_pitch_target, abs(pitch))
        self.max_roll_target = max(self.max_roll_target, abs(roll))

    def capture(self):
        # Motor targets are updated on the host for every substep, so the full
        # frame cannot be replayed as a fixed CUDA graph.
        self.graph = None

    def simulate(self):
        for substep in range(self.sim_substeps):
            self._update_motor_targets(self.sim_time + substep * self.sim_dt)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise ValueError("Body poses must remain finite")

        joint_types = self.model.joint_type.numpy()
        joint_dof_dim = self.model.joint_dof_dim.numpy()
        joint_labels = [label.rsplit("/", 1)[-1] for label in self.model.joint_label]
        if "ankle_dual_axis_gimbal_roll" not in joint_labels or "ankle_dual_axis_gimbal_pitch" not in joint_labels:
            raise ValueError("Missing passive dual-axis ankle gimbal")
        universal_count = int(np.sum((joint_types == int(newton.JointType.D6)) & (joint_dof_dim[:, 1] == 2)))
        if universal_count != 6:
            raise ValueError("Expected six two-axis universal pushrod joints")
        if self.max_pitch_target < 0.95 * _PITCH_AMPLITUDE or self.max_roll_target < 0.95 * _ROLL_AMPLITUDE:
            raise ValueError("Motor target oscillation did not cover the requested range")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--camera-view",
            choices=sorted(_CAMERA_VIEWS),
            default="perspective",
            help="Camera view to use for GL/USD/RTX viewers.",
        )
        parser.set_defaults(num_frames=180)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
