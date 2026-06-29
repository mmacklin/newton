#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare VBD articulation solves and Kamino on complex linkages."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import warp as wp

import newton

EXTERNAL = Path("/home/horde/external-assets")
ATTACHED_DEMO = (
    EXTERNAL
    / "kamino_linkage_demos_20260625"
    / "src"
    / "newton"
    / "examples"
    / "kamino"
    / "example_kamino_robot_foot.py"
)


@dataclass(frozen=True)
class ModeSpec:
    label: str
    solver: str
    device: str
    vbd_solve: str | None = None
    iterations: int | None = None
    relaxation: float = 1.0
    joint_ke: float = 2.0e5
    joint_kd: float = 5.0e2
    diagonal_regularization: float = 0.0
    tangential_stiffness_scale: float = 1.0
    contact_normal_stiffness_scale: float = 1.0
    contact_normal_damping_scale: float = 1.0
    contact_hard: bool = True
    friction_epsilon: float = 1.0e-2
    drive_kp: float = 10.0
    drive_kd: float = 2.0
    joint_viscous_damping: float = 0.044
    disabled_drive_suffixes: tuple[str, ...] = ()
    target_frequency_scale: float = 1.0
    robot_foot_geometry: str = "source"
    kamino_joint_stabilization: float = 0.01
    cpu_graph: bool = False
    cuda_graph: bool = False


ROBOT_FOOT_MODES = {
    "kamino_cpu": ModeSpec("Kamino CPU", "kamino", "cpu"),
    "vbd_local_cpu": ModeSpec("VBD local CPU", "vbd", "cpu", "local", 8),
    "vbd_sparse_cpu": ModeSpec("VBD sparse direct CPU", "vbd", "cpu", "block_sparse_joints", 8),
    "kamino_corrected_cpu": ModeSpec(
        "Kamino CPU, compatible linkage",
        "kamino",
        "cpu",
        robot_foot_geometry="compatible",
        kamino_joint_stabilization=0.5,
    ),
    "vbd_local_corrected_cpu": ModeSpec(
        "VBD local CPU, compatible linkage",
        "vbd",
        "cpu",
        "local",
        8,
        joint_ke=5.0e4,
        joint_kd=1.25e2,
        robot_foot_geometry="compatible",
    ),
    "vbd_sparse_corrected_cpu": ModeSpec(
        "VBD sparse direct CPU, compatible linkage",
        "vbd",
        "cpu",
        "block_sparse_joints",
        8,
        joint_ke=5.0e4,
        joint_kd=1.25e2,
        robot_foot_geometry="compatible",
    ),
}
ROBOT_FOOT_MODES = {name: replace(spec, cpu_graph=spec.solver == "vbd") for name, spec in ROBOT_FOOT_MODES.items()}

DR_LEGS_MODES = {
    "kamino_cpu": ModeSpec("Kamino CPU", "kamino", "cpu", relaxation=0.65),
    "vbd_local_cpu": ModeSpec("VBD local CPU", "vbd", "cpu", "local", 8, 0.65),
    "vbd_sparse_cpu": ModeSpec("VBD sparse direct CPU", "vbd", "cpu", "block_sparse_joints", 8, 0.65),
    "kamino_free_ankle_cpu": ModeSpec(
        "Kamino free ankle CPU",
        "kamino",
        "cpu",
        relaxation=0.65,
        joint_kd=0.0,
        disabled_drive_suffixes=("j6_l_i", "j6_r_i"),
    ),
    "vbd_local_free_ankle_cpu": ModeSpec(
        "VBD local free ankle CPU",
        "vbd",
        "cpu",
        "local",
        8,
        0.65,
        joint_kd=0.0,
        disabled_drive_suffixes=("j6_l_i", "j6_r_i"),
    ),
    "vbd_sparse_free_ankle_cpu": ModeSpec(
        "VBD sparse direct free ankle CPU",
        "vbd",
        "cpu",
        "block_sparse_joints",
        8,
        0.65,
        joint_kd=0.0,
        disabled_drive_suffixes=("j6_l_i", "j6_r_i"),
    ),
}
DR_LEGS_MODES = {name: replace(spec, cpu_graph=spec.solver == "vbd") for name, spec in DR_LEGS_MODES.items()}
DR_LEGS_MODES.update(
    {
        "kamino_free_ankle_cuda": replace(
            DR_LEGS_MODES["kamino_free_ankle_cpu"],
            label="Kamino free ankle CUDA",
            device="cuda:0",
            cpu_graph=False,
            cuda_graph=True,
        ),
        "vbd_sparse_free_ankle_cuda": replace(
            DR_LEGS_MODES["vbd_sparse_free_ankle_cpu"],
            label="VBD sparse direct free ankle CUDA",
            device="cuda:0",
            cpu_graph=False,
            cuda_graph=True,
        ),
    }
)


def _load_attached_robot_foot() -> ModuleType:
    if not ATTACHED_DEMO.is_file():
        raise FileNotFoundError(f"Missing attached robot-foot demo: {ATTACHED_DEMO}")
    spec = importlib.util.spec_from_file_location("attached_kamino_robot_foot", ATTACHED_DEMO)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {ATTACHED_DEMO}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _single_closed_loop_articulation(builder: newton.ModelBuilder, label: str) -> None:
    joint_count = len(builder.joint_articulation)
    for joint_index in range(joint_count):
        builder.joint_articulation[joint_index] = 0
    builder.articulation_start = [0]
    builder.articulation_end = [joint_count]
    builder.articulation_label = [label]
    builder.articulation_world = [builder.current_world]


def build_robot_foot_model(device: str, geometry: str = "source") -> tuple[newton.Model, ModuleType, dict[str, str]]:
    source = _load_attached_robot_foot()
    if geometry == "compatible":
        source._FOOT_LEFT_PIVOT = source._FOOT_LEFT_PIVOT.copy()
        source._FOOT_RIGHT_PIVOT = source._FOOT_RIGHT_PIVOT.copy()
        source._FOOT_LEFT_PIVOT[2] = 0.0
        source._FOOT_RIGHT_PIVOT[2] = 0.0

        def _add_rod_ball_joint(builder, parent, child, parent_pos, child_pos, rod_q, label):
            return builder.add_joint_ball(
                parent=parent,
                child=child,
                parent_xform=wp.transform(p=parent_pos, q=rod_q),
                child_xform=wp.transform(p=child_pos, q=wp.quat_identity()),
                armature=0.002,
                friction=0.002,
                label=label,
            )

        source._add_rod_universal = _add_rod_ball_joint
    elif geometry != "source":
        raise ValueError(f"Unknown robot-foot geometry {geometry!r}")
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverKamino.register_custom_attributes(builder)
    motor_labels = source._add_robot_foot(builder)
    _single_closed_loop_articulation(builder, "three_pushrod_robot_foot")
    builder.color()
    model = builder.finalize(device=device, skip_validation_joints=True)
    return model, source, motor_labels


def _joint_index(model: newton.Model, short_label: str) -> int:
    for index, label in enumerate(model.joint_label):
        if label == short_label or label.rsplit("/", 1)[-1] == short_label:
            return index
    raise ValueError(f"Missing joint {short_label!r}")


def _body_index(model: newton.Model, short_label: str) -> int:
    for index, label in enumerate(model.body_label):
        if label.rsplit("/", 1)[-1] == short_label:
            return index
    raise ValueError(f"Missing body {short_label!r}")


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64) / float(np.dot(q, q))


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _quat_mul(_quat_mul(q, np.array([v[0], v[1], v[2], 0.0])), _quat_inv(q))[:3]


def _quat_rotvec(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    vector_norm = float(np.linalg.norm(q[:3]))
    if vector_norm < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    return q[:3] * (2.0 * math.atan2(vector_norm, float(q[3])) / vector_norm)


def _transform_point(body_q: np.ndarray, local: np.ndarray) -> np.ndarray:
    return body_q[:3] + _quat_rotate(body_q[3:7], local)


def _joint_anchor_residuals(model: newton.Model, state: newton.State, labels: set[str] | None = None) -> dict:
    body_q = state.body_q.numpy()
    rest_q = model.body_q.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_type = model.joint_type.numpy()
    joint_x_p = model.joint_X_p.numpy()
    joint_x_c = model.joint_X_c.numpy()
    joint_axis = model.joint_axis.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    joint_dof_dim = model.joint_dof_dim.numpy()

    linear_terms = []
    angular_terms = []
    per_joint = {}
    for joint_index, label in enumerate(model.joint_label):
        if labels is not None and label not in labels:
            continue
        child = int(joint_child[joint_index])
        parent = int(joint_parent[joint_index])
        joint_kind = int(joint_type[joint_index])
        if child < 0:
            continue

        x_child = _transform_point(body_q[child], joint_x_c[joint_index, :3])
        q_child = _quat_mul(body_q[child, 3:7], joint_x_c[joint_index, 3:7])
        child_rest_q = _quat_mul(rest_q[child, 3:7], joint_x_c[joint_index, 3:7])
        if parent >= 0:
            x_parent = _transform_point(body_q[parent], joint_x_p[joint_index, :3])
            q_parent = _quat_mul(body_q[parent, 3:7], joint_x_p[joint_index, 3:7])
            parent_rest_q = _quat_mul(rest_q[parent, 3:7], joint_x_p[joint_index, 3:7])
        else:
            x_parent = joint_x_p[joint_index, :3]
            q_parent = joint_x_p[joint_index, 3:7]
            parent_rest_q = q_parent

        linear_projector = np.eye(3)
        angular_projector = np.eye(3)
        qd_start = int(joint_qd_start[joint_index])
        if joint_kind == int(newton.JointType.PRISMATIC):
            axis = _quat_rotate(q_parent, joint_axis[qd_start])
            axis /= np.linalg.norm(axis)
            linear_projector -= np.outer(axis, axis)
        elif joint_kind == int(newton.JointType.REVOLUTE):
            axis = _quat_rotate(q_parent, joint_axis[qd_start])
            axis /= np.linalg.norm(axis)
            angular_projector -= np.outer(axis, axis)
        elif joint_kind == int(newton.JointType.D6):
            linear_count = int(joint_dof_dim[joint_index, 0])
            angular_count = int(joint_dof_dim[joint_index, 1])
            for axis_index in range(linear_count):
                axis = _quat_rotate(q_parent, joint_axis[qd_start + axis_index])
                axis /= np.linalg.norm(axis)
                linear_projector -= np.outer(axis, axis)
            for axis_index in range(angular_count):
                axis = _quat_rotate(q_parent, joint_axis[qd_start + linear_count + axis_index])
                axis /= np.linalg.norm(axis)
                angular_projector -= np.outer(axis, axis)

        linear = np.zeros(3)
        angular = np.zeros(3)
        if joint_kind in (
            int(newton.JointType.BALL),
            int(newton.JointType.FIXED),
            int(newton.JointType.REVOLUTE),
            int(newton.JointType.PRISMATIC),
            int(newton.JointType.D6),
        ):
            linear = linear_projector @ (x_child - x_parent)
            linear_terms.append(linear)
        if joint_kind in (
            int(newton.JointType.FIXED),
            int(newton.JointType.REVOLUTE),
            int(newton.JointType.PRISMATIC),
            int(newton.JointType.D6),
        ):
            q_rel = _quat_mul(_quat_inv(q_parent), q_child)
            q_rel_rest = _quat_mul(_quat_inv(parent_rest_q), child_rest_q)
            q_error = _quat_mul(q_rel, _quat_inv(q_rel_rest))
            angular = angular_projector @ _quat_rotate(q_parent, _quat_rotvec(q_error))
            angular_terms.append(angular)

        per_joint[label] = {
            "linear_m": float(np.linalg.norm(linear)),
            "angular_rad": float(np.linalg.norm(angular)),
        }

    linear_values = np.concatenate(linear_terms) if linear_terms else np.zeros(0)
    angular_values = np.concatenate(angular_terms) if angular_terms else np.zeros(0)
    return {
        "linear_norm_m": float(np.linalg.norm(linear_values)),
        "angular_norm_rad": float(np.linalg.norm(angular_values)),
        "linear_max_m": float(max((value["linear_m"] for value in per_joint.values()), default=0.0)),
        "angular_max_rad": float(max((value["angular_rad"] for value in per_joint.values()), default=0.0)),
        "per_joint": per_joint,
    }


def _revolute_coordinate(model: newton.Model, state: newton.State, joint_index: int) -> float:
    body_q = state.body_q.numpy()
    rest_q = model.body_q.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_x_p = model.joint_X_p.numpy()
    joint_x_c = model.joint_X_c.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    joint_axis = model.joint_axis.numpy()

    parent = int(joint_parent[joint_index])
    child = int(joint_child[joint_index])
    if child < 0:
        raise ValueError("Motor coordinate requires a child body")

    if parent >= 0:
        parent_frame = _quat_mul(body_q[parent, 3:7], joint_x_p[joint_index, 3:7])
        parent_rest = _quat_mul(rest_q[parent, 3:7], joint_x_p[joint_index, 3:7])
    else:
        parent_frame = np.asarray(joint_x_p[joint_index, 3:7], dtype=np.float64)
        parent_rest = parent_frame
    child_frame = _quat_mul(body_q[child, 3:7], joint_x_c[joint_index, 3:7])
    child_rest = _quat_mul(rest_q[child, 3:7], joint_x_c[joint_index, 3:7])
    relative = _quat_mul(_quat_inv(parent_frame), child_frame)
    relative_rest = _quat_mul(_quat_inv(parent_rest), child_rest)
    delta = _quat_mul(relative, _quat_inv(relative_rest))
    axis = np.asarray(joint_axis[int(joint_qd_start[joint_index])], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    return float(np.dot(_quat_rotvec(delta), axis))


def _update_robot_foot_targets(
    source: ModuleType,
    model: newton.Model,
    control: newton.Control,
    motor_target_indices: dict[str, int],
    motor_qd_indices: dict[str, int],
    target_q: np.ndarray,
    target_qd: np.ndarray,
    sim_time: float,
    frequency_scale: float = 1.0,
) -> dict[str, float]:
    pitch_frequency = source._PITCH_FREQUENCY * frequency_scale
    roll_frequency = source._ROLL_FREQUENCY * frequency_scale
    pitch_phase = 2.0 * math.pi * pitch_frequency * sim_time
    roll_phase = 2.0 * math.pi * roll_frequency * sim_time
    pitch = source._PITCH_AMPLITUDE * math.sin(pitch_phase)
    roll = source._ROLL_AMPLITUDE * math.sin(roll_phase)
    pitch_velocity = source._PITCH_AMPLITUDE * 2.0 * math.pi * pitch_frequency * math.cos(pitch_phase)
    roll_velocity = source._ROLL_AMPLITUDE * 2.0 * math.pi * roll_frequency * math.cos(roll_phase)

    target_q[motor_target_indices["pitch"]] = pitch
    target_q[motor_target_indices["roll"]] = roll
    target_qd[motor_qd_indices["pitch"]] = pitch_velocity
    target_qd[motor_qd_indices["roll"]] = roll_velocity
    control.joint_target_q.assign(target_q)
    control.joint_target_qd.assign(target_qd)
    return {"pitch": pitch, "roll": roll}


def _make_solver(model: newton.Model, spec: ModeSpec):
    if spec.solver == "kamino":
        config = newton.solvers.SolverKamino.Config.from_model(model)
        config.use_collision_detector = False
        config.use_fk_solver = False
        config.angular_velocity_damping = 0.04
        config.constraints.alpha = spec.kamino_joint_stabilization
        config.dynamics.preconditioning = True
        config.padmm.primal_tolerance = 5.0e-5
        config.padmm.dual_tolerance = 5.0e-5
        config.padmm.compl_tolerance = 5.0e-5
        config.padmm.max_iterations = 120
        config.padmm.rho_0 = 0.1
        config.padmm.use_acceleration = True
        config.padmm.warmstart_mode = "containers"
        return newton.solvers.SolverKamino(model=model, config=config)

    if spec.solver == "vbd":
        return newton.solvers.SolverVBD(
            model,
            iterations=spec.iterations or 8,
            friction_epsilon=spec.friction_epsilon,
            rigid_articulation_solve=spec.vbd_solve or "local",
            rigid_articulation_relaxation=spec.relaxation,
            rigid_articulation_diagonal_regularization=spec.diagonal_regularization,
            rigid_contact_tangential_stiffness_scale=spec.tangential_stiffness_scale,
            rigid_contact_hard=spec.contact_hard,
            rigid_avbd_alpha=0.0,
            rigid_avbd_beta=0.0,
            rigid_joint_linear_ke=spec.joint_ke,
            rigid_joint_angular_ke=spec.joint_ke,
            rigid_joint_linear_kd=spec.joint_kd,
            rigid_joint_angular_kd=spec.joint_kd,
        )

    raise ValueError(f"Unsupported solver {spec.solver!r}")


def _make_dr_legs_solver(model: newton.Model, spec: ModeSpec):
    if spec.solver == "kamino":
        config = newton.solvers.SolverKamino.Config.from_model(model)
        config.use_collision_detector = False
        config.use_fk_solver = False
        config.constraints.delta = 1.0e-3
        config.padmm.max_iterations = 200
        config.padmm.primal_tolerance = 1.0e-4
        config.padmm.dual_tolerance = 1.0e-4
        config.padmm.compl_tolerance = 1.0e-4
        return newton.solvers.SolverKamino(model=model, config=config)
    if spec.contact_normal_stiffness_scale != 1.0:
        model.shape_material_ke.assign(
            (model.shape_material_ke.numpy() * spec.contact_normal_stiffness_scale).astype(np.float32)
        )
    if spec.contact_normal_damping_scale != 1.0:
        model.shape_material_kd.assign(
            (model.shape_material_kd.numpy() * spec.contact_normal_damping_scale).astype(np.float32)
        )
    return _make_solver(model, spec)


def build_dr_legs_model(
    device: str,
    *,
    drive_kp: float = 10.0,
    drive_kd: float = 2.0,
    joint_viscous_damping: float = 0.044,
    disabled_drive_suffixes: tuple[str, ...] = (),
) -> newton.Model:
    asset_path = newton.utils.download_asset("disneyresearch")
    asset_file = asset_path / "dr_legs" / "usd" / "dr_legs_with_meshes_and_boxes.usda"
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverKamino.register_custom_attributes(builder)
    builder.request_contact_attributes("force")
    builder.default_shape_cfg.margin = 1.0e-6
    builder.default_shape_cfg.gap = 1.0e-2
    builder.add_usd(
        str(asset_file),
        joint_ordering=None,
        force_show_colliders=True,
        force_position_velocity_actuation=True,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    _single_closed_loop_articulation(builder, "dr_legs_closed_joint_graph")
    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device, skip_validation_joints=True)
    model.rigid_contact_max = 72

    target_mode = model.joint_target_mode.numpy()
    active = target_mode != int(newton.JointTargetMode.NONE)
    model.joint_target_ke.assign(np.where(active, drive_kp, 0.0).astype(np.float32))
    model.joint_target_kd.assign(np.where(active, drive_kd, 0.0).astype(np.float32))
    if disabled_drive_suffixes:
        target_ke = model.joint_target_ke.numpy()
        target_kd = model.joint_target_kd.numpy()
        qd_starts = model.joint_qd_start.numpy()
        for joint, label in enumerate(model.joint_label):
            if label.endswith(disabled_drive_suffixes):
                dof = int(qd_starts[joint])
                target_ke[dof] = 0.0
                target_kd[dof] = 0.0
        model.joint_target_ke.assign(target_ke)
        model.joint_target_kd.assign(target_kd)
    model.joint_armature.fill_(0.011)
    model.joint_damping.fill_(joint_viscous_damping)
    body_q = model.body_q.numpy().copy()
    body_q[:, 2] += 0.4
    model.body_q.assign(body_q)
    return model


def _cycle_joint_labels(model: newton.Model) -> set[str]:
    parents: dict[int, int] = {}

    def find(node: int) -> int:
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(a: int, b: int) -> bool:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return False
        parents[root_b] = root_a
        return True

    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    cycle_labels = set()
    for index, label in enumerate(model.joint_label):
        parent = int(joint_parent[index])
        child = int(joint_child[index])
        if not union(parent, child):
            cycle_labels.add(label)
    return cycle_labels


def _quat_tilt_deg(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    x, y, _, _ = q
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(float(np.clip(up_z, -1.0, 1.0))))


def run_robot_foot_mode(spec: ModeSpec, frames: int, timing_skip_frames: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mode": spec.label,
        "solver": spec.solver,
        "device": spec.device,
        "vbd_solve": spec.vbd_solve,
        "iterations": spec.iterations,
        "relaxation": spec.relaxation,
        "joint_ke": spec.joint_ke,
        "joint_kd": spec.joint_kd,
        "diagonal_regularization": spec.diagonal_regularization,
        "tangential_stiffness_scale": spec.tangential_stiffness_scale,
        "contact_normal_stiffness_scale": spec.contact_normal_stiffness_scale,
        "contact_normal_damping_scale": spec.contact_normal_damping_scale,
        "frames": frames,
        "fps": 50,
        "dt": 0.004,
        "substeps_per_frame": 5,
        "target_frequency_scale": spec.target_frequency_scale,
        "robot_foot_geometry": spec.robot_foot_geometry,
        "timing_method": "cpu_graph_solver" if spec.cpu_graph else "synchronized_dispatch",
    }
    try:
        model, source, motor_labels = build_robot_foot_model(spec.device, spec.robot_foot_geometry)
        state_0 = model.state()
        state_1 = model.state()
        state_1.assign(state_0)
        control = model.control()
        target_q = model.joint_target_q.numpy().copy()
        target_qd = model.joint_target_qd.numpy().copy()
        target_starts = model.joint_target_q_start.numpy()
        qd_starts = model.joint_qd_start.numpy()
        motor_joint_indices = {name: _joint_index(model, label) for name, label in motor_labels.items()}
        output_joint_indices = {
            "pitch": _joint_index(model, "ankle_dual_axis_gimbal_pitch"),
            "roll": _joint_index(model, "ankle_dual_axis_gimbal_roll"),
        }
        motor_target_indices = {name: int(target_starts[index]) for name, index in motor_joint_indices.items()}
        motor_qd_indices = {name: int(qd_starts[index]) for name, index in motor_joint_indices.items()}
        closure_labels = {
            label
            for label in model.joint_label
            if "pushrod_to_foot_universal" in label or "pushrod_to_gimbal_universal" in label
        }
        foot_body = _body_index(model, "foot_platform")
        solver = _make_solver(model, spec)

        graphs = None
        if spec.cpu_graph:
            if not model.device.is_cpu:
                raise ValueError("CPU graph timing requires the CPU device")

            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, row["dt"])
            wp.synchronize_device(model.device)

            state_0 = model.state()
            state_1 = model.state()
            state_1.assign(state_0)
            control = model.control()
            solver = _make_solver(model, spec)

            with wp.ScopedCapture(device=model.device) as capture_0:
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, row["dt"])
            with wp.ScopedCapture(device=model.device) as capture_1:
                state_1.clear_forces()
                solver.step(state_1, state_0, control, None, row["dt"])
            graphs = (capture_0.graph, capture_1.graph)
            wp.synchronize_device(model.device)

        _update_robot_foot_targets(
            source,
            model,
            control,
            motor_target_indices,
            motor_qd_indices,
            target_q,
            target_qd,
            0.0,
            spec.target_frequency_scale,
        )
        if spec.solver == "kamino":
            solver.step(state_0, state_1, control, None, row["dt"])
            solver.reset(state_0)
            state_1.assign(state_0)

        initial_closure = _joint_anchor_residuals(model, state_0, closure_labels)
        initial_all = _joint_anchor_residuals(model, state_0)
        closure_norms = []
        closure_maxes = []
        all_linear_norms = []
        all_angular_norms = []
        pitch_errors = []
        roll_errors = []
        pitch_targets = []
        roll_targets = []
        pitch_motor_positions = []
        roll_motor_positions = []
        output_pitch_positions = []
        output_roll_positions = []
        foot_angular_speeds = []
        step_times_us = []
        kamino_converged = []
        kamino_iterations = []
        kamino_dual_residuals = []
        sim_time = 0.0
        completed_substeps = 0
        current_state = state_0

        for frame in range(frames):
            for substep in range(row["substeps_per_frame"]):
                substep_time = sim_time + substep * row["dt"]
                targets = _update_robot_foot_targets(
                    source,
                    model,
                    control,
                    motor_target_indices,
                    motor_qd_indices,
                    target_q,
                    target_qd,
                    substep_time,
                    spec.target_frequency_scale,
                )
                start = time.perf_counter()
                if graphs is not None:
                    graph_index = completed_substeps % 2
                    wp.capture_launch(graphs[graph_index])
                    current_state = state_1 if graph_index == 0 else state_0
                else:
                    state_0.clear_forces()
                    solver.step(state_0, state_1, control, None, row["dt"])
                wp.synchronize_device(model.device)
                elapsed_us = (time.perf_counter() - start) * 1.0e6
                if spec.solver == "kamino":
                    status = solver._solver_kamino.solver_fd.data.status.numpy()[0]
                    kamino_converged.append(bool(status["converged"]))
                    kamino_iterations.append(int(status["iterations"]))
                    kamino_dual_residuals.append(float(status["r_d"]))
                if graphs is None:
                    state_0, state_1 = state_1, state_0
                    current_state = state_0
                completed_substeps += 1
                if frame >= timing_skip_frames:
                    step_times_us.append(elapsed_us)

                closure = _joint_anchor_residuals(model, current_state, closure_labels)
                all_residual = _joint_anchor_residuals(model, current_state)
                closure_norms.append(closure["linear_norm_m"])
                closure_maxes.append(closure["linear_max_m"])
                all_linear_norms.append(all_residual["linear_norm_m"])
                all_angular_norms.append(all_residual["angular_norm_rad"])
                pitch_motor = _revolute_coordinate(model, current_state, motor_joint_indices["pitch"])
                roll_motor = _revolute_coordinate(model, current_state, motor_joint_indices["roll"])
                pitch_targets.append(targets["pitch"])
                roll_targets.append(targets["roll"])
                pitch_motor_positions.append(pitch_motor)
                roll_motor_positions.append(roll_motor)
                output_pitch_positions.append(_revolute_coordinate(model, current_state, output_joint_indices["pitch"]))
                output_roll_positions.append(_revolute_coordinate(model, current_state, output_joint_indices["roll"]))
                pitch_errors.append(pitch_motor - targets["pitch"])
                roll_errors.append(roll_motor - targets["roll"])
                foot_angular_speeds.append(float(np.linalg.norm(current_state.body_qd.numpy()[foot_body, 3:6])))
            sim_time += 1.0 / row["fps"]

        body_q = current_state.body_q.numpy()
        final_closure = _joint_anchor_residuals(model, current_state, closure_labels)
        final_all = _joint_anchor_residuals(model, current_state)
        row.update(
            status="ok" if np.all(np.isfinite(body_q)) else "nonfinite",
            body_count=model.body_count,
            joint_count=model.joint_count,
            joint_dof_count=model.joint_dof_count,
            articulation_count=model.articulation_count,
            closure_joint_count=len(closure_labels),
            initial_closure=initial_closure,
            initial_all=initial_all,
            final_closure=final_closure,
            final_all=final_all,
            rms_closure_um=float(np.sqrt(np.mean(np.square(closure_norms))) * 1.0e6),
            max_closure_um=float(np.max(closure_maxes) * 1.0e6),
            rms_all_linear_um=float(np.sqrt(np.mean(np.square(all_linear_norms))) * 1.0e6),
            rms_all_angular_deg=float(np.degrees(np.sqrt(np.mean(np.square(all_angular_norms))))),
            rms_pitch_tracking_deg=float(np.degrees(np.sqrt(np.mean(np.square(pitch_errors))))),
            rms_roll_tracking_deg=float(np.degrees(np.sqrt(np.mean(np.square(roll_errors))))),
            pitch_target_range_deg=float(np.degrees(np.ptp(pitch_targets))),
            roll_target_range_deg=float(np.degrees(np.ptp(roll_targets))),
            pitch_motor_range_deg=float(np.degrees(np.ptp(pitch_motor_positions))),
            roll_motor_range_deg=float(np.degrees(np.ptp(roll_motor_positions))),
            output_pitch_range_deg=float(np.degrees(np.ptp(output_pitch_positions))),
            output_roll_range_deg=float(np.degrees(np.ptp(output_roll_positions))),
            max_foot_angular_speed=float(np.max(foot_angular_speeds)),
            mean_step_us=float(statistics.fmean(step_times_us)) if step_times_us else None,
            p50_step_us=float(np.percentile(step_times_us, 50.0)) if step_times_us else None,
            p90_step_us=float(np.percentile(step_times_us, 90.0)) if step_times_us else None,
            kamino_converged_fraction=(float(np.mean(kamino_converged)) if kamino_converged else None),
            kamino_p50_iterations=(float(np.percentile(kamino_iterations, 50.0)) if kamino_iterations else None),
            kamino_p50_dual_residual=(
                float(np.percentile(kamino_dual_residuals, 50.0)) if kamino_dual_residuals else None
            ),
        )
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=30))
    return row


def run_dr_legs_mode(spec: ModeSpec, frames: int, timing_skip_frames: int) -> dict[str, Any]:
    dt = 0.01
    substeps_per_frame = 2
    row: dict[str, Any] = {
        "mode": spec.label,
        "solver": spec.solver,
        "device": spec.device,
        "vbd_solve": spec.vbd_solve,
        "iterations": spec.iterations,
        "relaxation": spec.relaxation,
        "joint_ke": spec.joint_ke,
        "joint_kd": spec.joint_kd,
        "diagonal_regularization": spec.diagonal_regularization,
        "tangential_stiffness_scale": spec.tangential_stiffness_scale,
        "contact_normal_stiffness_scale": spec.contact_normal_stiffness_scale,
        "contact_normal_damping_scale": spec.contact_normal_damping_scale,
        "frames": frames,
        "fps": 50,
        "dt": dt,
        "substeps_per_frame": substeps_per_frame,
        "contact_pipeline": "Newton",
        "drive_kp": spec.drive_kp,
        "drive_kd": spec.drive_kd,
        "disabled_drive_suffixes": list(spec.disabled_drive_suffixes),
        "kamino_joint_armature": 0.011 if spec.solver == "kamino" else None,
        "kamino_joint_damping": spec.joint_viscous_damping if spec.solver == "kamino" else None,
        "timing_method": (
            "cuda_graph_end_to_end"
            if spec.cuda_graph
            else "cpu_graph_solver_plus_dispatched_collision"
            if spec.cpu_graph
            else "synchronized_dispatch"
        ),
    }
    try:
        model = build_dr_legs_model(
            spec.device,
            drive_kp=spec.drive_kp,
            drive_kd=spec.drive_kd,
            joint_viscous_damping=spec.joint_viscous_damping,
            disabled_drive_suffixes=spec.disabled_drive_suffixes,
        )
        state_0 = model.state()
        state_1 = model.state()
        state_1.assign(state_0)
        control = model.control()
        pipeline = newton.CollisionPipeline(model)
        contacts = model.contacts(collision_pipeline=pipeline)
        solver = _make_dr_legs_solver(model, spec)
        closure_labels = _cycle_joint_labels(model)
        pelvis = _body_index(model, "pelvis")

        graphs = None
        if spec.cuda_graph or spec.cpu_graph:
            if spec.cuda_graph and not model.device.is_cuda:
                raise ValueError("CUDA graph timing requires a CUDA device")
            if spec.cpu_graph and not model.device.is_cpu:
                raise ValueError("CPU graph timing requires the CPU device")

            # Compile every collision and solver kernel before stream capture, then
            # recreate mutable runtime state so the measured trajectory starts clean.
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            wp.synchronize_device(model.device)

            state_0 = model.state()
            state_1 = model.state()
            state_1.assign(state_0)
            control = model.control()
            pipeline = newton.CollisionPipeline(model)
            contacts = model.contacts(collision_pipeline=pipeline)
            solver = _make_dr_legs_solver(model, spec)

        initial_closure = _joint_anchor_residuals(model, state_0, closure_labels)

        if spec.cuda_graph or spec.cpu_graph:
            if spec.cpu_graph:
                pipeline.collide(state_0, contacts)
            with wp.ScopedCapture(device=model.device) as capture_0:
                state_0.clear_forces()
                if spec.cuda_graph:
                    pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
            if spec.cpu_graph:
                pipeline.collide(state_1, contacts)
            with wp.ScopedCapture(device=model.device) as capture_1:
                state_1.clear_forces()
                if spec.cuda_graph:
                    pipeline.collide(state_1, contacts)
                solver.step(state_1, state_0, control, contacts, dt)
            graphs = (capture_0.graph, capture_1.graph)
            wp.synchronize_device(model.device)

        closure_norms = []
        closure_maxes = []
        all_linear_norms = []
        pelvis_heights = []
        pelvis_tilts = []
        contact_counts = []
        max_body_speeds = []
        collision_times_us = []
        solver_times_us = []
        graph_step_times_us = []
        kamino_converged = []
        kamino_iterations = []
        kamino_primal_residuals = []
        kamino_dual_residuals = []
        kamino_complementarity_residuals = []
        failure_substep = None
        completed_substeps = 0
        current_state = state_0

        for frame in range(frames):
            for _ in range(substeps_per_frame):
                if graphs is not None:
                    graph_index = completed_substeps % 2
                    if spec.cpu_graph:
                        collision_state = state_0 if graph_index == 0 else state_1
                        collision_start = time.perf_counter()
                        pipeline.collide(collision_state, contacts)
                        wp.synchronize_device(model.device)
                        collision_us = (time.perf_counter() - collision_start) * 1.0e6
                    graph_start = time.perf_counter()
                    wp.capture_launch(graphs[graph_index])
                    wp.synchronize_device(model.device)
                    replay_us = (time.perf_counter() - graph_start) * 1.0e6
                    if spec.cuda_graph:
                        graph_step_us = replay_us
                        collision_us = None
                        solver_us = None
                    else:
                        graph_step_us = None
                        solver_us = replay_us
                    current_state = state_1 if graph_index == 0 else state_0
                else:
                    state_0.clear_forces()
                    collision_start = time.perf_counter()
                    pipeline.collide(state_0, contacts)
                    wp.synchronize_device(model.device)
                    collision_us = (time.perf_counter() - collision_start) * 1.0e6
                    solver_start = time.perf_counter()
                    solver.step(state_0, state_1, control, contacts, dt)
                    wp.synchronize_device(model.device)
                    solver_us = (time.perf_counter() - solver_start) * 1.0e6
                    state_0, state_1 = state_1, state_0
                    current_state = state_0
                if spec.solver == "kamino":
                    status = solver._solver_kamino.solver_fd.data.status.numpy()[0]
                    kamino_converged.append(bool(status["converged"]))
                    kamino_iterations.append(int(status["iterations"]))
                    kamino_primal_residuals.append(float(status["r_p"]))
                    kamino_dual_residuals.append(float(status["r_d"]))
                    kamino_complementarity_residuals.append(float(status["r_c"]))
                completed_substeps += 1
                if frame >= timing_skip_frames:
                    if spec.cuda_graph:
                        graph_step_times_us.append(graph_step_us)
                    else:
                        collision_times_us.append(collision_us)
                        solver_times_us.append(solver_us)

                poses = current_state.body_q.numpy()
                velocities = current_state.body_qd.numpy()
                if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(velocities)):
                    failure_substep = completed_substeps
                    break
                closure = _joint_anchor_residuals(model, current_state, closure_labels)
                all_residual = _joint_anchor_residuals(model, current_state)
                closure_norms.append(closure["linear_norm_m"])
                closure_maxes.append(closure["linear_max_m"])
                all_linear_norms.append(all_residual["linear_norm_m"])
                pelvis_heights.append(float(poses[pelvis, 2]))
                pelvis_tilts.append(_quat_tilt_deg(poses[pelvis, 3:7]))
                max_body_speeds.append(float(np.max(np.linalg.norm(velocities[:, :3], axis=1))))
                contact_counts.append(int(contacts.rigid_contact_count.numpy()[0]))
            if failure_substep is not None:
                break

        final_body_q = current_state.body_q.numpy()
        finite = bool(np.all(np.isfinite(final_body_q)))
        row.update(
            status="ok" if finite else "nonfinite",
            completed_substeps=completed_substeps,
            failure_substep=failure_substep,
            body_count=model.body_count,
            joint_count=model.joint_count,
            joint_dof_count=model.joint_dof_count,
            articulation_count=model.articulation_count,
            closure_joint_count=len(closure_labels),
            closure_joint_labels=sorted(closure_labels),
            initial_closure=initial_closure,
            final_closure=_joint_anchor_residuals(model, current_state, closure_labels) if finite else None,
            final_all=_joint_anchor_residuals(model, current_state) if finite else None,
            rms_closure_um=float(np.sqrt(np.mean(np.square(closure_norms))) * 1.0e6),
            rms_closure_per_joint_um=float(np.sqrt(np.mean(np.square(closure_norms)) / len(closure_labels)) * 1.0e6),
            max_closure_um=float(np.max(closure_maxes) * 1.0e6),
            rms_all_linear_um=float(np.sqrt(np.mean(np.square(all_linear_norms))) * 1.0e6),
            final_pelvis_height_m=float(pelvis_heights[-1]),
            min_pelvis_height_m=float(np.min(pelvis_heights)),
            max_pelvis_tilt_deg=float(np.max(pelvis_tilts)),
            max_body_speed_mps=float(np.max(max_body_speeds)),
            mean_contacts=float(np.mean(contact_counts)),
            max_contacts=int(np.max(contact_counts)),
            mean_collision_us=float(statistics.fmean(collision_times_us)) if collision_times_us else None,
            p50_collision_us=float(np.percentile(collision_times_us, 50.0)) if collision_times_us else None,
            p90_collision_us=float(np.percentile(collision_times_us, 90.0)) if collision_times_us else None,
            mean_solver_us=float(statistics.fmean(solver_times_us)) if solver_times_us else None,
            p50_solver_us=float(np.percentile(solver_times_us, 50.0)) if solver_times_us else None,
            p90_solver_us=float(np.percentile(solver_times_us, 90.0)) if solver_times_us else None,
            mean_graph_step_us=float(statistics.fmean(graph_step_times_us)) if graph_step_times_us else None,
            p50_graph_step_us=float(np.percentile(graph_step_times_us, 50.0)) if graph_step_times_us else None,
            p90_graph_step_us=float(np.percentile(graph_step_times_us, 90.0)) if graph_step_times_us else None,
            kamino_converged_fraction=(float(np.mean(kamino_converged)) if kamino_converged else None),
            kamino_p50_iterations=(float(np.percentile(kamino_iterations, 50.0)) if kamino_iterations else None),
            kamino_p90_iterations=(float(np.percentile(kamino_iterations, 90.0)) if kamino_iterations else None),
            kamino_p50_primal_residual=(
                float(np.percentile(kamino_primal_residuals, 50.0)) if kamino_primal_residuals else None
            ),
            kamino_p50_dual_residual=(
                float(np.percentile(kamino_dual_residuals, 50.0)) if kamino_dual_residuals else None
            ),
            kamino_p90_dual_residual=(
                float(np.percentile(kamino_dual_residuals, 90.0)) if kamino_dual_residuals else None
            ),
            kamino_p50_complementarity_residual=(
                float(np.percentile(kamino_complementarity_residuals, 50.0))
                if kamino_complementarity_residuals
                else None
            ),
        )
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=30))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("robot-foot", "dr-legs"), default="robot-foot")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--timing-skip-frames", type=int, default=10)
    mode_choices = tuple(dict.fromkeys((*ROBOT_FOOT_MODES, *DR_LEGS_MODES)))
    parser.add_argument("--modes", nargs="*", choices=mode_choices, default=list(ROBOT_FOOT_MODES))
    parser.add_argument("--output", type=Path, default=Path("reports/vbd_complex_linkages/results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = {
        "robot-foot": ROBOT_FOOT_MODES,
        "dr-legs": DR_LEGS_MODES,
    }[args.scenario]
    runner = {
        "robot-foot": run_robot_foot_mode,
        "dr-legs": run_dr_legs_mode,
    }[args.scenario]
    rows = []
    for mode_name in args.modes:
        row = runner(modes[mode_name], args.frames, args.timing_skip_frames)
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    payload = {
        "scenario": args.scenario,
        "source": {
            "robot-foot": "attached_internal_three_pushrod_robot_foot",
            "dr-legs": "newton_assets_disney_research_dr_legs",
        }[args.scenario],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
