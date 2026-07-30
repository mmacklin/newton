#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the trained DR Legs walk policy with Newton rigid-body solvers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch  # noqa: TID253
import warp as wp
import yaml

import newton
from reports.vbd_complex_linkages.bench_complex_linkages import (
    ModeSpec,
    _body_index,
    _cycle_joint_labels,
    _joint_anchor_residuals,
    _make_dr_legs_solver,
    _single_closed_loop_articulation,
)


@dataclass(frozen=True)
class PolicyConfig:
    action_scale: float = 0.4
    contact_duration: float = 0.3
    phase_embedding_k: int = 2
    pd_kp: float = 15.0
    pd_kd: float = 0.6
    pd_armature: float = 0.01
    path_deviation_scale: float = 0.1
    linear_path_error_limit: float = 0.1
    standing_height: float = 0.265
    height_error_scale: float = 0.05
    sim_dt: float = 0.004
    control_decimation: int = 5
    body_pose_offset_z: float = 0.265
    usd_model: str = "dr_legs/usd/dr_legs_with_meshes_and_boxes.usda"
    policy_file: str = "drlegs_walk.pt"

    @property
    def env_dt(self) -> float:
        return self.sim_dt * self.control_decimation

    @property
    def phase_rate(self) -> float:
        return 1.0 / (2.0 * self.contact_duration)


CASE_CONFIGS = {
    "kamino": ModeSpec("Kamino", "kamino", "cpu"),
    "local_i8": ModeSpec("VBD local, 8 iterations", "vbd", "cpu", "local", 8, 0.65),
    "local_i32": ModeSpec("VBD local, 32 iterations", "vbd", "cpu", "local", 32, 0.65),
    "sparse_i8": ModeSpec("VBD sparse direct, 8 iterations", "vbd", "cpu", "block_sparse_joints", 8, 0.65),
    "sparse_no_armature_i8": ModeSpec(
        "VBD sparse direct, no armature, 8 iterations",
        "vbd",
        "cpu",
        "block_sparse_joints",
        8,
        0.65,
    ),
}


def _load_config(asset_path: Path) -> PolicyConfig:
    path = asset_path / "dr_legs" / "rl_policies" / "drlegs_walk.yaml"
    if not path.exists():
        return PolicyConfig()
    values = yaml.safe_load(path.read_text()) or {}
    fields = PolicyConfig.__dataclass_fields__
    return PolicyConfig(**{name: values[name] for name in fields if name in values})


def _build_mlp(state_dict: dict[str, torch.Tensor], prefix: str = "actor") -> torch.nn.Sequential:
    indices = sorted({int(key.split(".")[1]) for key in state_dict if key.startswith(f"{prefix}.")})
    layers: list[torch.nn.Module] = []
    for layer_number, index in enumerate(indices):
        weight = state_dict[f"{prefix}.{index}.weight"]
        bias = state_dict[f"{prefix}.{index}.bias"]
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if layer_number < len(indices) - 1:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers)


def load_policy(path: Path) -> Any:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor = _build_mlp(checkpoint["model_state_dict"])
    actor.eval()
    normalizer = checkpoint.get("obs_norm_state_dict")
    if normalizer is None:
        return actor
    mean = normalizer["_mean"].float()
    std = normalizer["_std"].float()

    def policy(observation: torch.Tensor) -> torch.Tensor:
        return actor((observation - mean) / (std + 1.0e-2))

    return policy


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dtype=np.float64,
    )


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array((-q[0], -q[1], -q[2], q[3]), dtype=np.float64) / float(np.dot(q, q))


def _quat_rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    pure = np.array((vector[0], vector[1], vector[2], 0.0), dtype=np.float64)
    return _quat_mul(_quat_mul(q, pure), _quat_inv(q))[:3]


def _quat_rotate_inv(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return _quat_rotate(_quat_inv(q), vector)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array((0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)), dtype=np.float64)


def _quat_to_rotation9d(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dtype=np.float32,
    )


def _projected_yaw(q: np.ndarray) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (z * w + x * y), w * w + x * x - y * y - z * z)


def _tilt_deg(q: np.ndarray) -> float:
    up = _quat_rotate(q, np.array((0.0, 0.0, 1.0)))
    return math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))


def _body_com_position(model: newton.Model, poses: np.ndarray, body: int) -> np.ndarray:
    return poses[body, :3] + _quat_rotate(poses[body, 3:7], model.body_com.numpy()[body])


def _add_child_body_armature(model: newton.Model) -> None:
    """Approximate joint armature as isotropic child-link rotational inertia."""
    inertia = model.body_inertia.numpy().copy()
    joint_child = model.joint_child.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    joint_type = model.joint_type.numpy()
    armature = model.joint_armature.numpy()
    target_mode = model.joint_target_mode.numpy()

    for joint in range(model.joint_count):
        dof = int(joint_qd_start[joint])
        if target_mode[dof] == int(newton.JointTargetMode.NONE):
            continue
        if joint_type[joint] != int(newton.JointType.REVOLUTE):
            raise ValueError("The DR Legs armature approximation expects revolute actuators")
        child = int(joint_child[joint])
        inertia[child] += armature[dof] * np.eye(3)

    model.body_inertia.assign(inertia.astype(np.float32))
    model.body_inv_inertia.assign(np.linalg.inv(inertia).astype(np.float32))


def build_model(device: str, config: PolicyConfig, *, armature_mode: str) -> newton.Model:
    asset_path = newton.utils.download_asset("disneyresearch")
    source = asset_path / config.usd_model
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverKamino.register_custom_attributes(builder)
    builder.request_contact_attributes("force")
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0
    builder.add_usd(
        str(source),
        joint_ordering=None,
        force_show_colliders=True,
        force_position_velocity_actuation=True,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    _single_closed_loop_articulation(builder, "dr_legs_policy_articulation")
    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device, skip_validation_joints=True)
    model.rigid_contact_max = 72
    model.set_gravity((0.0, 0.0, -9.81))

    target_mode = model.joint_target_mode.numpy()
    actuated = target_mode != int(newton.JointTargetMode.NONE)
    model.joint_target_ke.assign(np.where(actuated, config.pd_kp, 0.0).astype(np.float32))
    model.joint_target_kd.assign(np.where(actuated, config.pd_kd, 0.0).astype(np.float32))
    model.joint_armature.assign(np.where(actuated, config.pd_armature, 0.0).astype(np.float32))
    model.joint_damping.zero_()
    if armature_mode == "isotropic_child_body":
        _add_child_body_armature(model)
    elif armature_mode not in ("native", "unsupported"):
        raise ValueError(f"Unknown armature mode {armature_mode!r}")

    body_q = model.body_q.numpy().copy()
    body_q[:, 2] += config.body_pose_offset_z
    model.body_q.assign(body_q)
    return model


class ObservationState:
    def __init__(self, config: PolicyConfig, root_position: np.ndarray):
        self.config = config
        self.phase = 0.0
        self.path_heading = 0.0
        self.path_position = root_position[:2].astype(np.float64).copy()
        self.desired_position = self.path_position.copy()
        self.action_history = np.zeros(12, dtype=np.float32)
        self.action_history_prev = np.zeros(12, dtype=np.float32)
        frequencies = []
        offsets = []
        for harmonic in range(1, config.phase_embedding_k + 1):
            frequencies.extend((2.0 * math.pi * harmonic, 2.0 * math.pi * harmonic))
            offsets.extend((0.5 * math.pi, 0.0))
        self.frequencies = np.asarray(frequencies)
        self.offsets = np.asarray(offsets)

    def build(
        self,
        root_position: np.ndarray,
        root_quat: np.ndarray,
        root_velocity: np.ndarray,
        joint_q: np.ndarray,
        previous_action: np.ndarray,
        command_xy: np.ndarray,
        command_yaw: float,
    ) -> np.ndarray:
        cfg = self.config
        self.phase = (self.phase + cfg.env_dt * cfg.phase_rate) % 1.0
        mid_heading = self.path_heading + 0.5 * cfg.env_dt * command_yaw
        self.path_position += _quat_rotate(_yaw_quat(mid_heading), np.r_[command_xy, 0.0])[:2] * cfg.env_dt
        self.desired_position += _quat_rotate(_yaw_quat(mid_heading), np.r_[command_xy, 0.0])[:2] * cfg.env_dt
        self.path_heading += cfg.env_dt * command_yaw

        difference = self.path_position - root_position[:2]
        difference_norm = float(np.linalg.norm(difference))
        if difference_norm > cfg.linear_path_error_limit:
            self.path_position[:] = root_position[:2] + difference * (cfg.linear_path_error_limit / difference_norm)

        path_quat = _yaw_quat(self.path_heading)
        root_in_path = _quat_mul(_quat_inv(path_quat), root_quat)
        path_difference = np.r_[root_position[:2] - self.path_position, 0.0]
        deviation_path = _quat_rotate_inv(path_quat, path_difference)[:2] / cfg.path_deviation_scale
        root_heading = _projected_yaw(root_in_path)
        deviation_heading = _quat_rotate_inv(_yaw_quat(root_heading), np.r_[-deviation_path, 0.0])[:2]
        command_linear_root = _quat_rotate_inv(root_in_path, np.r_[command_xy, 0.0])
        command_angular_root = _quat_rotate_inv(root_in_path, np.array((0.0, 0.0, command_yaw)))
        root_linear = _quat_rotate_inv(root_quat, root_velocity[:3])
        root_angular = _quat_rotate_inv(root_quat, root_velocity[3:])
        phase_encoding = np.sin(self.phase * self.frequencies + self.offsets)

        self.action_history_prev[:] = self.action_history
        self.action_history[:] = cfg.action_scale * previous_action
        height_error = (root_position[2] - cfg.standing_height) / cfg.height_error_scale
        return np.concatenate(
            (
                _quat_to_rotation9d(root_in_path),
                deviation_path,
                deviation_heading,
                command_xy,
                np.array((command_yaw,)),
                command_linear_root,
                command_angular_root,
                phase_encoding,
                root_linear,
                root_angular,
                np.array((cfg.standing_height, height_error)),
                joint_q,
                self.action_history,
                self.action_history_prev,
            )
        ).astype(np.float32)


def _make_solver(model: newton.Model, spec: ModeSpec):
    if spec.solver == "kamino":
        return _make_dr_legs_solver(model, spec)
    return newton.solvers.SolverVBD(
        model,
        iterations=spec.iterations or 8,
        friction_epsilon=1.0e-2,
        rigid_articulation_solve=spec.vbd_solve or "local",
        rigid_articulation_relaxation=spec.relaxation,
        rigid_contact_hard=True,
        rigid_avbd_alpha=0.0,
        rigid_avbd_beta=0.0,
        rigid_joint_linear_ke=2.0e5,
        rigid_joint_angular_ke=2.0e5,
        rigid_joint_linear_kd=5.0e2,
        rigid_joint_angular_kd=5.0e2,
    )


def _summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "rms": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(array * array))),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def run_case(
    case: str,
    policy: Any,
    config: PolicyConfig,
    *,
    control_steps: int,
    stand_steps: int,
    forward_speed: float,
    on_control_step: Callable[[newton.Model, newton.State, float, dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    spec = CASE_CONFIGS[case]
    if spec.solver == "kamino":
        armature_mode = "native"
    elif "no_armature" in case:
        armature_mode = "unsupported"
    else:
        armature_mode = "isotropic_child_body"
    model = build_model(spec.device, config, armature_mode=armature_mode)
    state_0 = model.state()
    state_1 = model.state()
    state_1.assign(state_0)
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = model.contacts(collision_pipeline=pipeline)
    solver = _make_solver(model, spec)
    pelvis = _body_index(model, "pelvis")
    closure_labels = _cycle_joint_labels(model)

    target_mode = model.joint_target_mode.numpy()
    actuated_indices = np.flatnonzero(target_mode != int(newton.JointTargetMode.NONE))
    if actuated_indices.shape != (12,):
        raise ValueError(f"Expected 12 actuated DOFs, found {actuated_indices.shape[0]}")
    target_q = control.joint_target_q.numpy().copy()
    target_qd = control.joint_target_qd.numpy().copy()

    initial_poses = state_0.body_q.numpy().copy()
    root_position = _body_com_position(model, initial_poses, pelvis)
    observation_state = ObservationState(config, root_position)
    previous_action = np.zeros(12, dtype=np.float32)

    heights: list[float] = []
    tilts: list[float] = []
    forward_velocities: list[float] = []
    velocity_errors: list[float] = []
    path_errors: list[float] = []
    closure_errors_um: list[float] = []
    action_norms: list[float] = []
    target_errors: list[float] = []
    contact_counts: list[float] = []
    policy_times_us: list[float] = []
    collision_times_us: list[float] = []
    solver_times_us: list[float] = []
    fall_control_step = None
    completed_substeps = 0
    sim_time = 0.0

    for control_step in range(control_steps):
        newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)
        joint_q = state_0.joint_q.numpy().astype(np.float32)
        poses = state_0.body_q.numpy()
        velocities = state_0.body_qd.numpy()
        root_position = _body_com_position(model, poses, pelvis)
        root_quat = poses[pelvis, 3:7].astype(np.float64)
        root_velocity = velocities[pelvis].astype(np.float64)
        command_xy = np.array((forward_speed if control_step >= stand_steps else 0.0, 0.0))
        observation = observation_state.build(
            root_position,
            root_quat,
            root_velocity,
            joint_q,
            previous_action,
            command_xy,
            0.0,
        )
        if observation.shape != (94,):
            raise ValueError(f"Policy observation has shape {observation.shape}, expected (94,)")

        policy_start = time.perf_counter()
        with torch.no_grad():
            action = policy(torch.from_numpy(observation).unsqueeze(0)).squeeze(0).numpy()
        policy_times_us.append((time.perf_counter() - policy_start) * 1.0e6)
        previous_action = action.astype(np.float32)
        target_q[actuated_indices] = config.action_scale * previous_action
        target_qd[actuated_indices] = 0.0
        control.joint_target_q.assign(target_q)
        control.joint_target_qd.assign(target_qd)

        finite = True
        for _ in range(config.control_decimation):
            state_0.clear_forces()
            collision_start = time.perf_counter()
            pipeline.collide(state_0, contacts)
            wp.synchronize_device(model.device)
            collision_times_us.append((time.perf_counter() - collision_start) * 1.0e6)
            solver_start = time.perf_counter()
            solver.step(state_0, state_1, control, contacts, config.sim_dt)
            wp.synchronize_device(model.device)
            solver_times_us.append((time.perf_counter() - solver_start) * 1.0e6)
            state_0, state_1 = state_1, state_0
            completed_substeps += 1
            sim_time += config.sim_dt
            finite = bool(np.isfinite(state_0.body_q.numpy()).all() and np.isfinite(state_0.body_qd.numpy()).all())
            if not finite:
                break
        if not finite:
            fall_control_step = control_step
            break

        newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)
        poses = state_0.body_q.numpy()
        velocities = state_0.body_qd.numpy()
        root_position = _body_com_position(model, poses, pelvis)
        root_quat = poses[pelvis, 3:7].astype(np.float64)
        root_velocity = velocities[pelvis].astype(np.float64)
        root_velocity_local = _quat_rotate_inv(root_quat, root_velocity[:3])
        height = float(root_position[2])
        tilt = _tilt_deg(root_quat)
        closure = _joint_anchor_residuals(model, state_0, closure_labels)
        current_joint_q = state_0.joint_q.numpy()
        target_error = current_joint_q[actuated_indices] - target_q[actuated_indices]

        heights.append(height)
        tilts.append(tilt)
        forward_velocities.append(float(root_velocity_local[0]))
        velocity_errors.append(float(root_velocity_local[0] - command_xy[0]))
        path_errors.append(float(np.linalg.norm(root_position[:2] - observation_state.desired_position)))
        closure_errors_um.append(float(closure["linear_norm_m"] * 1.0e6))
        action_norms.append(float(np.linalg.norm(previous_action)))
        target_errors.append(float(np.sqrt(np.mean(target_error * target_error))))
        contact_counts.append(float(contacts.rigid_contact_count.numpy()[0]))
        if on_control_step is not None:
            on_control_step(
                model,
                state_0,
                sim_time,
                {
                    "height_m": height,
                    "tilt_deg": tilt,
                    "closure_um": closure_errors_um[-1],
                    "contact_count": contact_counts[-1],
                },
            )

        if height < 0.12 or tilt > 60.0:
            fall_control_step = control_step
            break

    completed_control_steps = len(heights)
    walk_slice = slice(min(stand_steps, completed_control_steps), completed_control_steps)
    walk_velocity_errors = velocity_errors[walk_slice]
    walk_velocities = forward_velocities[walk_slice]
    poses = state_0.body_q.numpy()
    final_root = _body_com_position(model, poses, pelvis) if np.isfinite(poses).all() else np.full(3, np.nan)
    initial_root = _body_com_position(model, initial_poses, pelvis)

    return {
        "case": case,
        "label": spec.label,
        "solver": spec.solver,
        "vbd_solve": spec.vbd_solve,
        "iterations": spec.iterations,
        "device": spec.device,
        "armature_mode": armature_mode,
        "status": "complete" if completed_control_steps == control_steps else "fell",
        "control_steps_requested": control_steps,
        "control_steps_completed": completed_control_steps,
        "completed_substeps": completed_substeps,
        "fall_time_s": None if fall_control_step is None else fall_control_step * config.env_dt,
        "simulated_time_s": sim_time,
        "stand_time_s": stand_steps * config.env_dt,
        "forward_command_mps": forward_speed,
        "forward_displacement_m": float(final_root[0] - initial_root[0]),
        "lateral_displacement_m": float(final_root[1] - initial_root[1]),
        "height_m": _summarize(heights),
        "tilt_deg": _summarize(tilts),
        "forward_velocity_mps": _summarize(walk_velocities),
        "forward_velocity_error_mps": _summarize(walk_velocity_errors),
        "path_error_m": _summarize(path_errors),
        "closure_error_um": _summarize(closure_errors_um),
        "action_l2": _summarize(action_norms),
        "actuated_target_error_rad": _summarize(target_errors),
        "contact_count": _summarize(contact_counts),
        "policy_p50_us": float(np.percentile(policy_times_us, 50.0)),
        "collision_p50_us": float(np.percentile(collision_times_us, 50.0)),
        "solver_p50_us": float(np.percentile(solver_times_us, 50.0)),
        "solver_p90_us": float(np.percentile(solver_times_us, 90.0)),
        "step_p50_us": float(np.percentile(np.asarray(collision_times_us) + np.asarray(solver_times_us), 50.0)),
        "actuated_dof_indices": actuated_indices.tolist(),
        "closure_joint_labels": sorted(closure_labels),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=CASE_CONFIGS, default=list(CASE_CONFIGS))
    parser.add_argument("--control-steps", type=int, default=400)
    parser.add_argument("--stand-steps", type=int, default=50)
    parser.add_argument("--forward-speed", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/vbd_complex_linkages/dr_legs_policy_results.json"),
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    asset_path = newton.utils.download_asset("disneyresearch")
    config = _load_config(asset_path)
    policy_path = asset_path / "dr_legs" / "rl_policies" / config.policy_file
    policy = load_policy(policy_path)
    rows = []
    for case in args.cases:
        print(f"Running DR Legs policy case={case}", flush=True)
        row = run_case(
            case,
            policy,
            config,
            control_steps=args.control_steps,
            stand_steps=args.stand_steps,
            forward_speed=args.forward_speed,
        )
        rows.append(row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)

    payload = {
        "source": "Newton Disney Research asset package",
        "policy_file": config.policy_file,
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "policy_observation_dim": 94,
        "policy_action_dim": 12,
        "config": {name: getattr(config, name) for name in config.__dataclass_fields__},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
