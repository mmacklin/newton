#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smoke benchmark for VBD local versus sparse articulation on humanoids with contacts."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.utils
from newton import JointTargetMode


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
    return _quat_mul(_quat_mul(q, np.array([v[0], v[1], v[2], 0.0])), _quat_inv(q))[0:3]


def _quat_rotvec(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    vector_norm = float(np.linalg.norm(q[0:3]))
    if vector_norm < 1.0e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vector_norm, float(q[3]))
    return q[0:3] * (angle / vector_norm)


def _transform_point(xform: np.ndarray, local: np.ndarray) -> np.ndarray:
    return xform[0:3] + _quat_rotate(xform[3:7], local)


def _joint_split_residual(model: newton.Model, state: newton.State) -> tuple[float, float]:
    body_q = state.body_q.numpy()
    body_q_rest = model.body_q.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_type = model.joint_type.numpy()
    joint_x_p = model.joint_X_p.numpy()
    joint_x_c = model.joint_X_c.numpy()
    joint_axis = model.joint_axis.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    joint_dof_dim = model.joint_dof_dim.numpy()

    linear_residuals = []
    angular_residuals = []
    for joint_index in range(model.joint_count):
        child = int(joint_child[joint_index])
        parent = int(joint_parent[joint_index])
        jt = int(joint_type[joint_index])
        if child < 0:
            continue

        x_child = _transform_point(body_q[child], joint_x_c[joint_index, 0:3])
        q_child = _quat_mul(body_q[child, 3:7], joint_x_c[joint_index, 3:7])
        child_rest_q = _quat_mul(body_q_rest[child, 3:7], joint_x_c[joint_index, 3:7])
        if parent >= 0:
            x_parent = _transform_point(body_q[parent], joint_x_p[joint_index, 0:3])
            q_parent = _quat_mul(body_q[parent, 3:7], joint_x_p[joint_index, 3:7])
            parent_rest_q = _quat_mul(body_q_rest[parent, 3:7], joint_x_p[joint_index, 3:7])
        else:
            x_parent = joint_x_p[joint_index, 0:3]
            q_parent = joint_x_p[joint_index, 3:7]
            parent_rest_q = q_parent

        P_lin = np.eye(3)
        P_ang = np.eye(3)
        qd_start = int(joint_qd_start[joint_index])
        if jt == int(newton.JointType.PRISMATIC):
            axis = _quat_rotate(q_parent, joint_axis[qd_start])
            axis /= np.linalg.norm(axis)
            P_lin = P_lin - np.outer(axis, axis)
        elif jt == int(newton.JointType.REVOLUTE):
            axis = _quat_rotate(q_parent, joint_axis[qd_start])
            axis /= np.linalg.norm(axis)
            P_ang = P_ang - np.outer(axis, axis)
        elif jt == int(newton.JointType.D6):
            lin_count = int(joint_dof_dim[joint_index, 0])
            ang_count = int(joint_dof_dim[joint_index, 1])
            for axis_index in range(lin_count):
                axis = _quat_rotate(q_parent, joint_axis[qd_start + axis_index])
                axis /= np.linalg.norm(axis)
                P_lin = P_lin - np.outer(axis, axis)
            for axis_index in range(ang_count):
                axis = _quat_rotate(q_parent, joint_axis[qd_start + lin_count + axis_index])
                axis /= np.linalg.norm(axis)
                P_ang = P_ang - np.outer(axis, axis)

        if jt != int(newton.JointType.FREE):
            linear_residuals.append(P_lin @ (x_child - x_parent))

        if jt in (
            int(newton.JointType.CABLE),
            int(newton.JointType.FIXED),
            int(newton.JointType.REVOLUTE),
            int(newton.JointType.PRISMATIC),
            int(newton.JointType.D6),
        ):
            q_rel = _quat_mul(_quat_inv(q_parent), q_child)
            q_rel_rest = _quat_mul(_quat_inv(parent_rest_q), child_rest_q)
            q_err = _quat_mul(q_rel, _quat_inv(q_rel_rest))
            angular_world = _quat_rotate(q_parent, _quat_rotvec(q_err))
            angular_residuals.append(P_ang @ angular_world)

    linear = np.concatenate(linear_residuals) if linear_residuals else np.zeros(0)
    angular = np.concatenate(angular_residuals) if angular_residuals else np.zeros(0)
    return float(np.linalg.norm(linear)), float(np.linalg.norm(angular))


def build_humanoid(robot: str, *, add_ground: bool) -> newton.Model:
    robot_builder = newton.ModelBuilder()
    robot_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1.0e-5
    )

    if robot == "h1":
        robot_builder.default_shape_cfg.ke = 2.0e3
        robot_builder.default_shape_cfg.kd = 1.0e2
        robot_builder.default_shape_cfg.kf = 1.0e3
        robot_builder.default_shape_cfg.mu = 0.75
        asset_path = newton.utils.download_asset("unitree_h1")
        robot_builder.add_usd(
            str(asset_path / "usd_structured" / "h1.usda"),
            ignore_paths=["/GroundPlane"],
            enable_self_collisions=False,
        )
        robot_builder.approximate_meshes("bounding_box")
        for i in range(len(robot_builder.joint_target_ke)):
            robot_builder.joint_target_ke[i] = 150.0
            robot_builder.joint_target_kd[i] = 5.0
            robot_builder.joint_target_mode[i] = int(JointTargetMode.POSITION)
    elif robot == "g1":
        robot_builder.default_shape_cfg.ke = 1.0e3
        robot_builder.default_shape_cfg.kd = 2.0e2
        robot_builder.default_shape_cfg.kf = 1.0e3
        robot_builder.default_shape_cfg.mu = 0.75
        asset_path = newton.utils.download_asset("unitree_g1")
        robot_builder.add_usd(
            str(asset_path / "usd_structured" / "g1_29dof_with_hand_rev_1_0.usda"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.2)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )
        for i in range(6, robot_builder.joint_dof_count):
            robot_builder.joint_target_ke[i] = 500.0
            robot_builder.joint_target_kd[i] = 10.0
            robot_builder.joint_target_mode[i] = int(JointTargetMode.POSITION)
        robot_builder.approximate_meshes("bounding_box")
    else:
        raise ValueError(f"Unsupported robot: {robot}")

    builder = newton.ModelBuilder()
    builder.add_builder(robot_builder)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 2.0e2 if robot == "g1" else 1.0e2
    if add_ground:
        builder.add_ground_plane()
    builder.color()
    return builder.finalize(device="cpu")


def solver_stiffness_kwargs(joint_stiffness: str) -> dict:
    if joint_stiffness == "default":
        return {}
    if joint_stiffness == "fixed_high":
        return {
            "rigid_joint_adaptive_stiffness": False,
            "rigid_joint_linear_ke": 1.0e8,
            "rigid_joint_angular_ke": 1.0e6,
            "rigid_joint_linear_kd": 0.0,
            "rigid_joint_angular_kd": 0.0,
        }
    raise ValueError(f"Unsupported joint stiffness preset: {joint_stiffness}")


def run_mode(
    model: newton.Model,
    mode: str,
    steps: int,
    iterations: int,
    dt: float,
    solver_kwargs: dict,
) -> dict:
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_articulation_solve=mode,
        rigid_body_contact_buffer_size=512,
        **solver_kwargs,
    )

    max_contacts = 0
    first_contact_step = None
    ok = True
    start = time.perf_counter()
    for step in range(steps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        max_contacts = max(max_contacts, contact_count)
        if contact_count > 0 and first_contact_step is None:
            first_contact_step = step
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

        body_q = state_0.body_q.numpy()
        body_qd = state_0.body_qd.numpy()
        if not np.isfinite(body_q).all() or not np.isfinite(body_qd).all():
            ok = False
            break

    elapsed_ms = 1.0e3 * (time.perf_counter() - start)
    body_q = state_0.body_q.numpy()
    body_qd = state_0.body_qd.numpy()
    linear_residual, angular_residual = _joint_split_residual(model, state_0)
    return {
        "mode": mode,
        "ok": ok,
        "steps_completed": step + 1,
        "elapsed_ms": elapsed_ms,
        "mean_step_ms": elapsed_ms / float(step + 1),
        "first_contact_step": first_contact_step,
        "max_contacts": max_contacts,
        "min_body_z": float(np.min(body_q[:, 2])),
        "max_linear_speed": float(np.max(np.linalg.norm(body_qd[:, 0:3], axis=1))),
        "joint_linear_l2": linear_residual,
        "joint_angular_l2": angular_residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", nargs="+", default=["h1", "g1"], choices=["h1", "g1"])
    parser.add_argument("--modes", nargs="+", default=["local", "block_sparse_joints"])
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--contact-mode", default="ground", choices=["ground", "none"])
    parser.add_argument("--joint-stiffness", default="default", choices=["default", "fixed_high"])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("humanoid_contact_cpu.json"))
    args = parser.parse_args()

    results = []
    solver_kwargs = solver_stiffness_kwargs(args.joint_stiffness)
    for robot in args.robots:
        model = build_humanoid(robot, add_ground=args.contact_mode == "ground")
        dt = 1.0 / 240.0 if robot == "h1" else 1.0 / 360.0
        for mode in args.modes:
            result = run_mode(model, mode, args.steps, args.iterations, dt, solver_kwargs)
            result.update(
                {
                    "robot": robot,
                    "contact_mode": args.contact_mode,
                    "joint_stiffness": args.joint_stiffness,
                    "body_count": model.body_count,
                    "joint_count": model.joint_count,
                    "joint_dof_count": model.joint_dof_count,
                    "shape_count": model.shape_count,
                    "dt": dt,
                    "iterations": args.iterations,
                }
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True))

    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
