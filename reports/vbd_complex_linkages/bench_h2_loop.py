#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Unitree's public H2 closed-loop model with VBD articulation solves."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton

DEFAULT_ASSET_DIR = Path("/home/horde/external-assets/unitree_ros_h2/robots/h2_description")
SOURCE_URL = "https://github.com/unitreerobotics/unitree_ros/blob/master/robots/h2_description/H2_loop.urdf"
SOURCE_COMMIT = "aa0f5c68b5aba347bad409e71b6430407da758d7"

CLOSURE_PAIRS = {
    "left_ankle": ("left_ankle_connect_rhs", "left_ankle_connect_lhs"),
    "right_ankle": ("right_ankle_connect_lhs", "right_ankle_connect_rhs"),
    "left_knee": ("left_knee_connect_rhs", "left_knee_connect_lhs"),
    "right_knee": ("right_knee_connect_lhs", "right_knee_connect_rhs"),
    "right_waist": ("waist_connect_rhs", "torso_connect_rhs"),
    "left_waist": ("waist_connect_lhs", "torso_connect_lhs"),
}

# The public URDF represents each loop rod with two fixed endpoint links.  Match
# the companion MJCF by making the driven endpoint spherical, then connecting
# the opposite endpoint to the distal point on that finite-length rod.
PROXIMAL_BALL_JOINTS = {
    "left_ankle_A_rod_joint",
    "right_ankle_A_rod_joint",
    "left_knee_connect_lhs_joint",
    "right_knee_connect_rhs_joint",
    "torso_connect_rhs_joint",
    "torso_connect_lhs_joint",
}

DRIVE_PHASES = {
    "left_ankle_A_joint": (0.12, 0.0),
    "right_ankle_A_joint": (0.12, math.pi),
    "left_knee_motor_joint": (0.15, 0.0),
    "right_knee_motor_joint": (0.15, math.pi),
    "torso_constraint_R_joint": (0.08, 0.0),
    "torso_constraint_L_joint": (0.08, math.pi),
}

CASE_CONFIGS = {
    "local_i8": ("VBD local, 8 iterations", "local", 8),
    "local_i32": ("VBD local, 32 iterations", "local", 32),
    "sparse_i8": ("VBD sparse direct, 8 iterations", "block_sparse_joints", 8),
}


def _short_label(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _prepare_urdf(asset_dir: Path, *, with_visuals: bool) -> str:
    root = ET.parse(asset_dir / "H2_loop.urdf").getroot()
    converted = set()
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name in PROXIMAL_BALL_JOINTS:
            joint.set("type", "ball")
            converted.add(name)
    if converted != PROXIMAL_BALL_JOINTS:
        missing = sorted(PROXIMAL_BALL_JOINTS - converted)
        raise ValueError(f"Missing H2 loop-rod joints: {missing}")
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "collision" or (not with_visuals and child.tag == "visual"):
                parent.remove(child)
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename and not Path(filename).is_absolute():
            mesh.set("filename", str((asset_dir / filename).resolve()))
    return ET.tostring(root, encoding="unicode")


def _find_label(labels: list[str], short: str) -> int:
    matches = [index for index, label in enumerate(labels) if _short_label(label) == short]
    if len(matches) != 1:
        raise ValueError(f"Expected one label ending in {short!r}, found {len(matches)}")
    return matches[0]


def _quat_between(a: np.ndarray, b: np.ndarray) -> wp.quat:
    a = a.astype(np.float64) / np.linalg.norm(a)
    b = b.astype(np.float64) / np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < -0.999999:
        axis = np.cross(a, np.array((1.0, 0.0, 0.0)))
        if np.linalg.norm(axis) < 1.0e-8:
            axis = np.cross(a, np.array((0.0, 1.0, 0.0)))
        axis /= np.linalg.norm(axis)
        return wp.quat_from_axis_angle(wp.vec3(*axis), math.pi)
    cross = np.cross(a, b)
    scale = math.sqrt((1.0 + dot) * 2.0)
    return wp.quat(*(cross / scale), 0.5 * scale)


def _build_tree_builder(urdf_source: str, *, with_visuals: bool) -> newton.ModelBuilder:
    builder = newton.ModelBuilder(gravity=0.0)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e5,
        limit_kd=1.0e3,
        friction=1.0e-5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder.add_urdf(
            urdf_source,
            floating=False,
            hide_visuals=not with_visuals,
            parse_visuals_as_colliders=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            mesh_maxhullvert=16,
            force_position_velocity_actuation=False,
        )
    return builder


def build_model(asset_dir: Path, device: str, *, with_visuals: bool) -> tuple[newton.Model, np.ndarray, np.ndarray]:
    urdf_source = _prepare_urdf(asset_dir, with_visuals=with_visuals)
    tree_builder = _build_tree_builder(urdf_source, with_visuals=with_visuals)
    tree_builder.color()
    tree_model = tree_builder.finalize(device="cpu")
    tree_state = tree_model.state()
    newton.eval_fk(tree_model, tree_model.joint_q, tree_model.joint_qd, tree_state)
    initial_q = tree_state.body_q.numpy().copy()
    initial_qd = tree_state.body_qd.numpy().copy()

    builder = _build_tree_builder(urdf_source, with_visuals=with_visuals)

    body_labels = list(builder.body_label)
    for name, (parent_name, child_name) in CLOSURE_PAIRS.items():
        parent = _find_label(body_labels, parent_name)
        child = _find_label(body_labels, child_name)
        child_anchor = _inverse_transform_point(initial_q[child], initial_q[parent, :3])
        builder.add_joint_ball(
            parent=parent,
            child=child,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(*child_anchor), wp.quat_identity()),
            label=f"h2_loop_{name}",
        )
        if with_visuals:
            length = float(np.linalg.norm(child_anchor))
            builder.add_shape_capsule(
                body=child,
                xform=wp.transform(
                    wp.vec3(*(0.5 * child_anchor)),
                    _quat_between(np.array((0.0, 0.0, 1.0)), child_anchor),
                ),
                radius=0.004,
                half_height=0.5 * length,
                as_site=True,
                color=wp.vec3(0.1, 0.22, 0.25),
                label=f"h2_loop_{name}_rod",
            )

    for joint_name in DRIVE_PHASES:
        joint_index = _find_label(list(builder.joint_label), joint_name)
        dof_start = builder.joint_qd_start[joint_index]
        builder.joint_target_ke[dof_start] = 1.0e4
        builder.joint_target_kd[dof_start] = 2.0e2
        builder.joint_target_mode[dof_start] = int(newton.JointTargetMode.POSITION)

    builder.articulation_start.clear()
    builder.articulation_end.clear()
    builder.articulation_label.clear()
    builder.articulation_world.clear()
    builder.joint_articulation = [-1] * len(builder.joint_articulation)
    builder.add_articulation(
        list(range(len(builder.joint_type))),
        label="Unitree H2 closed-loop articulation",
        allow_closed_loops=True,
    )
    builder.color()
    model = builder.finalize(device=device)
    model.body_q.assign(initial_q)
    model.body_qd.assign(initial_qd)
    return model, initial_q, initial_qd


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


def _transform_point(body_q: np.ndarray, local: np.ndarray) -> np.ndarray:
    pure = np.array((local[0], local[1], local[2], 0.0), dtype=np.float64)
    rotated = _quat_mul(_quat_mul(body_q[3:7], pure), _quat_inv(body_q[3:7]))[:3]
    return body_q[:3] + rotated


def _inverse_transform_point(body_q: np.ndarray, world: np.ndarray) -> np.ndarray:
    offset = world - body_q[:3]
    pure = np.array((offset[0], offset[1], offset[2], 0.0), dtype=np.float64)
    inverse = _quat_inv(body_q[3:7])
    return _quat_mul(_quat_mul(inverse, pure), body_q[3:7])[:3]


def closure_errors(model: newton.Model, state: newton.State) -> dict[str, float]:
    body_q = state.body_q.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_x_p = model.joint_X_p.numpy()
    joint_x_c = model.joint_X_c.numpy()
    per_joint = {}
    vectors = []
    for joint_index, label in enumerate(model.joint_label):
        short = _short_label(label)
        if not short.startswith("h2_loop_"):
            continue
        parent = int(joint_parent[joint_index])
        child = int(joint_child[joint_index])
        error = _transform_point(body_q[child], joint_x_c[joint_index, :3]) - _transform_point(
            body_q[parent], joint_x_p[joint_index, :3]
        )
        vectors.append(error)
        per_joint[short.removeprefix("h2_loop_")] = float(np.linalg.norm(error) * 1.0e6)
    values = np.asarray(list(per_joint.values()), dtype=np.float64)
    return {
        "aggregate_um": float(np.linalg.norm(np.concatenate(vectors)) * 1.0e6),
        "joint_rms_um": float(np.sqrt(np.mean(values * values))),
        "joint_max_um": float(np.max(values)),
        **{f"{name}_um": value for name, value in per_joint.items()},
    }


def _set_targets(
    model: newton.Model,
    control: newton.Control,
    target_q: np.ndarray,
    target_indices: dict[str, int],
    sim_time: float,
    period: float,
) -> None:
    phase = 2.0 * math.pi * sim_time / period
    for name, (amplitude, offset) in DRIVE_PHASES.items():
        target_q[target_indices[name]] = amplitude * math.sin(phase + offset)
    control.joint_target_q.assign(target_q)


def run_case(
    asset_dir: Path,
    mode: str,
    device: str,
    *,
    frames: int,
    iterations: int,
    substeps: int,
    period: float,
    relaxation: float,
    joint_ke: float,
    joint_kd: float,
    with_visuals: bool = False,
    viewer=None,
) -> tuple[dict, newton.Model, newton.State]:
    model, initial_q, initial_qd = build_model(asset_dir, device, with_visuals=with_visuals)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    state_0.body_q.assign(initial_q)
    state_0.body_qd.assign(initial_qd)
    state_1.assign(state_0)
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=relaxation,
        rigid_avbd_alpha=0.0,
        rigid_avbd_beta=0.0,
        rigid_joint_linear_ke=joint_ke,
        rigid_joint_angular_ke=joint_ke,
        rigid_joint_linear_kd=joint_kd,
        rigid_joint_angular_kd=joint_kd,
    )
    target_q = control.joint_target_q.numpy().copy()
    target_starts = model.joint_target_q_start.numpy()
    joint_labels = list(model.joint_label)
    target_indices = {name: int(target_starts[_find_label(joint_labels, name)]) for name in DRIVE_PHASES}

    if viewer is not None:
        viewer.set_model(model)

    frame_dt = 1.0 / 60.0
    dt = frame_dt / substeps
    sim_time = 0.0
    errors = [closure_errors(model, state_0)]
    step_times_us = []
    max_speed = 0.0
    finite = True
    for _frame in range(frames):
        for _substep in range(substeps):
            _set_targets(model, control, target_q, target_indices, sim_time, period)
            state_0.clear_forces()
            start = time.perf_counter()
            solver.step(state_0, state_1, control, None, dt)
            wp.synchronize_device(model.device)
            step_times_us.append((time.perf_counter() - start) * 1.0e6)
            state_0, state_1 = state_1, state_0
            sim_time += dt
            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()
            finite = bool(np.isfinite(body_q).all() and np.isfinite(body_qd).all())
            if not finite:
                break
            max_speed = max(max_speed, float(np.max(np.linalg.norm(body_qd[:, :3], axis=1))))
            errors.append(closure_errors(model, state_0))
        if viewer is not None and finite:
            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.end_frame()
        if not finite:
            break

    aggregate = np.asarray([row["aggregate_um"] for row in errors], dtype=np.float64)
    joint_max = np.asarray([row["joint_max_um"] for row in errors], dtype=np.float64)
    stable = bool(finite and max_speed < 5.0 and float(np.max(aggregate)) < 1.0e4)
    metrics = {
        "mode": mode,
        "device": device,
        "status": "ok" if stable else ("divergent" if finite else "nonfinite"),
        "finite": finite,
        "stable": stable,
        "frames_requested": frames,
        "substeps_completed": len(step_times_us),
        "iterations": iterations,
        "substeps_per_frame": substeps,
        "dt": dt,
        "relaxation": relaxation,
        "joint_ke": joint_ke,
        "joint_kd": joint_kd,
        "body_count": model.body_count,
        "joint_count": model.joint_count,
        "joint_dof_count": model.joint_dof_count,
        "articulation_count": model.articulation_count,
        "closure_count": len(CLOSURE_PAIRS),
        "initial_closure": errors[0],
        "final_closure": errors[-1],
        "rms_aggregate_closure_um": float(np.sqrt(np.mean(aggregate * aggregate))),
        "max_aggregate_closure_um": float(np.max(aggregate)),
        "max_joint_closure_um": float(np.max(joint_max)),
        "max_body_speed_mps": max_speed,
        "mean_step_us": float(statistics.fmean(step_times_us)),
        "p50_step_us": float(np.percentile(step_times_us, 50.0)),
        "p90_step_us": float(np.percentile(step_times_us, 90.0)),
    }
    return metrics, model, state_0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--cases", nargs="+", choices=CASE_CONFIGS, default=list(CASE_CONFIGS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--period", type=float, default=2.8)
    parser.add_argument("--relaxation", type=float, default=0.65)
    parser.add_argument("--joint-ke", type=float, default=2.0e5)
    parser.add_argument("--joint-kd", type=float, default=5.0e2)
    parser.add_argument("--output", type=Path, default=Path("reports/vbd_complex_linkages/h2_loop_results.json"))
    args = parser.parse_args()

    rows = []
    for case in args.cases:
        label, mode, iterations = CASE_CONFIGS[case]
        print(f"Running H2 case={case} device={args.device}", flush=True)
        metrics, _, _ = run_case(
            args.asset_dir,
            mode,
            args.device,
            frames=args.frames,
            iterations=iterations,
            substeps=args.substeps,
            period=args.period,
            relaxation=args.relaxation,
            joint_ke=args.joint_ke,
            joint_kd=args.joint_kd,
        )
        metrics["case"] = case
        metrics["label"] = label
        rows.append(metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)

    payload = {
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "asset_license": "BSD-3-Clause",
        "conversion": ("six finite-length rods reconstructed with a proximal BALL joint and a distal BALL closure"),
        "closure_pairs": CLOSURE_PAIRS,
        "drive_phases": DRIVE_PHASES,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
