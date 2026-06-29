#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU performance comparison for real robot articulations.

The benchmark intentionally separates three paths:

* ``vbd_local``: existing SolverVBD per-body local rigid solve.
* ``vbd_sparse``: SolverVBD block-sparse rigid articulation solve.
* ``mujoco_cpu``: Newton's SolverMuJoCo wrapper around MuJoCo-C.
* ``mujoco_cpu_raw``: direct ``mj_step`` on SolverMuJoCo's internal model/data.

The raw MuJoCo path is the closest apples-to-MuJoCo throughput number. The
wrapped MuJoCo path includes Newton<->MuJoCo state conversion and is a useful
bound for wrapper overhead.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.utils
from newton import JointTargetMode


def build_robot(robot: str, *, add_ground: bool, device: str) -> newton.Model:
    robot_builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(robot_builder)
    robot_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3,
        limit_kd=1.0e1,
        friction=1.0e-5,
    )

    if robot == "g1":
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
    elif robot == "h1":
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
    else:
        raise ValueError(f"Unsupported robot: {robot}")

    builder = newton.ModelBuilder()
    builder.add_builder(robot_builder)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 2.0e2 if robot == "g1" else 1.0e2
    if add_ground:
        builder.add_ground_plane()
    builder.color()
    return builder.finalize(device=device)


def solver_stiffness_kwargs(joint_stiffness: str) -> dict:
    if joint_stiffness == "default":
        return {}
    if joint_stiffness == "fixed_high":
        return {
            "rigid_joint_linear_ke": 1.0e8,
            "rigid_joint_angular_ke": 1.0e6,
            "rigid_joint_linear_kd": 0.0,
            "rigid_joint_angular_kd": 0.0,
        }
    raise ValueError(f"Unsupported joint stiffness preset: {joint_stiffness}")


def make_runtime(args: argparse.Namespace, robot: str, mode: str) -> dict:
    use_contacts = args.contact_mode == "ground"
    if args.device != "cpu" and mode.startswith("mujoco_"):
        raise ValueError("MuJoCo CPU benchmark modes require --device cpu")
    model = build_robot(robot, add_ground=use_contacts, device=args.device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts() if use_contacts else None
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    if mode == "vbd_local":
        solver = newton.solvers.SolverVBD(
            model,
            iterations=args.vbd_iterations,
            rigid_articulation_solve="local",
            rigid_body_contact_buffer_size=args.contact_buffer_size,
            **solver_stiffness_kwargs(args.joint_stiffness),
        )
    elif mode == "vbd_sparse":
        solver = newton.solvers.SolverVBD(
            model,
            iterations=args.vbd_iterations,
            rigid_articulation_solve="block_sparse_joints",
            rigid_articulation_relaxation=args.rigid_articulation_relaxation,
            rigid_body_contact_buffer_size=args.contact_buffer_size,
            **solver_stiffness_kwargs(args.joint_stiffness),
        )
    elif mode in ("mujoco_cpu", "mujoco_cpu_raw"):
        solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=True,
            disable_contacts=not use_contacts,
            solver=args.mujoco_solver,
            integrator=args.mujoco_integrator,
            iterations=args.mujoco_iterations,
            ls_iterations=args.mujoco_ls_iterations,
            njmax=args.mujoco_njmax,
            nconmax=args.mujoco_nconmax,
            cone=args.mujoco_cone,
            impratio=args.mujoco_impratio,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return {
        "model": model,
        "state_0": state_0,
        "state_1": state_1,
        "control": control,
        "contacts": contacts,
        "solver": solver,
        "use_contacts": use_contacts,
    }


def _step_vbd(runtime: dict, dt: float) -> None:
    model = runtime["model"]
    state_0 = runtime["state_0"]
    state_1 = runtime["state_1"]
    control = runtime["control"]
    contacts = runtime["contacts"]
    solver = runtime["solver"]

    state_0.clear_forces()
    if runtime["use_contacts"]:
        assert contacts is not None
        model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    runtime["state_0"], runtime["state_1"] = state_1, state_0


def _step_vbd_fixed_buffers(runtime: dict, dt: float) -> None:
    model = runtime["model"]
    state_0 = runtime["state_0"]
    state_1 = runtime["state_1"]
    control = runtime["control"]
    contacts = runtime["contacts"]
    solver = runtime["solver"]

    state_0.clear_forces()
    if runtime["use_contacts"]:
        assert contacts is not None
        model.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)


def _step_mujoco_cpu(runtime: dict, dt: float) -> None:
    state_0 = runtime["state_0"]
    state_1 = runtime["state_1"]
    control = runtime["control"]
    contacts = runtime["contacts"]
    solver = runtime["solver"]

    solver.step(state_0, state_1, control, contacts, dt)
    runtime["state_0"], runtime["state_1"] = state_1, state_0


def _step_mujoco_cpu_raw(runtime: dict, dt: float) -> None:
    solver = runtime["solver"]
    solver.mj_model.opt.timestep = dt
    solver._mujoco.mj_step(solver.mj_model, solver.mj_data)


def _time_loop(fn, runtime: dict, steps: int, dt: float) -> list[float]:
    device = runtime["model"].device
    times_us = []
    for _ in range(steps):
        start = time.perf_counter()
        fn(runtime, dt)
        wp.synchronize_device(device)
        times_us.append((time.perf_counter() - start) * 1.0e6)
    return times_us


def _time_vbd_graph(runtime: dict, steps: int, dt: float) -> tuple[list[float], str | None]:
    device = runtime["model"].device
    try:
        with wp.ScopedCapture(device=device) as capture:
            _step_vbd_fixed_buffers(runtime, dt)
        graph = capture.graph
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"

    times_us = []
    for _ in range(steps):
        start = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize_device(device)
        times_us.append((time.perf_counter() - start) * 1.0e6)
    return times_us, None


def summarize_times(times_us: list[float]) -> dict:
    arr = np.asarray(times_us, dtype=np.float64)
    return {
        "mean_step_us": float(statistics.fmean(times_us)),
        "p50_step_us": float(np.percentile(arr, 50.0)),
        "p90_step_us": float(np.percentile(arr, 90.0)),
        "p95_step_us": float(np.percentile(arr, 95.0)),
        "min_step_us": float(np.min(arr)),
        "max_step_us": float(np.max(arr)),
    }


def run_case(args: argparse.Namespace, robot: str, mode: str) -> dict:
    runtime = make_runtime(args, robot, mode)
    model = runtime["model"]

    if mode.startswith("vbd_"):
        step_fn = _step_vbd
    elif mode == "mujoco_cpu":
        step_fn = _step_mujoco_cpu
    elif mode == "mujoco_cpu_raw":
        step_fn = _step_mujoco_cpu_raw
    else:
        raise ValueError(mode)

    for _ in range(args.warmup):
        step_fn(runtime, args.dt)
    wp.synchronize_device(model.device)

    times_us = _time_loop(step_fn, runtime, args.steps, args.dt)
    result = {
        "robot": robot,
        "mode": mode,
        "contact_mode": args.contact_mode,
        "joint_stiffness": args.joint_stiffness,
        "body_count": model.body_count,
        "joint_count": model.joint_count,
        "joint_dof_count": model.joint_dof_count,
        "shape_count": model.shape_count,
        "dt": args.dt,
        "steps": args.steps,
        "warmup": args.warmup,
        "vbd_iterations": args.vbd_iterations if mode.startswith("vbd_") else None,
        "mujoco_iterations": args.mujoco_iterations if mode.startswith("mujoco_") else None,
        **summarize_times(times_us),
    }

    if (args.graph or args.cpu_graph) and mode.startswith("vbd_"):
        graph_runtime = make_runtime(args, robot, mode)
        for _ in range(args.warmup):
            _step_vbd_fixed_buffers(graph_runtime, args.dt)
        wp.synchronize_device(graph_runtime["model"].device)
        graph_times, graph_error = _time_vbd_graph(graph_runtime, args.steps, args.dt)
        graph_prefix = "cpu_graph" if graph_runtime["model"].device.is_cpu else "cuda_graph"
        result[f"{graph_prefix}_error"] = graph_error
        if graph_times:
            for key, value in summarize_times(graph_times).items():
                result[f"{graph_prefix}_{key}"] = value
            result[f"{graph_prefix}_speedup_vs_python"] = (
                result["mean_step_us"] / result[f"{graph_prefix}_mean_step_us"]
            )

    return result


def add_comparisons(rows: list[dict]) -> list[dict]:
    by_robot: dict[tuple[str, str, str], dict[str, dict]] = {}
    for row in rows:
        key = (row["robot"], row["contact_mode"], row["joint_stiffness"])
        by_robot.setdefault(key, {})[row["mode"]] = row

    for modes in by_robot.values():
        raw = modes.get("mujoco_cpu_raw")
        wrapped = modes.get("mujoco_cpu")
        for row in modes.values():
            if raw is not None:
                row["ratio_vs_mujoco_raw"] = row["mean_step_us"] / raw["mean_step_us"]
            if wrapped is not None:
                row["ratio_vs_mujoco_wrapped"] = row["mean_step_us"] / wrapped["mean_step_us"]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", nargs="+", default=["g1"], choices=["g1", "h1"])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["vbd_local", "vbd_sparse", "mujoco_cpu", "mujoco_cpu_raw"],
        choices=[
            "vbd_local",
            "vbd_sparse",
            "mujoco_cpu",
            "mujoco_cpu_raw",
        ],
    )
    parser.add_argument("--contact-mode", choices=["none", "ground"], default="ground")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--joint-stiffness", choices=["default", "fixed_high"], default="fixed_high")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dt", type=float, default=1.0 / 360.0)
    parser.add_argument("--vbd-iterations", type=int, default=3)
    parser.add_argument("--rigid-articulation-relaxation", type=float, default=0.65)
    parser.add_argument("--contact-buffer-size", type=int, default=512)
    parser.add_argument("--mujoco-solver", default="newton")
    parser.add_argument("--mujoco-integrator", default="implicitfast")
    parser.add_argument("--mujoco-iterations", type=int, default=100)
    parser.add_argument("--mujoco-ls-iterations", type=int, default=50)
    parser.add_argument("--mujoco-njmax", type=int, default=300)
    parser.add_argument("--mujoco-nconmax", type=int, default=150)
    parser.add_argument("--mujoco-cone", default="elliptic")
    parser.add_argument("--mujoco-impratio", type=float, default=100.0)
    parser.add_argument("--cpu-graph", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for robot in args.robots:
        for mode in args.modes:
            rows.append(run_case(args, robot, mode))
    rows = add_comparisons(rows)
    payload = {"rows": rows}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
