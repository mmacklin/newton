#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark and render public synthetic sparse-VBD validation mechanisms."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp

import newton
from reports.vbd_complex_linkages.bench_complex_linkages import _joint_anchor_residuals
from reports.vbd_complex_linkages.render_complex_linkages import _as_rgb, _capture_frame, _make_viewer, _write_video


def _body_inertia(mass: float, length: float, width: float) -> wp.mat33:
    i_long = mass * width * width / 6.0
    i_cross = mass * (length * length + width * width) / 12.0
    return wp.mat33(i_long, 0.0, 0.0, 0.0, i_cross, 0.0, 0.0, 0.0, i_cross)


def _bar_body(
    builder: newton.ModelBuilder,
    start: np.ndarray,
    end: np.ndarray,
    label: str,
    color: wp.vec3,
) -> int:
    direction = end - start
    length = float(np.linalg.norm(direction))
    angle = math.atan2(float(direction[1]), float(direction[0]))
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
    mass = length
    body = builder.add_link(
        xform=wp.transform(wp.vec3(*(0.5 * (start + end))), rotation),
        mass=mass,
        inertia=_body_inertia(mass, length, 0.08),
        label=label,
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
    builder.add_shape_box(body, hx=0.5 * length, hy=0.04, hz=0.04, cfg=cfg, color=color)
    return body


def _circle_intersection(a: np.ndarray, b: np.ndarray, radius_a: float, radius_b: float) -> np.ndarray:
    delta = b - a
    distance = float(np.linalg.norm(delta))
    along = (radius_a * radius_a - radius_b * radius_b + distance * distance) / (2.0 * distance)
    height = math.sqrt(max(radius_a * radius_a - along * along, 0.0))
    direction = delta / distance
    normal = np.array((-direction[1], direction[0]), dtype=np.float64)
    return a + along * direction + height * normal


def build_four_bar(device: str) -> tuple[newton.Model, int]:
    builder = newton.ModelBuilder(gravity=0.0)
    left_ground = np.array((-0.6, 0.0), dtype=np.float64)
    right_ground = np.array((0.6, 0.0), dtype=np.float64)
    crank_length = 0.42
    coupler_length = 0.92
    rocker_length = 0.56
    crank_angle = math.radians(50.0)
    crank_tip = left_ground + crank_length * np.array((math.cos(crank_angle), math.sin(crank_angle)))
    coupler_tip = _circle_intersection(crank_tip, right_ground, coupler_length, rocker_length)

    def point2(value: np.ndarray) -> np.ndarray:
        return np.array((value[0], value[1], 1.0), dtype=np.float64)

    crank = _bar_body(builder, point2(left_ground), point2(crank_tip), "crank", wp.vec3(0.15, 0.48, 0.90))
    coupler = _bar_body(builder, point2(crank_tip), point2(coupler_tip), "coupler", wp.vec3(0.12, 0.72, 0.38))
    rocker = _bar_body(builder, point2(right_ground), point2(coupler_tip), "rocker", wp.vec3(0.90, 0.34, 0.24))

    axis = newton.Axis.Z
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=crank,
            parent_xform=wp.transform(wp.vec3(*point2(left_ground))),
            child_xform=wp.transform(wp.vec3(-0.5 * crank_length, 0.0, 0.0)),
            axis=axis,
            target_ke=2.0e4,
            target_kd=2.0e2,
            label="crank_drive",
        ),
        builder.add_joint_revolute(
            parent=crank,
            child=coupler,
            parent_xform=wp.transform(wp.vec3(0.5 * crank_length, 0.0, 0.0)),
            child_xform=wp.transform(wp.vec3(-0.5 * coupler_length, 0.0, 0.0)),
            axis=axis,
            label="crank_coupler",
        ),
        builder.add_joint_revolute(
            parent=coupler,
            child=rocker,
            parent_xform=wp.transform(wp.vec3(0.5 * coupler_length, 0.0, 0.0)),
            child_xform=wp.transform(wp.vec3(0.5 * rocker_length, 0.0, 0.0)),
            axis=axis,
            label="coupler_rocker",
        ),
        builder.add_joint_revolute(
            parent=-1,
            child=rocker,
            parent_xform=wp.transform(wp.vec3(*point2(right_ground))),
            child_xform=wp.transform(wp.vec3(-0.5 * rocker_length, 0.0, 0.0)),
            axis=axis,
            label="rocker_ground_closure",
        ),
    ]
    builder.add_articulation(joints, label="four_bar", allow_closed_loops=True)
    builder.color()
    return builder.finalize(device=device), joints[0]


def build_cable_chain(device: str, segment_count: int = 32, gravity: float = 0.0) -> newton.Model:
    builder = newton.ModelBuilder(gravity=gravity)
    points, quaternions = newton.utils.create_straight_cable_points_and_quaternions(
        start=wp.vec3(-1.6, 0.0, 1.4),
        direction=wp.vec3(1.0, 0.0, 0.0),
        length=3.2,
        num_segments=segment_count,
    )
    cfg = newton.ModelBuilder.ShapeConfig(
        density=600.0,
        has_shape_collision=False,
    )
    bodies, cable_joints = builder.add_rod(
        positions=points,
        quaternions=quaternions,
        radius=0.025,
        cfg=cfg,
        stretch_stiffness=1.0e6,
        stretch_damping=1.0e2,
        bend_stiffness=2.0e2,
        bend_damping=2.0,
        wrap_in_articulation=False,
        body_frame_origin="com",
        color=wp.vec3(0.12, 0.60, 0.82),
        label="cable_chain",
    )
    visual_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
    segment_half_length = 1.6 / segment_count
    for body in bodies:
        builder.add_shape_box(
            body,
            hx=0.022,
            hy=0.022,
            hz=segment_half_length,
            cfg=visual_cfg,
            color=wp.vec3(0.12, 0.60, 0.82),
        )
    root_pose = builder.body_q[bodies[0]]
    root_joint = builder.add_joint_fixed(
        parent=-1,
        child=bodies[0],
        parent_xform=root_pose,
        child_xform=wp.transform_identity(),
        label="cable_root",
    )
    builder.add_articulation(sorted([root_joint, *cable_joints]), label="cable_chain")
    builder.color()
    return builder.finalize(device=device)


def _make_solver(model: newton.Model, mode: str, iterations: int, relaxation: float = 1.0):
    return newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=relaxation,
        rigid_avbd_alpha=0.0,
        rigid_avbd_beta=0.0,
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_angular_ke=2.0e2,
        rigid_joint_linear_kd=1.0e2,
        rigid_joint_angular_kd=2.0,
    )


def _perturb_cable(state: newton.State, amplitude: float = 0.06) -> None:
    poses = state.body_q.numpy().copy()
    for body in range(1, len(poses)):
        phase = float(body)
        poses[body, 1] += amplitude * math.sin(0.73 * phase)
        poses[body, 2] += 0.6 * amplitude * math.cos(0.41 * phase)
    state.body_q.assign(poses)


def _run_cable_residual(mode: str, iterations: int) -> float:
    model = build_cable_chain("cpu")
    state_0 = model.state()
    state_1 = model.state()
    _perturb_cable(state_0)
    solver = _make_solver(model, mode, iterations)
    solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)
    return _joint_anchor_residuals(model, state_1)["linear_norm_m"]


def _time_cable_step(mode: str, iterations: int, repeats: int) -> float:
    model = build_cable_chain("cpu")
    state_0 = model.state()
    state_1 = model.state()
    solver = _make_solver(model, mode, iterations)
    control = model.control()
    times = []
    for repeat in range(repeats + 5):
        state_0.clear_forces()
        start = time.perf_counter()
        solver.step(state_0, state_1, control, None, 1.0 / 120.0)
        wp.synchronize_device(model.device)
        elapsed = 1.0e6 * (time.perf_counter() - start)
        state_0, state_1 = state_1, state_0
        if repeat >= 5:
            times.append(elapsed)
    return float(statistics.median(times))


def benchmark_cable(repeats: int) -> list[dict]:
    rows = []
    for iterations in (1, 2, 4, 8, 16):
        for mode in ("local", "block_sparse_joints"):
            rows.append(
                {
                    "mode": mode,
                    "iterations": iterations,
                    "residual_m": _run_cable_residual(mode, iterations),
                    "p50_step_us": _time_cable_step(mode, iterations, repeats),
                }
            )
    for row in rows:
        local_candidates = [
            candidate
            for candidate in rows
            if candidate["mode"] == "local" and candidate["p50_step_us"] <= row["p50_step_us"]
        ]
        if local_candidates:
            best_local = min(local_candidates, key=lambda candidate: candidate["residual_m"])
            row["best_local_residual_at_budget_m"] = best_local["residual_m"]
            row["residual_reduction_at_budget"] = best_local["residual_m"] / max(row["residual_m"], 1.0e-12)
    return rows


def _set_four_bar_target(model: newton.Model, control: newton.Control, joint: int, target: float) -> None:
    values = control.joint_target_q.numpy()
    values[int(model.joint_target_q_start.numpy()[joint])] = target
    control.joint_target_q.assign(values)


def _render_mode(scenario: str, mode: str, frames: int) -> tuple[list[np.ndarray], dict]:
    if scenario == "four-bar":
        model, drive_joint = build_four_bar("cuda:0")
        iterations = 8
    else:
        model = build_cable_chain("cuda:0", gravity=-9.81)
        drive_joint = -1
        iterations = 4
    state_0 = model.state()
    state_1 = model.state()
    if scenario == "cable":
        _perturb_cable(state_0, amplitude=0.03)
    control = model.control()
    solver = _make_solver(model, mode, iterations, relaxation=0.8 if scenario == "cable" else 1.0)
    viewer = _make_viewer(model, state_0, "dr-legs", 960, 540)
    rendered = []
    residuals = []
    sim_time = 0.0
    try:
        for _frame in range(frames):
            for _substep in range(4):
                if drive_joint >= 0:
                    _set_four_bar_target(model, control, drive_joint, 0.35 * math.sin(2.0 * math.pi * sim_time / 2.5))
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, 1.0 / 240.0)
                state_0, state_1 = state_1, state_0
                sim_time += 1.0 / 240.0
            residual_um = 1.0e6 * _joint_anchor_residuals(model, state_0)["linear_norm_m"]
            residuals.append(residual_um)
            label = f"{mode.replace('_', ' ')}  residual {residual_um:.1f} um"
            rendered.append(_capture_frame(viewer, state_0, sim_time, label))
    finally:
        viewer.close()
    return rendered, {
        "mode": mode,
        "iterations": iterations,
        "rms_residual_um": float(np.sqrt(np.mean(np.square(residuals)))),
        "max_residual_um": float(np.max(residuals)),
        "finite": bool(np.isfinite(state_0.body_q.numpy()).all()),
    }


def render_comparisons(output_dir: Path, frames: int, scenarios: list[str]) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in scenarios:
        with tempfile.TemporaryDirectory(prefix=f"vbd-{scenario}-") as temp_dir:
            temp_path = Path(temp_dir)
            captures = {}
            metrics = {}
            for mode in ("local", "block_sparse_joints"):
                frame_path = temp_path / f"{mode}.npy"
                metric_path = temp_path / f"{mode}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--render-worker",
                    scenario,
                    mode,
                    "--frames",
                    str(frames),
                    "--worker-frames",
                    str(frame_path),
                    "--worker-metrics",
                    str(metric_path),
                ]
                environment = dict(os.environ)
                environment.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
                subprocess.run(command, check=True, env=environment)
                captures[mode] = np.load(frame_path, mmap_mode="r")
                metrics[mode] = json.loads(metric_path.read_text())

            local_frames = captures["local"]
            sparse_frames = captures["block_sparse_joints"]
            combined = [
                np.concatenate((_as_rgb(a), _as_rgb(b)), axis=1)
                for a, b in zip(local_frames, sparse_frames, strict=True)
            ]
            video = output_dir / f"{scenario}_local_vs_sparse.mp4"
            poster = output_dir / f"{scenario}_local_vs_sparse.jpg"
            _write_video(video, combined, 60)
            imageio.imwrite(poster, combined[len(combined) // 2])
        rows.append(
            {
                "scenario": scenario,
                "video": f"videos/{video.name}",
                "poster": f"videos/{poster.name}",
                "local": metrics["local"],
                "sparse": metrics["block_sparse_joints"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--scenarios", nargs="+", choices=("four-bar", "cable"), default=["four-bar", "cable"])
    parser.add_argument("--render-worker", nargs=2, metavar=("SCENARIO", "MODE"))
    parser.add_argument("--worker-frames", type=Path)
    parser.add_argument("--worker-metrics", type=Path)
    parser.add_argument("--video-dir", type=Path, default=Path.home() / "reports" / "vbd-complex-linkages" / "videos")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/vbd_complex_linkages/visual_validation_results.json")
    )
    args = parser.parse_args()

    if args.render_worker:
        if args.worker_frames is None or args.worker_metrics is None:
            parser.error("--render-worker requires --worker-frames and --worker-metrics")
        scenario, mode = args.render_worker
        rendered, metrics = _render_mode(scenario, mode, args.frames)
        np.save(args.worker_frames, np.asarray(rendered, dtype=np.uint8))
        args.worker_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        return

    existing = json.loads(args.output.read_text()) if args.output.exists() else {}
    payload = {
        "cable_convergence": existing.get("cable_convergence", [])
        if args.skip_benchmark
        else benchmark_cable(args.timing_repeats),
        "visuals": existing.get("visuals", [])
        if args.skip_render
        else render_comparisons(args.video_dir, args.frames, args.scenarios),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
