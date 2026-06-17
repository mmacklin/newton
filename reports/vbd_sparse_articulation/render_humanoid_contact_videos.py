#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render ViewerGL videos for the VBD humanoid contact smoke cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp
from bench_vbd_humanoid_contact import _joint_split_residual, build_humanoid, solver_stiffness_kwargs

import newton
import newton.viewer

WIDTH = 960
HEIGHT = 540
FPS = 30

_WP_EMPTY = wp.empty


def _install_cpu_pinned_fallback() -> None:
    """Let ViewerGL run on CPU-only hosts where Warp pinned allocation needs CUDA."""

    def empty_with_fallback(*args, **kwargs):
        try:
            return _WP_EMPTY(*args, **kwargs)
        except (AttributeError, RuntimeError):
            if kwargs.get("pinned") and str(kwargs.get("device", "cpu")) == "cpu":
                fallback_kwargs = dict(kwargs)
                fallback_kwargs["pinned"] = False
                return _WP_EMPTY(*args, **fallback_kwargs)
            raise

    wp.empty = empty_with_fallback


def _as_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.shape[-1] == 4:
        return np.ascontiguousarray(frame[:, :, :3])
    return np.ascontiguousarray(frame)


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(_as_rgb(frame))


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return frame

    image = Image.fromarray(_as_rgb(frame))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    pad = 8
    draw.rectangle(
        (0, 0, box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad),
        fill=(20, 24, 28),
    )
    draw.text((pad, pad), label, fill=(245, 248, 250), font=font)
    return np.asarray(image)


def _setup_camera(viewer: newton.viewer.ViewerGL, state: newton.State) -> None:
    body_pos = state.body_q.numpy()[:, 0:3]
    min_bounds = np.min(body_pos, axis=0)
    max_bounds = np.max(body_pos, axis=0)
    center = 0.5 * (min_bounds + max_bounds)
    extent = max(float(np.max(max_bounds - min_bounds)), 1.0)

    target = center + np.array([0.0, 0.0, 0.10], dtype=np.float64)
    eye = target + np.array([1.45 * extent, -2.10 * extent, 0.72 * extent], dtype=np.float64)
    front = target - eye
    front /= np.linalg.norm(front)
    yaw = float(np.degrees(np.arctan2(front[1], front[0])))
    pitch = float(np.degrees(np.arcsin(front[2])))
    viewer.set_camera(pos=wp.vec3(float(eye[0]), float(eye[1]), float(eye[2])), pitch=pitch, yaw=yaw)
    if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
        viewer.camera.fov = 38.0


def render_case(
    *,
    robot: str,
    mode: str,
    joint_stiffness: str,
    steps: int,
    iterations: int,
    width: int,
    height: int,
    label: str,
) -> tuple[list[np.ndarray], dict]:
    model = build_humanoid(robot, add_ground=True)
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
        **solver_stiffness_kwargs(joint_stiffness),
    )

    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
    frames: list[np.ndarray] = []
    max_contacts = 0
    first_contact_step = None
    ok = True
    sim_time = 0.0
    dt = 1.0 / 240.0 if robot == "h1" else 1.0 / 360.0

    try:
        viewer.set_model(model)
        _setup_camera(viewer, state_0)

        for step in range(steps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            contact_count = int(contacts.rigid_contact_count.numpy()[0])
            max_contacts = max(max_contacts, contact_count)
            if contact_count > 0 and first_contact_step is None:
                first_contact_step = step

            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
            sim_time += dt

            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.log_contacts(contacts, state_0)
            viewer.end_frame()

            frame = viewer.get_frame().numpy()
            frames.append(_label_frame(frame, label))

            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()
            if not np.isfinite(body_q).all() or not np.isfinite(body_qd).all():
                ok = False
                break
    finally:
        viewer.close()

    if not frames:
        raise RuntimeError(f"captured no frames for {label}")
    stacked = np.asarray(frames)
    if int(stacked.max()) == int(stacked.min()):
        raise RuntimeError(f"captured blank frames for {label}")

    body_q = state_0.body_q.numpy()
    body_qd = state_0.body_qd.numpy()
    linear_residual, angular_residual = _joint_split_residual(model, state_0)
    metrics = {
        "robot": robot,
        "mode": mode,
        "joint_stiffness": joint_stiffness,
        "steps_completed": len(frames),
        "dt": dt,
        "iterations": iterations,
        "ok": ok,
        "first_contact_step": first_contact_step,
        "max_contacts": max_contacts,
        "min_body_z": float(np.min(body_q[:, 2])),
        "max_linear_speed": float(np.max(np.linalg.norm(body_qd[:, 0:3], axis=1))),
        "joint_linear_l2": linear_residual,
        "joint_angular_l2": angular_residual,
    }
    return frames, metrics


def _combine_side_by_side(left: list[np.ndarray], right: list[np.ndarray], label: str) -> list[np.ndarray]:
    count = min(len(left), len(right))
    frames = []
    for i in range(count):
        combined = np.concatenate((_as_rgb(left[i]), _as_rgb(right[i])), axis=1)
        frames.append(_label_frame(combined, label))
    return frames


def main() -> None:
    _install_cpu_pinned_fallback()

    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", nargs="+", default=["h1", "g1"], choices=["h1", "g1"])
    parser.add_argument("--modes", nargs="+", default=["local", "block_sparse_joints"], choices=["local", "block_sparse_joints"])
    parser.add_argument("--joint-stiffness", nargs="+", default=["default", "fixed_high"], choices=["default", "fixed_high"])
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("videos"))
    args = parser.parse_args()

    metrics: list[dict] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for robot in args.robots:
        for joint_stiffness in args.joint_stiffness:
            rendered: dict[str, list[np.ndarray]] = {}
            for mode in args.modes:
                stem = f"{robot}_{joint_stiffness}_{mode}"
                label = f"{robot.upper()} {joint_stiffness} {mode}"
                frames, run_metrics = render_case(
                    robot=robot,
                    mode=mode,
                    joint_stiffness=joint_stiffness,
                    steps=args.steps,
                    iterations=args.iterations,
                    width=args.width,
                    height=args.height,
                    label=label,
                )
                video_path = args.output_dir / f"{stem}.mp4"
                _write_video(video_path, frames, args.fps)
                poster_path = args.output_dir / f"{stem}.jpg"
                imageio.imwrite(poster_path, _as_rgb(frames[len(frames) // 2]))
                run_metrics["video"] = video_path.name
                run_metrics["poster"] = poster_path.name
                metrics.append(run_metrics)
                rendered[mode] = frames
                print(json.dumps(run_metrics, sort_keys=True))

            if "local" in rendered and "block_sparse_joints" in rendered:
                side_label = f"{robot.upper()} {joint_stiffness}: local left, sparse right"
                side_frames = _combine_side_by_side(
                    rendered["local"],
                    rendered["block_sparse_joints"],
                    side_label,
                )
                side_stem = f"{robot}_{joint_stiffness}_local_vs_sparse"
                _write_video(args.output_dir / f"{side_stem}.mp4", side_frames, args.fps)
                imageio.imwrite(args.output_dir / f"{side_stem}.jpg", _as_rgb(side_frames[len(side_frames) // 2]))

    metrics_path = args.output_dir / "humanoid_contact_video_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
