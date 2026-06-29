#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render ViewerGL videos for the complex-linkage comparison."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp

import newton
import newton.viewer
from reports.vbd_complex_linkages.bench_complex_linkages import (
    DR_LEGS_MODES,
    ROBOT_FOOT_MODES,
    _joint_anchor_residuals,
    _joint_index,
    _make_dr_legs_solver,
    _make_solver,
    _update_robot_foot_targets,
    build_dr_legs_model,
    build_robot_foot_model,
)

_WP_EMPTY = wp.empty


def _install_cpu_pinned_fallback() -> None:
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


def _restore_wp_empty() -> None:
    wp.empty = _WP_EMPTY


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
    except ImportError:
        return _as_rgb(frame)

    image = Image.fromarray(_as_rgb(frame))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    padding = 8
    draw.rectangle((0, 0, box[2] - box[0] + 2 * padding, box[3] - box[1] + 2 * padding), fill=(20, 24, 28))
    draw.text((padding, padding), label, fill=(245, 248, 250), font=font)
    return np.asarray(image)


def _setup_camera(viewer: newton.viewer.ViewerGL, state: newton.State, scenario: str) -> None:
    positions = state.body_q.numpy()[:, :3]
    center = np.mean(positions, axis=0)
    extent = max(float(np.max(np.linalg.norm(positions - center, axis=1))), 0.25)
    if scenario == "robot-foot":
        eye = center + np.array((1.8 * extent, -2.8 * extent, 1.3 * extent))
    else:
        eye = center + np.array((2.5 * extent, -3.5 * extent, 1.8 * extent))
    direction = center - eye
    direction /= np.linalg.norm(direction)
    yaw = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    pitch = math.degrees(math.asin(float(direction[2])))
    viewer.set_camera(wp.vec3(*eye), pitch, yaw)
    if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
        viewer.camera.fov = 38.0


def _make_viewer(model: newton.Model, state: newton.State, scenario: str, width: int, height: int):
    _install_cpu_pinned_fallback()
    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
    try:
        viewer.set_model(model)
    finally:
        _restore_wp_empty()
    viewer.picking_enabled = False
    viewer.wind = None
    _setup_camera(viewer, state, scenario)
    return viewer


def _capture_frame(viewer: newton.viewer.ViewerGL, state: newton.State, sim_time: float, label: str) -> np.ndarray:
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    viewer.end_frame()
    _install_cpu_pinned_fallback()
    try:
        frame = viewer.get_frame().numpy()
    finally:
        _restore_wp_empty()
    return _label_frame(frame, label)


def render_robot_foot(args: argparse.Namespace) -> dict:
    base_spec = ROBOT_FOOT_MODES[args.mode]
    spec = replace(base_spec, label=base_spec.label.replace(" CPU", ""), device=args.device)
    model, source, motor_labels = build_robot_foot_model(spec.device, spec.robot_foot_geometry)
    state_0 = model.state()
    state_1 = model.state()
    state_1.assign(state_0)
    control = model.control()
    solver = _make_solver(model, spec)
    target_q = model.joint_target_q.numpy().copy()
    target_qd = model.joint_target_qd.numpy().copy()
    target_starts = model.joint_target_q_start.numpy()
    qd_starts = model.joint_qd_start.numpy()
    motor_joints = {name: _joint_index(model, label) for name, label in motor_labels.items()}
    motor_targets = {name: int(target_starts[index]) for name, index in motor_joints.items()}
    motor_velocities = {name: int(qd_starts[index]) for name, index in motor_joints.items()}
    closure_labels = {
        label
        for label in model.joint_label
        if "pushrod_to_foot_universal" in label or "pushrod_to_gimbal_universal" in label
    }
    viewer = _make_viewer(model, state_0, "robot-foot", args.width, args.height)
    frames = []
    closure_um = []
    sim_time = 0.0
    try:
        for _ in range(args.frames):
            for _ in range(5):
                _update_robot_foot_targets(
                    source,
                    model,
                    control,
                    motor_targets,
                    motor_velocities,
                    target_q,
                    target_qd,
                    sim_time,
                )
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, 0.004)
                state_0, state_1 = state_1, state_0
                sim_time += 0.004
            residual = _joint_anchor_residuals(model, state_0, closure_labels)["linear_norm_m"] * 1.0e6
            closure_um.append(residual)
            frames.append(_capture_frame(viewer, state_0, sim_time, f"{spec.label}  closure {residual:.0f} um"))
    finally:
        viewer.close()
        _restore_wp_empty()
    return {
        "scenario": "robot-foot",
        "mode": spec.label,
        "device": spec.device,
        "frames": len(frames),
        "fps": 50,
        "finite": bool(np.isfinite(state_0.body_q.numpy()).all()),
        "rms_closure_um": float(np.sqrt(np.mean(np.square(closure_um)))),
        "captured_frames": frames,
    }


def render_dr_legs(args: argparse.Namespace) -> dict:
    base_spec = DR_LEGS_MODES[args.mode]
    spec = replace(base_spec, label=base_spec.label.replace(" CPU", ""), device=args.device)
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
    closure_labels = {
        "/DR_Legs/Joints/j6_l_o",
        "/DR_Legs/Joints/j6_r_o",
        "/DR_Legs/Joints/j9_l_i",
        "/DR_Legs/Joints/j9_l_o",
        "/DR_Legs/Joints/j9_r_i",
        "/DR_Legs/Joints/j9_r_o",
    }
    viewer = _make_viewer(model, state_0, "dr-legs", args.width, args.height)
    frames = []
    closure_um = []
    sim_time = 0.0
    failure_frame = None
    try:
        for frame_index in range(args.frames):
            for _ in range(2):
                state_0.clear_forces()
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, 0.01)
                state_0, state_1 = state_1, state_0
                sim_time += 0.01
            if not np.isfinite(state_0.body_q.numpy()).all():
                failure_frame = frame_index
                break
            residual = _joint_anchor_residuals(model, state_0, closure_labels)["linear_norm_m"] * 1.0e6
            closure_um.append(residual)
            contact_count = int(contacts.rigid_contact_count.numpy()[0])
            label = f"{spec.label}  closure {residual:.0f} um  contacts {contact_count}"
            frames.append(_capture_frame(viewer, state_0, sim_time, label))
    finally:
        viewer.close()
        _restore_wp_empty()
    return {
        "scenario": "dr-legs",
        "mode": spec.label,
        "device": spec.device,
        "frames": len(frames),
        "fps": 50,
        "finite": failure_frame is None,
        "failure_frame": failure_frame,
        "rms_closure_um": float(np.sqrt(np.mean(np.square(closure_um)))) if closure_um else None,
        "captured_frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("robot-foot", "dr-legs"), required=True)
    mode_choices = tuple(dict.fromkeys((*ROBOT_FOOT_MODES, *DR_LEGS_MODES)))
    parser.add_argument("--mode", choices=mode_choices, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "reports" / "vbd-complex-linkages" / "videos")
    args = parser.parse_args()
    result = render_robot_foot(args) if args.scenario == "robot-foot" else render_dr_legs(args)
    frames = result.pop("captured_frames")
    if not frames:
        raise RuntimeError("No finite frames were captured")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scenario}_{args.mode}".replace("-", "_")
    video_path = args.output_dir / f"{stem}.mp4"
    poster_path = args.output_dir / f"{stem}.jpg"
    _write_video(video_path, frames, int(result["fps"]))
    imageio.imwrite(poster_path, frames[len(frames) // 2])
    result["video"] = f"videos/{video_path.name}"
    result["poster"] = f"videos/{poster_path.name}"
    metrics_path = args.output_dir / f"{stem}.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
