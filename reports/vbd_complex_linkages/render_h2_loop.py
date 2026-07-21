#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the public Unitree H2 loop model under VBD articulation solves."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp

import newton
import newton.viewer
from reports.vbd_complex_linkages.bench_h2_loop import (
    CASE_CONFIGS,
    DEFAULT_ASSET_DIR,
    DRIVE_PHASES,
    _find_label,
    _set_targets,
    build_model,
    closure_errors,
)
from reports.vbd_complex_linkages.render_complex_linkages import _label_frame, _make_viewer, _write_video


def _capture_frame_cpu(viewer: newton.viewer.ViewerGL, state: newton.State, sim_time: float, label: str) -> np.ndarray:
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    viewer.end_frame()

    gl = viewer.renderer.__class__.gl
    width = viewer.renderer._screen_width
    height = viewer.renderer._screen_height
    frame = np.empty((height, width, 3), dtype=np.uint8)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, viewer.renderer._frame_fbo)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    gl.glReadPixels(
        0,
        0,
        width,
        height,
        gl.GL_RGB,
        gl.GL_UNSIGNED_BYTE,
        ctypes.c_void_p(frame.ctypes.data),
    )
    gl.glFinish()
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    return _label_frame(np.ascontiguousarray(frame[::-1]), label)


def render_case(args: argparse.Namespace, case: str) -> dict:
    label, mode, iterations = CASE_CONFIGS[case]
    model, _, _ = build_model(args.asset_dir, args.device, with_visuals=True)
    state_0 = model.state()
    state_1 = model.state()
    state_1.assign(state_0)
    control = model.control()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=iterations,
        rigid_articulation_solve=mode,
        rigid_articulation_relaxation=args.relaxation,
        rigid_avbd_alpha=0.0,
        rigid_avbd_beta=0.0,
        rigid_joint_linear_ke=args.joint_ke,
        rigid_joint_angular_ke=args.joint_ke,
        rigid_joint_linear_kd=args.joint_kd,
        rigid_joint_angular_kd=args.joint_kd,
    )
    target_q = control.joint_target_q.numpy().copy()
    target_starts = model.joint_target_q_start.numpy()
    joint_labels = list(model.joint_label)
    target_indices = {name: int(target_starts[_find_label(joint_labels, name)]) for name in DRIVE_PHASES}

    viewer = _make_viewer(model, state_0, "h2-loop", args.width, args.height)
    viewer.set_camera(wp.vec3(2.25, -3.1, 0.15), -5.0, 126.0)
    if hasattr(viewer, "camera") and hasattr(viewer.camera, "fov"):
        viewer.camera.fov = 34.0

    frame_dt = 1.0 / 60.0
    dt = frame_dt / args.substeps
    sim_time = 0.0
    frames = []
    aggregate_um = []
    finite = True
    try:
        for _ in range(args.frames):
            for _ in range(args.substeps):
                _set_targets(model, control, target_q, target_indices, sim_time, args.period)
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, dt)
                state_0, state_1 = state_1, state_0
                sim_time += dt
            finite = bool(np.isfinite(state_0.body_q.numpy()).all())
            if not finite:
                break
            error = closure_errors(model, state_0)["aggregate_um"]
            aggregate_um.append(error)
            frames.append(_capture_frame_cpu(viewer, state_0, sim_time, f"{label}  closure {error:.0f} um"))
    finally:
        viewer.close()

    if not frames:
        raise RuntimeError(f"No finite H2 frames captured for {case}")
    stem = f"h2_loop_{case}"
    video_path = args.output_dir / f"{stem}.mp4"
    poster_path = args.output_dir / f"{stem}.jpg"
    _write_video(video_path, frames, 60)
    imageio.imwrite(poster_path, frames[len(frames) // 2])
    return {
        "case": case,
        "label": label,
        "device": args.device,
        "frames": len(frames),
        "finite": finite,
        "rms_closure_um": float(np.sqrt(np.mean(np.square(aggregate_um)))),
        "video": f"videos/{video_path.name}",
        "poster": f"videos/{poster_path.name}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--case", choices=CASE_CONFIGS, default="sparse_i8")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--period", type=float, default=2.8)
    parser.add_argument("--relaxation", type=float, default=0.65)
    parser.add_argument("--joint-ke", type=float, default=2.0e5)
    parser.add_argument("--joint-kd", type=float, default=5.0e2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "reports" / "vbd-complex-linkages" / "videos",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = render_case(args, args.case)
    output = args.output_dir / f"h2_loop_{args.case}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
