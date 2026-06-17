#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render a ViewerGL video for the cable cross-slide table example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import warp as wp

import newton.viewer
from newton.examples.cable.example_cable_cross_slide_table import Example

WIDTH = 960
HEIGHT = 540
FPS = 60

_WP_EMPTY = wp.empty


def _float_tag(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def _decimal_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


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
    except Exception:
        return _as_rgb(frame)

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


def render_video(args: argparse.Namespace) -> dict:
    sim_viewer = newton.viewer.ViewerNull(num_frames=args.frames)
    example_args = SimpleNamespace(
        device=args.device,
        rigid_articulation_solve=args.rigid_articulation_solve,
        cable_bend_stiffness=args.cable_bend_stiffness,
        cable_bend_damping=args.cable_bend_damping,
        rigid_articulation_relaxation=args.rigid_articulation_relaxation,
        contact_ke=args.contact_ke,
        contact_kd=args.contact_kd,
        contact_mu=args.contact_mu,
        rigid_gap_scale=args.rigid_gap_scale,
        cable_gap_scale=args.cable_gap_scale,
        rigid_contact_k_start=args.rigid_contact_k_start,
        rigid_avbd_beta=args.rigid_avbd_beta,
        sim_substeps=args.sim_substeps,
        sim_iterations=args.sim_iterations,
    )
    example = Example(sim_viewer, example_args)
    saved_states: list[tuple[float, np.ndarray]] = []
    post_step_bounds_passed = True
    first_post_step_error = ""
    first_post_step_error_frame = None
    min_cable_z = float("inf")
    max_cable_z = float("-inf")
    max_cable_abs_xy = 0.0
    for frame_index in range(args.frames):
        example.step()
        body_q = example.state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise RuntimeError(f"non-finite body transform at frame {frame_index}")
        cable_pos = body_q[[int(body) for body in example.cable_bodies], 0:3]
        min_cable_z = min(min_cable_z, float(np.min(cable_pos[:, 2])))
        max_cable_z = max(max_cable_z, float(np.max(cable_pos[:, 2])))
        max_cable_abs_xy = max(max_cable_abs_xy, float(np.max(np.abs(cable_pos[:, 0:2]))))
        try:
            example.test_post_step()
        except Exception as exc:
            post_step_bounds_passed = False
            if first_post_step_error_frame is None:
                first_post_step_error_frame = frame_index
                first_post_step_error = str(exc)
            if not args.allow_post_step_failure:
                raise
        saved_states.append((example.sim_time, example.state_0.body_q.numpy().copy()))

    final_test_passed = None
    final_test_error = ""
    if args.require_final_test:
        final_test_passed = True
        try:
            example.test_final()
        except Exception as exc:
            final_test_passed = False
            final_test_error = str(exc)
            raise

    _install_cpu_pinned_fallback()
    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    frames: list[np.ndarray] = []
    try:
        try:
            viewer.set_model(example.model)
        finally:
            _restore_wp_empty()
        viewer.picking_enabled = False
        viewer.wind = None
        viewer.set_camera(
            pos=wp.vec3(0.0, 0.0, 0.8),
            pitch=-90.0,
            yaw=90.0,
        )

        for frame_index, (sim_time, body_q) in enumerate(saved_states):
            example.state_0.body_q.assign(body_q)
            viewer.begin_frame(sim_time)
            viewer.log_state(example.state_0)
            viewer.end_frame()
            _install_cpu_pinned_fallback()
            try:
                frame = viewer.get_frame().numpy()
            finally:
                _restore_wp_empty()
            frames.append(
                _label_frame(
                    frame,
                    "XY cable table - "
                    f"{args.rigid_articulation_solve} - bend {args.cable_bend_stiffness:.0e} - "
                    f"ke {args.contact_ke:.0e} - "
                    f"k0 {args.rigid_contact_k_start:.0e} - "
                    f"relax {args.rigid_articulation_relaxation:g} - "
                    f"frame {frame_index + 1}/{args.frames}",
                )
            )
    finally:
        viewer.close()
        _restore_wp_empty()

    if not frames:
        raise RuntimeError("captured no frames")
    stacked = np.asarray(frames)
    if int(stacked.max()) == int(stacked.min()):
        raise RuntimeError("captured blank frames")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"cable_cross_slide_table_{args.rigid_articulation_solve}"
        f"_bend_{_float_tag(args.cable_bend_stiffness)}"
        f"_ke_{_float_tag(args.contact_ke)}"
        f"_kd_{_float_tag(args.contact_kd)}"
        f"_k0_{_float_tag(args.rigid_contact_k_start)}"
        f"_relax_{_decimal_tag(args.rigid_articulation_relaxation)}"
    )
    video_path = args.output_dir / f"{stem}.mp4"
    poster_path = args.output_dir / f"{stem}.jpg"
    _write_video(video_path, frames, args.fps)
    imageio.imwrite(poster_path, _as_rgb(frames[len(frames) // 2]))

    metrics = {
        "video": video_path.name,
        "poster": poster_path.name,
        "frames": len(frames),
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "mode": args.rigid_articulation_solve,
        "device": args.device,
        "cable_bend_stiffness": args.cable_bend_stiffness,
        "cable_bend_damping": args.cable_bend_damping,
        "rigid_articulation_relaxation": args.rigid_articulation_relaxation,
        "contact_ke": args.contact_ke,
        "contact_kd": args.contact_kd,
        "contact_mu": args.contact_mu,
        "rigid_gap_scale": args.rigid_gap_scale,
        "cable_gap_scale": args.cable_gap_scale,
        "rigid_contact_k_start": args.rigid_contact_k_start,
        "rigid_avbd_beta": args.rigid_avbd_beta,
        "sim_substeps": args.sim_substeps,
        "sim_iterations": args.sim_iterations,
        "sim_time": example.sim_time,
        "table_tracking_max_error": example.table_tracking_max_error,
        "table_tracking_rms_error": float(
            np.sqrt(example.table_tracking_error_sq_sum / max(1, example.table_tracking_sample_count))
        ),
        "final_test_passed": final_test_passed,
        "final_test_error": final_test_error,
        "final_test_enforced": args.require_final_test,
        "post_step_bounds_passed": post_step_bounds_passed,
        "first_post_step_error_frame": first_post_step_error_frame,
        "first_post_step_error": first_post_step_error,
        "min_cable_z": min_cable_z,
        "max_cable_z": max_cable_z,
        "max_cable_abs_xy": max_cable_abs_xy,
    }
    metrics_path = args.output_dir / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rigid-articulation-solve", default="block_sparse_joints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("videos"))
    parser.add_argument("--require-final-test", action="store_true")
    parser.add_argument("--allow-post-step-failure", action="store_true")
    parser.add_argument("--cable-bend-stiffness", type=float, default=1.0e-5)
    parser.add_argument("--cable-bend-damping", type=float, default=1.0e-2)
    parser.add_argument("--rigid-articulation-relaxation", type=float, default=0.65)
    parser.add_argument("--contact-ke", type=float, default=1.0e5)
    parser.add_argument("--contact-kd", type=float, default=0.0)
    parser.add_argument("--contact-mu", type=float, default=1.0)
    parser.add_argument("--rigid-gap-scale", type=float, default=5.0)
    parser.add_argument("--cable-gap-scale", type=float, default=2.0)
    parser.add_argument("--rigid-contact-k-start", type=float, default=1.0e2)
    parser.add_argument("--rigid-avbd-beta", type=float, default=1.0e5)
    parser.add_argument("--sim-substeps", type=int, default=10)
    parser.add_argument("--sim-iterations", type=int, default=5)
    args = parser.parse_args()
    render_video(args)


if __name__ == "__main__":
    main()
