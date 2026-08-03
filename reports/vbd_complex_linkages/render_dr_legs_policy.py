#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render trained DR Legs policy rollouts from the benchmark harness."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch  # noqa: TID253
import warp as wp

import newton
from reports.vbd_complex_linkages.bench_dr_legs_policy import (
    CASE_CONFIGS,
    _body_index,
    _load_config,
    load_policy,
    run_case,
)
from reports.vbd_complex_linkages.render_complex_linkages import _make_viewer, _write_video
from reports.vbd_complex_linkages.render_h2_loop import _capture_frame_cpu


class PolicyCapture:
    def __init__(self, case: str, width: int, height: int):
        self.case = case
        self.width = width
        self.height = height
        self.viewer = None
        self.pelvis = None
        self.frames: list[np.ndarray] = []

    def __call__(
        self,
        model: newton.Model,
        state: newton.State,
        sim_time: float,
        metrics: dict[str, float],
    ) -> None:
        if self.viewer is None:
            self.viewer = _make_viewer(model, state, "dr-legs", self.width, self.height)
            self.pelvis = _body_index(model, "pelvis")
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
                self.viewer.camera.fov = 36.0

        center = state.body_q.numpy()[self.pelvis, :3] + np.array((0.0, 0.0, 0.03))
        eye = center + np.array((0.62, -0.95, 0.38))
        direction = center - eye
        direction /= np.linalg.norm(direction)
        yaw = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
        pitch = math.degrees(math.asin(float(direction[2])))
        self.viewer.set_camera(wp.vec3(*eye), pitch, yaw)

        label = (
            f"{CASE_CONFIGS[self.case].label}  t={sim_time:.2f}s  "
            f"closure {metrics['closure_um']:.0f} um  tilt {metrics['tilt_deg']:.1f} deg"
        )
        self.frames.append(_capture_frame_cpu(self.viewer, state, sim_time, label))

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASE_CONFIGS, default="sparse_i8")
    parser.add_argument("--control-steps", type=int, default=400)
    parser.add_argument("--stand-steps", type=int, default=50)
    parser.add_argument("--forward-speed", type=float, default=0.2)
    parser.add_argument(
        "--vbd-armature-mode",
        choices=("isotropic_child_body", "rank_one_child_body", "coupled_joint", "unsupported"),
        default=None,
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "reports" / "vbd-complex-linkages" / "videos",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    asset_path = newton.utils.download_asset("disneyresearch")
    config = _load_config(asset_path)
    policy = load_policy(asset_path / "dr_legs" / "rl_policies" / config.policy_file)
    capture = PolicyCapture(args.case, args.width, args.height)
    try:
        result = run_case(
            args.case,
            policy,
            config,
            control_steps=args.control_steps,
            stand_steps=args.stand_steps,
            forward_speed=args.forward_speed,
            armature_mode_override=args.vbd_armature_mode,
            on_control_step=capture,
        )
    finally:
        capture.close()

    if not capture.frames:
        raise RuntimeError(f"No DR Legs policy frames captured for {args.case}")
    minimum_frames = 100
    if len(capture.frames) < minimum_frames:
        capture.frames.extend([capture.frames[-1]] * (minimum_frames - len(capture.frames)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"dr_legs_policy_{args.case}"
    if args.vbd_armature_mode is not None:
        stem += f"_{args.vbd_armature_mode}"
    video_path = args.output_dir / f"{stem}.mp4"
    poster_path = args.output_dir / f"{stem}.jpg"
    _write_video(video_path, capture.frames, 50)
    imageio.imwrite(poster_path, capture.frames[min(len(capture.frames) // 2, result["control_steps_completed"] - 1)])
    metadata = {
        "case": args.case,
        "video": f"videos/{video_path.name}",
        "poster": f"videos/{poster_path.name}",
        "rendered_frames": len(capture.frames),
        "vbd_armature_mode": args.vbd_armature_mode,
        "rollout": result,
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
