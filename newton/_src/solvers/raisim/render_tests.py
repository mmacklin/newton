#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render MP4 videos for each SolverRaisim test scenario.

Uses ViewerGL in headless mode to capture frames and encodes them to
MP4 with imageio-ffmpeg.

Usage::

    python3 -m newton._src.solvers.raisim.render_tests
    python3 -m newton._src.solvers.raisim.render_tests --output-dir ~/reports/raisim
    python3 -m newton._src.solvers.raisim.render_tests --config sphere_rest g1_height
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import warp as wp

wp.config.enable_backward = False

import imageio_ffmpeg as ffmpeg  # noqa: E402

import newton  # noqa: E402
import newton.utils  # noqa: E402
import newton.viewer  # noqa: E402

# -----------------------------------------------------------------------
# Camera helper
# -----------------------------------------------------------------------


def _look_at(viewer, cam_pos, target=(0.0, 0.0, 0.5)):
    """Point the camera at *target* from *cam_pos* (Z-up)."""
    dx = target[0] - cam_pos[0]
    dy = target[1] - cam_pos[1]
    dz = target[2] - cam_pos[2]
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.sqrt(dx * dx + dy * dy)))
    viewer.set_camera(pos=wp.vec3(*cam_pos), pitch=pitch, yaw=yaw)


# -----------------------------------------------------------------------
# Generic renderer
# -----------------------------------------------------------------------


def render_scenario(
    viewer,
    model,
    solver,
    output_path: str,
    num_frames: int = 300,
    dt: float = 1.0 / 360.0,
    fps: int = 60,
    cam_pos=(2.0, -1.5, 1.2),
    cam_target=(0.0, 0.0, 0.5),
    substeps: int = 1,
):
    """Simulate and render to MP4 using a shared ViewerGL."""
    viewer.set_model(model)
    _look_at(viewer, cam_pos, cam_target)

    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    w = viewer.camera.width
    h = viewer.camera.height
    even_w = w if w % 2 == 0 else w + 1
    even_h = h if h % 2 == 0 else h + 1
    target_image = wp.empty(shape=(h, w, 3), dtype=wp.uint8, device=wp.get_device())

    writer = ffmpeg.write_frames(
        output_path, size=(even_w, even_h), fps=fps, codec="libx264", macro_block_size=1, quality=5
    )
    writer.send(None)

    sim_time = 0.0
    frame_dt = dt * substeps
    t0 = time.perf_counter()

    for frame in range(num_frames):
        for _ in range(substeps):
            s0.clear_forces()
            model.collide(s0, contacts)
            solver.step(s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0

        viewer.begin_frame(sim_time)
        viewer.log_state(s0)
        viewer.end_frame()

        frame_data = viewer.get_frame(target_image=target_image)
        frame_np = frame_data.numpy()
        if even_w != w or even_h != h:
            padded = np.zeros((even_h, even_w, 3), dtype=np.uint8)
            padded[:h, :w] = frame_np
            frame_np = padded
        writer.send(frame_np)

        sim_time += frame_dt
        if (frame + 1) % 100 == 0:
            print(f"    Frame {frame + 1}/{num_frames}")

    writer.close()
    wp.synchronize_device()
    elapsed = time.perf_counter() - t0

    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    bq = s0.body_q.numpy()
    print(f"    {num_frames} frames in {elapsed:.1f}s -> {output_path}")
    print(f"    Final body[0] z={bq[0, 2]:.4f}")


# -----------------------------------------------------------------------
# Scene builders
# -----------------------------------------------------------------------


def _build_sphere_rest():
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(p=wp.vec3(0, 0, 1.0), q=wp.quat_identity()), mass=1.0)
    builder.add_shape_sphere(body=b, radius=0.3)
    return builder.finalize()


def _build_box_rest():
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(p=wp.vec3(0, 0, 1.0), q=wp.quat_identity()), mass=1.0)
    builder.add_shape_box(body=b, hx=0.2, hy=0.2, hz=0.2)
    return builder.finalize()


def _build_free_fall():
    builder = newton.ModelBuilder()
    b = builder.add_body(xform=wp.transform(p=wp.vec3(0, 0, 5.0), q=wp.quat_identity()), mass=1.0)
    builder.add_shape_sphere(body=b, radius=0.3)
    return builder.finalize()


def _build_sphere_stack():
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    r = 0.15
    for i in range(5):
        b = builder.add_body(
            xform=wp.transform(p=wp.vec3(0, 0, r + 0.01 + i * (2.0 * r + 0.05)), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_sphere(body=b, radius=r)
    return builder.finalize()


def _build_g1():
    asset_path = newton.utils.download_asset("unitree_g1")
    g1 = newton.ModelBuilder()
    g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
    g1.default_shape_cfg.ke = 1.0e3
    g1.default_shape_cfg.kd = 2.0e2
    g1.default_shape_cfg.kf = 1.0e3
    g1.default_shape_cfg.mu = 0.75
    g1.add_usd(
        str(asset_path / "usd" / "g1_isaac.usd"),
        xform=wp.transform(wp.vec3(0, 0, 0.8)),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    for i in range(6, g1.joint_dof_count):
        g1.joint_target_ke[i] = 500.0
        g1.joint_target_kd[i] = 10.0
    g1.approximate_meshes("bounding_box")
    builder = newton.ModelBuilder()
    builder.replicate(g1, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 2.0e2
    builder.add_ground_plane()
    return builder.finalize()


def _build_h1():
    asset_path = newton.utils.download_asset("unitree_h1")
    h1 = newton.ModelBuilder()
    h1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
    h1.default_shape_cfg.ke = 2.0e3
    h1.default_shape_cfg.kd = 1.0e2
    h1.default_shape_cfg.kf = 1.0e3
    h1.default_shape_cfg.mu = 0.75
    h1.add_usd(
        str(asset_path / "usd_structured" / "h1.usda"),
        ignore_paths=["/GroundPlane"],
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    for i in range(h1.joint_dof_count):
        h1.joint_target_ke[i] = 500.0
        h1.joint_target_kd[i] = 10.0
    h1.approximate_meshes("bounding_box")
    builder = newton.ModelBuilder()
    builder.replicate(h1, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    return builder.finalize()


def _build_energy_drop():
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(p=wp.vec3(0, 0, 2.0), q=wp.quat_identity()), mass=1.0)
    builder.add_shape_sphere(body=b, radius=0.3)
    return builder.finalize()


def _build_anymal_d():
    asset_path = newton.utils.download_asset("anybotics_anymal_d")
    ab = newton.ModelBuilder()
    ab.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
    ab.default_shape_cfg.ke = 2.0e3
    ab.default_shape_cfg.kd = 1.0e2
    ab.default_shape_cfg.kf = 1.0e3
    ab.default_shape_cfg.mu = 0.75
    ab.add_usd(
        str(asset_path / "usd" / "anymal_d.usda"),
        xform=wp.transform(wp.vec3(0, 0, 0.62)),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    for i in range(ab.joint_dof_count):
        ab.joint_target_ke[i] = 2000.0
        ab.joint_target_kd[i] = 40.0
    ab.approximate_meshes("bounding_box")
    builder = newton.ModelBuilder()
    builder.replicate(ab, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    return builder.finalize()


def _build_go2():
    asset_path = newton.utils.download_asset("unitree_go2")
    go2 = newton.ModelBuilder()
    go2.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
    go2.default_shape_cfg.ke = 2.0e3
    go2.default_shape_cfg.kd = 1.0e2
    go2.default_shape_cfg.kf = 1.0e3
    go2.default_shape_cfg.mu = 0.75
    go2.add_usd(
        str(asset_path / "usd" / "go2.usda"),
        xform=wp.transform(wp.vec3(0, 0, 0.35)),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    for i in range(go2.joint_dof_count):
        go2.joint_target_ke[i] = 500.0
        go2.joint_target_kd[i] = 10.0
    go2.approximate_meshes("bounding_box")
    builder = newton.ModelBuilder()
    builder.replicate(go2, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    return builder.finalize()


def _build_box_tower():
    half = 0.1
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    for i in range(20):
        z = half + 0.001 + i * (2.0 * half + 0.002)
        b = builder.add_body(xform=wp.transform(p=wp.vec3(0, 0, z), q=wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body=b, hx=half, hy=half, hz=half)
    return builder.finalize()


def _build_box_pyramid():
    half = 0.1
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.6
    builder.add_ground_plane()
    rows = 5
    for row in range(rows):
        n = rows - row
        for col in range(n):
            x = (col - (n - 1) / 2.0) * (2.0 * half + 0.005)
            z = half + 0.001 + row * (2.0 * half + 0.002)
            b = builder.add_body(xform=wp.transform(p=wp.vec3(x, 0, z), q=wp.quat_identity()), mass=1.0)
            builder.add_shape_box(body=b, hx=half, hy=half, hz=half)
    return builder.finalize()


# -----------------------------------------------------------------------
# Configuration table
# -----------------------------------------------------------------------

CONFIGS = {
    "sphere_rest": {
        "label": "A1: Sphere resting height",
        "build": _build_sphere_rest,
        "frames": 360,
        "cam_pos": (1.5, -1.0, 0.8),
        "cam_target": (0.0, 0.0, 0.3),
    },
    "box_rest": {
        "label": "A2: Box resting orientation",
        "build": _build_box_rest,
        "frames": 360,
        "cam_pos": (1.5, -1.0, 0.8),
        "cam_target": (0.0, 0.0, 0.3),
    },
    "free_fall": {
        "label": "A3: Free-fall",
        "build": _build_free_fall,
        "frames": 180,
        "cam_pos": (3.0, -2.0, 4.0),
        "cam_target": (0.0, 0.0, 3.0),
    },
    "sphere_stack": {
        "label": "C3: Sphere stack",
        "build": _build_sphere_stack,
        "frames": 360,
        "cam_pos": (1.5, -1.0, 0.5),
        "cam_target": (0.0, 0.0, 0.3),
    },
    "g1_height": {
        "label": "A5: G1 maintains height",
        "build": _build_g1,
        "frames": 360,
        "cam_pos": (2.0, -1.5, 1.2),
        "cam_target": (0.0, 0.0, 0.8),
    },
    "h1_height": {
        "label": "A6: H1 maintains height",
        "build": _build_h1,
        "frames": 360,
        "cam_pos": (2.0, -1.5, 1.2),
        "cam_target": (0.0, 0.0, 0.8),
    },
    "energy_drop": {
        "label": "C1: Free body energy",
        "build": _build_energy_drop,
        "frames": 360,
        "cam_pos": (2.0, -1.5, 1.5),
        "cam_target": (0.0, 0.0, 0.5),
    },
    "anymal_d": {
        "label": "D1: Anymal D quadruped",
        "build": _build_anymal_d,
        "frames": 360,
        "cam_pos": (1.5, -1.0, 0.8),
        "cam_target": (0.0, 0.0, 0.4),
    },
    "go2": {
        "label": "D2: Unitree Go2 quadruped",
        "build": _build_go2,
        "frames": 360,
        "cam_pos": (1.2, -0.8, 0.5),
        "cam_target": (0.0, 0.0, 0.25),
    },
    "box_tower": {
        "label": "D3: Box tower (20 boxes)",
        "build": _build_box_tower,
        "frames": 360,
        "cam_pos": (2.0, -1.5, 2.0),
        "cam_target": (0.0, 0.0, 1.5),
    },
    "box_pyramid": {
        "label": "D4: Box pyramid (15 boxes)",
        "build": _build_box_pyramid,
        "frames": 360,
        "cam_pos": (1.5, -1.0, 0.5),
        "cam_target": (0.0, 0.0, 0.3),
    },
}


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Render test scenario MP4 videos for SolverRaisim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=list(CONFIGS.keys()),
        choices=list(CONFIGS.keys()),
        help="Scenarios to render.",
    )
    parser.add_argument("--output-dir", default=os.path.expanduser("~/reports/raisim"), help="Output directory.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--fps", type=int, default=60, help="Video frame rate.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)

    for name in args.config:
        cfg = CONFIGS[name]
        print(f"\n{'=' * 60}")
        print(f"  {cfg['label']}")
        print(f"{'=' * 60}")

        model = cfg["build"]()
        print(f"  bodies={model.body_count}  joints={model.joint_count}")

        solver = newton.solvers.SolverRaisim(model)
        mp4_path = os.path.join(args.output_dir, f"{name}.mp4")
        render_scenario(
            viewer,
            model,
            solver,
            mp4_path,
            num_frames=cfg["frames"],
            fps=args.fps,
            cam_pos=cfg["cam_pos"],
            cam_target=cfg["cam_target"],
        )

    viewer.close()
    print(f"\nAll videos saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
