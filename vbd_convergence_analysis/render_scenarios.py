#!/usr/bin/env python3
"""Render visualization images of the t-shirt drop simulation at various timesteps.

Generates 3D views of particle positions at key frames for use in the
convergence analysis report. Produces per-frame images, a progression grid,
and a baseline-vs-Chebyshev comparison panel.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Ensure the worktree root is on sys.path so `import newton` resolves.
WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import warp as wp
from pxr import Usd

import newton
import newton.usd
from newton import ModelBuilder
from newton.solvers import SolverVBD

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

ASSET_PATH = os.path.join(WORKTREE, "newton", "examples", "assets", "unisex_shirt.usd")

# Key frames to capture (0-indexed).
DEFAULT_KEY_FRAMES = [0, 10, 20, 30]


def _load_shirt_mesh():
    """Return (vertices, indices) for the t-shirt mesh."""
    usd_stage = Usd.Stage.Open(ASSET_PATH)
    usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
    shirt_mesh = newton.usd.get_mesh(usd_prim)
    vertices = [wp.vec3(v) for v in shirt_mesh.vertices]
    indices = shirt_mesh.indices
    return vertices, indices


def build_scenario(
    seed: int,
    iterations: int = 10,
    chebyshev_rho: float = 0.0,
):
    """Build a randomized t-shirt drop scenario (cm-scale).

    Returns a dict with model, solver, states, control, contacts, collision
    pipeline, dt, and substeps.
    """
    rng = np.random.default_rng(seed)

    drop_height = rng.uniform(40.0, 120.0)
    rot_angles = rng.uniform(-np.pi, np.pi, size=3)
    lateral_offset = rng.uniform(-30.0, 30.0, size=2)

    # cm-scale physics
    gravity_cm = -981.0
    tri_ke = 1e4
    tri_ka = 1e4
    tri_kd = 1.5e-6
    bending_ke = 5.0
    bending_kd = 1e-2
    density = 0.02

    scene = ModelBuilder(gravity=gravity_cm)
    vertices, indices = _load_shirt_mesh()

    qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(rot_angles[0]))
    qy = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(rot_angles[1]))
    qz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rot_angles[2]))
    rot = wp.mul(qz, wp.mul(qy, qx))
    pos = wp.vec3(float(lateral_offset[0]), float(lateral_offset[1]), float(drop_height))

    scene.add_cloth_mesh(
        vertices=vertices,
        indices=indices,
        rot=rot,
        pos=pos,
        vel=wp.vec3(0.0, 0.0, 0.0),
        density=density,
        scale=1.0,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=tri_kd,
        edge_ke=bending_ke,
        edge_kd=bending_kd,
        particle_radius=0.5,
    )

    scene.color()
    scene.add_ground_plane()
    model = scene.finalize(requires_grad=False)

    model.soft_contact_ke = 1e4
    model.soft_contact_kd = 1e-2
    model.soft_contact_mu = 0.25

    solver = SolverVBD(
        model,
        iterations=iterations,
        particle_tri_material_model="neohookean",
        particle_enable_self_contact=False,
    )
    if chebyshev_rho == "auto" or (isinstance(chebyshev_rho, (int, float)) and chebyshev_rho > 0):
        solver.chebyshev_rho = chebyshev_rho

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.5)
    contacts = collision_pipeline.contacts()

    sim_substeps = 10
    frame_dt = 1.0 / 60.0
    sim_dt = frame_dt / sim_substeps

    return {
        "model": model,
        "solver": solver,
        "state_0": state_0,
        "state_1": state_1,
        "control": control,
        "contacts": contacts,
        "collision_pipeline": collision_pipeline,
        "dt": sim_dt,
        "sim_substeps": sim_substeps,
        "indices": indices,
    }


def simulate_and_capture(scenario: dict, num_frames: int, key_frames: list[int]):
    """Run the simulation for *num_frames* and capture particle positions at *key_frames*.

    Returns a dict mapping frame index to an (N, 3) numpy array of particle
    positions.
    """
    model = scenario["model"]
    solver = scenario["solver"]
    state_0 = scenario["state_0"]
    state_1 = scenario["state_1"]
    control = scenario["control"]
    contacts = scenario["contacts"]
    collision_pipeline = scenario["collision_pipeline"]
    dt = scenario["dt"]
    substeps = scenario["sim_substeps"]

    captures: dict[int, np.ndarray] = {}

    # Capture frame 0 (initial state) before any simulation.
    if 0 in key_frames:
        captures[0] = state_0.particle_q.numpy().copy()

    for frame in range(1, num_frames + 1):
        for _sub in range(substeps):
            state_0.clear_forces()
            state_1.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

        if frame in key_frames:
            captures[frame] = state_0.particle_q.numpy().copy()

    return captures


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_frame(
    positions: np.ndarray,
    tri_faces: np.ndarray,
    title: str = "",
    elev: float = 30.0,
    azim: float = -60.0,
    figsize: tuple[float, float] = (6, 5),
    face_color: str = "#a0c4e8",
    edge_color: str = "#2060a0",
    alpha: float = 0.7,
) -> plt.Figure:
    """Render a single frame as a 3D mesh surface plot.

    Args:
        positions: (N, 3) array of vertex positions.
        tri_faces: (T, 3) array of triangle vertex indices.
        title: Figure title.
        elev: Elevation angle for 3D view.
        azim: Azimuth angle for 3D view.
        figsize: Figure size in inches.
        face_color: Color of triangle faces.
        edge_color: Color of triangle edges.
        alpha: Face transparency.

    Returns:
        The matplotlib Figure.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Build polygon collection from triangles.
    verts_per_tri = positions[tri_faces]  # (T, 3, 3)
    poly = Poly3DCollection(
        verts_per_tri,
        alpha=alpha,
        facecolor=face_color,
        edgecolor=edge_color,
        linewidths=0.15,
    )
    ax.add_collection3d(poly)

    # Set axis limits based on data.
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins).max() / 2.0 * 1.2  # add 20% margin
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(max(center[2] - span, -5.0), center[2] + span)

    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_zlabel("Z [cm]")
    ax.view_init(elev=elev, azim=azim)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig


def render_progression_grid(
    captures: dict[int, np.ndarray],
    tri_faces: np.ndarray,
    suptitle: str = "T-Shirt Drop Simulation Progression",
    elev: float = 30.0,
    azim: float = -60.0,
) -> plt.Figure:
    """Render a grid of frames showing the simulation progression.

    Args:
        captures: Mapping from frame index to (N, 3) positions.
        tri_faces: (T, 3) triangle indices.
        suptitle: Super-title for the figure.
        elev: Elevation angle.
        azim: Azimuth angle.

    Returns:
        The matplotlib Figure.
    """
    sorted_frames = sorted(captures.keys())
    n = len(sorted_frames)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)

    # Compute global axis limits across all frames.
    all_pos = np.concatenate(list(captures.values()), axis=0)
    g_min = all_pos.min(axis=0)
    g_max = all_pos.max(axis=0)
    g_center = (g_min + g_max) / 2.0
    g_span = (g_max - g_min).max() / 2.0 * 1.2

    for idx, frame in enumerate(sorted_frames):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        positions = captures[frame]
        verts_per_tri = positions[tri_faces]
        poly = Poly3DCollection(
            verts_per_tri,
            alpha=0.7,
            facecolor="#a0c4e8",
            edgecolor="#2060a0",
            linewidths=0.15,
        )
        ax.add_collection3d(poly)

        ax.set_xlim(g_center[0] - g_span, g_center[0] + g_span)
        ax.set_ylim(g_center[1] - g_span, g_center[1] + g_span)
        ax.set_zlim(max(g_center[2] - g_span, -5.0), g_center[2] + g_span)

        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Frame {frame}", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def render_comparison(
    captures_a: dict[int, np.ndarray],
    captures_b: dict[int, np.ndarray],
    tri_faces: np.ndarray,
    label_a: str = "Baseline",
    label_b: str = "Chebyshev",
    color_a: str = "#a0c4e8",
    color_b: str = "#e8a0a0",
    elev: float = 30.0,
    azim: float = -60.0,
) -> plt.Figure:
    """Render a side-by-side comparison of two solver configurations.

    Each row corresponds to one key frame; left column is config A, right is
    config B.

    Args:
        captures_a: Frame captures for configuration A.
        captures_b: Frame captures for configuration B.
        tri_faces: (T, 3) triangle indices.
        label_a: Label for configuration A.
        label_b: Label for configuration B.
        color_a: Face color for configuration A.
        color_b: Face color for configuration B.
        elev: Elevation angle.
        azim: Azimuth angle.

    Returns:
        The matplotlib Figure.
    """
    # Use frames common to both captures.
    common_frames = sorted(set(captures_a.keys()) & set(captures_b.keys()))
    n_rows = len(common_frames)
    if n_rows == 0:
        fig = plt.figure()
        fig.text(0.5, 0.5, "No common frames to compare", ha="center")
        return fig

    fig = plt.figure(figsize=(10, 4.5 * n_rows))
    fig.suptitle(
        f"{label_a} vs {label_b} Comparison",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )

    # Global axis limits.
    all_pos = np.concatenate(
        list(captures_a.values()) + list(captures_b.values()), axis=0
    )
    g_min = all_pos.min(axis=0)
    g_max = all_pos.max(axis=0)
    g_center = (g_min + g_max) / 2.0
    g_span = (g_max - g_min).max() / 2.0 * 1.2

    for row, frame in enumerate(common_frames):
        for col, (captures, color, label) in enumerate(
            [
                (captures_a, color_a, label_a),
                (captures_b, color_b, label_b),
            ]
        ):
            ax = fig.add_subplot(n_rows, 2, row * 2 + col + 1, projection="3d")
            positions = captures[frame]
            verts_per_tri = positions[tri_faces]
            poly = Poly3DCollection(
                verts_per_tri,
                alpha=0.7,
                facecolor=color,
                edgecolor="#404040",
                linewidths=0.15,
            )
            ax.add_collection3d(poly)

            ax.set_xlim(g_center[0] - g_span, g_center[0] + g_span)
            ax.set_ylim(g_center[1] - g_span, g_center[1] + g_span)
            ax.set_zlim(max(g_center[2] - g_span, -5.0), g_center[2] + g_span)

            ax.set_xlabel("X", fontsize=8)
            ax.set_ylabel("Y", fontsize=8)
            ax.set_zlabel("Z", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{label} - Frame {frame}", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Render visualization images of the t-shirt drop simulation."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"),
        help="Directory to save rendered images.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds to simulate.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Total number of simulation frames.",
    )
    parser.add_argument(
        "--key-frames",
        type=int,
        nargs="+",
        default=DEFAULT_KEY_FRAMES,
        help="Frame indices at which to capture snapshots.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="VBD iterations per substep.",
    )
    parser.add_argument(
        "--chebyshev-rho",
        type=float,
        default=0.95,
        help="Chebyshev spectral radius for comparison (0 = disabled).",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="Elevation angle for 3D view.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="Azimuth angle for 3D view.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved images.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    wp.init()

    # Load the mesh indices once (shared across all scenarios).
    _, raw_indices = _load_shirt_mesh()
    tri_faces = np.array(raw_indices).reshape(-1, 3)

    for seed in args.seeds:
        print(f"\n{'=' * 60}")
        print(f"Seed {seed} -- Baseline (chebyshev_rho=0)")
        print(f"{'=' * 60}")

        # -- Baseline run --
        scenario_bl = build_scenario(seed=seed, iterations=args.iterations, chebyshev_rho=0.0)
        captures_bl = simulate_and_capture(scenario_bl, args.num_frames, args.key_frames)

        # Save individual frames.
        for frame, positions in sorted(captures_bl.items()):
            fig = render_frame(
                positions,
                tri_faces,
                title=f"Seed {seed} - Frame {frame} (Baseline)",
                elev=args.elev,
                azim=args.azim,
            )
            path = os.path.join(args.output_dir, f"seed{seed}_baseline_frame{frame:03d}.png")
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {path}")

        # Save progression grid.
        fig_grid = render_progression_grid(
            captures_bl,
            tri_faces,
            suptitle=f"Seed {seed} - Baseline Progression",
            elev=args.elev,
            azim=args.azim,
        )
        grid_path = os.path.join(args.output_dir, f"seed{seed}_baseline_progression.png")
        fig_grid.savefig(grid_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_grid)
        print(f"  Saved {grid_path}")

        # -- Chebyshev run --
        if args.chebyshev_rho > 0:
            print(f"\nSeed {seed} -- Chebyshev (rho={args.chebyshev_rho})")
            scenario_ch = build_scenario(
                seed=seed,
                iterations=args.iterations,
                chebyshev_rho=args.chebyshev_rho,
            )
            captures_ch = simulate_and_capture(scenario_ch, args.num_frames, args.key_frames)

            # Save individual Chebyshev frames.
            for frame, positions in sorted(captures_ch.items()):
                fig = render_frame(
                    positions,
                    tri_faces,
                    title=f"Seed {seed} - Frame {frame} (Chebyshev rho={args.chebyshev_rho})",
                    elev=args.elev,
                    azim=args.azim,
                    face_color="#e8a0a0",
                    edge_color="#a02020",
                )
                path = os.path.join(args.output_dir, f"seed{seed}_chebyshev_frame{frame:03d}.png")
                fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved {path}")

            # Save Chebyshev progression grid.
            fig_grid_ch = render_progression_grid(
                captures_ch,
                tri_faces,
                suptitle=f"Seed {seed} - Chebyshev (rho={args.chebyshev_rho}) Progression",
                elev=args.elev,
                azim=args.azim,
            )
            grid_ch_path = os.path.join(
                args.output_dir, f"seed{seed}_chebyshev_progression.png"
            )
            fig_grid_ch.savefig(grid_ch_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig_grid_ch)
            print(f"  Saved {grid_ch_path}")

            # Save side-by-side comparison.
            fig_cmp = render_comparison(
                captures_bl,
                captures_ch,
                tri_faces,
                label_a="Baseline",
                label_b=f"Chebyshev (rho={args.chebyshev_rho})",
                elev=args.elev,
                azim=args.azim,
            )
            cmp_path = os.path.join(args.output_dir, f"seed{seed}_comparison.png")
            fig_cmp.savefig(cmp_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig_cmp)
            print(f"  Saved {cmp_path}")

    print(f"\nAll images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
