#!/usr/bin/env python3
"""VBD Convergence Analysis Test Script.

Spawns the t-shirt mesh in randomized positions and measures VBD solver
convergence across a distribution of scenarios. Produces JSON data files
consumed by the report generator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Ensure the worktree root is on sys.path so `import newton` resolves
WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

import warp as wp
from pxr import Usd

import newton
import newton.usd
from newton import ModelBuilder
from newton.solvers import SolverVBD


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def create_scenario(
    seed: int,
    iterations: int = 10,
    material_model: str = "neohookean",
    gravity_scale: float = 1.0,
    stiffness_scale: float = 1.0,
    drop_height_range: tuple = (40.0, 120.0),
    rotation_range: float = np.pi,
    chebyshev_rho: float = 0.0,
    jacobi_mode: bool = False,
    step_length: float = 1.0,
    enable_self_contact: bool = False,
    sim_substeps: int = 10,
):
    """Build one randomized t-shirt drop scenario.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    iterations : int
        VBD iteration count per substep.
    material_model : str
        Triangle material model ("neohookean" or "stvk").
    gravity_scale : float
        Multiplier on Earth gravity (in cm/s^2, sim is in cm).
    stiffness_scale : float
        Multiplier on elastic stiffness.
    drop_height_range : tuple
        (min, max) drop height in cm.
    rotation_range : float
        Maximum random rotation in radians (applied about each axis).

    Returns
    -------
    dict with keys: model, solver, state_0, state_1, control, contacts, dt, params
    """
    rng = np.random.default_rng(seed)

    # Randomize parameters
    drop_height = rng.uniform(*drop_height_range)
    rot_angles = rng.uniform(-rotation_range, rotation_range, size=3)
    lateral_offset = rng.uniform(-30.0, 30.0, size=2)

    # Physics parameters (cm-scale, matching franka example)
    gravity_cm = -981.0 * gravity_scale
    tri_ke = 1e4 * stiffness_scale
    tri_ka = 1e4 * stiffness_scale
    tri_kd = 1.5e-6
    bending_ke = 5.0 * stiffness_scale
    bending_kd = 1e-2
    density = 0.02

    scene = ModelBuilder(gravity=gravity_cm)

    # Load t-shirt mesh
    asset_path = os.path.join(WORKTREE, "newton", "examples", "assets", "unisex_shirt.usd")
    usd_stage = Usd.Stage.Open(asset_path)
    usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
    shirt_mesh = newton.usd.get_mesh(usd_prim)
    vertices = [wp.vec3(v) for v in shirt_mesh.vertices]
    indices = shirt_mesh.indices

    # Build rotation quaternion from random Euler angles
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

    # Contact parameters
    model.soft_contact_ke = 1e4
    model.soft_contact_kd = 1e-2
    model.soft_contact_mu = 0.25

    self_contact_kwargs = {}
    if enable_self_contact:
        self_contact_kwargs = dict(
            particle_self_contact_radius=0.2,
            particle_self_contact_margin=0.2,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
        )

    solver = SolverVBD(
        model,
        iterations=iterations,
        particle_tri_material_model=material_model,
        particle_enable_self_contact=enable_self_contact,
        **self_contact_kwargs,
    )
    solver.track_convergence = True
    if chebyshev_rho == "auto" or (isinstance(chebyshev_rho, (int, float)) and chebyshev_rho > 0):
        solver.chebyshev_rho = chebyshev_rho
    if jacobi_mode:
        solver.jacobi_mode = True
    if step_length < 1.0:
        solver.step_length = step_length

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.5)
    contacts = collision_pipeline.contacts()

    frame_dt = 1.0 / 60.0
    sim_dt = frame_dt / sim_substeps

    params = {
        "seed": seed,
        "iterations": iterations,
        "material_model": material_model,
        "gravity_scale": gravity_scale,
        "stiffness_scale": stiffness_scale,
        "drop_height": float(drop_height),
        "rot_angles": rot_angles.tolist(),
        "lateral_offset": lateral_offset.tolist(),
        "tri_ke": tri_ke,
        "tri_ka": tri_ka,
        "density": density,
        "sim_dt": sim_dt,
        "sim_substeps": sim_substeps,
        "particle_count": model.particle_count,
        "tri_count": model.tri_count,
        "chebyshev_rho": chebyshev_rho,
        "jacobi_mode": jacobi_mode,
        "step_length": step_length,
        "enable_self_contact": enable_self_contact,
    }

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
        "params": params,
    }


def run_scenario(scenario: dict, num_frames: int = 30) -> dict:
    """Run a scenario for a given number of frames and return convergence data.

    Parameters
    ----------
    scenario : dict
        Created by ``create_scenario``.
    num_frames : int
        Number of frames to simulate.

    Returns
    -------
    dict with convergence data and scenario parameters.
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

    solver.reset_convergence_data()
    t0 = time.time()

    for frame in range(num_frames):
        for sub in range(substeps):
            state_0.clear_forces()
            state_1.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    elapsed = time.time() - t0
    conv_data = solver.get_convergence_data()

    # Check for NaN at end
    final_pos = state_0.particle_q.numpy()
    has_nan = bool(np.any(np.isnan(final_pos)))

    return {
        "params": scenario["params"],
        "convergence": conv_data,
        "elapsed_seconds": elapsed,
        "has_nan": has_nan,
        "num_frames": num_frames,
        "total_substeps": num_frames * substeps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VBD Convergence Analysis")
    parser.add_argument("--num-scenarios", type=int, default=8, help="Number of randomized scenarios")
    parser.add_argument("--num-frames", type=int, default=30, help="Frames per scenario")
    parser.add_argument("--iterations", type=int, default=10, help="VBD iterations per substep")
    parser.add_argument("--material", type=str, default="neohookean", choices=["neohookean", "stvk"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--seed-offset", type=int, default=0, help="Base seed offset")
    parser.add_argument("--gravity-scales", type=float, nargs="+", default=[1.0],
                        help="Gravity scale factors to test")
    parser.add_argument("--stiffness-scales", type=float, nargs="+", default=[1.0],
                        help="Stiffness scale factors to test")
    parser.add_argument("--iteration-counts", type=int, nargs="+", default=None,
                        help="Multiple iteration counts to compare")
    parser.add_argument("--chebyshev-rho", type=str, default="0.0",
                        help="Chebyshev acceleration spectral radius (0=disabled, auto=adaptive, float=manual)")
    parser.add_argument("--jacobi", action="store_true", help="Use Jacobi mode (non-GS)")
    parser.add_argument("--step-length", type=float, default=1.0,
                        help="Newton step length (under-relaxation, e.g. 0.7, 0.5)")
    parser.add_argument("--self-contact", action="store_true",
                        help="Enable self-contact (matching franka cloth example)")
    args = parser.parse_args()

    # Parse chebyshev_rho
    if args.chebyshev_rho == "auto":
        args.chebyshev_rho_val = "auto"
    else:
        args.chebyshev_rho_val = float(args.chebyshev_rho)

    wp.init()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "convergence_results.json",
        )

    iteration_counts = args.iteration_counts or [args.iterations]

    all_results = []

    for gravity_scale in args.gravity_scales:
        for stiffness_scale in args.stiffness_scales:
            for iter_count in iteration_counts:
                for i in range(args.num_scenarios):
                    seed = args.seed_offset + i
                    print(f"\n{'='*60}")
                    print(f"Scenario {i+1}/{args.num_scenarios} | seed={seed} | "
                          f"iters={iter_count} | gravity={gravity_scale:.1f}x | "
                          f"stiffness={stiffness_scale:.1f}x | material={args.material}")
                    print(f"{'='*60}")

                    try:
                        scenario = create_scenario(
                            seed=seed,
                            iterations=iter_count,
                            material_model=args.material,
                            gravity_scale=gravity_scale,
                            stiffness_scale=stiffness_scale,
                            chebyshev_rho=args.chebyshev_rho_val,
                            jacobi_mode=args.jacobi,
                            step_length=args.step_length,
                            enable_self_contact=args.self_contact,
                        )
                        result = run_scenario(scenario, num_frames=args.num_frames)
                        all_results.append(result)

                        # Print summary
                        conv = result["convergence"]
                        if len(conv) > 0:
                            last_step = conv[-1]
                            iters = last_step["iteration_residuals"]
                            if len(iters) > 0:
                                first_res = iters[0].get("rms_force_residual", iters[0].get("rms_displacement", 0))
                                last_res = iters[-1].get("rms_force_residual", iters[-1].get("rms_displacement", 0))
                                ratio = last_res / first_res if first_res > 0 else float("inf")
                                print(f"  RMS force residual: {first_res:.6e} -> {last_res:.6e} "
                                      f"(ratio: {ratio:.4f})")
                                print(f"  Has NaN: {result['has_nan']}")
                                print(f"  Time: {result['elapsed_seconds']:.2f}s")

                    except Exception as e:
                        print(f"  ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        all_results.append({
                            "params": {"seed": seed, "iterations": iter_count,
                                       "gravity_scale": gravity_scale,
                                       "stiffness_scale": stiffness_scale,
                                       "material_model": args.material},
                            "error": str(e),
                        })

    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nDone. {len(all_results)} scenarios completed.")


if __name__ == "__main__":
    main()
