#!/usr/bin/env python3
"""Trajectory-based VBD convergence test.

Records reference trajectories under baseline solver, then replays each
snapshot under different solver configurations to measure per-iteration
convergence from identical starting states.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

import warp as wp

from vbd_convergence_analysis.run_convergence_test import create_scenario
from newton._src.solvers.vbd.particle_vbd_kernels import compute_force_residual


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789]
NUM_FRAMES = 60
SNAPSHOT_INTERVAL = 10  # save every Nth substep
ITERATIONS = 10
DROP_HEIGHT_RANGE = (5.0, 20.0)  # low drop to ensure ground contact within simulation

METHODS = {
    "Baseline GS": dict(step_length=1.0),
    "Alpha 0.3": dict(step_length=0.3),
    "Alpha 0.5": dict(step_length=0.5),
    "Alpha 0.7": dict(step_length=0.7),
    "Alpha 0.9": dict(step_length=0.9),
    "Chebyshev Auto": dict(chebyshev_rho="auto"),
    "Jacobi": dict(jacobi_mode=True),
}


# ---------------------------------------------------------------------------
# Reference trajectory recording
# ---------------------------------------------------------------------------

def record_reference_trajectory(seed: int, num_frames: int = NUM_FRAMES) -> list[dict]:
    """Run baseline simulation, save (pos, vel) at sampled substeps.

    Returns list of snapshot dicts with numpy arrays.
    """
    scenario = create_scenario(
        seed=seed, iterations=ITERATIONS, enable_self_contact=True,
        drop_height_range=DROP_HEIGHT_RANGE,
    )
    solver = scenario["solver"]
    solver.avbd_beta = 0.0  # constant contact penalty (no ramp during iterations)
    state0 = scenario["state_0"]
    state1 = scenario["state_1"]
    control = scenario["control"]
    contacts = scenario["contacts"]
    collision_pipeline = scenario["collision_pipeline"]
    dt = scenario["dt"]
    substeps = scenario["sim_substeps"]

    solver.track_convergence = False

    snapshots = []
    substep_idx = 0

    for frame in range(num_frames):
        for sub in range(substeps):
            if substep_idx % SNAPSHOT_INTERVAL == 0:
                pos_snap = state0.particle_q.numpy().copy()
                snapshots.append({
                    "substep_idx": substep_idx,
                    "particle_q": pos_snap,
                    "particle_qd": state0.particle_qd.numpy().copy(),
                    "z_min": float(pos_snap[:, 2].min()),
                })
            state0.clear_forces()
            state1.clear_forces()
            collision_pipeline.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt)
            state0, state1 = state1, state0
            substep_idx += 1

    return snapshots


# ---------------------------------------------------------------------------
# Energy decomposition
# ---------------------------------------------------------------------------

def compute_energy_decomposition(solver, state, dt: float) -> dict:
    """Evaluate per-energy-type force residual at the current state.

    Runs the residual kernel with different energy subsets zeroed out.
    Returns dict with RMS residual for each component.
    """
    model = solver.model
    n = model.particle_count
    flags = model.particle_flags.numpy()
    mass = model.particle_mass.numpy()
    active = (flags & 1).astype(bool) & (mass > 0)

    if active.sum() == 0:
        return {"full": 0.0, "inertia_only": 0.0, "no_bending": 0.0, "no_elastic": 0.0}

    buf = solver._force_residual_norms
    zero_tri = wp.zeros_like(model.tri_materials)
    zero_edge = wp.zeros_like(model.edge_bending_properties)

    def _eval(tri_mats, edge_props):
        solver.particle_forces.zero_()
        wp.launch(
            kernel=compute_force_residual, dim=n,
            inputs=[
                dt, solver.particle_q_prev, state.particle_q,
                model.particle_mass, solver.inertia, model.particle_flags,
                model.tri_indices, model.tri_poses, tri_mats, model.tri_areas,
                solver._tri_material_model,
                model.edge_indices, model.edge_rest_angle, model.edge_rest_length,
                edge_props,
                model.tet_indices, model.tet_poses, model.tet_materials,
                solver.particle_adjacency, solver.particle_forces,
            ],
            outputs=[buf], device=solver.device,
        )
        r = buf.numpy()
        return float(np.sqrt(np.mean(r[active] ** 2)))

    return {
        "full": _eval(model.tri_materials, model.edge_bending_properties),
        "inertia_only": _eval(zero_tri, zero_edge),
        "no_bending": _eval(model.tri_materials, zero_edge),
        "no_elastic": _eval(zero_tri, model.edge_bending_properties),
    }


# ---------------------------------------------------------------------------
# Method evaluation on snapshots
# ---------------------------------------------------------------------------

def evaluate_method_on_snapshots(
    seed: int,
    snapshots: list[dict],
    method_config: dict,
) -> list[dict]:
    """For each snapshot, restore state, run one substep, collect convergence.

    Returns list of dicts with iteration_residuals per snapshot.
    """
    results = []

    for snap in snapshots:
        # Create a fresh scenario+solver with the method's config
        scenario = create_scenario(
            seed=seed,
            iterations=ITERATIONS,
            enable_self_contact=True,
            drop_height_range=DROP_HEIGHT_RANGE,
            **method_config,
        )
        solver = scenario["solver"]
        solver.avbd_beta = 0.0  # constant contact penalty
        state0 = scenario["state_0"]
        state1 = scenario["state_1"]
        control = scenario["control"]
        contacts = scenario["contacts"]
        collision_pipeline = scenario["collision_pipeline"]
        dt = scenario["dt"]

        # Restore saved state
        wp.copy(
            state0.particle_q,
            wp.from_numpy(
                snap["particle_q"].astype(np.float32),
                dtype=wp.vec3,
                device=solver.device,
            ),
        )
        wp.copy(
            state0.particle_qd,
            wp.from_numpy(
                snap["particle_qd"].astype(np.float32),
                dtype=wp.vec3,
                device=solver.device,
            ),
        )

        # Run one substep with convergence tracking
        solver.track_convergence = True
        solver.reset_convergence_data()

        state0.clear_forces()
        state1.clear_forces()
        collision_pipeline.collide(state0, contacts)
        solver.step(state0, state1, control, contacts, dt)

        conv = solver.get_convergence_data()
        if conv:
            iters = conv[0]["iteration_residuals"]
            curve = [it["rms_force_residual"] for it in iters]
        else:
            curve = []

        # Compute per-energy decomposition at the final state (after iterations)
        # The solver's internal state (inertia, particle_q_prev) is still valid
        energy_decomp = compute_energy_decomposition(solver, state1, dt)

        # Also check ground contact count
        try:
            n_contacts = int(contacts.soft_contact_count.numpy()[0])
        except Exception:
            n_contacts = 0

        results.append({
            "substep_idx": snap["substep_idx"],
            "iteration_residuals": curve,
            "energy_decomposition": energy_decomp,
            "contact_count": n_contacts,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trajectory-based VBD convergence test")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    args = parser.parse_args()

    wp.init()

    num_frames = args.num_frames

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output or os.path.join(base_dir, "trajectory_convergence_v2.json")

    all_results = {
        "metadata": {
            "seeds": args.seeds,
            "num_frames": num_frames,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "iterations": ITERATIONS,
            "self_contact": True,
        },
        "methods": {},
    }

    # Phase 1: Record reference trajectories
    trajectories = {}
    for seed in args.seeds:
        print(f"\n=== Recording reference trajectory (seed={seed}) ===")
        t0 = time.time()
        snapshots = record_reference_trajectory(seed, num_frames=num_frames)
        elapsed = time.time() - t0
        trajectories[seed] = snapshots
        print(f"  Recorded {len(snapshots)} snapshots in {elapsed:.1f}s")

    # Phase 2: Evaluate each method on all snapshots
    for method_name, method_config in METHODS.items():
        print(f"\n=== Evaluating: {method_name} ===")
        all_curves = []

        all_decomps = []
        all_contacts = []

        for seed in args.seeds:
            print(f"  Seed {seed}...")
            t0 = time.time()
            results = evaluate_method_on_snapshots(
                seed, trajectories[seed], method_config,
            )
            elapsed = time.time() - t0

            for r in results:
                if r["iteration_residuals"]:
                    all_curves.append(r["iteration_residuals"])
                if "energy_decomposition" in r:
                    all_decomps.append(r["energy_decomposition"])
                all_contacts.append(r.get("contact_count", 0))

            print(f"    {len(results)} snapshots in {elapsed:.1f}s")

        # Summary
        if all_curves:
            arr = np.array(all_curves)
            med = np.median(arr, axis=0)
            ratio = med[-1] / med[0] if med[0] > 1e-15 else float("inf")
            print(f"  Median curve: {med[0]:.4f} -> {med[-1]:.4f} (ratio {ratio:.4f})")

        all_results["methods"][method_name] = {
            "config": method_config,
            "curves": all_curves,
            "energy_decompositions": all_decomps,
            "contact_counts": all_contacts,
        }

    # Save
    # Convert any non-serializable types
    def _clean(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(all_results, f, default=_clean)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
