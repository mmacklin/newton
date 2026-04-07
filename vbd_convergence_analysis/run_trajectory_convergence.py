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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789]
NUM_FRAMES = 30
SNAPSHOT_INTERVAL = 10  # save every Nth substep
ITERATIONS = 10

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
    )
    solver = scenario["solver"]
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
                snapshots.append({
                    "substep_idx": substep_idx,
                    "particle_q": state0.particle_q.numpy().copy(),
                    "particle_qd": state0.particle_qd.numpy().copy(),
                })
            state0.clear_forces()
            state1.clear_forces()
            collision_pipeline.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt)
            state0, state1 = state1, state0
            substep_idx += 1

    return snapshots


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
            **method_config,
        )
        solver = scenario["solver"]
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

        results.append({
            "substep_idx": snap["substep_idx"],
            "iteration_residuals": curve,
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
