#!/usr/bin/env python3
"""Full rollout comparison for VBD solver methods.

Runs independent simulations for each method, measuring per-frame
force residual to show accumulated simulation quality impact.
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
NUM_FRAMES = 60
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
# Single rollout with per-frame residual measurement
# ---------------------------------------------------------------------------

def run_rollout(seed: int, method_config: dict, num_frames: int = NUM_FRAMES) -> dict:
    """Run full simulation, measure per-frame force residual.

    After each frame, runs a 1-iteration probe substep to evaluate
    the force residual at the current state.
    """
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
    substeps = scenario["sim_substeps"]

    # Also create a probe solver for residual evaluation
    probe_scenario = create_scenario(
        seed=seed,
        iterations=1,
        enable_self_contact=True,
        drop_height_range=DROP_HEIGHT_RANGE,
    )
    probe_solver = probe_scenario["solver"]
    probe_solver.avbd_beta = 0.0
    probe_state0 = probe_scenario["state_0"]
    probe_state1 = probe_scenario["state_1"]
    probe_control = probe_scenario["control"]
    probe_contacts = probe_scenario["contacts"]
    probe_pipeline = probe_scenario["collision_pipeline"]

    per_frame_residuals = []

    solver.track_convergence = False

    for frame in range(num_frames):
        # Run the actual simulation frame
        for sub in range(substeps):
            state0.clear_forces()
            state1.clear_forces()
            collision_pipeline.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt)
            state0, state1 = state1, state0

        # Probe: evaluate force residual at current state
        # Copy current state to probe solver's state
        wp.copy(probe_state0.particle_q, state0.particle_q)
        wp.copy(probe_state0.particle_qd, state0.particle_qd)

        probe_solver.track_convergence = True
        probe_solver.reset_convergence_data()

        probe_state0.clear_forces()
        probe_state1.clear_forces()
        probe_pipeline.collide(probe_state0, probe_contacts)
        probe_solver.step(
            probe_state0, probe_state1, probe_control, probe_contacts, dt,
        )

        conv = probe_solver.get_convergence_data()
        if conv:
            # Iteration 0 residual = force residual before any VBD corrections
            residual = conv[0]["iteration_residuals"][0]["rms_force_residual"]
        else:
            residual = float("nan")

        per_frame_residuals.append(residual)

    # Check for NaN
    final_pos = state0.particle_q.numpy()
    has_nan = bool(np.any(np.isnan(final_pos)))

    return {
        "per_frame_residuals": per_frame_residuals,
        "has_nan": has_nan,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VBD rollout comparison")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    args = parser.parse_args()

    wp.init()

    num_frames = args.num_frames

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output or os.path.join(base_dir, "rollout_comparison_v2.json")

    all_results = {
        "metadata": {
            "seeds": args.seeds,
            "num_frames": num_frames,
            "iterations": ITERATIONS,
            "self_contact": True,
        },
        "methods": {},
    }

    for method_name, method_config in METHODS.items():
        print(f"\n=== {method_name} ===")
        seed_results = {}

        for seed in args.seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            t0 = time.time()
            result = run_rollout(seed, method_config, num_frames=num_frames)
            elapsed = time.time() - t0

            residuals = result["per_frame_residuals"]
            print(
                f"done in {elapsed:.1f}s  "
                f"(first={residuals[0]:.4f}, last={residuals[-1]:.4f}, "
                f"nan={result['has_nan']})"
            )

            seed_results[str(seed)] = {
                "per_frame_residuals": residuals,
                "has_nan": result["has_nan"],
            }

        all_results["methods"][method_name] = {
            "config": method_config,
            "seeds": seed_results,
        }

    # Save
    with open(output_path, "w") as f:
        json.dump(all_results, f)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
