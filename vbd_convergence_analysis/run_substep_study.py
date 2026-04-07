#!/usr/bin/env python3
"""Substeps vs iterations trade-off study.

For each (substeps, iterations) configuration, runs a full rollout
measuring GPU time and per-frame force residual. All runs use the
convergence-fixed solver (quadratic self-contact, c=kd damping,
avbd_beta=0, alpha=0.7).
"""
from __future__ import annotations

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
DROP_HEIGHT_RANGE = (5.0, 20.0)
STEP_LENGTH = 0.7

# (substeps, iterations) configs grouped by total solver calls
CONFIGS = [
    # Total = 10
    (1, 10), (2, 5), (5, 2), (10, 1),
    # Total = 20
    (1, 20), (2, 10), (5, 4), (10, 2), (20, 1),
    # Total = 30
    (1, 30), (3, 10), (5, 6), (10, 3), (15, 2), (30, 1),
    # Total = 60
    (2, 30), (5, 12), (10, 6), (20, 3), (30, 2),
    # Baseline (expensive)
    (10, 10),
]


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def run_rollout(seed: int, substeps: int, iterations: int) -> dict:
    """Run one full rollout, return per-frame residuals and GPU times."""
    scenario = create_scenario(
        seed=seed,
        iterations=iterations,
        enable_self_contact=True,
        drop_height_range=DROP_HEIGHT_RANGE,
        step_length=STEP_LENGTH,
        sim_substeps=substeps,
    )
    solver = scenario["solver"]
    solver.avbd_beta = 0.0
    solver.track_convergence = False

    state0 = scenario["state_0"]
    state1 = scenario["state_1"]
    control = scenario["control"]
    contacts = scenario["contacts"]
    pipeline = scenario["collision_pipeline"]
    dt = scenario["dt"]

    # Probe solver for residual evaluation.  Use a FIXED reference dt
    # (sim_substeps=1 -> dt=1/60) so that the inertia term m/dt^2 is
    # consistent across configs.  Otherwise more substeps = larger m/dt^2
    # = higher residual for the same positional accuracy.
    PROBE_SUBSTEPS = 1
    probe_sc = create_scenario(
        seed=seed, iterations=1, enable_self_contact=True,
        drop_height_range=DROP_HEIGHT_RANGE, sim_substeps=PROBE_SUBSTEPS,
    )
    probe_solver = probe_sc["solver"]
    probe_solver.avbd_beta = 0.0
    probe_s0 = probe_sc["state_0"]
    probe_s1 = probe_sc["state_1"]
    probe_control = probe_sc["control"]
    probe_contacts = probe_sc["contacts"]
    probe_pipeline = probe_sc["collision_pipeline"]
    probe_dt = probe_sc["dt"]

    per_frame_residuals = []
    per_frame_times = []

    for frame in range(NUM_FRAMES):
        wp.synchronize()
        t0 = time.perf_counter()

        for sub in range(substeps):
            state0.clear_forces()
            state1.clear_forces()
            pipeline.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt)
            state0, state1 = state1, state0

        wp.synchronize()
        frame_time = time.perf_counter() - t0
        per_frame_times.append(frame_time)

        # Probe: measure residual at current state
        wp.copy(probe_s0.particle_q, state0.particle_q)
        wp.copy(probe_s0.particle_qd, state0.particle_qd)
        probe_solver.track_convergence = True
        probe_solver.reset_convergence_data()
        probe_s0.clear_forces()
        probe_s1.clear_forces()
        probe_pipeline.collide(probe_s0, probe_contacts)
        probe_solver.step(probe_s0, probe_s1, probe_control, probe_contacts, probe_dt)

        conv = probe_solver.get_convergence_data()
        if conv:
            residual = conv[0]["iteration_residuals"][0]["rms_force_residual"]
        else:
            residual = float("nan")
        per_frame_residuals.append(residual)

    has_nan = bool(np.any(np.isnan(state0.particle_q.numpy())))

    return {
        "per_frame_residuals": per_frame_residuals,
        "per_frame_times": per_frame_times,
        "has_nan": has_nan,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wp.init()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "substep_study_results.json")

    results = {
        "metadata": {
            "seeds": SEEDS,
            "num_frames": NUM_FRAMES,
            "step_length": STEP_LENGTH,
            "drop_height_range": list(DROP_HEIGHT_RANGE),
            "self_contact": True,
            "avbd_beta": 0.0,
        },
        "configs": [],
    }

    for substeps, iterations in CONFIGS:
        total = substeps * iterations
        label = f"{substeps}sub x {iterations}iter (total={total})"
        print(f"\n=== {label} ===")

        seed_results = {}
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            result = run_rollout(seed, substeps, iterations)
            elapsed = sum(result["per_frame_times"])
            med_res = float(np.median(result["per_frame_residuals"]))
            med_time = float(np.median(result["per_frame_times"]))
            print(f"done ({elapsed:.1f}s total, {med_time*1000:.1f}ms/frame, "
                  f"median_res={med_res:.1f}, nan={result['has_nan']})")

            seed_results[str(seed)] = {
                "per_frame_residuals": result["per_frame_residuals"],
                "per_frame_times": result["per_frame_times"],
                "has_nan": result["has_nan"],
            }

        results["configs"].append({
            "substeps": substeps,
            "iterations": iterations,
            "total_calls": total,
            "seeds": seed_results,
        })

    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
