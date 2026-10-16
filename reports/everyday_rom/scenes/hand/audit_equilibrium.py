"""Compare recorded hold q and modal force with the prescribed static target."""

import argparse
import json
from pathlib import Path

import numpy as np
from hand_geometry import transform_points
from individual import FINGERS, finger_patches


def audit_saved(run: Path, starts, release_duration):
    """Check every recorded force, including the complete unforced intervals."""
    rows = []
    for arm, enabled in (("actuated", True), ("no-actuation", False)):
        with np.load(run / arm / "actuation.npz", allow_pickle=False) as a:
            forces, target, times, stiffness = a["forces"], a["q_target"], a["times"], a["stiffness"]
            dt = float(a["dt"])
            assert dt > 0 and np.isfinite(dt)
            assert forces.shape == target.shape == (round(12 / dt), 48)
            assert np.isfinite(forces).all() and np.isfinite(target).all() and np.isfinite(stiffness).all()
            assert bool(a["enabled"]) == enabled
            assert float(a["release_duration_s"]) == release_duration
            np.testing.assert_array_equal(times, np.arange(len(forces)) * dt)
            np.testing.assert_array_equal(a["starts"], starts)
            np.testing.assert_array_equal(forces, target * stiffness if enabled else np.zeros_like(forces))
            zero_samples = 0
            for i, start in enumerate(starts):
                begin = int(round((start + 0.9 + release_duration) / dt))
                end = int(round(starts[i + 1] / dt)) if i + 1 < len(starts) else len(times)
                np.testing.assert_array_equal(forces[begin:end], 0)
                np.testing.assert_array_equal(target[begin:end], 0)
                zero_samples += end - begin
            rows.append({"arm": arm, "finite": True, "formula_exact_fp32": True, "released_zero_samples": zero_samples})
    return {"passed": True, "rows": rows, "release_duration_s": release_duration}


def audit(trial, targets):
    with np.load(trial / "actuated/substeps.npz") as a, np.load(trial / "no-actuation/substeps.npz") as b:
        qa, qb, ts = a["q"], b["q"], a["times"]
        ba, bb = a["body_q"], b["body_q"]
    with np.load(trial / "actuated/original_states.npz") as a:
        rest, phi, body, k = a["rest"], a["vertex_phi"].astype(float), int(a["hand_body"]), a["mode_stiffness"]
    with np.load(trial / "actuated/actuation.npz") as a, np.load(trial / "no-actuation/actuation.npz") as b:
        forces, prescribed, starts = a["forces"], a["q_target"], a["starts"]
        amplitude = float(a["amplitude"])
        np.testing.assert_array_equal(b["forces"], 0)
        np.testing.assert_allclose(forces, prescribed * k, rtol=2e-7, atol=1e-6)
        exact_force_reference = np.array_equal(forces, prescribed * k)
    with np.load(targets) as a:
        coords = a["q"] * amplitude
    patches = finger_patches(rest)
    rows = []
    for i, (name, start) in enumerate(zip(FINGERS, starts, strict=True)):
        hold = (ts >= start + 0.75) & (ts < start + 0.89)
        delta = qa[hold].mean(axis=0) - qb[hold].mean(axis=0)
        desired = coords[i]
        predicted = np.einsum("nmc,m->nc", phi, desired)
        actual = np.einsum("nmc,m->nc", phi, delta)
        mask = patches[i]
        mid = np.flatnonzero(hold)[len(np.flatnonzero(hold)) // 2]
        local_a = rest + np.einsum("nmc,m->nc", phi, qa[mid])
        local_b = rest + np.einsum("nmc,m->nc", phi, qb[mid])
        world = transform_points(local_a, ba[mid, body]) - transform_points(local_b, bb[mid, body])
        frame_only = transform_points(local_b, ba[mid, body]) - transform_points(local_b, bb[mid, body])
        force_error = float(np.max(np.abs(forces[hold] - desired * k)))
        rows.append(
            {
                "finger": name,
                "prescribed_local_centroid_travel_m": float(np.linalg.norm(predicted[mask].mean(axis=0))),
                "solved_local_centroid_travel_m": float(np.linalg.norm(actual[mask].mean(axis=0))),
                "prescribed_local_max_distal_m": float(np.linalg.norm(predicted[mask], axis=1).max()),
                "solved_local_max_distal_m": float(np.linalg.norm(actual[mask], axis=1).max()),
                "solved_world_max_distal_m": float(np.linalg.norm(world[mask], axis=1).max()),
                "frame_only_max_distal_m": float(np.linalg.norm(frame_only[mask], axis=1).max()),
                "q_minus_target_control_relative_norm": float(
                    np.linalg.norm(delta - desired) / np.linalg.norm(desired)
                ),
                "modal_response_gain": float(np.dot(delta, desired) / np.dot(desired, desired)),
                "surface_difference_from_target_m": float(np.linalg.norm(actual - predicted, axis=1).max()),
                "max_force_minus_K_target": force_error,
                "force_over_K_target_relative_error": float(
                    np.linalg.norm(forces[hold].mean(axis=0) / k - desired) / np.linalg.norm(desired)
                ),
                "checks": {
                    "force_formula_exact_fp32": exact_force_reference,
                    "unit_not_double_response": abs(np.dot(delta, desired) / np.dot(desired, desired) - 1) < 0.02,
                    "target_surface_match": np.linalg.norm(actual - predicted, axis=1).max() < 0.0002,
                },
            }
        )
    # Cast NumPy bools without losing numerical precision in the evidence.
    for row in rows:
        row["checks"] = {k: bool(v) for k, v in row["checks"].items()}
    return {
        "explanation": "Centroid travel and maximum distal vertex travel are labeled separately. Both are body-local relative to the matched no-actuation control, not floating-frame motion.",
        "all_checks_passed": all(all(r["checks"].values()) for r in rows),
        "rows": rows,
        "negative_control_forces_exactly_zero": True,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("trial", type=Path)
    p.add_argument("targets", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = audit(args.trial, args.targets)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
