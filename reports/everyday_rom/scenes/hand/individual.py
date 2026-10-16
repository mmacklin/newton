# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Five separated contact-only presses of the accepted derived hand ROM."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import warp as wp
from hand_contact import ACCEPTED, PIN, RUNTIME, HandContact, digest, private_path, recover, smooth_cpu

import newton

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TARGET_XZ = np.array([[0.053, 0.078], [0.030, 0.126], [0.004, 0.129], [-0.021, 0.113], [-0.037, 0.102]])
HALF = np.array([0.004, 0.008, 0.006])
STARTS = 1.0 + np.arange(5) * 3.5
DURATION = 18.5
DEPTHS = (0.0055, 0.0065, 0.008, 0.0065, 0.006)


def finger_patches(rest):
    """Disjoint distal surface regions, identified on the fixed derived rest mesh."""
    x, _, z = rest.T
    return np.array(
        [
            (x > 0.045) & (z > 0.055),
            (x >= 0.016) & (x <= 0.045) & (z > 0.105),
            (x >= -0.009) & (x < 0.016) & (z > 0.115),
            (x >= -0.030) & (x < -0.009) & (z > 0.100),
            (x < -0.030) & (z > 0.085),
        ]
    )


def trajectory(rest, depths, duration=DURATION):
    """Precompute only rigid probe positions [m], never hand/modal coordinates."""
    ts = np.arange(round(duration * 240) + 1) / 240
    positions = np.tile([*TARGET_XZ[0, :1], 0.04, 0.12 + TARGET_XZ[0, 1]], (len(ts), 1))
    ends = []
    for i, (x, z) in enumerate(TARGET_XZ):
        footprint = (np.abs(rest[:, 0] - x) < HALF[0]) & (np.abs(rest[:, 2] - z) < HALF[2])
        surface_y = float(rest[footprint, 1].max())
        end_y = surface_y + HALF[1] - depths[i]
        ends.append(end_y)
        pulse = smooth_cpu(ts, STARTS[i], 1.1) * (1 - smooth_cpu(ts, STARTS[i] + 1.5, 0.2))
        positions[:, 1] -= (0.04 - end_y) * pulse
        if i:
            # Lateral travel starts only after complete retraction and recovery.
            shift = smooth_cpu(ts, STARTS[i] - 0.8, 0.6)
            positions[:, 0] += (x - TARGET_XZ[i - 1, 0]) * shift
            positions[:, 2] += (z - TARGET_XZ[i - 1, 1]) * shift
    return positions.astype(np.float32), ends


@wp.kernel
def drive_individual(
    clock: wp.array[int],
    path: wp.array[wp.vec3],
    dt: float,
    support: int,
    pusher: int,
    b0: wp.array[wp.transform],
    b1: wp.array[wp.transform],
    v0: wp.array[wp.spatial_vector],
    v1: wp.array[wp.spatial_vector],
):
    body = support
    p, v = wp.vec3(0.0, 0.0, 0.12), wp.vec3(0.0)
    if wp.tid() == 1:
        body = pusher
        slot = wp.min(clock[0], path.shape[0] - 2)
        p = path[slot]
        v = (path[slot + 1] - p) / dt
    b0[body] = wp.transform(p, wp.quat_identity())
    b1[body] = b0[body]
    v0[body] = wp.spatial_vector(v, wp.vec3(0.0))
    v1[body] = v0[body]


@wp.kernel
def contact_regions(
    cursor: wp.array[int],
    raw: wp.array[int],
    s0: wp.array[int],
    s1: wp.array[int],
    sample0: wp.array[int],
    sample1: wp.array[int],
    hand: int,
    pusher: int,
    labels: wp.array[int],
    saved: wp.array2d[int],
):
    for i in range(wp.min(raw[0], s0.shape[0])):
        sample = int(-1)
        if s0[i] == hand and s1[i] == pusher:
            sample = sample0[i]
        elif s1[i] == hand and s0[i] == pusher:
            sample = sample1[i]
        if sample >= 0 and sample < labels.shape[0]:
            saved[cursor[0], labels[sample] + 1] += 1


class IndividualHand(HandContact):
    def __init__(self, *, depths=DEPTHS, **kwargs):
        self.depths = np.asarray(depths)
        self.probe_half = HALF
        super().__init__(duration=DURATION, **kwargs)
        self.meshes[-1]["name"] = "individual_finger_probe"

    def prepare_drive(self, device):
        path, self.end_y = trajectory(self.rest, self.depths, self.duration)
        self.path = wp.array(path, dtype=wp.vec3, device=device)
        if self.audited:
            labels = np.full(len(self.basis.sample_points), -1, dtype=np.int32)
            for i, patch in enumerate(finger_patches(self.rest)):
                labels[self.used[patch]] = i
            self.sample_labels = wp.array(labels, dtype=int, device=device)
            self.finger_contacts = wp.zeros((self.capacity, 6), dtype=int, device=device)
        # The first recorded state must also contain the true retracted probe.
        self.drive_step()

    def drive_step(self):
        wp.launch(
            drive_individual,
            2,
            inputs=[
                self.cursor,
                self.path,
                self.dt,
                self.support,
                self.pusher,
                self.s0.body_q,
                self.s1.body_q,
                self.s0.body_qd,
                self.s1.body_qd,
            ],
            device=self.model.device,
        )

    def audit_contact_regions(self):
        c = self.contacts
        wp.launch(
            contact_regions,
            1,
            inputs=[
                self.cursor,
                c.rigid_contact_count,
                c.rigid_contact_shape0,
                c.rigid_contact_shape1,
                c.rigid_contact_elastic_sample0,
                c.rigid_contact_elastic_sample1,
                self.hand_shape,
                self.pusher_shape,
                self.sample_labels,
                self.finger_contacts,
            ],
            device=self.model.device,
        )

    def export(self, output):
        result = super().export(output)
        # These inherited baseline bar fields do not describe the narrow probe.
        result.pop("travel_m")
        result.pop("bar_height_m")
        result.update(
            probe_depths_m=self.depths.tolist(),
            probe_half_extents_m=HALF.tolist(),
            targets_xz_m=TARGET_XZ.tolist(),
            probe_end_y_m=self.end_y,
        )
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(
            output / "finger_contacts.npz",
            counts=self.finger_contacts.numpy(),
            columns=["outside_distal_regions", *FINGERS],
            patches=finger_patches(self.rest),
            targets_xz=TARGET_XZ,
        )
        meta_path = output / "scene.json"
        meta = json.loads(meta_path.read_text())
        meta.update(
            slug="hand-individual-finger-presses",
            poster_frame=round((STARTS[1] + 1.3) * 60),
            sequence=[
                {
                    "finger": name,
                    "start_s": float(start),
                    "hold_s": [float(start + 1.1), float(start + 1.5)],
                    "fully_retracted_s": float(start + 1.7),
                    "depth_m": float(self.depths[i]),
                }
                for i, (name, start) in enumerate(zip(FINGERS, STARTS, strict=True))
            ],
        )
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        return result


def paired_metrics(main, control):
    with np.load(main / "original_states.npz") as a, np.load(control / "original_states.npz") as b:
        rest, phi = a["rest"], a["vertex_phi"].astype(float)
        np.testing.assert_array_equal(rest, b["rest"])
        np.testing.assert_array_equal(phi, b["vertex_phi"])
        pusher = int(a["pusher_body"])
    with np.load(main / "substeps.npz") as sa, np.load(control / "substeps.npz") as sb:
        qa, qb, ts = sa["q"], sb["q"], sa["times"]
        contacts, control_contacts = sa["contacts"], sb["contacts"]
        np.testing.assert_array_equal(sa["body_q"][:, pusher], sb["body_q"][:, pusher])
    baseline = qa[(ts >= 0.7) & (ts <= 0.95)].mean(axis=0)
    patches = finger_patches(rest)
    old_patches = [
        (rest[:, 2] > 0.09) & (rest[:, 0] >= lo) & (rest[:, 0] < hi)
        for lo, hi in ((-0.06, -0.015), (-0.015, 0.008), (0.008, 0.03), (0.03, 0.06))
    ]
    response, causal, distal, old_distal = [], [], [], []
    for lo in range(0, len(qa), 32):
        dis = np.linalg.norm(recover(np.zeros_like(rest), phi, qa[lo : lo + 32] - baseline), axis=-1)
        delta = np.linalg.norm(recover(np.zeros_like(rest), phi, qa[lo : lo + 32] - qb[lo : lo + 32]), axis=-1)
        response.extend(dis.max(axis=1))
        causal.extend(delta.max(axis=1))
        distal.extend(np.stack([delta[:, p].max(axis=1) for p in patches], axis=1))
        old_distal.extend(np.stack([delta[:, p].max(axis=1) for p in old_patches], axis=1))
    response, causal, distal, old_distal = map(np.asarray, (response, causal, distal, old_distal))
    counts = np.load(main / "finger_contacts.npz")["counts"]
    per_finger = []
    for i, (name, start) in enumerate(zip(FINGERS, STARTS, strict=True)):
        load = (ts >= start) & (ts <= start + 1.7)
        hold = (ts >= start + 1.15) & (ts <= start + 1.5)
        released = (ts >= start + 2.7) & (ts <= start + 3.2)
        peak = float(distal[load, i].max())
        residual = float(distal[released, i].max())
        contact_peak = counts[load].max(axis=0)
        checks = {
            "target_causal_response_3mm": peak >= 0.003,
            "sustained_target_contact": bool((counts[hold, i + 1] > 0).all()),
            "no_other_finger_contact": bool(np.delete(contact_peak[1:], i).max() == 0),
            "no_outside_region_contact": int(contact_peak[0]) == 0,
            "released_no_contacts": bool((contacts[released, 1] == 0).all()),
            "released_recovery_90pct": residual <= 0.1 * peak,
            "released_recovery_1mm": residual <= 0.001,
        }
        per_finger.append(
            {
                "finger": name,
                "checks": checks,
                "passed": all(checks.values()),
                "causal_response_m": peak,
                "recovery_residual_m": residual,
                "hold_target_contact_min": int(counts[hold, i + 1].min()),
                "contact_peaks_by_region": contact_peak.tolist(),
                "causal_peaks_all_fingers_m": distal[load].max(axis=0).tolist(),
            }
        )
    peak, causal_peak = float(response.max()), float(causal.max())
    residual = float(response[ts >= DURATION - 0.5].max())
    checks = {
        "response_3mm": peak >= 0.003,
        "causal_response_3mm": causal_peak >= 0.003,
        "distal_three_patches_1mm": int((old_distal.max(axis=0) >= 0.001).sum()) >= 3,
        "recovery_90pct": residual <= 0.1 * peak,
        "recovery_1mm": residual <= 0.001,
        "negative_no_pusher_contacts": int(control_contacts[:, 1].max()) == 0,
        "sustained_hold_contact": all(p["checks"]["sustained_target_contact"] for p in per_finger),
        "all_contacts_classified": bool(np.array_equal(counts.sum(axis=1), contacts[:, 1])),
        "all_fingers_causal_and_released": all(p["passed"] for p in per_finger),
    }
    np.savez_compressed(
        main / "paired_response.npz",
        times=ts,
        baseline_response=response,
        contact_causal_response=causal,
        distal_patch_response=old_distal,
        finger_response=distal,
        finger_names=FINGERS,
    )
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "response_m": peak,
        "causal_response_m": causal_peak,
        "recovery_residual_m": residual,
        "recovery_fraction": 1 - residual / peak,
        "per_finger": per_finger,
        "distal_spatial_patch_peaks_m": old_distal.max(axis=0).tolist(),
    }


def benchmark(kind, repeats, depths):
    cls = HandContact if kind == "baseline" else IndividualHand
    kwargs = {} if kind == "baseline" else {"depths": depths}
    rows, runs = [], []
    for repeat in range(repeats + 1):
        example = cls(audited=False, **kwargs)
        assert example.graph is not None and not hasattr(example, "audit")
        metric_names = (
            "initial_residual_norm",
            "solve_residual_norm",
            "applied_residual_norm",
            "update_norm",
            "update_max",
        )
        assert all(len(getattr(example.solver, "elastic_body_block_" + name)) == 0 for name in metric_names)
        wp.synchronize_device(example.model.device)
        samples = []
        for _ in range(example.frames):
            start = time.perf_counter_ns()
            wp.capture_launch(example.graph)
            wp.synchronize_device(example.model.device)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        if repeat:
            rows.append(float(np.mean(samples)))
            # Profiling and final state readbacks are outside the timed sequence.
            final_q = example.s0.joint_q.numpy()
            with wp.ScopedTimer("uncaptured_frame", cuda_filter=wp.TIMING_ALL, print=False) as timer:
                example.simulate()
            groups = defaultdict(lambda: {"calls": 0, "total_ms": 0.0})
            for item in timer.timing_results:
                groups[item.name]["calls"] += 1
                groups[item.name]["total_ms"] += item.elapsed
            runs.append(
                {
                    "frame_ms": samples,
                    "mean_ms": rows[-1],
                    "median_ms": float(np.median(samples)),
                    "p95_ms": float(np.quantile(samples, 0.95)),
                    "final_joint_q": final_q.tolist(),
                    "uncaptured_TIMING_ALL": [
                        dict(name=k, **v) for k, v in sorted(groups.items(), key=lambda pair: -pair[1]["total_ms"])
                    ],
                }
            )
        del example
    return {
        "kind": kind,
        "ms_per_frame": rows,
        "median_ms_per_frame": float(np.median(rows)),
        "frames_per_repeat": 510 if kind == "baseline" else round(DURATION * 60),
        "substeps_per_frame": 4,
        "iterations": 20,
        "modes": 32,
        "cuda_graph": True,
        "solver_metrics": False,
        "geometry_diagnostics": False,
        "state_trace": False,
        "frame_readback": False,
        "timer": "synchronized wall clock per CUDA graph frame; all frames retained; one discarded full-run warmup",
        "profile_scope": "one separate uncaptured TIMING_ALL frame after each full trajectory",
        "runs": runs,
        "working_target_ms": 4.17,
        "target_met": bool(np.median(rows) <= 4.17),
        "runtime_commit": PIN,
        "runtime_path": str(Path(newton.__file__).resolve()),
        "source_hashes": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    }


def snapshot(output):
    source = output / "source"
    source.mkdir(parents=True)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source / path.name)
    shutil.copy2(ACCEPTED / "ATTRIBUTION.md", output / "ATTRIBUTION.md")
    shutil.copy2(ACCEPTED / "derived_rom.npz", output / "derived_rom.npz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("baseline", "benchmark", "audit", "validate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", nargs=5, type=float, default=DEPTHS)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    output = private_path(args.output)
    if output.exists():
        raise ValueError("Never overwrite evidence")
    if Path(newton.__file__).resolve() != RUNTIME / "newton/__init__.py":
        raise AssertionError("Wrong Newton runtime")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RUNTIME, text=True).strip()
    if revision != PIN:
        raise AssertionError("Runtime pin changed")
    if args.repeats < 1 or not all(0 < depth < 0.015 for depth in args.depths):
        raise ValueError("Positive repeats and bounded probe depths required")
    output.mkdir(parents=True)
    snapshot(output)
    provenance = {
        "runtime_commit": PIN,
        "runtime_path": str(Path(newton.__file__).resolve()),
        "derived_rom_sha256": digest(ACCEPTED / "derived_rom.npz"),
        "source_origin": str(ACCEPTED / "source"),
        "sequence": list(FINGERS),
        "starts_s": STARTS.tolist(),
        "duration_s": DURATION,
        "target_xz_m": TARGET_XZ.tolist(),
        "probe_half_extents_m": HALF.tolist(),
        "depths_m": args.depths,
        "original_thresholds_unchanged": True,
        "finite_principal_strain_gate": 0.15,
        "minimum_global_response_m": 0.003,
        "minimum_per_finger_causal_response_m": 0.003,
        "source_hashes": {p.name: digest(p) for p in (output / "source").glob("*.py")},
        "publication": "staged only; derived asset permission and limitations in ATTRIBUTION.md",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.action in ("baseline", "benchmark"):
        result = benchmark(args.action, args.repeats, args.depths)
    else:
        summaries = {}
        for name, enabled in (("pusher", True), ("no-pusher", False)):
            example = IndividualHand(depths=args.depths, enabled=enabled)
            seconds = example.run()
            summaries[name] = example.export(output / name) | {"audited_simulation_seconds": seconds}
            del example
        paired = paired_metrics(output / "pusher", output / "no-pusher")
        result = {
            "arms": summaries,
            "paired": paired,
            "test_passed": all(s["geometry_passed"] for s in summaries.values()) and paired["passed"],
        }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2), flush=True)
    if args.action in ("audit", "validate") and not result["test_passed"]:
        raise SystemExit(1)
    if args.action == "validate":
        benchmark_output = output / "benchmark"
        benchmark_output.mkdir()
        snapshot(benchmark_output)
        timings = benchmark("benchmark", args.repeats, args.depths)
        (benchmark_output / "results.json").write_text(json.dumps(timings, indent=2) + "\n")
        print(json.dumps({"benchmark_ms_per_frame": timings["ms_per_frame"]}), flush=True)


if __name__ == "__main__":
    main()
