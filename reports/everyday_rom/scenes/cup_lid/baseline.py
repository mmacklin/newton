# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Simulation-only timing of the pinned, unmodified supported cup example."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.basic import _reduced_elastic_cap as geometry
from newton.examples.basic import example_basic_reduced_elastic_cup_straw as supported

PIN = "8228bf6f690fff4dbb359d5de6447b09cb77c1e7"
FRAMES = 570
STAGE = Path("output/everyday-rom/cup-lid").resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def provenance():
    runtime = Path(newton.__file__).resolve().parent.parent
    head = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    runtime_tree = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD:newton"], text=True).strip()
    pinned_tree = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", f"{PIN}:newton"], text=True).strip()
    if runtime_tree != pinned_tree:
        raise RuntimeError(f"Expected unchanged Newton runtime from {PIN}; imported {runtime} at {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(runtime), "diff", "--name-only", "HEAD", "--", "newton"], text=True
    )
    if dirty.strip():
        raise RuntimeError(f"Runtime contains modifications: {dirty}")
    return {
        "runtime": str(runtime),
        "runtime_commit": head,
        "warp_version": wp.__version__,
        "supported_example_sha256": digest(supported.__file__),
        "supported_geometry_sha256": digest(geometry.__file__),
        "baseline_source_sha256": digest(__file__),
    }


def activities(timer, frames):
    groups = {}
    for entry in timer.timing_results:
        key = (entry.name, entry.filter)
        item = groups.setdefault(key, {"name": entry.name, "filter": int(entry.filter), "count": 0, "total_ms": 0.0})
        item["count"] += 1
        item["total_ms"] += entry.elapsed
    result = sorted(groups.values(), key=lambda item: item["total_ms"], reverse=True)
    for item in result:
        item["ms_per_frame"] = item["total_ms"] / frames
    return result


def snapshot(example):
    return {name: getattr(example.state_0, name).numpy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}


def benchmark(output, trials=3):
    output.mkdir(parents=True, exist_ok=True)
    result = provenance()
    wp.set_device("cuda:0")
    warmup = supported.Example(None)
    for _ in range(3):
        warmup.step()
    wp.synchronize()
    result.update(
        {
            "device": str(wp.get_device()),
            "device_name": wp.get_device().name,
            "config": asdict(warmup.cfg),
            "frames_per_trial": FRAMES,
            "timing_scope": "Continuous CUDA graph simulation only; synchronized ScopedTimer TIMING_ALL. No readbacks, metrics, trace kernels, surface recovery, or rendering inside timing scopes. Each trial starts from a fresh model/solver; no state restores.",
            "solver_metrics_enabled": False,
            "diagnostic_gpu_work": False,
            "trials": [],
            "kernel_profile": [],
        }
    )
    del warmup
    for trial in range(trials):
        example = supported.Example(None)
        assert example.graph is not None and not example.solver.elastic_solver_metrics
        assert not example.metrics_enabled
        with wp.ScopedTimer("cup_lid_graph", cuda_filter=wp.TIMING_ALL, synchronize=True, print=False) as timer:
            for _ in range(FRAMES):
                wp.capture_launch(example.graph)
        items = activities(timer, FRAMES)
        row = {
            "trial": trial,
            "wall_ms_per_frame": timer.elapsed / FRAMES,
            "cuda_ms_per_frame": sum(item["ms_per_frame"] for item in items),
            "activities": items,
        }
        result["trials"].append(row)
        np.savez_compressed(output / f"graph-final-{trial}.npz", **snapshot(example))
        print(json.dumps(row), flush=True)
        del example
    result["median_graph_wall_ms_per_frame"] = float(np.median([r["wall_ms_per_frame"] for r in result["trials"]]))
    result["median_graph_cuda_ms_per_frame"] = float(np.median([r["cuda_ms_per_frame"] for r in result["trials"]]))
    write_json(output / "performance.json", result)

    # Keep two independent continuous histories for graph and uncaptured attribution.
    graph, kernels = supported.Example(None), supported.Example(None)
    completed = 0
    saved = {}
    for label, start in (
        ("approach", 30),
        ("insert", 75),
        ("hold", 110),
        ("stir", 240),
        ("withdraw", 470),
        ("released", 550),
    ):
        for _ in range(start - completed):
            wp.capture_launch(graph.graph)
            wp.capture_launch(kernels.graph)
        repetitions = 8
        with wp.ScopedTimer(label + "_graph", cuda_filter=wp.TIMING_ALL, synchronize=True, print=False) as timer:
            for _ in range(repetitions):
                wp.capture_launch(graph.graph)
        graph_items = activities(timer, repetitions)
        row = {
            "phase": label,
            "start_frame": start,
            "frames": repetitions,
            "graph_wall_ms_per_frame": timer.elapsed / repetitions,
            "graph_cuda_ms_per_frame": sum(x["ms_per_frame"] for x in graph_items),
            "graph_activities": graph_items,
        }
        with wp.ScopedTimer(label + "_kernels", cuda_filter=wp.TIMING_ALL, synchronize=True, print=False) as timer:
            for _ in range(repetitions):
                kernels.simulate()
        row["uncaptured_wall_ms_per_frame"] = timer.elapsed / repetitions
        row["activities"] = activities(timer, repetitions)
        row["uncaptured_cuda_activity_ms_per_frame"] = sum(x["ms_per_frame"] for x in row["activities"])
        completed = start + repetitions
        a, b = snapshot(graph), snapshot(kernels)
        row["graph_vs_uncaptured_max_abs"] = {k: float(np.max(np.abs(a[k] - b[k]))) for k in a}
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=2e-6, rtol=2e-5)
            saved[f"{label}_{key}"] = a[key]
        row["matched_state_check_passed"] = True
        result["kernel_profile"].append(row)
        write_json(output / "performance.json", result)
        print(json.dumps({k: v for k, v in row.items() if "activities" not in k}), flush=True)
    np.savez_compressed(output / "profile-states.npz", **saved)
    result["passed"] = True
    write_json(output / "performance.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STAGE / "baseline")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    benchmark(args.output, args.trials)
