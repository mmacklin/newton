# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fresh 450-frame generic fused positive/control/repeat qualification, not timing."""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import warp as wp
from brush_glove_geometry import Config
from fused_toothbrush import Example
from verify_fused import STATE_KEYS, load, verify_pair


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def run_arm(folder, enabled):
    """Save actual state before metrics/assertions; never measure frame time."""
    folder.mkdir()
    example = Example(cfg=Config(substeps=4, iterations=16, elastic_relaxation=0.4, contact_enabled=enabled))
    example.source_files = (
        *example.source_files,
        "verify_fused.py",
        "qualify_fused.py",
        "test_toothbrush_scrub.py",
        "test_fused_trajectory.py",
    )
    model = {key: value.numpy().copy() for key, value in vars(example.model).items() if isinstance(value, wp.array)}
    np.savez_compressed(folder / "model.npz", **model)
    del model
    settled = {key: getattr(example.state_0, key).numpy().copy() for key in STATE_KEYS if key != "joint_enabled"}
    settled["joint_enabled"] = example.model.joint_enabled.numpy().copy()
    settled["equilibrium_joint_q"] = example.equilibrium_joint_q.copy()
    np.savez_compressed(folder / "settled.npz", **settled)
    saved = {
        key: np.lib.format.open_memmap(
            folder / f"raw-{key}.npy", mode="w+", dtype=settled[key].dtype, shape=(450, *settled[key].shape)
        )
        for key in STATE_KEYS
    }
    residuals = {key: [] for key in example.solver.elastic_mode_solve_metrics()}
    contacts, offsets = {"shape0": [], "shape1": []}, [0]
    error = None
    try:
        for frame in range(450):
            wp.capture_launch(example.graph)
            wp.synchronize()
            for key in STATE_KEYS:
                saved[key][frame] = (
                    example.model.joint_enabled.numpy()
                    if key == "joint_enabled"
                    else getattr(example.state_0, key).numpy()
                )
                saved[key].flush()
            for key, value in example.solver.elastic_mode_solve_metrics().items():
                residuals[key].append(value)
            count = int(example.contacts.rigid_contact_count.numpy()[0])
            offsets.append(offsets[-1] + min(count, example.contacts.rigid_contact_max))
            for key, values in contacts.items():
                values.append(getattr(example.contacts, "rigid_contact_" + key).numpy()[:count].copy())
            write(folder / "progress.json", {"saved_frames": frame + 1, "metrics_frames": len(example.records)})
            example.sim_time += 1 / 60
            example.timings.append(0.0)
            example.update_metrics()
            if (frame + 1) % 90 == 0:
                print(folder.name, frame + 1, "frames saved", flush=True)
    except Exception:
        error = traceback.format_exc()
    finally:
        np.savez_compressed(folder / "residuals.npz", **{key: np.asarray(values) for key, values in residuals.items()})
        np.savez_compressed(
            folder / "frame_contacts.npz",
            offsets=np.asarray(offsets),
            **{key: np.concatenate(values) if values else np.empty(0, np.int32) for key, values in contacts.items()},
        )
        example.capacity_audit.export(folder, 2040)
        example.field_audit.export(folder)
        write(folder / "records.json", example.records)
        if error:
            write(folder / "failure.json", {"passed": False, "traceback": error})
        if example.records:
            example.export(folder, "unqualified public-generic physical diagnostic", test_passed=False)
    if error:
        raise RuntimeError("Saved failing trajectory; no retry or later arm")
    # Exported arrays already exist even when either assertion path fails.
    example.test_final()
    write(
        folder / "runtime-gates.json",
        {"passed": True, "frames": 450, "private_dispatcher": False, "timing_claim": False},
    )
    del example
    return {"runtime_gates_passed": True}


def collect(output):
    output = Path(output)
    assert not output.exists(), "Use a new qualification directory"
    output.mkdir(parents=True)
    wp.set_device("cuda:0")
    write(
        output / "device.json",
        {
            "name": wp.get_device().name,
            "arch": wp.get_device().arch,
            "scope": "physical qualification only, no timing or renderer",
        },
    )
    try:
        run_arm(output / "positive", True)
        run_arm(output / "control", False)
        run_arm(output / "repeat", True)
        write(output / "collection.json", {"collected": True, "independent_audits_pending": True, "passed": None})
    except Exception:
        write(
            output / "collection.json",
            {"collected": False, "passed": False, "traceback": traceback.format_exc(), "automatic_retry": False},
        )
        raise


def audit(output):
    """Run the CPU audits after releasing the GPU lock; no new simulation."""
    output = Path(output)
    try:
        pair = verify_pair(output / "positive", output / "control", output / "pair-audit")
        repeated_pair = verify_pair(output / "repeat", output / "control", output / "repeat-pair-audit")
        a, b = (load(output / arm / "state.npz") for arm in ("positive", "repeat"))
        differences = {
            key: {
                "bitwise_equal": bool(np.array_equal(a[key], b[key])),
                "max_abs_difference": float(np.max(np.abs(a[key].astype(float) - b[key].astype(float)))),
            }
            for key in STATE_KEYS
        }
        result = {
            "passed": True,
            "pair": pair,
            "repeat_pair": repeated_pair,
            "repeat_comparison_diagnostic": differences,
            "private_trajectory_comparison": "not measured; not a gate",
            "performance_or_render_authorized": False,
        }
        write(output / "result.json", result)
        return result
    except Exception:
        write(
            output / "result.json",
            {
                "passed": False,
                "traceback": traceback.format_exc(),
                "automatic_retry": False,
                "timing_or_render_performed": False,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--collect-only",
        action="store_true",
        help="Save trajectories and live gates; independent CPU audit remains required",
    )
    mode.add_argument(
        "--verify-only", action="store_true", help="Audit an existing collection on CPU, without CUDA execution"
    )
    arguments = parser.parse_args()
    if not arguments.verify_only:
        collect(arguments.output)
    if not arguments.collect_only:
        audit(arguments.output)
