# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import warp as wp

import newton.viewer
from newton.examples.basic.example_basic_reduced_elastic_chair_stick_slip import Example as ChairExample
from newton.examples.basic.example_basic_reduced_elastic_dipper import Example as DipperExample
from newton.examples.basic.example_basic_reduced_elastic_gripper_contact import Example as GripperExample
from newton.examples.basic.example_basic_reduced_elastic_scraper_contact import Example as ScraperExample
from newton.examples.basic.example_basic_reduced_elastic_wall_contact import Example as WallExample

CASE_DEFS = {
    "dipper": (DipperExample, 120),
    "wall": (WallExample, 90),
    "gripper": (GripperExample, 120),
    "scraper": (ScraperExample, 120),
    "chair": (ChairExample, 240),
}

FIELDNAMES = (
    "case",
    "iterations",
    "substeps",
    "sim_dt",
    "frames",
    "passed",
    "error",
    "final_residual",
    "final_residual_ratio",
    "max_residual_ratio",
    "final_update_norm",
    "final_update_max",
    "max_update_norm",
    "max_update_max",
    "deformation",
    "contact_dropouts",
    "chatter",
    "visual_stability",
)


def _iteration_example(base_cls, iterations: int):
    class IterationExample(base_cls):
        solver_iterations = iterations

    IterationExample.__name__ = f"{base_cls.__name__}Iterations{iterations}"
    return IterationExample


def _attr(example: Any, name: str, default: float = 0.0) -> float:
    return float(getattr(example, name, default))


def _case_metrics(name: str, example: Any) -> dict[str, float]:
    if name == "dipper":
        return {
            "deformation": _attr(example, "max_tip_bend"),
            "contact_dropouts": 0.0,
            "chatter": _attr(example, "max_mode_accel"),
            "visual_stability": _attr(example, "max_mode_step"),
        }
    if name == "wall":
        return {
            "deformation": _attr(example, "max_compression"),
            "contact_dropouts": _attr(example, "settled_contact_dropouts"),
            "chatter": _attr(example, "settled_modal_accel_max"),
            "visual_stability": _attr(example, "settled_modal_step_max"),
        }
    if name == "gripper":
        settled_rel_z_range = _attr(example, "settled_rel_z_max") - _attr(example, "settled_rel_z_min")
        settled_x_range = _attr(example, "settled_part_x_max") - _attr(example, "settled_part_x_min")
        return {
            "deformation": min(_attr(example, "max_left_compression"), _attr(example, "max_right_compression")),
            "contact_dropouts": _attr(example, "settled_contact_dropouts"),
            "chatter": max(
                _attr(example, "max_settled_horizontal_speed"), _attr(example, "max_settled_vertical_speed")
            ),
            "visual_stability": max(settled_rel_z_range, settled_x_range),
        }
    if name == "scraper":
        return {
            "deformation": max(_attr(example, "max_vertical_compression"), _attr(example, "max_lateral_bend")),
            "contact_dropouts": _attr(example, "contact_dropouts_after_settle"),
            "chatter": _attr(example, "settled_modal_accel_max"),
            "visual_stability": _attr(example, "settled_modal_step_max"),
        }
    if name == "chair":
        normal_mean = example.contact_normal_speed_sum / max(example.contact_normal_speed_count, 1)
        return {
            "deformation": _attr(example, "max_displacement"),
            "contact_dropouts": _attr(example, "contact_dropouts_after_settle"),
            "chatter": float(normal_mean),
            "visual_stability": _attr(example, "max_contact_normal_speed"),
        }
    raise ValueError(f"unknown case {name!r}")


def run_case(name: str, base_cls, frames: int, iterations: int, substeps: int | None = None) -> dict[str, Any]:
    cls = _iteration_example(base_cls, iterations)
    viewer = newton.viewer.ViewerNull()
    example = cls(viewer, None)
    if substeps is not None:
        set_example_substeps(example, substeps)
    error = ""
    passed = True

    try:
        for _ in range(frames):
            example.step()
            example.render()
        example.test_final()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        passed = False
    finally:
        if hasattr(viewer, "close"):
            viewer.close()

    row = {
        "case": name,
        "iterations": iterations,
        "substeps": int(example.sim_substeps),
        "sim_dt": float(example.sim_dt),
        "frames": frames,
        "passed": int(passed),
        "error": error,
        "final_residual": _attr(example, "final_modal_solve_residual_norm"),
        "final_residual_ratio": _attr(example, "final_modal_solve_residual_ratio"),
        "max_residual_ratio": _attr(example, "max_modal_solve_residual_ratio"),
        "final_update_norm": _attr(example, "final_modal_update_norm"),
        "final_update_max": _attr(example, "final_modal_update_max"),
        "max_update_norm": _attr(example, "max_modal_update_norm"),
        "max_update_max": _attr(example, "max_modal_update_max"),
    }
    row.update(_case_metrics(name, example))
    return row


def set_example_substeps(example: Any, substeps: int):
    if substeps <= 0:
        raise ValueError(f"substeps must be positive, got {substeps}")
    example.sim_substeps = substeps
    example.sim_dt = example.frame_dt / substeps


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep reduced elastic example VBD iteration counts.")
    parser.add_argument("--device", default="cuda:0", help="Warp device to run on.")
    parser.add_argument("--cases", nargs="+", default=list(CASE_DEFS), choices=list(CASE_DEFS))
    parser.add_argument("--iterations", nargs="+", type=int, default=[8, 12, 16, 22, 32, 48, 72])
    parser.add_argument(
        "--substeps",
        nargs="+",
        type=int,
        default=None,
        help="Override simulation substeps per rendered frame. Defaults to each example's configured value.",
    )
    parser.add_argument("--output", default="reports/assets/reduced_elastic_iteration_sweep.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file, wp.ScopedDevice(args.device):
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        print(",".join(FIELDNAMES), flush=True)

        for case in args.cases:
            base_cls, frames = CASE_DEFS[case]
            for iterations in args.iterations:
                substeps_values = args.substeps if args.substeps is not None else [None]
                for substeps in substeps_values:
                    row = run_case(case, base_cls, frames, iterations, substeps)
                    writer.writerow(row)
                    file.flush()
                    print(",".join(str(row[name]) for name in FIELDNAMES), flush=True)


if __name__ == "__main__":
    main()
