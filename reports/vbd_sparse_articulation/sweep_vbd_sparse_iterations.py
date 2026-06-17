# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Residual sweep for VBD rigid articulation iterations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import warp as wp
from bench_vbd_sparse_articulation import (
    build_model,
    compute_contact_penetration,
    compute_joint_residual,
    make_solver,
    perturb_body_transforms,
)

import newton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scenarios", nargs="+", default=["chain_fixed", "chain_revolute", "loop_fixed"])
    parser.add_argument("--body-counts", nargs="+", type=int, default=[4, 32])
    parser.add_argument("--iterations", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--modes", nargs="+", default=["local", "block_sparse_joints"])
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--perturb", type=float, default=2.0e-2)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args()


def run_residual_case(args: argparse.Namespace, scenario: str, body_count: int, mode: str, iterations: int) -> dict:
    wp_device = wp.get_device(args.device)
    with wp.ScopedDevice(wp_device):
        model = build_model(scenario, body_count, args.device)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()

        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        if args.perturb > 0.0 and scenario != "contact_stack":
            wp.launch(
                kernel=perturb_body_transforms,
                dim=model.body_count,
                inputs=[state_0.body_q, model.body_inv_mass, args.perturb],
                device=wp_device,
            )

        solver = make_solver(model, mode, iterations)
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, args.dt)
        wp.synchronize_device()

        joint_l2, joint_max = compute_joint_residual(model, state_1)
        return {
            "scenario": scenario,
            "body_count": body_count,
            "mode": mode,
            "iterations": iterations,
            "joint_residual_l2": joint_l2,
            "joint_residual_max": joint_max,
            "contact_penetration_max": compute_contact_penetration(contacts),
        }


def add_comparisons(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, int, int], dict[str, dict]] = {}
    for row in rows:
        key = (row["scenario"], row["body_count"], row["iterations"])
        by_key.setdefault(key, {})[row["mode"]] = row

    for modes in by_key.values():
        baseline = modes.get("local")
        if baseline is None:
            continue
        local_residual = float(baseline["joint_residual_l2"])
        for row in modes.values():
            denom = max(float(row["joint_residual_l2"]), 1.0e-12)
            row["residual_reduction_vs_local"] = local_residual / denom
    return rows


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "| Scenario | Bodies | Iterations | Local residual | Sparse residual | Residual reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_key: dict[tuple[str, int, int], dict[str, dict]] = {}
    for row in rows:
        key = (row["scenario"], row["body_count"], row["iterations"])
        by_key.setdefault(key, {})[row["mode"]] = row

    for key in sorted(by_key):
        modes = by_key[key]
        local = modes.get("local")
        sparse = modes.get("block_sparse_joints")
        if local is None or sparse is None:
            continue
        scenario, body_count, iterations = key
        lines.append(
            f"| {scenario} | {body_count} | {iterations} | "
            f"{local['joint_residual_l2']:.6g} | {sparse['joint_residual_l2']:.6g} | "
            f"{sparse['residual_reduction_vs_local']:.3g} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    rows = []
    for scenario in args.scenarios:
        for body_count in args.body_counts:
            for iterations in args.iterations:
                for mode in args.modes:
                    rows.append(run_residual_case(args, scenario, body_count, mode, iterations))
    rows = add_comparisons(rows)

    payload = {"device": args.device, "rows": rows}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.markdown is not None:
        write_markdown(rows, args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
