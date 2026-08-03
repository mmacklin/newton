#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sweep Kamino DVI contact settings on the trained DR Legs policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch  # noqa: TID253

import newton
from reports.vbd_complex_linkages import bench_dr_legs_policy as benchmark

VARIANTS = {
    "baseline_c2": {
        "kamino_dvi_contact_iterations": 2,
        "kamino_dvi_contact_block_preconditioner": False,
        "kamino_dvi_contact_stabilization": 0.015,
    },
    "baseline_c4": {
        "kamino_dvi_contact_iterations": 4,
        "kamino_dvi_contact_block_preconditioner": False,
        "kamino_dvi_contact_stabilization": 0.015,
    },
    "baseline_c8": {
        "kamino_dvi_contact_iterations": 8,
        "kamino_dvi_contact_block_preconditioner": False,
        "kamino_dvi_contact_stabilization": 0.015,
    },
    "block_preconditioner_c4": {
        "kamino_dvi_contact_iterations": 4,
        "kamino_dvi_contact_block_preconditioner": True,
        "kamino_dvi_contact_stabilization": 0.015,
    },
    "gamma05_preconditioner_c4": {
        "kamino_dvi_contact_iterations": 4,
        "kamino_dvi_contact_block_preconditioner": True,
        "kamino_dvi_contact_stabilization": 0.05,
    },
    "gamma10_preconditioner_c4": {
        "kamino_dvi_contact_iterations": 4,
        "kamino_dvi_contact_block_preconditioner": True,
        "kamino_dvi_contact_stabilization": 0.1,
    },
    "gamma20_preconditioner_c4": {
        "kamino_dvi_contact_iterations": 4,
        "kamino_dvi_contact_block_preconditioner": True,
        "kamino_dvi_contact_stabilization": 0.2,
    },
    "gamma10_preconditioner_c5": {
        "kamino_dvi_contact_iterations": 5,
        "kamino_dvi_contact_block_preconditioner": True,
        "kamino_dvi_contact_stabilization": 0.1,
    },
    "gamma10_preconditioner_c6": {
        "kamino_dvi_contact_iterations": 6,
        "kamino_dvi_contact_block_preconditioner": True,
        "kamino_dvi_contact_stabilization": 0.1,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--control-steps", type=int, default=400)
    parser.add_argument("--stand-steps", type=int, default=50)
    parser.add_argument("--forward-speed", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/vbd_complex_linkages/dr_legs_dvi_contact_sweep.json"),
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    asset_path = newton.utils.download_asset("disneyresearch")
    config = benchmark._load_config(asset_path)
    policy_path = asset_path / "dr_legs" / "rl_policies" / config.policy_file
    policy = benchmark.load_policy(policy_path)
    base_spec = benchmark.CASE_CONFIGS["kamino_dvi"]
    rows = []
    for variant in args.variants:
        case = f"kamino_dvi_{variant}"
        benchmark.CASE_CONFIGS[case] = replace(base_spec, label=f"Kamino DVI {variant}", **VARIANTS[variant])
        print(f"Running {case}", flush=True)
        rows.append(
            benchmark.run_case(
                case,
                policy,
                config,
                control_steps=args.control_steps,
                stand_steps=args.stand_steps,
                forward_speed=args.forward_speed,
            )
        )
        print(json.dumps(rows[-1], indent=2, sort_keys=True), flush=True)

    payload = {
        "source": "Newton Disney Research asset package",
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "control_steps": args.control_steps,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
