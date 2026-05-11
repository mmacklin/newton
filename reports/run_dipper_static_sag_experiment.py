# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.basic.example_basic_reduced_elastic_dipper import Example as DipperExample

SAMPLE_TIMES_SHORT = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0)
SAMPLE_TIMES_LONG = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 45.0, 60.0)


@dataclass
class Variant:
    experiment: str
    label: str
    duration: float
    iterations: int = 36
    substeps: int = 1
    gravity_scale: float = 1.0
    primary_stiffness: float | None = None
    primary_damping: float | None = None
    primary_mass: float | None = None
    joint_adaptive_stiffness: bool = False
    joint_linear_ke: float = 1.0e5
    joint_angular_ke: float = 1.0e5
    joint_linear_k_start: float = 1.0e5
    joint_angular_k_start: float = 1.0e5


@dataclass
class Sample:
    experiment: str
    label: str
    frame: int
    time: float
    joint_linear_ke: float
    bend: float
    q1: float
    q2: float
    current_residual: float
    max_residual: float
    drive_stroke: float


def _set_mode_value(array: wp.array, mode_index: int, value: float):
    values = array.numpy()
    values[mode_index] = value
    array.assign(values)


def _configure_example(example: DipperExample, variant: Variant):
    example.drive_amplitude = 0.0
    example.sim_substeps = variant.substeps
    example.sim_dt = example.frame_dt / float(variant.substeps)
    example.solver = newton.solvers.SolverVBD(
        example.model,
        iterations=variant.iterations,
        rigid_joint_linear_k_start=variant.joint_linear_k_start,
        rigid_joint_angular_k_start=variant.joint_angular_k_start,
        rigid_joint_linear_ke=variant.joint_linear_ke,
        rigid_joint_angular_ke=variant.joint_angular_ke,
        rigid_joint_adaptive_stiffness=variant.joint_adaptive_stiffness,
    )

    if variant.gravity_scale != 1.0:
        example.model.set_gravity((0.0, 0.0, -example.gravity * variant.gravity_scale))

    mode_start = int(example.model.elastic_mode_start.numpy()[0])
    primary_mode = mode_start + 1
    if variant.primary_stiffness is not None:
        _set_mode_value(example.model.elastic_mode_stiffness, primary_mode, variant.primary_stiffness)
    if variant.primary_damping is not None:
        _set_mode_value(example.model.elastic_mode_damping, primary_mode, variant.primary_damping)
    if variant.primary_mass is not None:
        _set_mode_value(example.model.elastic_mode_mass, primary_mode, variant.primary_mass)


def _current_residual(example: DipperExample) -> float:
    return max(example._joint_residuals())


def run_variant(variant: Variant) -> list[Sample]:
    viewer = newton.viewer.ViewerNull()
    example = DipperExample(viewer, None)
    _configure_example(example, variant)

    sample_times = SAMPLE_TIMES_LONG if variant.duration > 10.0 else SAMPLE_TIMES_SHORT
    sample_frames = {int(round(t * example.fps)) for t in sample_times if t <= variant.duration + 1.0e-6}
    max_frame = int(round(variant.duration * example.fps))
    sample_frames.add(max_frame)
    samples: list[Sample] = []

    for frame in range(max_frame + 1):
        if frame > 0:
            example.step()

        if frame in sample_frames:
            q = example._mode_values()
            samples.append(
                Sample(
                    experiment=variant.experiment,
                    label=variant.label,
                    frame=frame,
                    time=frame / example.fps,
                    joint_linear_ke=variant.joint_linear_ke,
                    bend=example._tip_bend(),
                    q1=float(q[1]),
                    q2=float(q[2]),
                    current_residual=_current_residual(example),
                    max_residual=example.max_joint_residual,
                    drive_stroke=example._drive_stroke(),
                )
            )

    return samples


def _by_variant(samples: Iterable[Sample]) -> dict[tuple[str, str], list[Sample]]:
    grouped: dict[tuple[str, str], list[Sample]] = {}
    for sample in samples:
        grouped.setdefault((sample.experiment, sample.label), []).append(sample)
    for rows in grouped.values():
        rows.sort(key=lambda s: s.time)
    return grouped


def fit_exponential(rows: list[Sample], value_name: str = "bend") -> dict[str, float | None]:
    times = np.array([s.time for s in rows], dtype=float)
    values = np.array([getattr(s, value_name) for s in rows], dtype=float)
    mask = (times > 0.0) & (values > 0.0)
    times = times[mask]
    values = values[mask]
    if times.shape[0] < 3:
        return {"q_inf": None, "tau": None, "mse": None}

    y_max = float(np.max(values))
    if y_max <= 1.0e-6:
        return {"q_inf": 0.0, "tau": None, "mse": 0.0}

    q_min = y_max * 1.001
    q_max = max(y_max * 20.0, q_min + 1.0e-5)
    candidates = np.geomspace(q_min, q_max, 1200)

    best: tuple[float, float, float] | None = None
    for q_inf in candidates:
        y = 1.0 - values / q_inf
        if np.any(y <= 0.0):
            continue
        denom = float(np.dot(times, times))
        if denom <= 0.0:
            continue
        slope = float(np.dot(times, np.log(y)) / denom)
        if slope >= 0.0:
            continue
        tau = -1.0 / slope
        pred = q_inf * (1.0 - np.exp(-times / tau))
        mse = float(np.mean((pred - values) ** 2))
        if best is None or mse < best[0]:
            best = (mse, float(q_inf), float(tau))

    if best is None:
        return {"q_inf": None, "tau": None, "mse": None}
    return {"mse": best[0], "q_inf": best[1], "tau": best[2]}


def _sample_at(rows: list[Sample], time: float) -> Sample | None:
    for row in rows:
        if abs(row.time - time) < 1.0e-6:
            return row
    return None


def _last_rate(rows: list[Sample], value_name: str = "bend") -> float | None:
    if len(rows) < 2:
        return None
    prev = rows[-2]
    last = rows[-1]
    dt = last.time - prev.time
    if dt <= 0.0:
        return None
    return float((getattr(last, value_name) - getattr(prev, value_name)) / dt)


def summarize(samples: list[Sample]) -> list[dict[str, object]]:
    summary = []
    for (experiment, label), rows in _by_variant(samples).items():
        fit = fit_exponential(rows)
        row_1s = _sample_at(rows, 1.0)
        row_4s = _sample_at(rows, 4.0)
        row_10s = _sample_at(rows, 10.0)
        row_30s = _sample_at(rows, 30.0)
        row_60s = _sample_at(rows, 60.0)
        final = rows[-1]
        reaction_proxy_final = final.joint_linear_ke * final.current_residual
        summary.append(
            {
                "experiment": experiment,
                "label": label,
                "bend_1s": row_1s.bend if row_1s else None,
                "bend_4s": row_4s.bend if row_4s else None,
                "bend_10s": row_10s.bend if row_10s else None,
                "bend_30s": row_30s.bend if row_30s else None,
                "bend_60s": row_60s.bend if row_60s else None,
                "residual_1s": row_1s.current_residual if row_1s else None,
                "residual_4s": row_4s.current_residual if row_4s else None,
                "reaction_proxy_1s": row_1s.joint_linear_ke * row_1s.current_residual if row_1s else None,
                "reaction_proxy_4s": row_4s.joint_linear_ke * row_4s.current_residual if row_4s else None,
                "final_time": final.time,
                "bend_final": final.bend,
                "bend_rate_final": _last_rate(rows),
                "q1_final": final.q1,
                "q2_final": final.q2,
                "joint_linear_ke": final.joint_linear_ke,
                "reaction_proxy_final": reaction_proxy_final,
                "q_inf_fit": fit["q_inf"],
                "tau_fit": fit["tau"],
                "mse_fit": fit["mse"],
                "last_residual": rows[-1].current_residual,
            }
        )
    return summary


def _format_meters(value: object) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f} cm"


def _format_float(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _format_rate(value: object) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.3f} cm/s"


def _format_force(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f} N"


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def svg_plot(title: str, grouped_rows: dict[str, list[Sample]], value_name: str = "bend") -> str:
    width = 820
    height = 360
    left = 60
    right = 190
    top = 32
    bottom = 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2", "#db2777", "#475569"]

    max_time = max((row.time for rows in grouped_rows.values() for row in rows), default=1.0)
    max_value = max((getattr(row, value_name) for rows in grouped_rows.values() for row in rows), default=1.0)
    max_value = max(max_value, 1.0e-6)

    def sx(t: float) -> float:
        return left + (t / max_time) * plot_w

    def sy(v: float) -> float:
        return top + plot_h - (v / max_value) * plot_h

    value_label = "joint error [m]" if value_name == "current_residual" else "bend [m]"
    lines = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<text x="{left}" y="20" class="plot-title">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
        f'<text x="{left + plot_w - 24}" y="{height - 12}" class="tick">time [s]</text>',
        f'<text x="8" y="{top + 12}" class="tick">{value_label}</text>',
    ]

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + frac * plot_w
        t = frac * max_time
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>')
        lines.append(f'<text x="{x - 10:.1f}" y="{top + plot_h + 18}" class="tick">{t:.0f}</text>')
        y = top + plot_h - frac * plot_h
        v = frac * max_value
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 54}" y="{y + 4:.1f}" class="tick">{v:.3f}</text>')

    for idx, (label, rows) in enumerate(grouped_rows.items()):
        color = colors[idx % len(colors)]
        points = [(sx(row.time), sy(float(getattr(row, value_name)))) for row in rows]
        lines.append(f'<polyline points="{_polyline(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_y = top + 18 + idx * 20
        lines.append(
            f'<line x1="{left + plot_w + 22}" y1="{legend_y - 5}" x2="{left + plot_w + 44}" y2="{legend_y - 5}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(f'<text x="{left + plot_w + 52}" y="{legend_y}" class="legend">{escape(label)}</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "\n".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_csv(path: Path, samples: list[Sample]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "experiment",
                "label",
                "frame",
                "time_s",
                "joint_linear_ke",
                "bend_m",
                "q1_m",
                "q2_m",
                "current_residual_m",
                "max_residual_m",
                "drive_stroke_m",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.experiment,
                    sample.label,
                    sample.frame,
                    f"{sample.time:.8f}",
                    f"{sample.joint_linear_ke:.8f}",
                    f"{sample.bend:.10f}",
                    f"{sample.q1:.10f}",
                    f"{sample.q2:.10f}",
                    f"{sample.current_residual:.10f}",
                    f"{sample.max_residual:.10f}",
                    f"{sample.drive_stroke:.10f}",
                ]
            )


def write_report(path: Path, samples: list[Sample], summary: list[dict[str, object]], cache_key: str):
    by_exp: dict[str, dict[str, list[Sample]]] = {}
    for (experiment, label), rows in _by_variant(samples).items():
        by_exp.setdefault(experiment, {})[label] = rows

    def summary_rows(experiment: str) -> list[list[str]]:
        rows = []
        for row in summary:
            if row["experiment"] != experiment:
                continue
            rows.append(
                [
                    f"<code>{escape(str(row['label']))}</code>",
                    _format_meters(row["bend_1s"]),
                    _format_meters(row["bend_4s"]),
                    _format_meters(row["bend_10s"]),
                    _format_meters(row["bend_30s"]),
                    _format_meters(row["bend_60s"]),
                    f"{_format_meters(row['bend_final'])} @ {_format_float(row['final_time'], 0)}s",
                    _format_float(row["q1_final"], 4),
                    _format_float(row["q2_final"], 4),
                    _format_force(row["reaction_proxy_final"]),
                    _format_float(row["q_inf_fit"], 3),
                    _format_float(row["tau_fit"], 1),
                    _format_rate(row["bend_rate_final"]),
                    _format_float(row["last_residual"], 5),
                ]
            )
        return rows

    sections = []
    for experiment, label in [
        ("control", "Controls"),
        ("iterations_adaptive", "Adaptive-Penalty Iteration Sweep"),
        ("iterations_fixed", "Fixed-Penalty Iteration Sweep"),
        ("joint_stiffness", "Fixed Joint Stiffness Sweep"),
        ("substeps", "Timestep/Substep Sweep"),
        ("stiffness", "Primary Bending Stiffness Sweep"),
        ("damping", "Primary Bending Damping Sweep"),
        ("modal_mass", "Primary Modal Mass Sweep"),
    ]:
        if experiment not in by_exp:
            continue
        sections.append(f"<h2>{escape(label)}</h2>")
        sections.append(svg_plot(f"{label}: static sag", by_exp[experiment]))
        if experiment == "joint_stiffness":
            sections.append(svg_plot(f"{label}: joint error", by_exp[experiment], value_name="current_residual"))
        sections.append(
            _table(
                [
                    "variant",
                    "bend 1s",
                    "bend 4s",
                    "bend 10s",
                    "bend 30s",
                    "bend 60s",
                    "final bend",
                    "q1 final",
                    "q2 final",
                    "k*C final",
                    "fit q_inf",
                    "fit tau [s]",
                    "final rate",
                    "residual",
                ],
                summary_rows(experiment),
            )
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dipper Static Sag Experiment</title>
  <style>
    body {{ margin: 0; font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #0f172a; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px; }}
    h1 {{ font-size: 32px; margin: 0 0 8px; }}
    h2 {{ margin-top: 34px; font-size: 22px; }}
    p {{ max-width: 900px; }}
    code {{ background: #e2e8f0; border-radius: 4px; padding: 1px 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 24px; background: white; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #e2e8f0; }}
    svg {{ width: 100%; max-width: 980px; display: block; margin: 10px 0 12px; background: white; border: 1px solid #cbd5e1; }}
    .axis {{ stroke: #0f172a; stroke-width: 1.3; }}
    .grid {{ stroke: #cbd5e1; stroke-width: 0.8; }}
    .plot-title {{ font-size: 15px; font-weight: 600; fill: #0f172a; }}
    .tick, .legend {{ font-size: 12px; fill: #334155; }}
    .note {{ background: #e0f2fe; border-left: 4px solid #0284c7; padding: 12px 14px; max-width: 920px; }}
  </style>
</head>
<body>
<main>
  <h1>Dipper Static Sag Experiment</h1>
  <p>Generated {cache_key}. The piston is held at zero stroke in all runs. The measurement is the distance between the elastic tip attachment and the same point on the undeformed floating frame, so it isolates modal bending from rigid-frame motion.</p>
  <h2>Question</h2>
  <p>The observed dipper sag grows slowly even when the piston is fixed. This experiment checks whether that is caused by timestep size, modal mass/damping, beam stiffness, adaptive joint penalties, or convergence of the split rigid/elastic VBD coupling.</p>
  <h2>Method</h2>
  <ul>
    <li>Freeze the prismatic actuator at the horizontal configuration.</li>
    <li>Measure only modal tip bend, not rigid-frame rotation.</li>
    <li>Use a zero-gravity control to rule out numerical drift without load.</li>
    <li>Sweep VBD iterations with normal adaptive joint penalties and with joint penalties pinned to their caps.</li>
    <li>Sweep fixed joint stiffness at high iteration count to compare sag, joint error, and the penalty-reaction proxy <code>k*C</code>.</li>
    <li>Sweep substeps, primary bending stiffness, modal damping, and modal mass.</li>
    <li>Fit a first-order response to measured tip bend only as a compact rate diagnostic; the fit is not assumed to be the governing model.</li>
  </ul>
  <div class="note">
    <p><strong>Finding:</strong> the sag disappears with zero gravity and is nearly insensitive to substepping, so this is not free numerical drift or a timestep artifact. Modal mass and damping have little effect on the early response, so it is not primarily a modal oscillator settling slowly. The iteration sweeps are non-monotonic: with adaptive penalties the sag peaks around the mid-iteration range, while with fixed penalties high iteration counts drive the residual down and almost eliminate modal deflection. The fixed joint stiffness sweep shows the important failure mode: joint error drops roughly with stiffness and <code>k*C</code> stays in the same broad force range, but sag changes dramatically. This points to the current residual-driven split rigid/elastic coupling as the dominant issue: the elastic update is seeing constraint residual leakage, not the true converged joint reaction.</p>
  </div>
  <p>Raw data: <a href="assets/dipper_static_sag_experiment.csv?datetime={cache_key}">CSV</a> and <a href="assets/dipper_static_sag_summary.json?datetime={cache_key}">JSON summary</a>.</p>
  {"".join(sections)}
</main>
</body>
</html>
"""
    path.write_text(html)


def build_variants() -> list[Variant]:
    iteration_counts = (4, 8, 12, 18, 24, 36, 48, 72, 96, 144)
    variants: list[Variant] = [
        Variant("control", "gravity 1g", duration=60.0),
        Variant("control", "gravity 0g", duration=4.0, gravity_scale=0.0),
    ]
    variants.extend(
        Variant(
            "iterations_adaptive",
            f"{iters} iterations",
            duration=8.0,
            iterations=iters,
            joint_adaptive_stiffness=True,
            joint_linear_ke=3.0e6,
            joint_angular_ke=1.2e6,
        )
        for iters in iteration_counts
    )
    variants.extend(
        Variant(
            "iterations_fixed",
            f"{iters} iterations",
            duration=8.0,
            iterations=iters,
            joint_linear_ke=3.0e6,
            joint_angular_ke=1.2e6,
        )
        for iters in iteration_counts
    )
    variants.extend(
        Variant(
            "joint_stiffness",
            f"k_joint={joint_ke:g}",
            duration=10.0,
            iterations=144,
            joint_linear_ke=joint_ke,
            joint_angular_ke=joint_ke,
        )
        for joint_ke in (2.0e4, 5.0e4, 1.0e5, 2.0e5, 5.0e5, 1.0e6, 3.0e6)
    )
    variants.extend(
        Variant("substeps", f"{substeps} substeps", duration=4.0, substeps=substeps) for substeps in (1, 3, 12)
    )
    variants.extend(
        Variant("stiffness", f"k={stiffness:g}", duration=10.0, primary_stiffness=stiffness)
        for stiffness in (18.0, 180.0, 600.0)
    )
    variants.extend(
        Variant("damping", f"c={damping:g}", duration=10.0, primary_damping=damping) for damping in (0.0, 0.35, 3.5)
    )
    variants.extend(
        Variant("modal_mass", f"m={mass:g}", duration=10.0, primary_mass=mass) for mass in (0.006, 0.09, 0.9)
    )
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("reports/dipper_static_sag_experiment.html"))
    parser.add_argument("--assets", type=Path, default=Path("reports/assets"))
    args = parser.parse_args()

    cache_key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_samples: list[Sample] = []
    variants = build_variants()

    with wp.ScopedDevice(args.device):
        for index, variant in enumerate(variants, start=1):
            print(f"[{index:02d}/{len(variants):02d}] {variant.experiment}: {variant.label}", flush=True)
            all_samples.extend(run_variant(variant))

    summary = summarize(all_samples)
    args.assets.mkdir(parents=True, exist_ok=True)
    write_csv(args.assets / "dipper_static_sag_experiment.csv", all_samples)
    (args.assets / "dipper_static_sag_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(args.output, all_samples, summary, cache_key)
    print(f"cache_key={cache_key}", flush=True)
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
