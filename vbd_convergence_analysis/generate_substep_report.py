#!/usr/bin/env python3
"""Generate substeps vs iterations trade-off report."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

TOTAL_COLORS = {
    10: "#E53935",
    20: "#1E88E5",
    30: "#43A047",
    60: "#F4511E",
    100: "#8E24AA",
}


def _tc(total: int) -> str:
    return TOTAL_COLORS.get(total, "#757575")


# ---------------------------------------------------------------------------
# Data loading & metrics
# ---------------------------------------------------------------------------

def load_data(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compute_config_metrics(cfg: dict) -> dict:
    """Compute summary metrics for one (substeps, iterations) config."""
    all_residuals = []
    all_times = []

    for seed_str, sdata in cfg["seeds"].items():
        all_residuals.extend(sdata["per_frame_residuals"])
        all_times.extend(sdata["per_frame_times"])

    residuals = np.array(all_residuals)
    times = np.array(all_times)

    # Filter NaN/inf
    valid = np.isfinite(residuals)

    return {
        "substeps": cfg["substeps"],
        "iterations": cfg["iterations"],
        "total_calls": cfg["total_calls"],
        "median_residual": float(np.median(residuals[valid])) if valid.any() else float("nan"),
        "mean_residual": float(np.mean(residuals[valid])) if valid.any() else float("nan"),
        "p90_residual": float(np.percentile(residuals[valid], 90)) if valid.any() else float("nan"),
        "median_time_ms": float(np.median(times) * 1000),
        "mean_time_ms": float(np.mean(times) * 1000),
        "n_nan": int((~valid).sum()),
        "n_seeds": len(cfg["seeds"]),
    }


# ---------------------------------------------------------------------------
# Plotly chart generators
# ---------------------------------------------------------------------------

def make_summary_table(metrics: list[dict]) -> str:
    rows = []
    for m in sorted(metrics, key=lambda x: (x["total_calls"], x["substeps"])):
        label = f"{m['substeps']}x{m['iterations']}"
        nan_style = ' style="color:#c62828;font-weight:bold;"' if m["n_nan"] > 0 else ""
        rows.append(f"""<tr>
            <td><strong>{label}</strong></td>
            <td>{m['total_calls']}</td>
            <td>{m['substeps']}</td>
            <td>{m['iterations']}</td>
            <td>{m['median_time_ms']:.1f}</td>
            <td>{m['median_residual']:.1f}</td>
            <td>{m['p90_residual']:.1f}</td>
            <td{nan_style}>{m['n_nan']}</td>
        </tr>""")

    return f"""<table>
    <thead><tr>
        <th>Config</th><th>Total Calls</th><th>Substeps</th><th>Iters</th>
        <th>Time (ms/frame)</th><th>Median Residual</th><th>P90 Residual</th><th>NaN frames</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>"""


def make_pareto_plot(metrics: list[dict], plot_id: str = "pareto") -> str:
    """Scatter: time vs residual, colored by total calls."""
    traces = []

    by_total = {}
    for m in metrics:
        by_total.setdefault(m["total_calls"], []).append(m)

    for total in sorted(by_total.keys()):
        entries = by_total[total]
        x = [m["median_time_ms"] for m in entries]
        y = [m["median_residual"] for m in entries]
        labels = [f"{m['substeps']}x{m['iterations']}" for m in entries]
        color = _tc(total)

        traces.append(f"""{{
            x: {json.dumps(x)},
            y: {json.dumps(y)},
            mode: 'markers+text',
            name: 'total={total}',
            text: {json.dumps(labels)},
            textposition: 'top center',
            textfont: {{size: 9}},
            marker: {{size: 12, color: '{color}', line: {{width: 1, color: '#333'}}}},
        }}""")

    layout = """{
        title: 'Substep/Iteration Trade-off: Time vs Residual',
        xaxis: {title: 'GPU Time (ms / frame)'},
        yaxis: {title: 'Median Force Residual', type: 'log'},
        hovermode: 'closest',
        width: 1050, height: 550,
        legend: {x: 0.01, y: 0.99},
        margin: {l: 80, r: 30, t: 50, b: 50}
    }"""

    return f"""<div id="{plot_id}"></div>
    <script>Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});</script>"""


def make_isocost_plot(metrics: list[dict], plot_id: str = "isocost") -> str:
    """For each total cost, plot residual vs substep count."""
    traces = []

    by_total = {}
    for m in metrics:
        by_total.setdefault(m["total_calls"], []).append(m)

    for total in sorted(by_total.keys()):
        entries = sorted(by_total[total], key=lambda m: m["substeps"])
        if len(entries) < 2:
            continue
        x = [m["substeps"] for m in entries]
        y = [m["median_residual"] for m in entries]
        labels = [f"{m['substeps']}x{m['iterations']}" for m in entries]

        traces.append(f"""{{
            x: {json.dumps(x)},
            y: {json.dumps(y)},
            mode: 'lines+markers+text',
            name: 'total={total}',
            text: {json.dumps(labels)},
            textposition: 'top center',
            textfont: {{size: 8}},
            line: {{color: '{_tc(total)}', width: 2}},
            marker: {{size: 8}},
        }}""")

    layout = """{
        title: 'Iso-Cost Curves: Residual vs Substep Count (fixed total solver calls)',
        xaxis: {title: 'Substeps per Frame', type: 'log'},
        yaxis: {title: 'Median Force Residual', type: 'log'},
        hovermode: 'closest',
        width: 1050, height: 500,
        legend: {x: 0.01, y: 0.01},
        margin: {l: 80, r: 30, t: 50, b: 50}
    }"""

    return f"""<div id="{plot_id}"></div>
    <script>Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});</script>"""


def make_trajectory_plot(data: dict, plot_id: str = "traj") -> str:
    """Per-frame residual over time for selected configs."""
    # Pick representative configs
    key_configs = [(1, 10), (2, 5), (5, 2), (10, 1),
                   (5, 6), (10, 3), (10, 6), (10, 10)]
    traces = []

    palette = ["#E53935", "#1E88E5", "#43A047", "#F4511E",
               "#8E24AA", "#00ACC1", "#FF9800", "#795548"]

    idx = 0
    for cfg in data["configs"]:
        key = (cfg["substeps"], cfg["iterations"])
        if key not in key_configs:
            continue
        color = palette[idx % len(palette)]
        idx += 1
        label = f"{cfg['substeps']}x{cfg['iterations']} (t={cfg['total_calls']})"

        # Collect all seeds' residuals
        all_curves = []
        for sdata in cfg["seeds"].values():
            all_curves.append(sdata["per_frame_residuals"])
        arr = np.array(all_curves)
        med = np.median(arr, axis=0).tolist()
        x = list(range(1, len(med) + 1))

        traces.append(f"""{{
            x: {json.dumps(x)},
            y: {json.dumps(med)},
            mode: 'lines',
            name: '{label}',
            line: {{color: '{color}', width: 2}},
        }}""")

    layout = """{
        title: 'Per-Frame Residual Over Time (median across seeds)',
        xaxis: {title: 'Frame'},
        yaxis: {title: 'Force Residual', type: 'log'},
        hovermode: 'x unified',
        width: 1050, height: 500,
        legend: {x: 0.01, y: 0.99},
        margin: {l: 80, r: 30, t: 50, b: 50}
    }"""

    return f"""<div id="{plot_id}"></div>
    <script>Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});</script>"""


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(data: dict, output_path: str) -> None:
    metrics = [compute_config_metrics(cfg) for cfg in data["configs"]]
    meta = data.get("metadata", {})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Find best config (lowest residual per ms)
    valid_m = [m for m in metrics if np.isfinite(m["median_residual"])]
    if valid_m:
        best_efficiency = min(valid_m, key=lambda m: m["median_residual"] * m["median_time_ms"])
        best_residual = min(valid_m, key=lambda m: m["median_residual"])
    else:
        best_efficiency = best_residual = None

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VBD Substep/Iteration Trade-off Study</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #f5f5f5; color: #333; line-height: 1.6; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 1.8em; margin-bottom: 4px; }}
    h2 {{ font-size: 1.4em; margin: 32px 0 12px; border-bottom: 2px solid #1976D2; padding-bottom: 6px; }}
    .subtitle {{ color: #666; margin-bottom: 24px; }}
    .card {{ background: #fff; border-radius: 8px; padding: 20px; margin: 16px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
    th {{ background: #f5f5f5; font-weight: 600; font-size: 0.9em; }}
    td {{ font-size: 0.9em; }}
    .finding {{ background: #E3F2FD; border-left: 4px solid #1976D2; padding: 12px 16px;
                margin: 12px 0; border-radius: 0 6px 6px 0; }}
    code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body>
<div class="container">

<h1>Substep / Iteration Trade-off Study</h1>
<p class="subtitle">Generated {timestamp} | alpha={meta.get('step_length', 0.7)},
avbd_beta={meta.get('avbd_beta', 0)}, self-contact, {len(meta.get('seeds', []))} seeds,
{NUM_FRAMES} frames</p>

<h2>1. Summary</h2>
<div class="card">
"""
    if best_efficiency:
        html += f"""
    <div class="finding">
        <strong>Best efficiency</strong> (lowest residual x time):
        <code>{best_efficiency['substeps']}x{best_efficiency['iterations']}</code>
        ({best_efficiency['median_time_ms']:.1f} ms/frame, residual {best_efficiency['median_residual']:.1f})<br>
        <strong>Lowest residual</strong>:
        <code>{best_residual['substeps']}x{best_residual['iterations']}</code>
        ({best_residual['median_time_ms']:.1f} ms/frame, residual {best_residual['median_residual']:.1f})
    </div>
"""
    html += make_summary_table(metrics)
    html += """
</div>

<h2>2. Pareto Frontier: Time vs Residual</h2>
<div class="card">
    <p>Each point is one (substeps x iterations) configuration. Points closer to the
    bottom-left are better (lower time AND lower residual). Color = total solver calls per frame.</p>
"""
    html += make_pareto_plot(metrics)
    html += """
</div>

<h2>3. Iso-Cost Curves</h2>
<div class="card">
    <p>For a fixed compute budget (total solver calls), how does residual change
    as we shift work between more substeps vs more iterations?
    Each curve connects configs with the same total calls.</p>
"""
    html += make_isocost_plot(metrics)
    html += """
</div>

<h2>4. Per-Frame Residual Trajectories</h2>
<div class="card">
    <p>Force residual at each frame for selected configurations (median across seeds).
    Shows how error evolves over the simulation, especially during contact.</p>
"""
    html += make_trajectory_plot(data)
    html += f"""
</div>

<h2>5. Methodology</h2>
<div class="card">
    <p>Each configuration runs a full 60-frame (1 second) t-shirt drop simulation
    with self-contact enabled. After each frame, a 1-iteration probe substep measures
    the force residual at the current state. GPU time is measured per frame using
    <code>wp.synchronize()</code> + <code>time.perf_counter()</code>.</p>
    <p>Solver settings: alpha={meta.get('step_length', 0.7)}, avbd_beta=0 (constant penalty),
    quadratic self-contact, c=kd damping. {len(meta.get('seeds', []))} random seeds
    (drop height {meta.get('drop_height_range', [5, 20])} cm).</p>
</div>

</div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report written to {output_path} ({len(html):,} bytes)")


NUM_FRAMES = 60


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "substep_study_results.json")
    output_path = os.path.join(base_dir, "substep_study_report.html")

    data = load_data(data_path)
    if data is None:
        print(f"ERROR: {data_path} not found. Run run_substep_study.py first.")
        sys.exit(1)

    generate_report(data, output_path)


if __name__ == "__main__":
    main()
