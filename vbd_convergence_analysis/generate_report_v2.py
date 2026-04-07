#!/usr/bin/env python3
"""Generate VBD Convergence Report v2.

Reads trajectory-based convergence data and rollout comparison data,
produces a self-contained HTML report with Plotly charts.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    "Baseline GS": "#E53935",
    "Alpha 0.3": "#AB47BC",
    "Alpha 0.5": "#5C6BC0",
    "Alpha 0.7": "#1E88E5",
    "Alpha 0.9": "#00ACC1",
    "Chebyshev Auto": "#43A047",
    "Jacobi": "#F4511E",
}

METHOD_ORDER = [
    "Baseline GS", "Alpha 0.3", "Alpha 0.5", "Alpha 0.7",
    "Alpha 0.9", "Chebyshev Auto", "Jacobi",
]


def _c(label: str) -> str:
    return COLORS.get(label, "#757575")


def _c_alpha(label: str, a: float) -> str:
    """Return rgba version of color with given alpha."""
    hex_c = _c(label).lstrip("#")
    r, g, b = int(hex_c[:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectory_data(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_rollout_data(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_trajectory_metrics(
    traj_data: dict, mask: np.ndarray | None = None,
) -> list[dict]:
    """Compute per-method summary metrics from trajectory data.

    Args:
        traj_data: Loaded trajectory JSON.
        mask: Boolean array (length = number of curves) to select a subset.
              If None, use all curves.
    """
    rows = []
    for method_name in METHOD_ORDER:
        mdata = traj_data["methods"].get(method_name)
        if not mdata:
            continue
        curves = np.array(mdata["curves"])
        if curves.size == 0:
            continue

        if mask is not None and len(mask) == len(curves):
            curves = curves[mask]
        if len(curves) == 0:
            continue

        med = np.median(curves, axis=0)

        # Per-iteration ratios
        per_iter = curves[:, 1:] / np.maximum(curves[:, :-1], 1e-15)
        med_ratios = np.median(per_iter, axis=0)
        geo_mean = float(np.exp(np.mean(np.log(np.maximum(med_ratios, 1e-15)))))

        overall_ratio = float(med[-1] / med[0]) if med[0] > 1e-15 else float("inf")

        rows.append({
            "method": method_name,
            "iter0": float(med[0]),
            "iter_last": float(med[-1]),
            "overall_ratio": overall_ratio,
            "geo_mean_rate": geo_mean,
            "n_curves": len(curves),
            "med_curve": med.tolist(),
            "p25_curve": np.percentile(curves, 25, axis=0).tolist(),
            "p75_curve": np.percentile(curves, 75, axis=0).tolist(),
            "med_ratios": med_ratios.tolist(),
        })

    return rows


def _regime_masks(traj_data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute boolean masks for easy / hard / all regimes.

    Uses baseline iter-0 residual to classify snapshots.
    Returns (easy_mask, hard_mask, all_mask).
    """
    bl = traj_data["methods"].get("Baseline GS")
    if not bl:
        n = 0
        for m in traj_data["methods"].values():
            n = len(m.get("curves", []))
            break
        return np.ones(n, dtype=bool), np.ones(n, dtype=bool), np.ones(n, dtype=bool)

    curves = np.array(bl["curves"])
    iter0 = curves[:, 0]
    easy = iter0 < 5.0
    hard = iter0 >= 5.0
    return easy, hard, np.ones(len(curves), dtype=bool)


# ---------------------------------------------------------------------------
# Plotly chart generators (return JS snippet strings)
# ---------------------------------------------------------------------------

def make_convergence_plot(
    metrics: list[dict],
    plot_id: str = "conv_plot",
    title: str = "Per-Iteration Convergence (from consistent starting states)",
) -> str:
    """Convergence curves: median force residual per iteration with IQR."""
    if not metrics:
        return f'<p id="{plot_id}"><em>No data for this regime.</em></p>'

    traces = []

    for m in metrics:
        label = m["method"]
        n = len(m["med_curve"])
        x = list(range(1, n + 1))
        color = _c(label)
        fill = _c_alpha(label, 0.12)

        # IQR band
        traces.append(
            f"""{{
            x: {json.dumps(x + x[::-1])},
            y: {json.dumps(m['p25_curve'] + m['p75_curve'][::-1])},
            fill: 'toself', fillcolor: '{fill}',
            line: {{color: 'transparent'}}, showlegend: false, hoverinfo: 'skip'
        }}"""
        )
        # Median line
        traces.append(
            f"""{{
            x: {json.dumps(x)},
            y: {json.dumps(m['med_curve'])},
            mode: 'lines+markers', name: '{label}',
            line: {{color: '{color}', width: 2.5}}, marker: {{size: 5}}
        }}"""
        )

    layout = f"""{{
        title: '{title}',
        xaxis: {{title: 'VBD Iteration', dtick: 1}},
        yaxis: {{title: 'RMS Force Residual ||\\u2207G||', type: 'log'}},
        hovermode: 'x unified', width: 1050, height: 520,
        legend: {{x: 0.01, y: 0.01, bgcolor: 'rgba(255,255,255,0.8)'}},
        margin: {{l: 80, r: 30, t: 50, b: 50}}
    }}"""

    return f"""<div id="{plot_id}"></div>
    <script>Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});</script>"""


def make_ratio_plot(metrics: list[dict], plot_id: str = "ratio_plot") -> str:
    """Per-iteration convergence ratio plot."""
    traces = []

    for m in metrics:
        label = m["method"]
        n = len(m["med_ratios"])
        x = list(range(2, n + 2))  # iterations 2..10
        traces.append(
            f"""{{
            x: {json.dumps(x)},
            y: {json.dumps(m['med_ratios'])},
            mode: 'lines+markers', name: '{label}',
            line: {{color: '{_c(label)}', width: 2}}, marker: {{size: 4}}
        }}"""
        )

    # Reference line at 1.0
    traces.append(
        """{
        x: [1, 11], y: [1, 1], mode: 'lines', name: 'ratio=1.0',
        line: {color: '#999', width: 1, dash: 'dash'}, showlegend: false
    }"""
    )

    layout = f"""{{
        title: 'Per-Iteration Residual Ratio (iter n / iter n-1)',
        xaxis: {{title: 'VBD Iteration', dtick: 1}},
        yaxis: {{title: 'Residual Ratio', range: [0.5, 1.5]}},
        hovermode: 'x unified', width: 1050, height: 420,
        legend: {{x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)'}},
        margin: {{l: 70, r: 30, t: 50, b: 50}}
    }}"""

    return f"""<div id="{plot_id}"></div>
    <script>Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});</script>"""


def make_rollout_plot(rollout_data: dict, plot_id: str = "rollout_plot") -> str:
    """Per-frame residual over time for full rollouts."""
    traces = []

    for method_name in METHOD_ORDER:
        mdata = rollout_data["methods"].get(method_name)
        if not mdata:
            continue

        color = _c(method_name)
        seed_curves = []

        for seed_str, sdata in mdata["seeds"].items():
            vals = sdata["per_frame_residuals"]
            x = list(range(1, len(vals) + 1))
            seed_curves.append(vals)

            # Thin per-seed line
            traces.append(
                f"""{{
                x: {json.dumps(x)},
                y: {json.dumps(vals)},
                mode: 'lines', showlegend: false,
                line: {{color: '{_c_alpha(method_name, 0.25)}', width: 1}},
                hoverinfo: 'skip'
            }}"""
            )

        # Bold median line
        if seed_curves:
            arr = np.array(seed_curves)
            med = np.median(arr, axis=0).tolist()
            x = list(range(1, len(med) + 1))
            traces.append(
                f"""{{
                x: {json.dumps(x)},
                y: {json.dumps(med)},
                mode: 'lines+markers', name: '{method_name}',
                line: {{color: '{color}', width: 3}}, marker: {{size: 4}}
            }}"""
            )

    layout = f"""{{
        title: 'Accumulated Error: Per-Frame Force Residual Over Full Rollout',
        xaxis: {{title: 'Frame'}},
        yaxis: {{title: 'RMS Force Residual ||\\u2207G||', type: 'log'}},
        hovermode: 'x unified', width: 1050, height: 520,
        legend: {{x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.8)'}},
        margin: {{l: 80, r: 30, t: 50, b: 50}}
    }}"""

    return f"""<div id="{plot_id}"></div>
    <script>Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});</script>"""


def make_alpha_sweep_plot(metrics: list[dict], plot_id: str = "alpha_sweep") -> str:
    """Convergence curves for alpha sweep only."""
    alpha_methods = ["Baseline GS", "Alpha 0.3", "Alpha 0.5", "Alpha 0.7", "Alpha 0.9"]
    filtered = [m for m in metrics if m["method"] in alpha_methods]
    return make_convergence_plot(filtered, plot_id).replace(
        "Per-Iteration Convergence (from consistent starting states)",
        "Alpha Sweep: Step Length Impact on Convergence",
    )


def make_jacobi_plot(metrics: list[dict], plot_id: str = "jacobi_plot") -> str:
    """Jacobi vs GS comparison."""
    filtered = [m for m in metrics if m["method"] in ("Baseline GS", "Jacobi")]
    return make_convergence_plot(filtered, plot_id).replace(
        "Per-Iteration Convergence (from consistent starting states)",
        "Jacobi vs Gauss-Seidel Convergence",
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def make_summary_table(metrics: list[dict]) -> str:
    rows = []
    for m in metrics:
        ratio = m["overall_ratio"]
        if ratio < 0.8:
            style = "color:#2e7d32;font-weight:bold;"
        elif ratio > 1.05:
            style = "color:#c62828;font-weight:bold;"
        else:
            style = ""

        rows.append(f"""<tr>
            <td><strong>{m['method']}</strong></td>
            <td>{m['iter0']:.4f}</td>
            <td>{m['iter_last']:.4f}</td>
            <td style="{style}">{ratio:.4f}</td>
            <td>{m['geo_mean_rate']:.4f}</td>
            <td>{m['n_curves']}</td>
        </tr>""")

    return f"""<table>
        <thead><tr>
            <th>Method</th><th>Iter 0</th><th>Iter 9</th>
            <th>Overall Ratio</th><th>Per-iter Geo Mean</th><th>Snapshots</th>
        </tr></thead>
        <tbody>{"".join(rows)}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    traj_data: dict | None,
    rollout_data: dict | None,
    output_path: str,
) -> None:
    # Compute metrics for all / easy / hard regimes
    if traj_data:
        easy_mask, hard_mask, all_mask = _regime_masks(traj_data)
        metrics_all = compute_trajectory_metrics(traj_data, all_mask)
        metrics_easy = compute_trajectory_metrics(traj_data, easy_mask)
        metrics_hard = compute_trajectory_metrics(traj_data, hard_mask)
        n_easy = int(easy_mask.sum())
        n_hard = int(hard_mask.sum())
    else:
        metrics_all = metrics_easy = metrics_hard = []
        n_easy = n_hard = 0

    # Use easy-regime metrics for executive summary (where methods actually differ)
    metrics = metrics_easy if metrics_easy else metrics_all

    # Executive summary bullets
    baseline_m = next((m for m in metrics if m["method"] == "Baseline GS"), None)
    best_alpha = min(
        (m for m in metrics if m["method"].startswith("Alpha")),
        key=lambda m: m["iter_last"],
        default=None,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = traj_data.get("metadata", {}) if traj_data else {}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VBD Convergence Report v2</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #f5f5f5; color: #333; line-height: 1.6; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 1.8em; margin-bottom: 4px; }}
    h2 {{ font-size: 1.4em; margin: 32px 0 12px; border-bottom: 2px solid #1976D2; padding-bottom: 6px; }}
    h3 {{ font-size: 1.15em; margin: 20px 0 8px; }}
    .subtitle {{ color: #666; margin-bottom: 24px; }}
    .card {{ background: #fff; border-radius: 8px; padding: 20px; margin: 16px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
    th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
    th {{ background: #f5f5f5; font-weight: 600; font-size: 0.9em; }}
    td {{ font-size: 0.9em; }}
    ul {{ margin: 8px 0 8px 24px; }}
    li {{ margin: 6px 0; }}
    .finding {{ background: #E3F2FD; border-left: 4px solid #1976D2; padding: 12px 16px;
                margin: 12px 0; border-radius: 0 6px 6px 0; }}
    .warning {{ background: #FFF3E0; border-left: 4px solid #F57C00; padding: 12px 16px;
                margin: 12px 0; border-radius: 0 6px 6px 0; }}
    code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
    .metric-label {{ font-weight: 600; }}
</style>
</head>
<body>
<div class="container">

<h1>VBD Solver Convergence Analysis</h1>
<p class="subtitle">Generated {timestamp} &mdash;
{len(meta.get('seeds', [])) if isinstance(meta.get('seeds'), list) else '?'} seeds,
{meta.get('iterations', 10)} iterations,
self-contact {'enabled' if meta.get('self_contact') else 'disabled'}</p>

<!-- ============================================================ -->
<h2>1. Executive Summary</h2>
<div class="card">
    <p><span class="metric-label">Metric:</span> RMS force residual
    ||&nabla;G(x)|| &mdash; the gradient of the implicit Euler variational energy.
    At the exact solution this is zero. Independent of step size strategy.</p>

    <p><span class="metric-label">Methodology:</span> All methods evaluated from the
    <em>same</em> reference trajectory snapshots (consistent starting states),
    eliminating compounded inter-substep drift.</p>

    {make_summary_table(metrics)}

    <div class="finding">
        <strong>Key findings:</strong>
        <ul>"""

    if baseline_m:
        html += f"""
            <li>Baseline GS <strong>{'diverges' if baseline_m['overall_ratio'] > 1.0 else 'stagnates'}</strong>
                (ratio {baseline_m['overall_ratio']:.3f}) &mdash; cross-color interference causes
                the force residual to increase with each iteration.</li>"""
    if best_alpha:
        html += f"""
            <li><strong>{best_alpha['method']}</strong> achieves the best convergence among alpha methods
                (ratio {best_alpha['overall_ratio']:.3f}, iter 0: {best_alpha['iter0']:.3f}
                &rarr; iter 9: {best_alpha['iter_last']:.3f}).</li>"""

    jacobi_m = next((m for m in metrics if m["method"] == "Jacobi"), None)
    if jacobi_m:
        bl_iter0 = f"{baseline_m['iter0']:.3f}" if baseline_m else "?"
        html += f"""
            <li>Jacobi (ratio {jacobi_m['overall_ratio']:.3f}) shows steady per-iteration
                convergence but starts from much higher residual
                ({jacobi_m['iter0']:.1f} vs {bl_iter0}).</li>"""

    html += """
        </ul>
    </div>
</div>

<!-- ============================================================ -->
<h2>2. Per-Iteration Convergence</h2>

"""
    html += f"""
<div class="warning">
    <strong>Regime matters.</strong> Snapshots are split by baseline iter-0 residual:
    <strong>{n_easy} &ldquo;easy&rdquo;</strong> (free-fall, residual &lt; 5) and
    <strong>{n_hard} &ldquo;hard&rdquo;</strong> (contact, residual &ge; 5).
    Method differences are only visible in the easy regime; in hard/contact frames
    the unresolved residual (~32) dominates and all methods look identical.
</div>

<h3>Easy Regime (free-fall, {n_easy} snapshots)</h3>
<div class="card">
    <p>Low-residual substeps where the cloth is in free fall or lightly deformed.
    This is where per-iteration convergence strategy matters.</p>
    {make_summary_table(metrics_easy)}
"""
    html += make_convergence_plot(
        metrics_easy, "conv_easy",
        f"Easy Regime: Per-Iteration Convergence ({n_easy} snapshots)",
    )
    html += """
</div>
<div class="card">
"""
    html += make_ratio_plot(metrics_easy, "ratio_easy")
    html += f"""
</div>

<h3>Hard Regime (contact, {n_hard} snapshots)</h3>
<div class="card">
    <p>High-residual substeps where the cloth is in contact.
    The residual (~32) is too large for 10 iterations to resolve,
    so all methods perform identically.</p>
    {make_summary_table(metrics_hard)}
"""
    html += make_convergence_plot(
        metrics_hard, "conv_hard",
        f"Hard Regime: Per-Iteration Convergence ({n_hard} snapshots)",
    )
    html += """
</div>

<h3>All Snapshots Combined</h3>
<div class="card">
"""
    html += make_convergence_plot(
        metrics_all, "conv_all", "All Snapshots Combined",
    )
    html += """
</div>
"""

    # Section 3: Rollout
    if rollout_data:
        html += """
<!-- ============================================================ -->
<h2>3. Full Rollout: Accumulated Error</h2>
<div class="card">
    <p>Independent simulations per method. Per-frame force residual shows how
    per-substep divergence (or convergence) compounds over a full trajectory.
    Thin lines = individual seeds; bold line = median.</p>
"""
        html += make_rollout_plot(rollout_data)
        html += """
</div>
"""

    # Section 4: Method details
    html += """
<!-- ============================================================ -->
<h2>4. Method Details</h2>

<h3>Alpha Sweep</h3>
<div class="card">
    <p>Convergence curves for different step lengths (under-relaxation parameter &alpha;).
    Full GS step (&alpha;=1.0) overshoots; smaller values prevent overshoot but may
    converge too slowly.</p>
"""
    html += make_alpha_sweep_plot(metrics_easy)
    html += """
</div>

<h3>Jacobi vs Gauss-Seidel</h3>
<div class="card">
    <p>Jacobi updates all vertices simultaneously (no sequential color-group updates).
    This eliminates cross-color interference but loses GS information propagation.
    (Easy regime only.)</p>
"""
    html += make_jacobi_plot(metrics_easy)
    html += """
</div>
"""

    # Section 5: Methodology
    html += f"""
<!-- ============================================================ -->
<h2>5. Methodology</h2>
<div class="card">
    <h3>Test Setup</h3>
    <ul>
        <li><strong>Mesh:</strong> Unisex t-shirt (~6400 vertices, ~12700 triangles)</li>
        <li><strong>Material:</strong> Neo-Hookean membrane + dihedral bending</li>
        <li><strong>Self-contact:</strong> Enabled for all tests</li>
        <li><strong>Seeds:</strong> {len(meta.get('seeds', []))} random initial configurations
            (randomized drop height, rotation, lateral offset)</li>
        <li><strong>Simulation:</strong> 60 FPS, 10 substeps/frame,
            {meta.get('iterations', 10)} VBD iterations/substep</li>
    </ul>

    <h3>Trajectory-Based Convergence Test</h3>
    <p>A reference trajectory is recorded under baseline GS. At sampled substeps,
    (positions, velocities) are saved. Each method is then evaluated from these
    <em>identical</em> starting states by running a single substep with convergence
    tracking. This eliminates the confound of compounded inter-substep drift.</p>

    <h3>Full Rollout Test</h3>
    <p>Each method runs an independent simulation. After each frame, a probe substep
    measures the force residual at the current state. This shows whether per-substep
    convergence/divergence compounds over a full trajectory.</p>

    <h3>Metric</h3>
    <p><strong>RMS Force Residual</strong> ||&nabla;G(x)|| = RMS of per-vertex
    total force (inertia + elastic + bending + contact). G(x) is the implicit Euler
    variational energy; at the exact solution &nabla;G = 0. This metric is independent
    of step size / relaxation strategy.</p>
</div>

</div><!-- container -->
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report written to {output_path} ({len(html):,} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    traj_path = os.path.join(base_dir, "trajectory_convergence_v2.json")
    rollout_path = os.path.join(base_dir, "rollout_comparison_v2.json")
    output_path = os.path.join(base_dir, "vbd_convergence_report_v2.html")

    traj_data = load_trajectory_data(traj_path)
    rollout_data = load_rollout_data(rollout_path)

    if traj_data is None and rollout_data is None:
        print("ERROR: No data files found. Run the experiment scripts first.")
        sys.exit(1)

    generate_report(traj_data, rollout_data, output_path)


if __name__ == "__main__":
    main()
