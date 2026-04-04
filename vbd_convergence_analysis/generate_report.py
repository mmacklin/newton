#!/usr/bin/env python3
"""Generate an HTML convergence analysis report from JSON results.

Reads convergence_results.json (or a specified file) and produces
vbd_convergence_report.html with interactive plots.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np


def load_results(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def make_plotly_cdn():
    return '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>'


def compute_summary_stats(results: list) -> dict:
    """Compute aggregate statistics across all scenarios."""
    stats = {
        "total_scenarios": len(results),
        "successful": sum(1 for r in results if "error" not in r),
        "nan_count": sum(1 for r in results if r.get("has_nan", False)),
        "errors": sum(1 for r in results if "error" in r),
    }

    # Aggregate per-iteration convergence curves
    all_convergence_curves = defaultdict(list)  # iteration_count -> list of per-iteration RMS displacement arrays

    for r in results:
        if "error" in r or "convergence" not in r:
            continue
        conv = r["convergence"]
        iter_count = r["params"].get("iterations", 0)

        for step_data in conv:
            iters = step_data.get("iteration_residuals", [])
            if len(iters) == 0:
                continue
            rms_curve = [it["rms_displacement"] for it in iters]
            all_convergence_curves[iter_count].append(rms_curve)

    stats["convergence_curves"] = dict(all_convergence_curves)
    return stats


def generate_convergence_per_iteration_plot(results: list, plot_id: str = "conv_per_iter") -> str:
    """Generate plot showing RMS displacement vs iteration number, averaged across steps and scenarios."""
    # Group by (iteration_count, seed)
    groups = defaultdict(lambda: defaultdict(list))
    for r in results:
        if "error" in r or "convergence" not in r:
            continue
        icount = r["params"]["iterations"]
        seed = r["params"]["seed"]
        for step_data in r["convergence"]:
            iters = step_data.get("iteration_residuals", [])
            if iters:
                groups[icount][seed].append([it["rms_displacement"] for it in iters])

    traces = []
    for icount in sorted(groups.keys()):
        all_curves = []
        for seed, curves_list in groups[icount].items():
            for curve in curves_list:
                if len(curve) == icount:
                    all_curves.append(curve)

        if not all_curves:
            continue

        arr = np.array(all_curves)
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        x = list(range(1, len(mean_curve) + 1))

        # Shaded IQR
        traces.append(f"""{{
            x: {json.dumps(x + x[::-1])},
            y: {json.dumps((p25.tolist() + p75[::-1].tolist()))},
            fill: 'toself',
            fillcolor: 'rgba(31, 119, 180, 0.15)',
            line: {{color: 'transparent'}},
            showlegend: false,
            name: 'IQR ({icount} iters)',
            hoverinfo: 'skip'
        }}""")

        # Mean line
        traces.append(f"""{{
            x: {json.dumps(x)},
            y: {json.dumps(mean_curve.tolist())},
            mode: 'lines+markers',
            name: '{icount} iterations (mean)',
            line: {{width: 2}}
        }}""")

    layout = """{
        title: 'VBD Convergence: RMS Displacement per Iteration',
        xaxis: {title: 'Iteration Number', dtick: 1},
        yaxis: {title: 'RMS Displacement (cm)', type: 'log'},
        hovermode: 'x unified',
        width: 900,
        height: 500
    }"""

    return f"""
    <div id="{plot_id}"></div>
    <script>
    Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});
    </script>
    """


def generate_convergence_ratio_plot(results: list, plot_id: str = "conv_ratio") -> str:
    """Plot the ratio of last/first iteration displacement to show convergence rate."""
    data_by_iter = defaultdict(list)

    for r in results:
        if "error" in r or "convergence" not in r:
            continue
        icount = r["params"]["iterations"]
        seed = r["params"]["seed"]

        for step_data in r["convergence"]:
            iters = step_data.get("iteration_residuals", [])
            if len(iters) >= 2:
                first = iters[0]["rms_displacement"]
                last = iters[-1]["rms_displacement"]
                if first > 1e-15:
                    ratio = last / first
                    data_by_iter[icount].append(ratio)

    traces = []
    for icount in sorted(data_by_iter.keys()):
        ratios = data_by_iter[icount]
        traces.append(f"""{{
            y: {json.dumps(ratios)},
            type: 'box',
            name: '{icount} iterations',
            boxpoints: 'outliers'
        }}""")

    layout = """{
        title: 'Convergence Ratio (Last/First Iteration RMS Displacement)',
        yaxis: {title: 'Ratio (lower = better convergence)', type: 'log'},
        width: 900,
        height: 500
    }"""

    return f"""
    <div id="{plot_id}"></div>
    <script>
    Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});
    </script>
    """


def generate_per_step_convergence_plot(results: list, plot_id: str = "per_step_conv") -> str:
    """Show how convergence evolves over simulation time (per step)."""
    # Pick first successful scenario for detailed view
    for r in results:
        if "error" not in r and "convergence" in r and len(r["convergence"]) > 0:
            selected = r
            break
    else:
        return "<p>No valid scenarios found for per-step convergence plot.</p>"

    conv = selected["convergence"]
    steps = []
    first_rms = []
    last_rms = []

    for step_data in conv:
        iters = step_data.get("iteration_residuals", [])
        if len(iters) >= 2:
            steps.append(step_data["step"])
            first_rms.append(iters[0]["rms_displacement"])
            last_rms.append(iters[-1]["rms_displacement"])

    traces = [
        f"""{{
            x: {json.dumps(steps)},
            y: {json.dumps(first_rms)},
            mode: 'lines',
            name: 'First iteration RMS',
            line: {{color: 'red'}}
        }}""",
        f"""{{
            x: {json.dumps(steps)},
            y: {json.dumps(last_rms)},
            mode: 'lines',
            name: 'Last iteration RMS',
            line: {{color: 'blue'}}
        }}""",
    ]

    seed = selected["params"]["seed"]
    layout = f"""{{
        title: 'Per-Step Convergence (Seed {seed})',
        xaxis: {{title: 'Substep'}},
        yaxis: {{title: 'RMS Displacement (cm)', type: 'log'}},
        hovermode: 'x unified',
        width: 900,
        height: 500
    }}"""

    return f"""
    <div id="{plot_id}"></div>
    <script>
    Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});
    </script>
    """


def generate_displacement_percentiles_plot(results: list, plot_id: str = "disp_pct") -> str:
    """Show p50, p95, p99 displacement across iterations for one scenario."""
    for r in results:
        if "error" not in r and "convergence" in r and len(r["convergence"]) > 10:
            selected = r
            break
    else:
        return "<p>No valid scenarios found for percentile plot.</p>"

    # Use a substep from the middle of the simulation
    mid = len(selected["convergence"]) // 2
    step_data = selected["convergence"][mid]
    iters = step_data.get("iteration_residuals", [])

    if len(iters) == 0:
        return "<p>No iteration data for percentile plot.</p>"

    x = list(range(1, len(iters) + 1))
    p50 = [it["p50_displacement"] for it in iters]
    p95 = [it["p95_displacement"] for it in iters]
    p99 = [it["p99_displacement"] for it in iters]
    max_d = [it["max_displacement"] for it in iters]

    traces = [
        f"""{{x: {json.dumps(x)}, y: {json.dumps(p50)}, mode: 'lines+markers', name: 'P50'}}""",
        f"""{{x: {json.dumps(x)}, y: {json.dumps(p95)}, mode: 'lines+markers', name: 'P95'}}""",
        f"""{{x: {json.dumps(x)}, y: {json.dumps(p99)}, mode: 'lines+markers', name: 'P99'}}""",
        f"""{{x: {json.dumps(x)}, y: {json.dumps(max_d)}, mode: 'lines+markers', name: 'Max'}}""",
    ]

    layout = f"""{{
        title: 'Displacement Percentiles per Iteration (Step {step_data["step"]})',
        xaxis: {{title: 'Iteration', dtick: 1}},
        yaxis: {{title: 'Displacement (cm)', type: 'log'}},
        hovermode: 'x unified',
        width: 900,
        height: 500
    }}"""

    return f"""
    <div id="{plot_id}"></div>
    <script>
    Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});
    </script>
    """


def generate_heatmap_plot(results: list, plot_id: str = "heatmap") -> str:
    """Heatmap of RMS displacement: x=iteration, y=substep, color=log(displacement)."""
    for r in results:
        if "error" not in r and "convergence" in r and len(r["convergence"]) > 10:
            selected = r
            break
    else:
        return "<p>No valid scenarios for heatmap.</p>"

    conv = selected["convergence"]
    n_iters = selected["params"]["iterations"]
    n_steps = len(conv)

    z = []
    for step_data in conv:
        row = []
        iters = step_data.get("iteration_residuals", [])
        for it in iters:
            val = it["rms_displacement"]
            row.append(np.log10(val) if val > 0 else -10)
        # Pad if needed
        while len(row) < n_iters:
            row.append(None)
        z.append(row)

    trace = f"""{{
        z: {json.dumps(z)},
        type: 'heatmap',
        colorscale: 'Viridis',
        colorbar: {{title: 'log10(RMS disp)'}},
        x: {json.dumps(list(range(1, n_iters+1)))},
    }}"""

    seed = selected["params"]["seed"]
    layout = f"""{{
        title: 'RMS Displacement Heatmap (Seed {seed})',
        xaxis: {{title: 'Iteration', dtick: 1}},
        yaxis: {{title: 'Substep'}},
        width: 900,
        height: 600
    }}"""

    return f"""
    <div id="{plot_id}"></div>
    <script>
    Plotly.newPlot('{plot_id}', [{trace}], {layout});
    </script>
    """


def generate_nan_check_table(results: list) -> str:
    """Table showing NaN status per scenario."""
    rows = []
    for r in results:
        if "error" in r:
            status = f'<span style="color:orange">ERROR: {r["error"][:50]}</span>'
        elif r.get("has_nan"):
            status = '<span style="color:red">NaN detected</span>'
        else:
            status = '<span style="color:green">OK</span>'

        params = r.get("params", {})
        rows.append(f"""<tr>
            <td>{params.get('seed', 'N/A')}</td>
            <td>{params.get('iterations', 'N/A')}</td>
            <td>{params.get('material_model', 'N/A')}</td>
            <td>{params.get('drop_height', 'N/A'):.1f}</td>
            <td>{params.get('particle_count', 'N/A')}</td>
            <td>{r.get('elapsed_seconds', 'N/A'):.2f}s</td>
            <td>{status}</td>
        </tr>""")

    return f"""
    <table class="scenario-table">
        <tr>
            <th>Seed</th><th>Iterations</th><th>Material</th>
            <th>Drop Height (cm)</th><th>Particles</th><th>Time</th><th>Status</th>
        </tr>
        {"".join(rows)}
    </table>
    """


def generate_convergence_rate_analysis(results: list) -> str:
    """Textual analysis of convergence behavior."""
    all_ratios = []
    all_first = []
    all_last = []
    stagnation_count = 0
    total_steps = 0

    for r in results:
        if "error" in r or "convergence" not in r:
            continue
        for step_data in r["convergence"]:
            iters = step_data.get("iteration_residuals", [])
            if len(iters) >= 2:
                total_steps += 1
                first = iters[0]["rms_displacement"]
                last = iters[-1]["rms_displacement"]
                all_first.append(first)
                all_last.append(last)
                if first > 1e-15:
                    ratio = last / first
                    all_ratios.append(ratio)
                    if ratio > 0.95:
                        stagnation_count += 1

    if not all_ratios:
        return "<p>Insufficient data for convergence rate analysis.</p>"

    ratios = np.array(all_ratios)
    analysis = f"""
    <div class="analysis-box">
        <h3>Convergence Rate Analysis</h3>
        <ul>
            <li><strong>Total substeps analyzed:</strong> {total_steps}</li>
            <li><strong>Mean convergence ratio (last/first):</strong> {np.mean(ratios):.4f}</li>
            <li><strong>Median convergence ratio:</strong> {np.median(ratios):.4f}</li>
            <li><strong>Stagnation rate (ratio > 0.95):</strong> {stagnation_count}/{total_steps} = {100*stagnation_count/total_steps:.1f}%</li>
            <li><strong>Mean first-iteration RMS displacement:</strong> {np.mean(all_first):.6e} cm</li>
            <li><strong>Mean last-iteration RMS displacement:</strong> {np.mean(all_last):.6e} cm</li>
            <li><strong>Effective reduction factor:</strong> {np.mean(all_last)/np.mean(all_first):.4f}x</li>
        </ul>
        <h4>Interpretation</h4>
        <p>A convergence ratio of 1.0 means no improvement between first and last iteration.
        Ratios below 0.5 indicate good convergence. Below 0.1 indicates excellent convergence.
        Stagnation (ratio > 0.95) suggests the solver is not making meaningful progress in later iterations.</p>
    </div>
    """
    return analysis


def generate_html_report(results: list, output_path: str, label: str = "Baseline"):
    """Generate the full HTML report."""
    stats = compute_summary_stats(results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>VBD Convergence Analysis Report - {label}</title>
    {make_plotly_cdn()}
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f8f9fa; color: #333; }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 40px; }}
        .summary-box {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
        .analysis-box {{ background: #e8f4f8; padding: 20px; border-radius: 8px; border-left: 4px solid #0077b6; margin: 20px 0; }}
        .scenario-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        .scenario-table th, .scenario-table td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        .scenario-table th {{ background: #16213e; color: white; }}
        .scenario-table tr:nth-child(even) {{ background: #f2f2f2; }}
        .plot-container {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>VBD Convergence Analysis Report - {label}</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary-box">
        <h2>Summary</h2>
        <ul>
            <li><strong>Total scenarios:</strong> {stats['total_scenarios']}</li>
            <li><strong>Successful:</strong> {stats['successful']}</li>
            <li><strong>NaN detected:</strong> {stats['nan_count']}</li>
            <li><strong>Errors:</strong> {stats['errors']}</li>
        </ul>
    </div>

    <h2>Scenario Status</h2>
    {generate_nan_check_table(results)}

    <h2>Convergence Analysis</h2>
    {generate_convergence_rate_analysis(results)}

    <div class="plot-container">
        <h2>RMS Displacement per VBD Iteration</h2>
        <p>Shows how the per-vertex RMS displacement decreases across VBD iterations within each substep,
        averaged across all substeps and scenarios. The shaded region is the interquartile range.</p>
        {generate_convergence_per_iteration_plot(results)}
    </div>

    <div class="plot-container">
        <h2>Convergence Ratio Distribution</h2>
        <p>Box plot of last/first iteration RMS displacement ratio. Lower values indicate better convergence.</p>
        {generate_convergence_ratio_plot(results)}
    </div>

    <div class="plot-container">
        <h2>Per-Step Convergence Over Time</h2>
        <p>Shows how first-iteration and last-iteration displacements evolve over the simulation timeline.</p>
        {generate_per_step_convergence_plot(results)}
    </div>

    <div class="plot-container">
        <h2>Displacement Percentiles per Iteration</h2>
        <p>P50, P95, P99, and max displacement across vertices at each iteration for a representative substep.</p>
        {generate_displacement_percentiles_plot(results)}
    </div>

    <div class="plot-container">
        <h2>Convergence Heatmap</h2>
        <p>Color shows log10(RMS displacement) at each (substep, iteration) pair. Darker = smaller displacement = better converged.</p>
        {generate_heatmap_plot(results)}
    </div>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--label", type=str, default="Baseline")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input or os.path.join(base_dir, "convergence_results.json")
    output_path = args.output or os.path.join(base_dir, "vbd_convergence_report.html")

    results = load_results(input_path)
    generate_html_report(results, output_path, label=args.label)


if __name__ == "__main__":
    main()
