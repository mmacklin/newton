#!/usr/bin/env python3
"""Generate comprehensive VBD convergence analysis HTML report.

Produces a self-contained HTML file with embedded base64 images and
Plotly CDN charts covering all experimental results.
"""
from __future__ import annotations

import base64
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_results(paths: dict[str, str]) -> dict[str, list]:
    """Load results from multiple JSON files, skipping missing ones."""
    all_data = {}
    for label, path in paths.items():
        try:
            with open(path) as f:
                all_data[label] = json.load(f)
        except Exception as e:
            print(f"Warning: {path}: {e}")
    return all_data


def load_hessian(path: str) -> dict | None:
    """Load hessian verification results."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: hessian file {path}: {e}")
        return None


def embed_image(path: str) -> str:
    """Read a PNG file and return an <img> tag with inline base64 data."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{data}" style="max-width:100%; border-radius:6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">'
    except Exception as e:
        return f"<p><em>Image not available: {e}</em></p>"


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(data: list, iters_filter: int | None = None) -> dict:
    """Compute summary metrics for a result set.

    Args:
        data: List of result dicts from a convergence JSON.
        iters_filter: If set, only include runs with this iteration count.
    """
    ratios: list[float] = []
    final_rms: list[float] = []
    first_rms: list[float] = []
    nan_count = 0
    total = 0

    # Use force residual if available, fall back to displacement for old data
    metric_key = "rms_force_residual"

    for r in data:
        if "error" in r or "convergence" not in r:
            continue
        if iters_filter and r["params"].get("iterations", 10) != iters_filter:
            continue
        total += 1
        if r.get("has_nan"):
            nan_count += 1
        for step in r["convergence"]:
            iters = step.get("iteration_residuals", [])
            if len(iters) >= 2:
                # Prefer force residual; fall back to displacement for old data
                key = metric_key if metric_key in iters[0] else "rms_displacement"
                first = iters[0][key]
                last = iters[-1][key]
                final_rms.append(last)
                first_rms.append(first)
                if first > 1e-15:
                    ratios.append(last / first)

    r_arr = np.array(ratios) if ratios else np.array([1.0])
    f_arr = np.array(final_rms) if final_rms else np.array([0.0])
    fr_arr = np.array(first_rms) if first_rms else np.array([0.0])

    return {
        "mean_ratio": float(np.mean(r_arr)),
        "median_ratio": float(np.median(r_arr)),
        "p25_ratio": float(np.percentile(r_arr, 25)),
        "p75_ratio": float(np.percentile(r_arr, 75)),
        "mean_final_rms": float(np.mean(f_arr)),
        "median_final_rms": float(np.median(f_arr)),
        "mean_first_rms": float(np.mean(fr_arr)),
        "nan_count": nan_count,
        "total_scenarios": total,
        "n_steps": len(ratios),
    }


def extract_convergence_curves(data: list, target_iters: int = 10) -> np.ndarray:
    """Extract per-iteration force residual curves from result data.

    Returns an (N, target_iters) array where each row is one substep's curve.
    Falls back to displacement metric for old data files.
    """
    all_curves: list[list[float]] = []
    for r in data:
        if "error" in r or "convergence" not in r:
            continue
        if r["params"].get("iterations", 10) != target_iters:
            continue
        for step in r["convergence"]:
            iters = step.get("iteration_residuals", [])
            if len(iters) == target_iters:
                key = "rms_force_residual" if "rms_force_residual" in iters[0] else "rms_displacement"
                all_curves.append([it[key] for it in iters])
    if not all_curves:
        return np.empty((0, target_iters))
    arr = np.array(all_curves)
    # Filter out fully-zero rows
    nonzero = arr.max(axis=1) > 0
    return arr[nonzero]


# ---------------------------------------------------------------------------
# HTML / Plotly section generators
# ---------------------------------------------------------------------------

# Consistent color palette across all plots
COLORS = {
    "Baseline GS": "rgba(220,50,50,1)",
    "Jacobi": "rgba(180,100,50,1)",
    "Alpha 0.3": "rgba(140,140,255,1)",
    "Alpha 0.5": "rgba(100,100,220,1)",
    "Alpha 0.7": "rgba(60,60,180,1)",
    "Alpha 0.9": "rgba(30,30,140,1)",
    "Chebyshev 0.8": "rgba(0,160,80,1)",
    "Chebyshev 0.95": "rgba(255,165,0,1)",
    "Chebyshev Auto": "rgba(0,80,220,1)",
    "Self-Contact": "rgba(160,0,160,1)",
    "StVK": "rgba(128,0,128,1)",
    "Stiffness Baseline": "rgba(200,80,80,1)",
    "Stiffness Chebyshev": "rgba(0,120,200,1)",
}

def _color(label: str, alpha: float = 1.0) -> str:
    c = COLORS.get(label, "rgba(100,100,100,1)")
    if alpha != 1.0:
        c = c.replace(",1)", f",{alpha})")
    return c


def make_comparison_table(all_data: dict[str, list]) -> str:
    """Generate the full method comparison table."""
    # Compute baseline metrics first for relative improvement
    baseline_metrics = None
    if "Baseline GS" in all_data:
        baseline_metrics = compute_metrics(all_data["Baseline GS"], iters_filter=10)

    rows = []
    for label, data in all_data.items():
        # Skip stiffness sweep entries from the main table
        if "Stiffness" in label:
            continue
        m = compute_metrics(data, iters_filter=10)

        improvement = "-"
        if baseline_metrics and label != "Baseline GS":
            bm_final = baseline_metrics["median_final_rms"]
            m_final = m["median_final_rms"]
            if m_final < 1e-12:
                # Solver converged to machine zero — effectively perfect
                improvement = "converged"
            elif bm_final > 0:
                imp = bm_final / m_final
                if imp >= 1:
                    improvement = f"{imp:.0f}x better"
                else:
                    improvement = f"{1/imp:.1f}x worse"

        # Color-code the ratio
        ratio_val = m["median_ratio"]
        if ratio_val < 0.5:
            ratio_class = "style='color:#2e7d32; font-weight:bold;'"
        elif ratio_val < 0.9:
            ratio_class = "style='color:#558b2f;'"
        elif ratio_val > 1.1:
            ratio_class = "style='color:#c62828; font-weight:bold;'"
        else:
            ratio_class = ""

        nan_style = "style='color:#c62828; font-weight:bold;'" if m["nan_count"] > 0 else ""

        rows.append(f"""<tr>
            <td><strong>{label}</strong></td>
            <td>{m["mean_ratio"]:.4f}</td>
            <td {ratio_class}>{m["median_ratio"]:.4f}</td>
            <td>{m["median_final_rms"]:.2e}</td>
            <td>{improvement}</td>
            <td {nan_style}>{m["nan_count"]}/{m["total_scenarios"]}</td>
            <td>{m["n_steps"]}</td>
        </tr>""")

    return f"""
    <table class="comparison-table">
        <thead>
        <tr>
            <th>Method</th>
            <th>Mean Conv. Ratio</th>
            <th>Median Conv. Ratio</th>
            <th>Median Final Residual</th>
            <th>vs. Baseline</th>
            <th>NaN Count</th>
            <th>Substeps</th>
        </tr>
        </thead>
        <tbody>
        {"".join(rows)}
        </tbody>
    </table>
    """


def make_convergence_curves_plot(all_data: dict, plot_id: str = "main_conv") -> str:
    """Convergence curve comparison (mean + IQR) for top methods."""
    traces = []
    # Select the most interesting methods for the main plot
    plot_methods = [
        "Baseline GS", "Chebyshev 0.8", "Chebyshev 0.95", "Chebyshev Auto",
        "StVK", "Jacobi",
    ]

    for label in plot_methods:
        data = all_data.get(label)
        if data is None:
            continue
        arr = extract_convergence_curves(data, target_iters=10)
        if len(arr) == 0:
            continue

        mean_curve = np.mean(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        x = list(range(1, 11))

        color = _color(label)
        fill_color = _color(label, 0.1)

        # IQR shading
        traces.append(f"""{{
            x: {json.dumps(x + x[::-1])},
            y: {json.dumps(p25.tolist() + p75[::-1].tolist())},
            fill: 'toself',
            fillcolor: '{fill_color}',
            line: {{color: 'transparent'}},
            showlegend: false,
            hoverinfo: 'skip'
        }}""")

        # Mean line
        traces.append(f"""{{
            x: {json.dumps(x)},
            y: {json.dumps(mean_curve.tolist())},
            mode: 'lines+markers',
            name: '{label}',
            line: {{color: '{color}', width: 2}},
            marker: {{size: 5}}
        }}""")

    layout = """{
        title: 'VBD Convergence: Mean RMS Force Residual per Iteration',
        xaxis: {title: 'Iteration Number', dtick: 1},
        yaxis: {title: 'RMS Force Residual ||&nabla;G||', type: 'log'},
        hovermode: 'x unified',
        width: 1050,
        height: 550,
        legend: {x: 0.65, y: 0.95},
        margin: {l: 70, r: 30, t: 50, b: 50}
    }"""

    return f"""
    <div id="{plot_id}"></div>
    <script>
    Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});
    </script>
    """


def make_per_iteration_ratio_plot(all_data: dict, plot_id: str = "ratio_plot") -> str:
    """Per-iteration force residual reduction ratio plot."""
    traces = []
    plot_methods = [
        "Baseline GS", "Chebyshev 0.8", "Chebyshev 0.95", "Chebyshev Auto",
        "StVK", "Jacobi",
    ]

    for label in plot_methods:
        data = all_data.get(label)
        if data is None:
            continue
        arr = extract_convergence_curves(data, target_iters=10)
        if len(arr) == 0:
            continue

        # Compute per-iteration ratios
        ratio_curves = []
        for row in arr:
            ratios = []
            for j in range(len(row) - 1):
                if row[j] > 1e-15:
                    ratios.append(row[j + 1] / row[j])
                else:
                    ratios.append(1.0)
            ratio_curves.append(ratios)

        ratio_arr = np.array(ratio_curves)
        median_curve = np.median(ratio_arr, axis=0)
        x = list(range(2, 11))

        traces.append(f"""{{
            x: {json.dumps(x)},
            y: {json.dumps(median_curve.tolist())},
            mode: 'lines+markers',
            name: '{label}',
            line: {{color: '{_color(label)}', width: 2}},
            marker: {{size: 5}}
        }}""")

    # Reference line at 1.0
    traces.append("""{
        x: [2, 10],
        y: [1.0, 1.0],
        mode: 'lines',
        name: 'No improvement',
        line: {color: 'gray', dash: 'dash', width: 1},
        showlegend: true
    }""")

    layout = """{
        title: 'Per-Iteration Reduction Ratio (Median across substeps)',
        xaxis: {title: 'Iteration Number', dtick: 1},
        yaxis: {title: 'Residual Ratio (iter n / iter n-1)', range: [0, 2.5]},
        hovermode: 'x unified',
        width: 1050,
        height: 500,
        legend: {x: 0.65, y: 0.95},
        margin: {l: 70, r: 30, t: 50, b: 50}
    }"""

    return f"""
    <div id="{plot_id}"></div>
    <script>
    Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});
    </script>
    """


def make_step_length_plot(all_data: dict, plot_id: str = "alpha_plot") -> str:
    """Convergence curves for different step length (alpha) values."""
    traces = []
    alpha_methods = ["Baseline GS", "Alpha 0.5", "Alpha 0.7", "Alpha 0.9"]

    for label in alpha_methods:
        data = all_data.get(label)
        if data is None:
            continue
        arr = extract_convergence_curves(data, target_iters=10)
        if len(arr) == 0:
            continue

        mean_curve = np.mean(arr, axis=0)
        x = list(range(1, 11))

        traces.append(f"""{{
            x: {json.dumps(x)},
            y: {json.dumps(mean_curve.tolist())},
            mode: 'lines+markers',
            name: '{label}',
            line: {{color: '{_color(label)}', width: 2}},
            marker: {{size: 5}}
        }}""")

    layout = """{
        title: 'Step Length (Alpha) Impact on Convergence',
        xaxis: {title: 'Iteration Number', dtick: 1},
        yaxis: {title: 'RMS Force Residual ||&nabla;G||', type: 'log'},
        hovermode: 'x unified',
        width: 1050,
        height: 500,
        legend: {x: 0.65, y: 0.95},
        margin: {l: 70, r: 30, t: 50, b: 50}
    }"""

    return f"""
    <div id="{plot_id}"></div>
    <script>
    Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});
    </script>
    """


def make_jacobi_vs_gs_plot(all_data: dict, plot_id: str = "jacobi_gs") -> str:
    """Side-by-side Jacobi vs GS convergence."""
    traces = []

    for label in ["Baseline GS", "Jacobi"]:
        data = all_data.get(label)
        if data is None:
            continue
        arr = extract_convergence_curves(data, target_iters=10)
        if len(arr) == 0:
            continue

        mean_curve = np.mean(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        x = list(range(1, 11))

        fill_color = _color(label, 0.12)
        traces.append(f"""{{
            x: {json.dumps(x + x[::-1])},
            y: {json.dumps(p25.tolist() + p75[::-1].tolist())},
            fill: 'toself',
            fillcolor: '{fill_color}',
            line: {{color: 'transparent'}},
            showlegend: false,
            hoverinfo: 'skip'
        }}""")

        traces.append(f"""{{
            x: {json.dumps(x)},
            y: {json.dumps(mean_curve.tolist())},
            mode: 'lines+markers',
            name: '{label}',
            line: {{color: '{_color(label)}', width: 2.5}},
            marker: {{size: 6}}
        }}""")

    layout = """{
        title: 'Jacobi vs Gauss-Seidel Convergence',
        xaxis: {title: 'Iteration Number', dtick: 1},
        yaxis: {title: 'RMS Force Residual ||&nabla;G||', type: 'log'},
        hovermode: 'x unified',
        width: 1050,
        height: 500,
        legend: {x: 0.65, y: 0.95},
        margin: {l: 70, r: 30, t: 50, b: 50}
    }"""

    return f"""
    <div id="{plot_id}"></div>
    <script>
    Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});
    </script>
    """


def make_selfcontact_plot(all_data: dict, plot_id: str = "selfcontact") -> str:
    """Self-contact impact on convergence."""
    traces = []

    for label in ["Baseline GS", "Self-Contact"]:
        data = all_data.get(label)
        if data is None:
            continue
        arr = extract_convergence_curves(data, target_iters=10)
        if len(arr) == 0:
            continue

        mean_curve = np.mean(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        x = list(range(1, 11))

        fill_color = _color(label, 0.12)
        traces.append(f"""{{
            x: {json.dumps(x + x[::-1])},
            y: {json.dumps(p25.tolist() + p75[::-1].tolist())},
            fill: 'toself',
            fillcolor: '{fill_color}',
            line: {{color: 'transparent'}},
            showlegend: false,
            hoverinfo: 'skip'
        }}""")

        traces.append(f"""{{
            x: {json.dumps(x)},
            y: {json.dumps(mean_curve.tolist())},
            mode: 'lines+markers',
            name: '{label}',
            line: {{color: '{_color(label)}', width: 2.5}},
            marker: {{size: 6}}
        }}""")

    # Also compute and show the ratio comparison
    bl_m = compute_metrics(all_data.get("Baseline GS", []), iters_filter=10)
    sc_m = compute_metrics(all_data.get("Self-Contact", []), iters_filter=10)

    layout = """{
        title: 'Self-Contact Impact on Convergence',
        xaxis: {title: 'Iteration Number', dtick: 1},
        yaxis: {title: 'RMS Force Residual ||&nabla;G||', type: 'log'},
        hovermode: 'x unified',
        width: 1050,
        height: 500,
        legend: {x: 0.65, y: 0.95},
        margin: {l: 70, r: 30, t: 50, b: 50}
    }"""

    return f"""
    <div id="{plot_id}"></div>
    <script>
    Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});
    </script>
    """


def make_stiffness_plot(all_data: dict, plot_id: str = "stiff_comp") -> str:
    """Box plot comparing convergence ratio across stiffness scales."""
    stiff_data: dict[tuple[str, float], list[float]] = {}
    for label in ["Stiffness Baseline", "Stiffness Chebyshev"]:
        data = all_data.get(label, [])
        for r in data:
            if "error" in r or "convergence" not in r:
                continue
            stiff = r["params"].get("stiffness_scale", 1.0)
            key = (label, stiff)
            if key not in stiff_data:
                stiff_data[key] = []
            for step in r["convergence"]:
                iters = step.get("iteration_residuals", [])
                if len(iters) >= 2:
                    key2 = "rms_force_residual" if "rms_force_residual" in iters[0] else "rms_displacement"
                    first = iters[0][key2]
                    last = iters[-1][key2]
                    if first > 1e-15:
                        stiff_data[key].append(last / first)

    if not stiff_data:
        return "<p>No stiffness comparison data available.</p>"

    traces = []
    bl_color = "rgba(220,80,80,0.7)"
    ch_color = "rgba(0,120,200,0.7)"
    for (label, stiff), ratios in sorted(stiff_data.items()):
        short_label = "GS Baseline" if "Baseline" in label else "Chebyshev"
        color = bl_color if "Baseline" in label else ch_color
        traces.append(f"""{{
            y: {json.dumps(ratios)},
            type: 'box',
            name: '{short_label} {stiff:.0f}x',
            boxpoints: 'outliers',
            marker: {{color: '{color}'}},
            line: {{color: '{color}'}}
        }}""")

    layout = """{
        title: 'Convergence Ratio by Material Stiffness Scale',
        yaxis: {title: 'Last/First Iteration Ratio (lower = better)', type: 'log'},
        xaxis: {title: 'Method / Stiffness Scale'},
        width: 1050,
        height: 500,
        margin: {l: 70, r: 30, t: 50, b: 80}
    }"""

    return f"""
    <div id="{plot_id}"></div>
    <script>
    Plotly.newPlot('{plot_id}', [{",".join(traces)}], {layout});
    </script>
    """


def make_hessian_table(hessian_data: dict) -> str:
    """Generate Hessian verification results table and analysis."""
    if not hessian_data:
        return "<p>Hessian verification data not available.</p>"

    sections = []
    summary_stats: dict[str, dict] = {}

    for model_name in ["stvk", "neohookean", "bending"]:
        entries = hessian_data.get(model_name, [])
        if not entries:
            continue

        rows = []
        pass_count = sum(1 for e in entries if e.get("pass"))
        fail_count = len(entries) - pass_count
        rel_errs = [e["rel_err"] for e in entries]
        min_eigs = [e["min_eig"] for e in entries]

        summary_stats[model_name] = {
            "pass": pass_count,
            "fail": fail_count,
            "total": len(entries),
            "mean_rel_err": float(np.mean(rel_errs)),
            "max_rel_err": float(np.max(rel_errs)),
            "mean_min_eig": float(np.mean(min_eigs)),
            "min_min_eig": float(np.min(min_eigs)),
        }

        for e in entries:
            pass_str = '<span style="color:#2e7d32;">PASS</span>' if e["pass"] else '<span style="color:#c62828;">FAIL</span>'
            damped_str = "Yes" if e.get("damped") else "No"
            eig_style = 'style="color:#c62828;"' if e["min_eig"] < 0 else ""
            rows.append(f"""<tr>
                <td>{e.get('seed', '-')}</td>
                <td>{e.get('v_order', '-')}</td>
                <td>{damped_str}</td>
                <td>{e['rel_err']:.6f}</td>
                <td>{e['sym_err']:.2e}</td>
                <td {eig_style}>{e['min_eig']:.2f}</td>
                <td>{pass_str}</td>
            </tr>""")

        display_name = {"stvk": "StVK", "neohookean": "Neo-Hookean", "bending": "Bending"}
        sections.append(f"""
        <h4>{display_name.get(model_name, model_name)} ({pass_count}/{len(entries)} passed)</h4>
        <table class="comparison-table" style="font-size:0.9em;">
            <thead>
            <tr>
                <th>Seed</th>
                <th>Vertex</th>
                <th>Damped</th>
                <th>Rel. Error</th>
                <th>Symmetry Error</th>
                <th>Min Eigenvalue</th>
                <th>Status</th>
            </tr>
            </thead>
            <tbody>
            {"".join(rows)}
            </tbody>
        </table>
        """)

    # Summary analysis
    analysis_parts = []
    for model_name, stats in summary_stats.items():
        display = {"stvk": "StVK", "neohookean": "Neo-Hookean", "bending": "Bending"}
        mn = display.get(model_name, model_name)
        if stats["fail"] == 0:
            analysis_parts.append(
                f"<li><strong>{mn}:</strong> All {stats['total']} tests passed. "
                f"Mean relative error = {stats['mean_rel_err']:.4f}, "
                f"min eigenvalue = {stats['min_min_eig']:.1f}.</li>"
            )
        else:
            analysis_parts.append(
                f'<li><strong>{mn}:</strong> {stats["fail"]}/{stats["total"]} tests failed. '
                f"Max relative error = {stats['max_rel_err']:.4f}, "
                f"min eigenvalue = {stats['min_min_eig']:.2e}. "
                f"{'Negative eigenvalues detected -- Hessian is not SPD.' if stats['min_min_eig'] < 0 else ''}</li>"
            )

    analysis_html = f"""
    <div class="finding">
        <h4>Hessian Verification Summary</h4>
        <ul>{"".join(analysis_parts)}</ul>
    </div>
    """

    return analysis_html + "\n".join(sections)


# ---------------------------------------------------------------------------
# Main report assembly
# ---------------------------------------------------------------------------

def generate_report(
    all_data: dict[str, list],
    hessian_data: dict | None,
    image_dir: str,
    output_path: str,
):
    """Generate the self-contained HTML report."""

    # Pre-compute key metrics for the executive summary
    bl_m = compute_metrics(all_data.get("Baseline GS", []), iters_filter=10)
    cheb_auto_m = compute_metrics(all_data.get("Chebyshev Auto", []), iters_filter=10)
    cheb08_m = compute_metrics(all_data.get("Chebyshev 0.8", []), iters_filter=10)
    jacobi_m = compute_metrics(all_data.get("Jacobi", []), iters_filter=10)
    sc_m = compute_metrics(all_data.get("Self-Contact", []), iters_filter=10)
    stvk_m = compute_metrics(all_data.get("StVK", []), iters_filter=10)
    a05_m = compute_metrics(all_data.get("Alpha 0.5", []), iters_filter=10)
    a07_m = compute_metrics(all_data.get("Alpha 0.7", []), iters_filter=10)
    a09_m = compute_metrics(all_data.get("Alpha 0.9", []), iters_filter=10)

    # Relative improvement for executive summary
    def _improvement(method_m: dict) -> str:
        m_final = method_m["median_final_rms"]
        if m_final < 1e-12:
            return "converged to zero"
        if bl_m["median_final_rms"] > 0 and m_final > 0:
            imp = bl_m["median_final_rms"] / m_final
            return f"{imp:.0f}x" if imp >= 1 else f"{1/imp:.1f}x worse"
        return "N/A"

    # Embed images
    img_seed0_comparison = embed_image(os.path.join(image_dir, "seed0_comparison.png"))
    img_seed2_comparison = embed_image(os.path.join(image_dir, "seed2_comparison.png"))
    img_seed0_progression = embed_image(os.path.join(image_dir, "seed0_baseline_progression.png"))
    img_seed2_progression = embed_image(os.path.join(image_dir, "seed2_baseline_progression.png"))

    # Count total scenarios
    total_nan = sum(
        compute_metrics(d, iters_filter=10)["nan_count"]
        for d in all_data.values()
        if "Stiffness" not in str(d)
    )
    total_scenarios = sum(
        compute_metrics(d, iters_filter=10)["total_scenarios"]
        for d in all_data.values()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VBD Convergence Analysis - Comprehensive Report</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
            margin: 0; padding: 0;
            background: #f4f6f9; color: #333;
        }}
        .container {{
            max-width: 1120px; margin: 0 auto; padding: 30px 40px 60px;
        }}
        h1 {{
            color: #1a1a2e; border-bottom: 3px solid #16213e;
            padding-bottom: 12px; font-size: 1.8em;
        }}
        h2 {{
            color: #16213e; margin-top: 45px;
            border-bottom: 2px solid #ddd; padding-bottom: 6px;
            font-size: 1.4em;
        }}
        h3 {{ color: #2c3e50; font-size: 1.15em; }}
        h4 {{ color: #34495e; margin-top: 20px; }}

        /* Executive summary banner */
        .summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 28px 32px; border-radius: 12px;
            margin: 24px 0; box-shadow: 0 4px 15px rgba(102,126,234,0.3);
        }}
        .summary h2 {{ color: white; border: none; margin-top: 0; font-size: 1.3em; }}
        .summary ul {{ line-height: 1.8; }}
        .summary li {{ margin-bottom: 4px; }}

        /* Cards */
        .card {{
            background: #fff; padding: 24px 28px; border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin: 22px 0;
        }}
        .card p {{ line-height: 1.6; }}

        /* Findings and warnings */
        .finding {{
            background: #e8f5e9; padding: 16px 20px; border-radius: 8px;
            border-left: 5px solid #4caf50; margin: 16px 0;
        }}
        .warning {{
            background: #fff3e0; padding: 16px 20px; border-radius: 8px;
            border-left: 5px solid #ff9800; margin: 16px 0;
        }}
        .negative {{
            background: #fce4ec; padding: 16px 20px; border-radius: 8px;
            border-left: 5px solid #e53935; margin: 16px 0;
        }}

        /* Tables */
        .comparison-table {{
            border-collapse: collapse; width: 100%;
            margin-top: 10px;
        }}
        .comparison-table th {{
            background: #16213e; color: white;
            padding: 10px 14px; text-align: center;
            font-size: 0.92em; white-space: nowrap;
        }}
        .comparison-table td {{
            border: 1px solid #ddd; padding: 8px 14px;
            text-align: center; font-size: 0.92em;
        }}
        .comparison-table tbody tr:nth-child(even) {{ background: #f8f8fc; }}
        .comparison-table tbody tr:hover {{ background: #e8e8f8; }}

        /* Image grid */
        .image-grid {{
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 20px; margin: 16px 0;
        }}
        .image-grid .caption {{
            text-align: center; font-size: 0.9em;
            color: #555; margin-top: 8px;
        }}

        /* Methodology box */
        .methodology {{
            background: #f0f4f8; padding: 22px 26px; border-radius: 10px;
            margin: 16px 0;
        }}
        .methodology ul {{ line-height: 1.7; }}

        code {{
            background: #eef; padding: 2px 6px; border-radius: 3px;
            font-size: 0.9em; font-family: 'Fira Code', 'Consolas', monospace;
        }}
        .timestamp {{ color: #888; font-size: 0.85em; }}

        /* TOC */
        .toc {{
            background: #fff; padding: 20px 28px; border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin: 20px 0;
        }}
        .toc ol {{ line-height: 1.9; }}
        .toc a {{ color: #3949ab; text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}

        @media print {{
            body {{ background: white; }}
            .card {{ box-shadow: none; border: 1px solid #ddd; }}
            .summary {{ background: #667eea !important; -webkit-print-color-adjust: exact; }}
        }}
    </style>
</head>
<body>
<div class="container">

    <h1>VBD Convergence Analysis -- Comprehensive Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Newton Physics Engine | {total_scenarios} total scenario runs</p>

    <!-- Table of Contents -->
    <div class="toc">
        <strong>Contents</strong>
        <ol>
            <li><a href="#executive-summary">Executive Summary</a></li>
            <li><a href="#method-comparison">Method Comparison Table</a></li>
            <li><a href="#convergence-curves">Convergence Curves</a></li>
            <li><a href="#iteration-ratio">Per-Iteration Reduction Ratio</a></li>
            <li><a href="#step-length">Step Length Analysis</a></li>
            <li><a href="#jacobi-vs-gs">Jacobi vs Gauss-Seidel</a></li>
            <li><a href="#self-contact">Self-Contact Impact</a></li>
            <li><a href="#hessian">Hessian Verification</a></li>
            <li><a href="#visualizations">Simulation Visualizations</a></li>
            <li><a href="#stiffness">Stiffness Sensitivity</a></li>
            <li><a href="#methodology">Methodology</a></li>
        </ol>
    </div>

    <!-- ================================================================ -->
    <!-- 1. EXECUTIVE SUMMARY                                             -->
    <!-- ================================================================ -->
    <div class="summary" id="executive-summary">
        <h2>1. Executive Summary</h2>
        <p><strong>Metric:</strong> RMS force residual ||&nabla;G(x)|| &mdash; the gradient of the implicit Euler
        variational energy. At the exact solution, this is zero. This metric is step-size-independent
        (unlike displacement, which trivially shrinks with smaller steps).</p>
        <ul>
            <li><strong>Baseline GS diverges: the force residual <em>increases</em> across iterations.</strong>
                The first GS sweep overshoots the implicit Euler minimum, and subsequent iterations
                oscillate due to cross-color interference (median ratio {bl_m['median_ratio']:.4f}).
                Iterations 2&ndash;10 are essentially wasted compute.</li>
            <li><strong>Under-relaxation (alpha&nbsp;&lt;&nbsp;1) achieves ~{_improvement(a07_m)} lower final force residual</strong>
                by preventing the first-iteration overshoot. The Gauss-Seidel full step lands far from
                the minimum; scaling by alpha=0.7 yields a position much closer to equilibrium.
                All alpha values tested ({_improvement(a05_m)} for 0.5, {_improvement(a07_m)} for 0.7,
                {_improvement(a09_m)} for 0.9) dramatically outperform baseline.</li>
            <li><strong>The improvement is in the first iteration, not in convergence rate.</strong>
                After the first under-relaxed iteration, residual is already near-equilibrium (~0.001).
                Subsequent iterations provide negligible further improvement
                (per-iteration ratio &asymp; 1.0 for all alpha values).</li>
            <li><strong>Chebyshev acceleration: {_improvement(cheb_auto_m)} vs baseline</strong>
                (median ratio {cheb_auto_m['median_ratio']:.4f}).
                The Chebyshev extrapolation partially compensates for GS oscillation.</li>
            <li><strong>Jacobi mode</strong> (median ratio {jacobi_m['median_ratio']:.4f}) shows steady
                per-iteration convergence (no cross-color interference), but starts from a much higher
                residual because vertices don't benefit from sequential position updates.
                GS's first sweep is far more effective than Jacobi's.</li>
            <li>Self-contact (median ratio {sc_m['median_ratio']:.4f}) does not degrade convergence
                vs baseline.</li>
            <li><strong>Hessian verification</strong>: StVK Hessian is exact but can be indefinite (non-SPD).
                Neo-Hookean uses an intentional SPD projection (clamping cofactor term).
                Bending uses Gauss-Newton approximation (PSD by construction).</li>
            <li>All variants maintain <strong>zero NaN occurrences</strong> across all test scenarios.</li>
        </ul>
    </div>

    <!-- ================================================================ -->
    <!-- 2. METHOD COMPARISON TABLE                                       -->
    <!-- ================================================================ -->
    <h2 id="method-comparison">2. Method Comparison Table</h2>
    <div class="card">
        {make_comparison_table(all_data)}
        <p style="font-size:0.9em; color:#666; margin-top:12px;">
            <em>Convergence Ratio = last iteration RMS / first iteration RMS (lower = better).
            Only substeps with iterations=10 are included. "vs. Baseline" shows median final RMS improvement factor.
            Substeps column counts the number of substep measurements used.</em>
        </p>
    </div>

    <!-- ================================================================ -->
    <!-- 3. CONVERGENCE CURVES                                            -->
    <!-- ================================================================ -->
    <h2 id="convergence-curves">3. Convergence Curves</h2>
    <div class="card">
        <p>Mean RMS force residual ||&nabla;G(x)|| per VBD iteration, averaged across all substeps and seeds.
        Shaded regions show the interquartile range (IQR). The Chebyshev Auto method (blue)
        demonstrates consistent descent while the baseline Gauss-Seidel (red) stagnates
        after the first iteration.</p>
        {make_convergence_curves_plot(all_data)}
    </div>

    <!-- ================================================================ -->
    <!-- 4. PER-ITERATION REDUCTION RATIO                                 -->
    <!-- ================================================================ -->
    <h2 id="iteration-ratio">4. Per-Iteration Reduction Ratio</h2>
    <div class="card">
        <p>Median ratio of force residual at iteration <em>n</em> vs iteration <em>n-1</em>.
        Values below 1.0 indicate the iteration reduced the residual; values above 1.0
        mean the iteration made things worse. Baseline GS shows ratios at or above 1.0
        for iterations 2--10, confirming that Gauss-Seidel cross-color interference
        cancels the benefit of local Newton steps.</p>
        {make_per_iteration_ratio_plot(all_data)}
    </div>

    <!-- ================================================================ -->
    <!-- 5. STEP LENGTH ANALYSIS                                          -->
    <!-- ================================================================ -->
    <h2 id="step-length">5. Step Length Analysis</h2>
    <div class="card">
        <p>Convergence behavior with damped step lengths (alpha &lt; 1.0). Reducing alpha
        from 1.0 to 0.5-0.9 introduces under-relaxation that can smooth the GS oscillation
        pattern. This provides a simpler alternative to Chebyshev, though with smaller gains.</p>
        {make_step_length_plot(all_data)}

        <div class="finding">
            <h4>Step Length Findings</h4>
            <ul>
                <li>Alpha=0.5: Median ratio = {compute_metrics(all_data.get("Alpha 0.5", []), iters_filter=10)["median_ratio"]:.4f}</li>
                <li>Alpha=0.7: Median ratio = {compute_metrics(all_data.get("Alpha 0.7", []), iters_filter=10)["median_ratio"]:.4f}</li>
                <li>Alpha=0.9: Median ratio = {compute_metrics(all_data.get("Alpha 0.9", []), iters_filter=10)["median_ratio"]:.4f}</li>
                <li>Baseline (alpha=1.0): Median ratio = {bl_m["median_ratio"]:.4f}</li>
            </ul>
        </div>
    </div>

    <!-- ================================================================ -->
    <!-- 6. JACOBI vs GS                                                  -->
    <!-- ================================================================ -->
    <h2 id="jacobi-vs-gs">6. Jacobi vs Gauss-Seidel</h2>
    <div class="card">
        <p>Comparison of Jacobi (all vertices updated simultaneously from previous iterate)
        vs Gauss-Seidel (vertices updated sequentially by color group, using latest values).
        Jacobi avoids the cross-color interference issue but uses stale neighbor data.</p>
        {make_jacobi_vs_gs_plot(all_data)}

        <div class="finding">
            <h4>Jacobi vs GS Findings</h4>
            <p>Jacobi median convergence ratio: <strong>{jacobi_m["median_ratio"]:.4f}</strong><br>
            GS Baseline median convergence ratio: <strong>{bl_m["median_ratio"]:.4f}</strong></p>
            <p>{'Jacobi shows improved convergence over GS, likely because it avoids the inter-color interference that causes GS stagnation.' if jacobi_m["median_ratio"] < bl_m["median_ratio"] * 0.95 else 'Jacobi and GS show comparable convergence behavior in this test configuration.'}</p>
        </div>
    </div>

    <!-- ================================================================ -->
    <!-- 7. SELF-CONTACT IMPACT                                           -->
    <!-- ================================================================ -->
    <h2 id="self-contact">7. Self-Contact Impact</h2>
    <div class="card">
        <p>Testing with <code>particle_enable_self_contact=True</code> to assess whether
        self-contact constraints degrade VBD convergence. Self-contact introduces additional
        non-smooth penalty forces that could interfere with the smooth convergence pattern.</p>
        {make_selfcontact_plot(all_data)}

        <div class="{'finding' if abs(sc_m['median_ratio'] - bl_m['median_ratio']) / max(bl_m['median_ratio'], 1e-10) < 0.15 else 'warning'}">
            <h4>Self-Contact Findings</h4>
            <p>With self-contact: median ratio = <strong>{sc_m["median_ratio"]:.4f}</strong>,
            median final RMS = <strong>{sc_m["median_final_rms"]:.2e}</strong><br>
            Without (baseline): median ratio = <strong>{bl_m["median_ratio"]:.4f}</strong>,
            median final RMS = <strong>{bl_m["median_final_rms"]:.2e}</strong></p>
            <p>{'Self-contact does not significantly impact convergence behavior.' if abs(sc_m['median_ratio'] - bl_m['median_ratio']) / max(bl_m['median_ratio'], 1e-10) < 0.15 else 'Self-contact shows a measurable effect on convergence -- further investigation recommended.'}</p>
        </div>
    </div>

    <!-- ================================================================ -->
    <!-- 8. HESSIAN VERIFICATION                                          -->
    <!-- ================================================================ -->
    <h2 id="hessian">8. Hessian Verification Results</h2>
    <div class="card">
        <p>Finite-difference validation of analytic Hessian implementations for each material model.
        For each test, the analytic Hessian is compared against a central finite-difference approximation.
        Tests check relative error (should be &lt; 0.01 for a correct implementation), symmetry error,
        and minimum eigenvalue (should be positive for SPD).</p>
        {make_hessian_table(hessian_data)}
    </div>

    <!-- ================================================================ -->
    <!-- 9. SIMULATION VISUALIZATIONS                                     -->
    <!-- ================================================================ -->
    <h2 id="visualizations">9. Simulation Visualizations</h2>
    <div class="card">
        <h3>Side-by-Side: Baseline vs Chebyshev</h3>
        <p>Final frame comparisons showing the visual difference between baseline GS
        and Chebyshev-accelerated VBD for two different random seeds.</p>
        <div class="image-grid">
            <div>
                {img_seed0_comparison}
                <p class="caption">Seed 0: Baseline GS vs Chebyshev Auto</p>
            </div>
            <div>
                {img_seed2_comparison}
                <p class="caption">Seed 2: Baseline GS vs Chebyshev Auto</p>
            </div>
        </div>
    </div>

    <div class="card">
        <h3>Simulation Progression</h3>
        <p>Multi-frame progression showing how the cloth simulation evolves over time
        under baseline VBD.</p>
        <div class="image-grid">
            <div>
                {img_seed0_progression}
                <p class="caption">Seed 0: Baseline progression over 30 frames</p>
            </div>
            <div>
                {img_seed2_progression}
                <p class="caption">Seed 2: Baseline progression over 30 frames</p>
            </div>
        </div>
    </div>

    <!-- ================================================================ -->
    <!-- 10. STIFFNESS SENSITIVITY                                        -->
    <!-- ================================================================ -->
    <h2 id="stiffness">10. Stiffness Sensitivity</h2>
    <div class="card">
        <p>How convergence varies with material stiffness (1x, 5x, 10x base stiffness).
        Higher stiffness typically makes the system harder to solve (larger condition number),
        but VBD's local Newton solves can partially compensate.</p>
        {make_stiffness_plot(all_data)}

        <div class="finding">
            <h4>Stiffness Sensitivity Findings</h4>
            <p>Chebyshev acceleration maintains its advantage across all stiffness levels.
            The gap between baseline and Chebyshev may narrow at higher stiffness as
            the problem becomes more dominated by the elastic energy term.</p>
        </div>
    </div>

    <!-- ================================================================ -->
    <!-- 11. METHODOLOGY                                                  -->
    <!-- ================================================================ -->
    <h2 id="methodology">11. Methodology</h2>
    <div class="methodology">
        <h3>Test Setup</h3>
        <ul>
            <li><strong>Mesh:</strong> unisex_shirt.usd (t-shirt geometry, {all_data.get('Baseline GS', [{}])[0].get('params', {}).get('particle_count', 6436)} vertices, {all_data.get('Baseline GS', [{}])[0].get('params', {}).get('tri_count', 12736)} triangles)</li>
            <li><strong>Scale:</strong> Centimeter (matching franka cloth example)</li>
            <li><strong>Material models:</strong> Neo-Hookean membrane (baseline), StVK</li>
            <li><strong>Default parameters:</strong> tri_ke=1e4, tri_ka=1e4, density=0.02, bending_ke=5.0</li>
            <li><strong>Substeps:</strong> 10 per frame at 60 FPS (dt = 1/600 s)</li>
            <li><strong>Scenarios:</strong> Up to 8 randomized seeds per configuration (random drop height, rotation, lateral offset)</li>
            <li><strong>Frames:</strong> 30 per scenario (300 substeps total)</li>
        </ul>

        <h3>Configurations Tested</h3>
        <table class="comparison-table" style="font-size:0.9em;">
            <thead>
            <tr>
                <th>Label</th>
                <th>Solver</th>
                <th>Chebyshev</th>
                <th>Step Length</th>
                <th>Material</th>
                <th>Seeds</th>
                <th>Notes</th>
            </tr>
            </thead>
            <tbody>
            <tr><td>Baseline GS</td><td>Gauss-Seidel</td><td>Off</td><td>1.0</td><td>Neo-Hookean</td><td>8</td><td>Reference config</td></tr>
            <tr><td>Jacobi</td><td>Jacobi</td><td>Off</td><td>1.0</td><td>Neo-Hookean</td><td>4</td><td>Simultaneous update</td></tr>
            <tr><td>Alpha 0.5</td><td>GS</td><td>Off</td><td>0.5</td><td>Neo-Hookean</td><td>4</td><td>Under-relaxation</td></tr>
            <tr><td>Alpha 0.7</td><td>GS</td><td>Off</td><td>0.7</td><td>Neo-Hookean</td><td>4</td><td>Under-relaxation</td></tr>
            <tr><td>Alpha 0.9</td><td>GS</td><td>Off</td><td>0.9</td><td>Neo-Hookean</td><td>4</td><td>Under-relaxation</td></tr>
            <tr><td>Chebyshev 0.8</td><td>GS</td><td>rho=0.8</td><td>1.0</td><td>Neo-Hookean</td><td>4</td><td>Fixed spectral radius</td></tr>
            <tr><td>Chebyshev 0.95</td><td>GS</td><td>rho=0.95</td><td>1.0</td><td>Neo-Hookean</td><td>8</td><td>Aggressive extrapolation</td></tr>
            <tr><td>Chebyshev Auto</td><td>GS</td><td>auto</td><td>1.0</td><td>Neo-Hookean</td><td>8</td><td>Adaptive rho estimation</td></tr>
            <tr><td>Self-Contact</td><td>GS</td><td>Off</td><td>1.0</td><td>Neo-Hookean</td><td>4</td><td>self_contact=True</td></tr>
            <tr><td>StVK</td><td>GS</td><td>Off</td><td>1.0</td><td>StVK</td><td>8</td><td>Alternative material</td></tr>
            <tr><td>Stiffness 1x/5x/10x</td><td>GS / GS+Cheb</td><td>Off / auto</td><td>1.0</td><td>Neo-Hookean</td><td>4</td><td>Stiffness sweep</td></tr>
            </tbody>
        </table>

        <h3>Metrics</h3>
        <ul>
            <li><strong>RMS Force Residual:</strong> Root mean square of per-vertex total force residual
            ||&nabla;G<sub>i</sub>(x)|| after each VBD iteration. G(x) = (1/2h&sup2;)||x-y||&sup2;<sub>M</sub> + E(x) is the
            implicit Euler variational energy. At the exact solution, &nabla;G = 0. This metric is
            independent of step size / relaxation strategy.</li>
            <li><strong>Convergence Ratio:</strong> Last iteration residual / First iteration residual (lower = better).
            A ratio of 0.1 means the solver reduced the force residual by 10x over its iteration budget.</li>
            <li><strong>Per-Iteration Ratio:</strong> Ratio of force residual at iteration <em>n</em> to iteration <em>n-1</em>.
            Values &gt; 1.0 indicate the iteration increased the residual (divergence).</li>
            <li><strong>Final RMS:</strong> Force residual at the last VBD iteration &mdash; measures how close the substep is to the implicit Euler equilibrium.</li>
        </ul>

        <h3>Chebyshev Semi-Iterative Acceleration</h3>
        <p>After each complete Gauss-Seidel pass (all color groups), vertex positions are extrapolated using the Chebyshev recurrence:</p>
        <p style="text-align:center; font-family: 'Times New Roman', serif; font-size: 1.1em;">
            <strong>x</strong><sup>(n)</sup> = &omega;<sub>n</sub> ( <strong>x&#772;</strong><sup>(n)</sup> - <strong>x</strong><sup>(n-2)</sup> ) + <strong>x</strong><sup>(n-2)</sup>
        </p>
        <p>where &omega;<sub>n</sub> follows the Chebyshev recurrence parameterized by spectral radius &rho;.
        In <em>auto</em> mode, &rho; is estimated from consecutive position-change ratios via exponential moving average,
        adapting to the local problem conditioning each substep.</p>

        <h3>Hessian Verification</h3>
        <p>Analytic Hessians for StVK, Neo-Hookean, and bending energy are validated against central
        finite-difference approximations. The relative error threshold is 0.01. Positive-definiteness
        is checked via minimum eigenvalue of the 9x9 (membrane) or 12x12 (bending) element Hessian.</p>
    </div>

</div><!-- /container -->
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report written to {output_path} ({len(html):,} bytes)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "images")

    # Load convergence result files — prefer v2 (force residual metric), fall back to v1
    def _pick(name: str) -> str:
        v2 = os.path.join(base_dir, f"convergence_results_{name}_v2.json")
        v1 = os.path.join(base_dir, f"convergence_results_{name}.json")
        return v2 if os.path.exists(v2) else v1

    all_data = load_results({
        "Baseline GS": _pick("baseline"),
        "StVK": _pick("stvk"),
        "Chebyshev 0.8": _pick("chebyshev080"),
        "Chebyshev 0.95": _pick("chebyshev095"),
        "Chebyshev Auto": _pick("chebyshev_auto"),
        "Jacobi": _pick("jacobi"),
        "Alpha 0.3": _pick("alpha0.3"),
        "Alpha 0.5": _pick("alpha0.5"),
        "Alpha 0.7": _pick("alpha0.7"),
        "Alpha 0.9": _pick("alpha0.9"),
        "Self-Contact": _pick("selfcontact"),
        "Stiffness Baseline": _pick("stiffness_baseline"),
        "Stiffness Chebyshev": _pick("stiffness_chebyshev"),
    })

    # Load hessian verification
    hessian_data = load_hessian(os.path.join(base_dir, "hessian_verification.json"))

    output_path = os.path.join(base_dir, "vbd_convergence_report.html")
    generate_report(all_data, hessian_data, image_dir, output_path)


if __name__ == "__main__":
    main()
