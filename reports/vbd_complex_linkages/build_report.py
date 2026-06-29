#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the static complex-linkage comparison report."""

from __future__ import annotations

import html
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports" / "vbd_complex_linkages"
OUTPUT = Path.home() / "reports" / ".hidden" / "vbd-complex-linkages"
VIDEO_SOURCE = Path.home() / "reports" / "vbd-complex-linkages" / "videos"
ASSET_SOURCE = SOURCE / "assets"
FORMULATION_DATA_SOURCE = SOURCE / "formulation_data"
GITHUB_BRANCH = "https://github.com/mmacklin/newton/tree/horde/vbd-sparse-articulation-main"
GITHUB_BLOB = "https://github.com/mmacklin/newton/blob/horde/vbd-sparse-articulation-main"
MEDIA_VERSION = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load(name: str) -> dict:
    return json.loads((SOURCE / name).read_text())


def _row(payload: dict, solver: str) -> dict:
    return next(row for row in payload["rows"] if row["solver"] == solver)


def _sweep_row(payload: dict, scenario: str, iterations: int, mode: str) -> dict:
    return next(
        row
        for row in payload["rows"]
        if row["scenario"] == scenario and row["iterations"] == iterations and row["mode"] == mode
    )


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


def _budget_reduction(row: dict) -> str:
    value = row.get("residual_reduction_at_budget")
    return "-" if value is None else f"{value:,.1f}x"


def _video(name: str, title: str, text: str) -> str:
    stem = name.removesuffix(".mp4")
    return f"""
    <article class="media">
      <h3>{html.escape(title)}</h3>
      <video controls muted loop playsinline preload="metadata" poster="videos/{stem}.jpg">
        <source src="videos/{name}?v={MEDIA_VERSION}" type="video/mp4">
      </video>
      <p>{html.escape(text)}</p>
    </article>"""


def _cable_convergence_svg(rows: list[dict]) -> str:
    width, height = 860, 340
    left, right, top, bottom = 76, 24, 24, 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    times = [row["p50_step_us"] / 1.0e3 for row in rows]
    residuals = [row["residual_m"] * 1.0e6 for row in rows]
    x_min, x_max = math.log10(min(times) * 0.85), math.log10(max(times) * 1.15)
    y_min, y_max = math.log10(min(residuals) * 0.65), math.log10(max(residuals) * 1.5)

    def x(value: float) -> float:
        return left + plot_width * (math.log10(value) - x_min) / (x_max - x_min)

    def y(value: float) -> float:
        return top + plot_height * (y_max - math.log10(value)) / (y_max - y_min)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Cable residual versus CPU wall time">',
        '<rect width="100%" height="100%" fill="#f7f8f7"/>',
    ]
    for tick in (0.2, 0.5, 1.0, 2.0):
        if min(times) * 0.85 <= tick <= max(times) * 1.15:
            px = x(tick)
            parts.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height - bottom}" stroke="#d7dce0"/>')
            parts.append(f'<text x="{px:.1f}" y="{height - bottom + 22}" text-anchor="middle">{tick:g}</text>')
    for tick in (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        if min(residuals) * 0.65 <= tick <= max(residuals) * 1.5:
            py = y(tick)
            parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" stroke="#d7dce0"/>')
            parts.append(f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end">{tick:g}</text>')
    colors = {"local": "#a6411d", "block_sparse_joints": "#006c67"}
    labels = {"local": "VBD local", "block_sparse_joints": "VBD sparse direct"}
    for mode in ("local", "block_sparse_joints"):
        mode_rows = sorted((row for row in rows if row["mode"] == mode), key=lambda row: row["p50_step_us"])
        points = " ".join(
            f"{x(row['p50_step_us'] / 1.0e3):.1f},{y(row['residual_m'] * 1.0e6):.1f}" for row in mode_rows
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[mode]}" stroke-width="3"/>')
        for row in mode_rows:
            px = x(row["p50_step_us"] / 1.0e3)
            py = y(row["residual_m"] * 1.0e6)
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{colors[mode]}"/>')
            parts.append(f'<text x="{px + 7:.1f}" y="{py - 7:.1f}" fill="{colors[mode]}">i{row["iterations"]}</text>')
        legend_x = 575 if mode == "local" else 700
        parts.append(
            f'<line x1="{legend_x}" y1="16" x2="{legend_x + 22}" y2="16" stroke="{colors[mode]}" stroke-width="3"/>'
        )
        parts.append(f'<text x="{legend_x + 28}" y="20">{labels[mode]}</text>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#171a1d"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#171a1d"/>',
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 10}" text-anchor="middle">CPU p50 substep [ms]</text>',
            f'<text transform="translate(18 {top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">Aggregate joint residual [µm]</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def main() -> None:
    foot = _load("robot_foot_compatible_results.json")
    foot_geometry = _load("robot_foot_geometry_diagnostic.json")
    g1 = _load("g1_ankle_results.json")
    dr_free_ankle = _load("dr_legs_free_ankle_results.json")
    dr_free_ankle_cuda = _load("dr_legs_free_ankle_cuda_results.json")
    matrix = _load("dr_legs_matrix_diagnostic.json")
    visual_validation = _load("visual_validation_results.json")
    iteration_sweep = json.loads((FORMULATION_DATA_SOURCE / "iteration_sweep_cpu_sparse.json").read_text())
    g1_cpu_graph = json.loads((FORMULATION_DATA_SOURCE / "g1_cpu_graph.json").read_text())
    g1_cuda_graph = json.loads((FORMULATION_DATA_SOURCE / "g1_cuda_graph.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    videos = OUTPUT / "videos"
    videos.mkdir(exist_ok=True)
    for path in VIDEO_SOURCE.iterdir():
        if path.suffix in (".mp4", ".jpg", ".json"):
            shutil.copy2(path, videos / path.name)
    for source_dir, output_name in ((ASSET_SOURCE, "assets"), (FORMULATION_DATA_SOURCE, "formulation_data")):
        output_dir = OUTPUT / output_name
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(source_dir, output_dir)
    for name in (
        "robot_foot_compatible_results.json",
        "robot_foot_geometry_diagnostic.json",
        "g1_ankle_results.json",
        "dr_legs_free_ankle_results.json",
        "dr_legs_free_ankle_cuda_results.json",
        "dr_legs_matrix_diagnostic.json",
        "visual_validation_results.json",
    ):
        shutil.copy2(SOURCE / name, OUTPUT / name)
    for stale_name in ("robot_foot_results.json", "dr_legs_results.json", "dr_legs_tuning"):
        stale_path = OUTPUT / stale_name
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
        elif stale_path.exists():
            stale_path.unlink()

    foot_rows = {solver: _row(foot, solver) for solver in ("kamino", "vbd")}
    foot_local = next(row for row in foot["rows"] if row.get("vbd_solve") == "local")
    foot_sparse = next(row for row in foot["rows"] if row.get("vbd_solve") == "block_sparse_joints")
    g1_kamino = _row(g1, "kamino")
    g1_local = next(row for row in g1["rows"] if row.get("vbd_solve") == "local")
    g1_sparse = next(row for row in g1["rows"] if row.get("vbd_solve") == "block_sparse_joints")
    dr_free_kamino = _row(dr_free_ankle, "kamino")
    dr_free_local = next(row for row in dr_free_ankle["rows"] if row.get("vbd_solve") == "local")
    dr_free_sparse = next(row for row in dr_free_ankle["rows"] if row.get("vbd_solve") == "block_sparse_joints")
    dr_cuda_kamino = _row(dr_free_ankle_cuda, "kamino")
    dr_cuda_sparse = next(row for row in dr_free_ankle_cuda["rows"] if row.get("vbd_solve") == "block_sparse_joints")
    visual_rows = {row["scenario"]: row for row in visual_validation["visuals"]}
    four_bar_visual = visual_rows["four-bar"]
    cable_visual = visual_rows["cable"]
    cable_convergence = visual_validation["cable_convergence"]
    cable_i4_sparse = next(
        row for row in cable_convergence if row["mode"] == "block_sparse_joints" and row["iterations"] == 4
    )
    cable_table_rows = "".join(
        "<tr>"
        f"<td>{'VBD local' if row['mode'] == 'local' else 'VBD sparse direct'}</td>"
        f"<td>{row['iterations']}</td>"
        f"<td>{row['p50_step_us'] / 1.0e3:.3f}</td>"
        f"<td>{row['residual_m'] * 1.0e6:,.3f}</td>"
        f"<td>{_budget_reduction(row)}</td>"
        "</tr>"
        for row in cable_convergence
    )
    synthetic = {
        scenario: {
            (iterations, mode): _sweep_row(iteration_sweep, scenario, iterations, mode)
            for iterations in (1, 8)
            for mode in ("local", "block_sparse_joints")
        }
        for scenario in ("chain_fixed", "chain_revolute", "loop_fixed")
    }
    cpu_graph_rows = {row["mode"]: row for row in g1_cpu_graph["rows"]}
    cuda_graph_rows = {row["mode"]: row for row in g1_cuda_graph["rows"]}

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = rf"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VBD Sparse Direct Articulation Solver</title>
<script>window.MathJax={{tex:{{inlineMath:[["\\(","\\)"]],displayMath:[["\\[","\\]"]]}},svg:{{fontCache:"global"}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
:root{{--ink:#171a1d;--muted:#596169;--line:#d7dce0;--paper:#fff;--soft:#f4f6f7;--accent:#006c67;--warn:#9b3d18}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:36px 28px 72px}} h1{{font-size:34px;line-height:1.1;margin:0 0 10px;letter-spacing:0}}
h2{{font-size:22px;margin:42px 0 14px;border-bottom:1px solid var(--line);padding-bottom:8px;letter-spacing:0}} h3{{font-size:16px;margin:0 0 10px}}
p{{max-width:850px}} .lede{{font-size:18px;color:#30363b}} .meta,.note{{color:var(--muted)}} .status{{border-left:4px solid var(--accent);padding:12px 16px;background:var(--soft);max-width:900px}}
.warn{{border-left-color:var(--warn)}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}} th,td{{text-align:right;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{background:var(--soft);font-size:13px}}
.media-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}} .media{{min-width:0}} video{{display:block;width:100%;aspect-ratio:16/9;background:#111}} code{{background:var(--soft);padding:1px 4px}}
.validation-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}} .validation-grid article{{min-width:0}}
.pill{{display:inline-block;border:1px solid var(--line);padding:2px 7px;margin-right:5px;font-size:12px}} a{{color:var(--accent)}}
figure{{margin:24px 0}} figure img{{display:block;width:100%;height:auto;border:1px solid var(--line);background:#f7f8f7}} figcaption{{margin-top:8px;color:var(--muted);font-size:13px}}
figure>svg{{display:block;width:100%;height:auto;border:1px solid var(--line);background:#f7f8f7}} figure>svg text{{font:12px system-ui,sans-serif}}
.formula-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin:18px 0}} .formula-grid>div{{border-top:3px solid var(--accent);padding:10px 14px;background:var(--soft)}}
.code-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin:18px 0}} pre{{margin:0;overflow:auto;background:#202724;color:#edf2ee;padding:16px;border-radius:4px;font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}} pre code{{background:none;padding:0}}
.code-links{{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 20px}} .code-links a{{border:1px solid var(--line);padding:7px 10px;text-decoration:none;background:var(--soft)}}
.equation-note{{border-left:4px solid var(--accent);padding:10px 14px;background:var(--soft);max-width:930px}}
.analysis-heading{{margin:26px 0 10px}} .glossary{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 28px;margin:14px 0 26px}}
.glossary>div{{border-top:1px solid var(--line);padding:10px 0}} .glossary dt{{font-weight:700}} .glossary dd{{margin:3px 0 0;color:var(--muted)}}
@media(max-width:800px){{main{{padding:24px 16px}}.media-grid,.validation-grid{{grid-template-columns:1fr}}table{{font-size:12px}}th,td{{padding:7px 5px}}}}
@media(max-width:800px){{.formula-grid,.code-grid,.glossary{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>VBD Sparse Direct Articulation Solver</h1>
<p class="lede">Formulation, sparse implementation, contact treatment, performance, and closed-loop mechanism comparisons for an articulation-wide maximal-coordinate primal solver.</p>
<p class="meta">Generated {timestamp}. Restricted source assets are not redistributed; this report contains aggregate measurements and rendered behavior.</p>

<h2>Summary</h2>
<div class="status"><strong>Sparse VBD gives the lowest closure error on the feasible driven linkages.</strong> On the compatible three-pushrod foot it reduces aggregate closure RMS by {foot_local["rms_closure_um"] / foot_sparse["rms_closure_um"]:.1f}x versus local VBD and {foot_rows["kamino"]["rms_closure_um"] / foot_sparse["rms_closure_um"]:.1f}x versus tuned Kamino. On the G1 ankle settling test the corresponding reductions are {g1_local["rms_closure_um"] / g1_sparse["rms_closure_um"]:.1f}x and {g1_kamino["rms_closure_um"] / g1_sparse["rms_closure_um"]:.1f}x.</div>
<p>Sparse VBD is also the fastest CPU path in the two contact-free mechanisms: median substep latency is {_fmt(foot_sparse["p50_step_us"] / 1.0e3, 3)} ms for the foot and {_fmt(g1_sparse["p50_step_us"] / 1.0e3, 3)} ms for the G1 ankle. In the larger DR Legs test its median CPU cost is {_fmt(dr_free_sparse["p50_solver_us"] / 1.0e3, 2)} ms for the solver and {_fmt((dr_free_sparse["p50_solver_us"] + dr_free_sparse["p50_collision_us"]) / 1.0e3, 2)} ms including collision; local VBD becomes nonfinite and Kamino reaches its iteration cap on most substeps.</p>

<h2>Solver landscape</h2>
<p>Two choices organize the methods compared here. A <em>reduced-coordinate</em> model stores joint coordinates and derives body poses from an articulation tree. A <em>maximal-coordinate</em> model stores every body pose independently and enforces joints between bodies. A <em>primal</em> solve updates positions or pose increments directly, while a <em>dual</em> or primal-dual solve introduces constraint reactions or multipliers.</p>
<figure>
  <img src="assets/solver-formulation-grid.svg?v=20260629-unified" alt="Two-by-two map of reduced and maximal coordinate solvers using primal and dual variables">
  <figcaption>VBD sparse direct occupies the maximal-coordinate, primal quadrant: independent body poses support closed loops, while the Newton step is solved in pose increments rather than joint multipliers. <a href="assets/solver-formulation-grid.svg">SVG</a> · <a href="assets/solver-formulation-grid.png">PNG</a></figcaption>
</figure>

<h2>Primal maximal-coordinate formulation</h2>
<p>For an articulation with bodies \(i=1,\ldots,n\), VBD stores world poses \(q_i\in SE(3)\). One nonlinear iteration introduces a six-vector tangent increment \(\Delta x_i=(\Delta p_i,\Delta\theta_i)\). The step objective combines inertia, joints and drives, and contact:</p>
\[
E(\mathbf{{q}})=\sum_i E_{{\mathrm{{inertia}},i}}(q_i)
+\sum_j E_{{\mathrm{{joint}},j}}(q_{{p(j)}},q_{{c(j)}})
+\sum_d E_{{\mathrm{{drive}},d}}(\mathbf{{q}})
+\sum_c E_{{\mathrm{{contact}},c}}(q_{{a(c)}},q_{{b(c)}}).
\]
<p>Linearizing in the body tangent space gives the articulation Newton system</p>
\[
H\,\Delta\mathbf{{x}}=-\mathbf{{g}},\qquad
\mathbf{{g}}=\nabla_{{\mathbf{{x}}}}E,\qquad
H\approx\nabla^2_{{\mathbf{{x}}}}E+\epsilon I,
\]
<p>followed by the manifold update \(q_i\leftarrow q_i\oplus\alpha\Delta x_i\). The relaxation \(\alpha\) is normally one; contact-heavy examples can use a smaller value because the contact set and local contact model remain fixed during the inner iteration.</p>

<div class="formula-grid">
  <div>
    <h3>Joint contribution</h3>
    <p>For a joint residual \(r_j\), projected Jacobians \(J_p,J_c\), and stiffness matrix \(K_j\), Gauss-Newton assembly adds</p>
    \[
    H_j=\begin{{bmatrix}}J_p^T K_jJ_p &amp; J_p^T K_jJ_c\\J_c^T K_jJ_p &amp; J_c^T K_jJ_c\end{{bmatrix}}.
    \]
    <p>The off-diagonal blocks couple the parent and child body updates. Closure joints are ordinary graph edges and enter the same assembly.</p>
  </div>
  <div>
    <h3>Local versus sparse VBD</h3>
    <p>Local VBD keeps only independent body blocks when computing an update. Sparse-direct VBD retains all joint-generated \(6\times6\) off-diagonal blocks within the articulation, then solves the coupled system with block Cholesky.</p>
    \[
    \Delta x_i^{{\mathrm{{local}}}}=-H_{{ii}}^{{-1}}g_i,
    \qquad
    \Delta\mathbf{{x}}^{{\mathrm{{sparse}}}}=-H_{{\mathrm{{art}}}}^{{-1}}\mathbf{{g}}.
    \]
  </div>
</div>

<h2>Sparse structure and contact treatment</h2>
<figure>
  <img src="assets/sparse-articulation-pattern.svg?v=20260629-unified" alt="Joint graph, six-by-six block articulation Hessian, and symbolic Cholesky fill pattern">
  <figcaption>Joint topology determines the off-diagonal matrix and factor pattern. Dynamic body-body contacts affect the right-hand side and outlined diagonal blocks, so they do not invalidate symbolic factorization. <a href="assets/sparse-articulation-pattern.svg">Open SVG</a></figcaption>
</figure>
<p>The matrix used by the current solver can be summarized as</p>
\[
H_{{\mathrm{{art}}}}=
H_{{\mathrm{{inertia}}}}+
\sum_j H_{{\mathrm{{joint}},j}}+
\operatorname{{blockdiag}}\!\left(\sum_c H_{{\mathrm{{contact}},c}}\right)+
\epsilon I.
\]
<p>Each active contact contributes current forces and torques to both incident bodies, so its gradient is present in \(-\mathbf{{g}}\) in the same VBD iteration as joint assembly. Only the per-body diagonal contact curvature is retained in \(H\); cross-body contact Hessian blocks are omitted. This is a deliberate quasi-Newton approximation: nonlinear contact is already revisited across VBD iterations and substeps, while the sparse factor pattern remains a function of static joint topology alone.</p>
<p>At model finalization, Newton groups each articulation's bodies and joints, applies a greedy minimum-degree order, computes symbolic Cholesky fill, and precomputes column and Schur-update slots. Per iteration, Warp only zeros numeric storage, assembles body and joint terms, factors the existing pattern, performs forward/back substitution, and applies the pose increments. Bodies not assigned to a multi-body articulation are represented as one-body articulations, so the same path covers isolated rigid bodies.</p>

<h2>Implementation and code</h2>
<div class="code-grid">
  <div>
    <h3>Public solver configuration</h3>
    <pre><code>solver = newton.solvers.SolverVBD(
    model,
    iterations=8,
    rigid_articulation_solve=&quot;block_sparse_joints&quot;,
    rigid_articulation_relaxation=1.0,
    rigid_articulation_diagonal_regularization=0.0,
)</code></pre>
  </div>
  <div>
    <h3>Per-iteration execution</h3>
    <pre><code># Static at model finalization
order = minimum_degree(joint_graph)
pattern = symbolic_cholesky(order, joint_graph)

# Numeric work each VBD iteration
assemble_body_and_contact_diagonals(H, g)
assemble_joint_blocks(H, g)
L = block_cholesky(H, pattern)
delta = solve(L, -g)
apply_pose_updates(delta, relaxation)</code></pre>
  </div>
</div>
<div class="code-links">
  <a href="{GITHUB_BRANCH}">GitHub branch</a>
  <a href="{GITHUB_BLOB}/newton/_src/solvers/vbd/rigid_sparse_articulation.py">Symbolic layout</a>
  <a href="{GITHUB_BLOB}/newton/_src/solvers/vbd/rigid_sparse_articulation_kernels.py">Warp assembly and solve kernels</a>
  <a href="{GITHUB_BLOB}/newton/_src/solvers/vbd/solver_vbd.py">Solver integration</a>
  <a href="{GITHUB_BLOB}/newton/tests/test_vbd_sparse_articulation.py">Regression tests</a>
  <a href="{GITHUB_BLOB}/reports/vbd_sparse_articulation/bench_vbd_sparse_articulation.py">Synthetic benchmark</a>
  <a href="{GITHUB_BLOB}/reports/vbd_sparse_articulation/bench_robot_perf.py">Robot performance harness</a>
</div>

<h2>Synthetic convergence evidence</h2>
<p>Before using full robot models, 32-body fixed, revolute, and low-width loop systems isolate articulation convergence. The reported composite residual is the L2 norm of per-joint anchor distance and orientation error after the requested number of VBD iterations; it is used only for same-model solver ratios.</p>
<table><thead><tr><th>Scenario</th><th>Local residual, i1</th><th>Sparse residual, i1</th><th>Local residual, i8</th><th>Sparse residual, i8</th><th>i8 reduction</th></tr></thead><tbody>
<tr><td>Fixed chain</td><td>{synthetic["chain_fixed"][(1, "local")]["joint_residual_l2"]:.6f}</td><td>{synthetic["chain_fixed"][(1, "block_sparse_joints")]["joint_residual_l2"]:.6f}</td><td>{synthetic["chain_fixed"][(8, "local")]["joint_residual_l2"]:.6f}</td><td>{synthetic["chain_fixed"][(8, "block_sparse_joints")]["joint_residual_l2"]:.6f}</td><td>{synthetic["chain_fixed"][(8, "block_sparse_joints")]["residual_reduction_vs_local"]:.2f}x</td></tr>
<tr><td>Revolute chain</td><td>{synthetic["chain_revolute"][(1, "local")]["joint_residual_l2"]:.6f}</td><td>{synthetic["chain_revolute"][(1, "block_sparse_joints")]["joint_residual_l2"]:.6f}</td><td>{synthetic["chain_revolute"][(8, "local")]["joint_residual_l2"]:.6f}</td><td>{synthetic["chain_revolute"][(8, "block_sparse_joints")]["joint_residual_l2"]:.6f}</td><td>{synthetic["chain_revolute"][(8, "block_sparse_joints")]["residual_reduction_vs_local"]:.2f}x</td></tr>
<tr><td>Fixed chain with closure</td><td>{synthetic["loop_fixed"][(1, "local")]["joint_residual_l2"]:.6f}</td><td>{synthetic["loop_fixed"][(1, "block_sparse_joints")]["joint_residual_l2"]:.6f}</td><td>{synthetic["loop_fixed"][(8, "local")]["joint_residual_l2"]:.6f}</td><td>{synthetic["loop_fixed"][(8, "block_sparse_joints")]["joint_residual_l2"]:.6f}</td><td>{synthetic["loop_fixed"][(8, "block_sparse_joints")]["residual_reduction_vs_local"]:.2f}x</td></tr>
</tbody></table>
<p>The direct articulation update is already better after one iteration and reaches roughly 2.1x lower residual by eight iterations. The result is consistent across tree and low-width closed-loop topology.</p>
<div class="status"><strong>Stiffness-ratio stress test.</strong> In an 8-body fixed-joint ring with linear stiffness \(10^6\) and angular stiffness \(1\), one local iteration leaves weighted joint energy 102.718, while one sparse-direct iteration leaves 0.879: a 116.9x reduction. This case is retained as a regression test because block-diagonal updates converge poorly when coupled joint modes have widely separated stiffness.</div>

<h3>Isolated backend snapshot</h3>
<p>A separate one-iteration, no-contact G1 benchmark isolates graph-replayed solver overhead on a 44-body, 44-joint model. It is not the same workload as the end-to-end examples below.</p>
<table><thead><tr><th>Backend</th><th>Mode</th><th>Graph p50 [µs]</th><th>Purpose</th></tr></thead><tbody>
<tr><td>CPU</td><td>VBD local</td><td>{_fmt(cpu_graph_rows["vbd_local"]["cpu_graph_p50_step_us"], 1)}</td><td>body-diagonal baseline</td></tr>
<tr><td>CPU</td><td>VBD sparse serial</td><td>{_fmt(cpu_graph_rows["vbd_sparse"]["cpu_graph_p50_step_us"], 1)}</td><td>6x6 block Cholesky</td></tr>
<tr><td>CUDA MIG</td><td>VBD local</td><td>{_fmt(cuda_graph_rows["vbd_local"]["cuda_graph_p50_step_us"], 1)}</td><td>existing local path</td></tr>
<tr><td>CUDA MIG</td><td>VBD sparse serial</td><td>{_fmt(cuda_graph_rows["vbd_sparse"]["cuda_graph_p50_step_us"], 1)}</td><td>serial CUDA baseline</td></tr>
<tr><td>CUDA MIG</td><td>VBD sparse cooperative</td><td>{_fmt(cuda_graph_rows["vbd_sparse_block32"]["cuda_graph_p50_step_us"], 1)}</td><td>one cooperative CTA per articulation</td></tr>
</tbody></table>
<p>The cooperative CUDA path is {cuda_graph_rows["vbd_sparse"]["cuda_graph_p50_step_us"] / cuda_graph_rows["vbd_sparse_block32"]["cuda_graph_p50_step_us"]:.1f}x faster than the serial sparse CUDA baseline. The CPU remains faster for one small articulation because the GPU has limited independent work; the later DR Legs table reports the integrated solver with contact.</p>
<p class="note">Raw formulation data: <a href="formulation_data/iteration_sweep_cpu_sparse.json">iteration sweep</a>, <a href="formulation_data/bench_cpu_sparse.json">CPU sparse cases</a>, <a href="formulation_data/g1_cpu_graph.json">G1 CPU graph</a>, and <a href="formulation_data/g1_cuda_graph.json">G1 CUDA graph</a>.</p>

<h2>Public visual validations</h2>
<p>These self-contained mechanisms use only Newton primitives and are generated by the public <a href="{GITHUB_BLOB}/reports/vbd_complex_linkages/bench_visual_validations.py">visual-validation harness</a>. Each video shows local VBD on the left and sparse-direct VBD on the right with identical models, timesteps, and iteration counts.</p>
<div class="validation-grid">
  <article>
    <h3>Driven closed four-bar</h3>
    <video controls muted loop playsinline preload="metadata" poster="{four_bar_visual["poster"]}"><source src="{four_bar_visual["video"]}?v={MEDIA_VERSION}" type="video/mp4"></video>
    <p>Three moving links, four revolute joints, and one explicit world closure. At eight iterations, local VBD records {_fmt(four_bar_visual["local"]["rms_residual_um"], 3)} µm RMS joint error; sparse direct records {_fmt(four_bar_visual["sparse"]["rms_residual_um"], 3)} µm.</p>
  </article>
  <article>
    <h3>Hanging 32-segment cable</h3>
    <video controls muted loop playsinline preload="metadata" poster="{cable_visual["poster"]}"><source src="{cable_visual["video"]}?v={MEDIA_VERSION}" type="video/mp4"></video>
    <p>A stiff cable chain swings under gravity from one fixed end. At four iterations, local VBD records {_fmt(cable_visual["local"]["rms_residual_um"], 1)} µm RMS connectivity error; sparse direct records {_fmt(cable_visual["sparse"]["rms_residual_um"], 1)} µm.</p>
  </article>
</div>

<h3 class="analysis-heading">Cable convergence per wall-clock time</h3>
<p>The cable is reset to the same perturbed state for each residual measurement. CPU latency is measured separately as median steady-state <code>solver.step()</code> time over 50 samples. The plot therefore compares achieved joint error against measured solver cost rather than assuming equal iterations imply equal work.</p>
<figure>{_cable_convergence_svg(cable_convergence)}<figcaption>Residual after one substep versus CPU wall time. At four sparse iterations, the solve reaches {cable_i4_sparse["residual_m"] * 1.0e6:.2f} µm in {cable_i4_sparse["p50_step_us"] / 1.0e3:.3f} ms; the best local configuration inside that measured time budget leaves {cable_i4_sparse["best_local_residual_at_budget_m"] * 1.0e6:.1f} µm, a {cable_i4_sparse["residual_reduction_at_budget"]:.0f}x larger residual.</figcaption></figure>
<table><thead><tr><th>Solver</th><th>Iterations</th><th>CPU p50 [ms]</th><th>Aggregate residual [µm]</th><th>Reduction vs best local in budget</th></tr></thead><tbody>
{cable_table_rows}
</tbody></table>
<p class="note">Raw data: <a href="visual_validation_results.json">visual validation and cable convergence JSON</a>. The i1 sparse point is slightly faster than local i1 in this run, so no slower local point exists strictly inside its measured budget; later sparse points use the best local row at or below their p50 latency.</p>

<h2>Evaluation methods</h2>
<p>CPU measurements use Warp's single-threaded CPU backend on one AMD EPYC 9B45 core. GPU measurements use an NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb partition. CPU p50 and p90 values are synchronized wall-clock substep times after warm-up and include Python dispatch. The DR Legs CUDA rows alternate two fixed-buffer CUDA graphs for the two state-buffer directions. Each graph contains force clear, Newton collision, and the complete solver step, and each replay is synchronized for wall-clock measurement.</p>
<h3 class="analysis-heading">Metrics and glossary</h3>
<dl class="glossary">
  <div><dt>Closure joint</dt><dd>A joint edge selected to close a cycle in the articulation graph. Its parent and child anchors should coincide in world space.</dd></div>
  <div><dt>Closure residual / geometric closure error</dt><dd>The Euclidean world-space separation between the two anchors of a selected closure joint. This solver-independent distance is reported in micrometers.</dd></div>
  <div><dt>Aggregate closure RMS</dt><dd>The time RMS of the root-sum-square of all selected closure-joint errors. It measures total mechanism closure error.</dd></div>
  <div><dt>Per-joint closure RMS</dt><dd>The aggregate squared closure error divided by the number of selected closures before taking the square root. It represents a typical closure-joint error.</dd></div>
  <div><dt>Peak / max joint closure</dt><dd>The largest individual closure-joint anchor separation observed at any sampled time.</dd></div>
  <div><dt>Final aggregate closure</dt><dd>The root-sum-square of all selected closure-joint errors at the final sample.</dd></div>
  <div><dt>Motor / PR RMS error</dt><dd>The RMS angular drive-tracking error. PR denotes the source model's two-motor pitch/roll drive mapping.</dd></div>
  <div><dt>Solver-native residual</dt><dd>An algorithm-specific stopping metric, such as Kamino's primal, dual, and complementarity residuals. It is not directly comparable to geometric closure error.</dd></div>
  <div><dt>p50 / p90 substep</dt><dd>Median and 90th-percentile synchronized wall-clock latency per simulation substep after warm-up.</dd></div>
  <div><dt>Speedup vs Kamino</dt><dd>Kamino p50 latency divided by the candidate's p50 latency on the same backend and workload. Kamino is 1.00x; values above one are faster.</dd></div>
</dl>
<p class="note">The same geometric closure calculation is applied to every solver and was cross-checked against Kamino's translational joint residual components. It is intentionally separate from each solver's native convergence residual.</p>

<h2>Three-pushrod robot foot</h2>
<p><span class="pill">8 bodies</span><span class="pill">11 joints</span><span class="pill">3 closure joints</span><span class="pill">22 DOFs</span></p>
<p>The comparison uses spherical pushrod ends and places the paired roll anchors in a compatible rotation plane. Every structural and closure joint belongs to one Newton articulation, and all modes receive the same 100° pitch and 80° roll motor sweeps. CPU measurements use <code>dt=0.004 s</code> and 5 substeps per 50 Hz frame. Both VBD modes use 8 iterations with matched <code>ke=50k</code>, <code>kd=125</code>; Kamino uses up to 120 PADMM iterations with joint stabilization <code>alpha=0.5</code>.</p>
<h3 class="analysis-heading">Error analysis</h3>
<table><thead><tr><th>CPU solver</th><th>Aggregate closure RMS [µm]</th><th>Pitch motor range [deg]</th><th>Roll motor range [deg]</th><th>Output pitch range [deg]</th><th>Output roll range [deg]</th></tr></thead><tbody>
<tr><td>Kamino</td><td>{_fmt(foot_rows["kamino"]["rms_closure_um"])}</td><td>{_fmt(foot_rows["kamino"]["pitch_motor_range_deg"])}</td><td>{_fmt(foot_rows["kamino"]["roll_motor_range_deg"])}</td><td>{_fmt(foot_rows["kamino"]["output_pitch_range_deg"])}</td><td>{_fmt(foot_rows["kamino"]["output_roll_range_deg"])}</td></tr>
<tr><td>VBD local, i8</td><td>{_fmt(foot_local["rms_closure_um"])}</td><td>{_fmt(foot_local["pitch_motor_range_deg"])}</td><td>{_fmt(foot_local["roll_motor_range_deg"])}</td><td>{_fmt(foot_local["output_pitch_range_deg"])}</td><td>{_fmt(foot_local["output_roll_range_deg"])}</td></tr>
<tr><td><strong>VBD sparse direct, i8</strong></td><td><strong>{_fmt(foot_sparse["rms_closure_um"])}</strong></td><td>{_fmt(foot_sparse["pitch_motor_range_deg"])}</td><td>{_fmt(foot_sparse["roll_motor_range_deg"])}</td><td>{_fmt(foot_sparse["output_pitch_range_deg"])}</td><td>{_fmt(foot_sparse["output_roll_range_deg"])}</td></tr>
</tbody></table>
<h3 class="analysis-heading">Performance</h3>
<table><thead><tr><th>Solver</th><th>p50 substep [ms]</th><th>p90 substep [ms]</th><th>Speedup vs Kamino</th></tr></thead><tbody>
<tr><td>Kamino</td><td>{_fmt(foot_rows["kamino"]["p50_step_us"] / 1.0e3, 3)}</td><td>{_fmt(foot_rows["kamino"]["p90_step_us"] / 1.0e3, 3)}</td><td>1.00x</td></tr>
<tr><td>VBD local, i8</td><td>{_fmt(foot_local["p50_step_us"] / 1.0e3, 3)}</td><td>{_fmt(foot_local["p90_step_us"] / 1.0e3, 3)}</td><td>{_fmt(foot_rows["kamino"]["p50_step_us"] / foot_local["p50_step_us"], 2)}x</td></tr>
<tr><td><strong>VBD sparse direct, i8</strong></td><td><strong>{_fmt(foot_sparse["p50_step_us"] / 1.0e3, 3)}</strong></td><td>{_fmt(foot_sparse["p90_step_us"] / 1.0e3, 3)}</td><td><strong>{_fmt(foot_rows["kamino"]["p50_step_us"] / foot_sparse["p50_step_us"], 2)}x</strong></td></tr>
</tbody></table>
<p>All three solvers produce comparable output motion. Kamino satisfies all PADMM stopping tolerances on {100.0 * foot_rows["kamino"]["kamino_converged_fraction"]:.1f}% of substeps, with {_fmt(foot_rows["kamino"]["kamino_p50_iterations"], 0)} median iterations and median dual residual {foot_rows["kamino"]["kamino_p50_dual_residual"]:.2e}.</p>
<div class="media-grid">
{_video("robot_foot_kamino_corrected_cpu.mp4", "Kamino", "Compatible linkage rendered on CUDA; CPU metrics are reported above.")}
{_video("robot_foot_vbd_local_corrected_cpu.mp4", "VBD local", "Compatible linkage, eight local iterations, rendered on CUDA.")}
{_video("robot_foot_vbd_sparse_corrected_cpu.mp4", "VBD sparse direct", "Compatible linkage, eight sparse-direct iterations, rendered on CUDA.")}
</div>
<h3>Geometry validation</h3>
<div class="status warn">The reference demonstration geometry used universal pushrod ends and non-coplanar paired roll anchors. At a 40° roll-motor angle, even an endpoint-distance-only fit leaves at least {foot_geometry["current_model"]["minimum_rod_length_residual_rss_mm"]:.1f} mm root-sum-square rod-length mismatch. The benchmark therefore uses spherical rod ends and compatible roll anchors so that solver accuracy is not confounded by an infeasible mechanism.</div>

<h2>G1 closed-loop ankle</h2>
<p><span class="pill">6 bodies</span><span class="pill">8 joints</span><span class="pill">2 closure joints</span><span class="pill">16 DOFs</span></p>
<p>This model represents one G1 ankle with both internal loop closures explicit. A CPU settling test holds the two driven A/B joints at fixed targets and measures closure throughout 180 frames. It uses <code>dt=1/240 s</code>, 4 substeps per 60 Hz frame, and 8 iterations for both VBD modes.</p>
<h3>Mechanism cutaway</h3>
<div class="media-grid">
{_video("g1_ankle_viewergl_cutaway.mp4", "Closed-loop ankle mechanism", "Diagnostic ViewerGL view retaining the foot mesh while rendering the outer leg shell as wireframe, exposing the moving internal rods. The visualization changes geometry visibility only, not dynamics.")}
</div>
<h3 class="analysis-heading">Error analysis</h3>
<table><thead><tr><th>CPU solver</th><th>Aggregate closure RMS [µm]</th><th>Peak joint closure [µm]</th><th>Final aggregate closure [µm]</th><th>Motor A RMS error [deg]</th><th>Motor B RMS error [deg]</th></tr></thead><tbody>
<tr><td>Kamino</td><td>{_fmt(g1_kamino["rms_closure_um"])}</td><td>{_fmt(g1_kamino["max_closure_um"])}</td><td>{_fmt(g1_kamino["final_closure"]["linear_norm_m"] * 1.0e6, 3)}</td><td>{_fmt(g1_kamino["rms_motor_a_tracking_deg"], 3)}</td><td>{_fmt(g1_kamino["rms_motor_b_tracking_deg"], 3)}</td></tr>
<tr><td>VBD local, i8</td><td>{_fmt(g1_local["rms_closure_um"])}</td><td>{_fmt(g1_local["max_closure_um"])}</td><td>{_fmt(g1_local["final_closure"]["linear_norm_m"] * 1.0e6, 3)}</td><td>{_fmt(g1_local["rms_motor_a_tracking_deg"], 3)}</td><td>{_fmt(g1_local["rms_motor_b_tracking_deg"], 3)}</td></tr>
<tr><td><strong>VBD sparse direct, i8</strong></td><td><strong>{_fmt(g1_sparse["rms_closure_um"])}</strong></td><td>{_fmt(g1_sparse["max_closure_um"])}</td><td><strong>{_fmt(g1_sparse["final_closure"]["linear_norm_m"] * 1.0e6, 3)}</strong></td><td>{_fmt(g1_sparse["rms_motor_a_tracking_deg"], 3)}</td><td>{_fmt(g1_sparse["rms_motor_b_tracking_deg"], 3)}</td></tr>
</tbody></table>
<p class="note">Sparse VBD has a brief 18.7 mm peak during the initial response, which is included in its RMS value, then settles to {_fmt(g1_sparse["final_closure"]["linear_norm_m"] * 1.0e6, 3)} µm aggregate closure.</p>
<h3 class="analysis-heading">Performance</h3>
<table><thead><tr><th>Solver</th><th>p50 substep [ms]</th><th>p90 substep [ms]</th><th>Speedup vs Kamino</th></tr></thead><tbody>
<tr><td>Kamino</td><td>{_fmt(g1_kamino["p50_step_us"] / 1.0e3, 3)}</td><td>{_fmt(g1_kamino["p90_step_us"] / 1.0e3, 3)}</td><td>1.00x</td></tr>
<tr><td>VBD local, i8</td><td>{_fmt(g1_local["p50_step_us"] / 1.0e3, 3)}</td><td>{_fmt(g1_local["p90_step_us"] / 1.0e3, 3)}</td><td>{_fmt(g1_kamino["p50_step_us"] / g1_local["p50_step_us"], 2)}x</td></tr>
<tr><td><strong>VBD sparse direct, i8</strong></td><td><strong>{_fmt(g1_sparse["p50_step_us"] / 1.0e3, 3)}</strong></td><td>{_fmt(g1_sparse["p90_step_us"] / 1.0e3, 3)}</td><td><strong>{_fmt(g1_kamino["p50_step_us"] / g1_sparse["p50_step_us"], 2)}x</strong></td></tr>
</tbody></table>
<h3>Dynamic drive stress test</h3>
<p>A separate CUDA test drives the same mechanism through time-varying pitch and roll using the reference model's two-motor PR mapping. The table reports tracking error in that mapped drive space and geometric loop closure.</p>
<table><thead><tr><th>Solver</th><th>RMS PR error [deg]</th><th>Aggregate closure RMS [µm]</th><th>Finite</th></tr></thead><tbody>
<tr><td>Kamino</td><td>14.321</td><td>2,240.5</td><td>yes</td></tr>
<tr><td>VBD local, i8</td><td>31.249</td><td>24,996.6</td><td>yes</td></tr>
<tr><td><strong>VBD sparse direct, i8</strong></td><td><strong>0.094</strong></td><td><strong>2.05</strong></td><td>yes</td></tr>
</tbody></table>
<p class="note"><strong>The extra roll visible in the Kamino clip is not root-body motion.</strong> The knee/root body is fixed to the world, and a direct pose diagnostic measures less than 0.001° root-orientation drift in both runs. The commanded roll range is -5.73° to +5.73°; Kamino's measured ankle roll reaches -8.82° to +12.98°, while sparse VBD stays at -5.76° to +5.73°. The excess Kamino motion is therefore ankle tracking and closure deformation, consistent with its 14.321° PR RMS error and 2.24 mm aggregate closure RMS.</p>
<div class="media-grid">
{_video("g1_ankle_kamino_tilt_kamino.mp4", "Kamino", "Dynamic high-gain PR drive sweep.")}
{_video("g1_ankle_local_tilt_pr_ik_viewergl_cuda_mesh.mp4", "VBD local", "The local update accumulates large linkage error.")}
{_video("g1_ankle_sparse_tilt_pr_ik_viewergl_cuda_mesh.mp4", "VBD sparse direct", "The coupled solve preserves the closed-loop motion.")}
</div>

<h2>DR Legs with ground contact</h2>
<p><span class="pill">31 bodies</span><span class="pill">36 revolute joints</span><span class="pill">6 graph-cycle closures</span></p>
<p>This contact test releases the two foot-to-inner-ankle drives while retaining ten hip and linkage position drives at <code>kp=10</code>, <code>kd=2</code>. Gravity causes the mechanism to tip forward, exercising the closed loops during changing ground contact. The simulation uses <code>dt=0.01 s</code>, two substeps per 50 Hz frame, and 8 iterations for both VBD modes.</p>
<h3 class="analysis-heading">Error analysis</h3>
<table><thead><tr><th>CPU solver</th><th>Status</th><th>Aggregate closure RMS [µm]</th><th>Per-joint closure RMS [µm]</th><th>Max joint closure [µm]</th><th>Max pelvis tilt [deg]</th></tr></thead><tbody>
<tr><td>Kamino free ankle</td><td>{dr_free_kamino["status"]}</td><td>{_fmt(dr_free_kamino["rms_closure_um"])}</td><td>{_fmt(dr_free_kamino["rms_closure_per_joint_um"])}</td><td>{_fmt(dr_free_kamino["max_closure_um"])}</td><td>{_fmt(dr_free_kamino["max_pelvis_tilt_deg"], 1)}</td></tr>
<tr><td>VBD local free ankle, i8</td><td>fails at substep {dr_free_local["failure_substep"]}</td><td>-</td><td>-</td><td>-</td><td>{_fmt(dr_free_local["max_pelvis_tilt_deg"], 1)}</td></tr>
<tr><td><strong>VBD sparse direct free ankle, i8</strong></td><td>{dr_free_sparse["status"]}</td><td><strong>{_fmt(dr_free_sparse["rms_closure_um"], 2)}</strong></td><td><strong>{_fmt(dr_free_sparse["rms_closure_per_joint_um"], 2)}</strong></td><td>{_fmt(dr_free_sparse["max_closure_um"], 1)}</td><td>{_fmt(dr_free_sparse["max_pelvis_tilt_deg"], 1)}</td></tr>
</tbody></table>
<p>Kamino reaches the 200-iteration PADMM cap on the median step and satisfies all three stopping tolerances on only {100.0 * dr_free_kamino["kamino_converged_fraction"]:.1f}% of substeps. Its median native residuals are <code>r_p={dr_free_kamino["kamino_p50_primal_residual"]:.2e}</code>, <code>r_d={dr_free_kamino["kamino_p50_dual_residual"]:.2e}</code>, and <code>r_c={dr_free_kamino["kamino_p50_complementarity_residual"]:.2e}</code>, against <code>1e-4</code> tolerances. The large dual residual, rather than the geometric metric itself, explains why increasing the nominal iteration budget did not produce tight closure here.</p>
<h3 class="analysis-heading">Performance</h3>
<table><thead><tr><th>Backend</th><th>Solver</th><th>Run status</th><th>p50 solver [ms]</th><th>p50 collision [ms]</th><th>End-to-end p50 [ms]</th><th>Speedup vs Kamino</th></tr></thead><tbody>
<tr><td>CPU</td><td>Kamino</td><td>complete</td><td>{_fmt(dr_free_kamino["p50_solver_us"] / 1.0e3, 3)}</td><td>{_fmt(dr_free_kamino["p50_collision_us"] / 1.0e3, 3)}</td><td>{_fmt((dr_free_kamino["p50_solver_us"] + dr_free_kamino["p50_collision_us"]) / 1.0e3, 3)}</td><td>1.00x</td></tr>
<tr><td>CPU</td><td>VBD local, i8</td><td>fails at {dr_free_local["failure_substep"]}</td><td>{_fmt(dr_free_local["p50_solver_us"] / 1.0e3, 3)}</td><td>{_fmt(dr_free_local["p50_collision_us"] / 1.0e3, 3)}</td><td>{_fmt((dr_free_local["p50_solver_us"] + dr_free_local["p50_collision_us"]) / 1.0e3, 3)}</td><td>{_fmt((dr_free_kamino["p50_solver_us"] + dr_free_kamino["p50_collision_us"]) / (dr_free_local["p50_solver_us"] + dr_free_local["p50_collision_us"]), 2)}x*</td></tr>
<tr><td>CPU</td><td><strong>VBD sparse direct, i8</strong></td><td>complete</td><td><strong>{_fmt(dr_free_sparse["p50_solver_us"] / 1.0e3, 3)}</strong></td><td>{_fmt(dr_free_sparse["p50_collision_us"] / 1.0e3, 3)}</td><td><strong>{_fmt((dr_free_sparse["p50_solver_us"] + dr_free_sparse["p50_collision_us"]) / 1.0e3, 3)}</strong></td><td><strong>{_fmt((dr_free_kamino["p50_solver_us"] + dr_free_kamino["p50_collision_us"]) / (dr_free_sparse["p50_solver_us"] + dr_free_sparse["p50_collision_us"]), 2)}x</strong></td></tr>
<tr><td>CUDA MIG</td><td>Kamino</td><td>complete</td><td>included</td><td>included</td><td>{_fmt(dr_cuda_kamino["p50_graph_step_us"] / 1.0e3, 3)}</td><td>1.00x</td></tr>
<tr><td>CUDA MIG</td><td>VBD local</td><td>not run after CPU failure</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>CUDA MIG</td><td><strong>VBD sparse direct, i8</strong></td><td>complete</td><td>included</td><td>included</td><td><strong>{_fmt(dr_cuda_sparse["p50_graph_step_us"] / 1.0e3, 3)}</strong></td><td><strong>{_fmt(dr_cuda_kamino["p50_graph_step_us"] / dr_cuda_sparse["p50_graph_step_us"], 2)}x</strong></td></tr>
</tbody></table>
<p class="note">DR Legs timings begin after 40 warm-up frames. CPU collision and solver are synchronized and timed separately; CPU end-to-end p50 is the sum of their medians. Each CUDA value is one synchronized replay of an end-to-end ping-pong graph, so collision and solver are included rather than timed separately. * Local VBD timing covers only its finite prefix and is not a successful-throughput result.</p>
<div class="media-grid">
{_video("dr_legs_kamino_free_ankle_cpu.mp4", "Kamino free ankle", "The two ankle drives are disabled while the ten hip and linkage drives remain active.")}
{_video("dr_legs_vbd_local_free_ankle_cpu.mp4", "VBD local free ankle", "Matched free-ankle configuration. Capture stops at the nonfinite failure.")}
{_video("dr_legs_vbd_sparse_free_ankle_cpu.mp4", "VBD sparse direct free ankle", "Matched free-ankle configuration; the driven linkage tips forward while preserving closure.")}
</div>

<h2>Numerical validation</h2>
<p>The sparse block-Cholesky implementation is checked directly against its assembled articulation matrix and obtains relative linear residual <strong>{matrix["sparse_relative_residual"]:.2e}</strong>. The rigid contact implementation is checked against finite differences of contact energy: the normal-only force has relative gradient error <strong>{matrix["normal_contact_gradient_relative_error"]:.2e}</strong>, and normal plus damping has relative error <strong>{matrix["normal_damping_contact_gradient_relative_error"]:.2e}</strong>.</p>
<p>The DR Legs sparse-direct run provides an end-to-end validation with joints and changing contact active together. It completes all {dr_free_sparse["completed_substeps"]} substeps with {_fmt(dr_free_sparse["rms_closure_um"], 3)} µm aggregate closure RMS and {_fmt(dr_free_sparse["max_body_speed_mps"], 2)} m/s maximum body speed.</p>

<h2>Interpretation</h2>
<p>For these configurations, articulation-wide sparse VBD gives lower geometric closure error than local VBD and Kamino. It is also faster than local VBD on the two contact-free CPU tests and is the only VBD mode to complete the DR Legs contact test. The result supports a unified maximal-coordinate rigid-body path in which joints receive a coupled direct solve while contact curvature remains block diagonal.</p>
<p>These are achieved-error comparisons, not equal-tolerance benchmarks. VBD uses a fixed eight nonlinear iterations, while Kamino uses residual-based PADMM stopping with different constraint and contact models. CPU timings include current Warp and Python dispatch overhead; CUDA graph timings retain graph launch and synchronization overhead. The single-articulation GPU workloads do not saturate the device.</p>
<p class="note">Reproducible data: <a href="robot_foot_compatible_results.json">compatible robot foot</a>, <a href="robot_foot_geometry_diagnostic.json">foot geometry check</a>, <a href="g1_ankle_results.json">G1 ankle</a>, <a href="dr_legs_free_ankle_results.json">DR Legs CPU</a>, <a href="dr_legs_free_ankle_cuda_results.json">DR Legs CUDA</a>, <a href="dr_legs_matrix_diagnostic.json">numerical checks</a>, and <a href="visual_validation_results.json">public visual validations</a>.</p>
</main></body></html>"""
    (OUTPUT / "index.html").write_text(body)
    print(OUTPUT / "index.html")


if __name__ == "__main__":
    main()
