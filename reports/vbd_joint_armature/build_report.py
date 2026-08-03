#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the standalone VBD coupled joint-armature report."""

from __future__ import annotations

import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports" / "vbd_complex_linkages"
OUTPUT = Path.home() / "reports" / "vbd-joint-armature"
VIDEO_SOURCE = Path.home() / "reports" / "vbd-complex-linkages" / "videos"
GITHUB_BLOB = "https://github.com/mmacklin/newton/blob/horde/vbd-sparse-articulation-main"


def _row(rows: list[dict], case: str) -> dict:
    return next(row for row in rows if row["case"] == case)


def _video(name: str, title: str, description: str, version: str) -> str:
    stem = name.removesuffix(".mp4")
    return f"""
    <article class="media">
      <h3>{html.escape(title)}</h3>
      <video controls muted loop playsinline preload="metadata" poster="videos/{stem}.jpg">
        <source src="videos/{name}?v={version}" type="video/mp4">
      </video>
      <p>{html.escape(description)}</p>
    </article>"""


def main() -> None:
    payload = json.loads((SOURCE / "dr_legs_policy_results.json").read_text())
    rows = payload["rows"]
    kamino = _row(rows, "kamino")
    coupled = _row(rows, "sparse_i8")
    isotropic = _row(rows, "sparse_isotropic_armature_i8")
    no_armature = _row(rows, "sparse_no_armature_i8")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    videos = OUTPUT / "videos"
    videos.mkdir(exist_ok=True)
    video_stems = ("dr_legs_policy_sparse_i8", "dr_legs_policy_sparse_isotropic_armature_i8")
    for stem in video_stems:
        for suffix in (".mp4", ".jpg", ".json"):
            shutil.copyfile(VIDEO_SOURCE / f"{stem}{suffix}", videos / f"{stem}{suffix}")
    shutil.copyfile(SOURCE / "dr_legs_policy_results.json", OUTPUT / "dr_legs_policy_results.json")

    generated = datetime.now(timezone.utc)
    timestamp = generated.strftime("%Y-%m-%d %H:%M UTC")
    media_version = generated.strftime("%Y%m%dT%H%M%SZ")
    body = rf"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coupled Joint Armature in Sparse VBD</title>
<script>window.MathJax={{tex:{{inlineMath:[["\\(","\\)"]],displayMath:[["\\[","\\]"]]}},svg:{{fontCache:"global"}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
:root{{--ink:#171a1d;--muted:#596169;--line:#d7dce0;--paper:#fff;--soft:#f4f6f7;--accent:#006c67;--old:#a84c22}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1050px;margin:auto;padding:36px 28px 72px}} h1{{font-size:34px;line-height:1.12;margin:0 0 10px;letter-spacing:0}}
h2{{font-size:22px;margin:40px 0 14px;border-bottom:1px solid var(--line);padding-bottom:8px;letter-spacing:0}} h3{{font-size:16px;margin:0 0 8px}}
p{{max-width:850px}} a{{color:var(--accent)}} .lede{{font-size:18px;color:#30363b}} .meta,.note{{color:var(--muted)}}
.summary{{border-left:4px solid var(--accent);background:var(--soft);padding:13px 16px;max-width:900px}}
.formula-grid,.media-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin:20px 0}}
.formula{{border-top:3px solid var(--accent);background:var(--soft);padding:14px 16px;min-width:0}} .formula.old{{border-top-color:var(--old)}}
.matrix{{margin:14px auto;border-collapse:separate;border-spacing:4px;font:600 15px ui-monospace,monospace}}
.matrix td{{min-width:110px;text-align:center;padding:12px 8px;background:#e5eeec;border:1px solid #c2d8d4}} .matrix.old td{{background:#f1e7e2;border-color:#dfc5b9}}
.matrix-label{{text-align:center;color:var(--muted);font-size:13px}} table.data{{border-collapse:collapse;width:100%;display:block;overflow-x:auto;font-variant-numeric:tabular-nums}}
.data th,.data td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} .data th:first-child,.data td:first-child{{text-align:left}} .data th{{background:var(--soft);font-size:13px}}
.media{{min-width:0}} video{{display:block;width:100%;aspect-ratio:16/9;background:#111}} code{{background:var(--soft);padding:1px 4px}}
pre{{overflow:auto;background:#202724;color:#edf2ee;padding:16px;border-radius:4px;font:13px/1.5 ui-monospace,monospace}}
@media(max-width:760px){{main{{padding:24px 16px}}.formula-grid,.media-grid{{grid-template-columns:1fr}}.matrix td{{min-width:78px}}}}
</style></head><body><main>
<h1>Coupled Joint Armature in Sparse VBD</h1>
<p class="lede">Why adding inertia to a child body is not equivalent to revolute-joint armature, and the parent-child rank-one term now assembled by the maximal-coordinate sparse solver.</p>
<p class="meta">Generated {timestamp}. Companion to the <a href="https://reports.mmacklin.com/vbd-complex-linkages/#joint-armature">full VBD sparse articulation report</a>.</p>

<h2>Result</h2>
<div class="summary"><strong>The cross block matters.</strong> On the trained DR Legs policy, replacing the isotropic child-body approximation with coupled relative-coordinate armature reduces pelvis tilt RMS from {isotropic["tilt_deg"]["rms"]:.2f}&deg; to {coupled["tilt_deg"]["rms"]:.2f}&deg;, lateral drift from {abs(isotropic["lateral_displacement_m"]):.3f} m to {abs(coupled["lateral_displacement_m"]):.3f} m, and closure RMS from {isotropic["closure_error_um"]["rms"]:.1f} &micro;m to {coupled["closure_error_um"]["rms"]:.1f} &micro;m. The corrected actuator tracking error, {coupled["actuated_target_error_rad"]["rms"]:.3f} rad RMS, matches Kamino's {kamino["actuated_target_error_rad"]["rms"]:.3f} rad.</div>

<h2>Physical term</h2>
<p>A revolute actuator's armature value \(a_j\) is rotor inertia in the joint coordinate. For joint axis \(\mathbf{{u}}\), parent angular velocity \(\boldsymbol{{\omega}}_p\), and child angular velocity \(\boldsymbol{{\omega}}_c\), the physical kinetic energy is</p>
\[
T_{{\mathrm{{arm}},j}}=\frac12a_j\dot q_j^2
=\frac12a_j\left[\mathbf{{u}}^T(\boldsymbol{{\omega}}_c-\boldsymbol{{\omega}}_p)\right]^2.
\]
<p>The armature responds to relative angular velocity. If parent and child rotate together, the term is exactly zero. In combined parent-child angular coordinates it contributes</p>
\[
M_{{\mathrm{{arm}},j}}=a_j
\begin{{bmatrix}}
\mathbf{{u}}\mathbf{{u}}^T &amp; -\mathbf{{u}}\mathbf{{u}}^T\\
-\mathbf{{u}}\mathbf{{u}}^T &amp; \mathbf{{u}}\mathbf{{u}}^T
\end{{bmatrix}}.
\]
<table class="matrix" aria-label="Coupled armature block matrix"><tr><td>+a uuᵀ<br>parent</td><td>-a uuᵀ<br>cross</td></tr><tr><td>-a uuᵀ<br>cross</td><td>+a uuᵀ<br>child</td></tr></table>
<p class="matrix-label">The complete contribution is rank one across the combined parent-child angular coordinates.</p>

<h2>Discretized VBD potential</h2>
<p>Let \(R_{{\mathrm{{rel}}}}=R_p^TR_c\) be the current relative joint orientation and \(\widehat R_{{\mathrm{{rel}}}}\) its forward-integrated inertial target. The SO(3) relative error and its scalar component along the revolute axis are</p>
\[
\boldsymbol{{\kappa}}=\log\!\left(R_{{\mathrm{{rel}}}}\widehat R_{{\mathrm{{rel}}}}^T\right),
\qquad e_j=\mathbf{{u}}_{{\mathrm{{local}}}}^T\boldsymbol{{\kappa}}.
\]
<p>The primal objective receives the positional inertial potential</p>
\[
\boxed{{E_{{\mathrm{{arm}},j}}=\frac{{a_j}}{{2\Delta t^2}}e_j^2}}.
\]
<p>Writing \(h=a_j/\Delta t^2\) and \(\mathbf{{j}}=J_{{SO(3)}}\mathbf{{u}}_{{\mathrm{{local}}}}\), its Gauss-Newton assembly is</p>
\[
H_{{pp}}{{+}}=h\mathbf{{j}}\mathbf{{j}}^T,\quad H_{{cc}}{{+}}=h\mathbf{{j}}\mathbf{{j}}^T,\quad
H_{{pc}}{{-}}=h\mathbf{{j}}\mathbf{{j}}^T,\quad H_{{cp}}{{-}}=h\mathbf{{j}}\mathbf{{j}}^T,
\]
\[
b_p{{+}}=he_j\mathbf{{j}},\qquad b_c{{-}}=he_j\mathbf{{j}}.
\]
<p class="note">The energy and first derivative use the nonlinear SO(3) relative rotation. The implementation retains the positive-semidefinite Gauss-Newton Hessian and omits the geometric \(e_j\nabla^2e_j\) term.</p>

<h2>Why child-body inertia differs</h2>
<div class="formula-grid">
  <section class="formula old"><h3>Child-body approximation</h3>
    <p>The previous approximation changed the child's local inertia:</p>
    \[I_c\leftarrow I_c+a_jI_3.\]
    <table class="matrix old" aria-label="Child-only approximation block matrix"><tr><td>0<br>parent</td><td>0<br>cross</td></tr><tr><td>0<br>cross</td><td>+a I₃<br>child</td></tr></table>
    <p>It penalizes absolute child rotation, changes common-mode whole-body inertia, and adds inertia around two axes unrelated to the revolute coordinate.</p>
  </section>
  <section class="formula"><h3>Coupled joint armature</h3>
    <p>The sparse solve adds both self blocks and the negative cross block without modifying either body's authored inertia.</p>
    <table class="matrix" aria-label="Coupled rank-one block matrix"><tr><td>+h jjᵀ<br>parent</td><td>-h jjᵀ<br>cross</td></tr><tr><td>-h jjᵀ<br>cross</td><td>+h jjᵀ<br>child</td></tr></table>
    <p>Common parent-child motion remains free, and the internal rotor inertia produces equal-and-opposite reactions while preserving total angular momentum.</p>
  </section>
</div>
<p>For a joint attached to the static world, a correctly oriented rank-one child update is equivalent at first order. For floating articulations, and especially closed-loop robots, omitting the parent and cross blocks changes the dynamics.</p>

<h2>Analytic check</h2>
<p>For two coaxial bodies with inertias \(I_p\), \(I_c\), armature \(a\), drive stiffness \(k\), target \(\theta_t\), and timestep \(\Delta t\), the expected one-step relative rotation is</p>
\[
\Delta\theta_{{\mathrm{{rel}}}}=
\frac{{k\theta_t}}{{k+\left(I_pI_c/(I_p+I_c)+a\right)/\Delta t^2}}.
\]
<p>The regression test predicts <code>1.6129032258e-4 rad</code> and measures <code>1.6129032874e-4 rad</code>. Physical angular-momentum error is <code>5.8e-12</code>.</p>

<h2>DR Legs policy ablation</h2>
<table class="data"><thead><tr><th>Solver / armature</th><th>Status</th><th>Tilt RMS [deg]</th><th>Target error RMS [rad]</th><th>Closure RMS [&micro;m]</th><th>Lateral drift [m]</th><th>CPU p50 [ms]</th></tr></thead><tbody>
<tr><td>Kamino PADMM / native</td><td>complete</td><td>{kamino["tilt_deg"]["rms"]:.2f}</td><td>{kamino["actuated_target_error_rad"]["rms"]:.3f}</td><td>{kamino["closure_error_um"]["rms"]:.1f}</td><td>{abs(kamino["lateral_displacement_m"]):.3f}</td><td>{kamino["step_p50_us"] / 1e3:.3f}</td></tr>
<tr><td>VBD sparse / none</td><td>falls at {no_armature["fall_time_s"]:.2f} s</td><td>{no_armature["tilt_deg"]["rms"]:.2f}</td><td>{no_armature["actuated_target_error_rad"]["rms"]:.3f}</td><td>{no_armature["closure_error_um"]["rms"]:.1f}</td><td>{abs(no_armature["lateral_displacement_m"]):.3f}</td><td>{no_armature["step_p50_us"] / 1e3:.3f}</td></tr>
<tr><td>VBD sparse / isotropic child</td><td>complete</td><td>{isotropic["tilt_deg"]["rms"]:.2f}</td><td>{isotropic["actuated_target_error_rad"]["rms"]:.3f}</td><td>{isotropic["closure_error_um"]["rms"]:.1f}</td><td>{abs(isotropic["lateral_displacement_m"]):.3f}</td><td>{isotropic["step_p50_us"] / 1e3:.3f}</td></tr>
<tr><td><strong>VBD sparse / coupled joint</strong></td><td><strong>complete</strong></td><td><strong>{coupled["tilt_deg"]["rms"]:.2f}</strong></td><td><strong>{coupled["actuated_target_error_rad"]["rms"]:.3f}</strong></td><td><strong>{coupled["closure_error_um"]["rms"]:.1f}</strong></td><td><strong>{abs(coupled["lateral_displacement_m"]):.3f}</strong></td><td><strong>{coupled["step_p50_us"] / 1e3:.3f}</strong></td></tr>
</tbody></table>
<div class="media-grid">
{_video("dr_legs_policy_sparse_isotropic_armature_i8.mp4", "Isotropic child-body ablation", "The robot completes the command but shows looser root motion and substantial lateral drift.", media_version)}
{_video("dr_legs_policy_sparse_i8.mp4", "Coupled joint armature", "The corrected rank-one parent-child coupling closely matches the trained controlled dynamics.", media_version)}
</div>

<h2>Implementation</h2>
<pre><code>solver = newton.solvers.SolverVBD(
    model,
    iterations=8,
    rigid_articulation_solve="block_sparse_joints",
    rigid_joint_armature=True,
)</code></pre>
<p><a href="{GITHUB_BLOB}/newton/_src/solvers/vbd/rigid_sparse_articulation_kernels.py#L1637">CUDA/scalar assembly</a> · <a href="{GITHUB_BLOB}/newton/_src/solvers/vbd/rigid_sparse_articulation_kernels.py#L2404">CPU serial assembly</a> · <a href="{GITHUB_BLOB}/newton/tests/test_vbd_sparse_articulation.py#L845">analytic regression</a> · <a href="dr_legs_policy_results.json">benchmark JSON</a></p>
</main></body></html>"""
    (OUTPUT / "index.html").write_text(body)


if __name__ == "__main__":
    main()
