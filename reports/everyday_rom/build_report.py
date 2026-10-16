# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build the everyday ROM gallery from measured simulation and render artifacts."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SLUGS = (
    "clog",
    "glasses-case",
    "toothbrush",
    "drink-bottle",
    "cup-lid",
    "hand",
    "straw",
    "spatula",
    "banana",
    "headphones",
)


def esc(value):
    return html.escape(str(value), quote=True)


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def presentation_metadata(slug, meta):
    """Correct gallery wording without mutating hash-protected scene metadata."""
    overrides = {
        "hand": {
            "title": "Pose-based hand",
            "description": "A replacement study: contact pushes the fingers of a pose-derived ROM. Dynamics and asset-publication checks are in progress.",
        },
        "glove": {
            "title": "Hollow glove: contact, carry, release and drop",
            "description": (
                "A rounded pusher bends the fingers of an open-cuff rubber-like glove. "
                "After unloading, the support carries it horizontally, then releases it to drop. "
                "The recorded support ascent occurs after release; the attached glove is not lifted."
            ),
        },
        "clog": {
            "description": (
                "An unclamped clog drops onto a rounded ground support, bends under heel and forefoot pads, "
                "then recoils while remaining on its rounded support after a 25 ms pad release."
            ),
        },
    }
    if slug == "cap" and meta.get("slug") != "cap_dynamics":
        overrides["cap"] = {"status": "bounded indentation; larger-dynamics follow-up outstanding"}
    if slug == "hand" and meta.get("publication_ready"):
        overrides.pop("hand")
    return meta | overrides.get(slug, {})


def render_current(directory, audit):
    if not audit or not (directory / "rtx.mp4").exists():
        return False
    inputs = audit.get("inputs_sha256", {})
    if not {"scene.json", "scene.npz", "metrics.json"}.issubset(inputs):
        return False
    for name, expected in inputs.items():
        if not (directory / name).exists():
            return False
        with (directory / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                return False
    return bool(audit.get("inputs_sha256"))


def replay_current(directory, audit):
    """Keep concatenated replays distinct from single-scene renderer audits."""
    if audit.get("artifact_type") != "normal_then_strain_concat" or not render_current(directory, audit):
        return False
    with (directory / "rtx.mp4").open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == audit.get("video_sha256")


def video_path(slug, folder=""):
    """Give every scene and camera/replay variant a distinct MP4 basename."""
    view = "normal-and-strain" if folder == "replay" else folder.replace("/", "-") or "normal"
    return f"videos/everyday-rom-{slug}-{view}.mp4"


def publish_videos(root, assets):
    """Publish byte-identical named copies without changing audited originals."""

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    manifest = {}
    for name, original in sorted(assets.items()):
        source, target = root / original, root / name
        expected = digest(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != expected:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".video-", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                shutil.copyfile(source, temporary)
                if digest(temporary) != expected:
                    raise ValueError(f"Video changed while publishing: {source}")
                temporary.chmod(0o644)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        manifest[name] = {"source": original, "sha256": expected, "size_bytes": target.stat().st_size}
    (root / "videos").mkdir(exist_ok=True)
    (root / "videos/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def compact_metrics(metrics):
    """Leave large per-frame arrays in the downloadable metrics artifact."""
    if isinstance(metrics, dict):
        return {key: compact_metrics(value) for key, value in metrics.items()}
    if isinstance(metrics, list) and len(metrics) > 20:
        return f"{len(metrics)} entries; see metrics.json"
    return metrics


def number(value, scale=1, suffix=""):
    if not isinstance(value, (int, float)):
        return "Not measured"
    return f"{value * scale:.3g}{suffix}"


def metric_table(metrics, timing_label="Simulation frame"):
    metrics = dict(metrics)
    timing_keys = (
        "warm_simulation_frame_ms",
        "graph_sim_frame_ms",
        "simulation_frame_ms",
        "warm_graph_mean_ms",
        "mean_warm_graph_frame_ms",
        "warm_graph_step_ms",
    )
    metrics.setdefault("sim_frame_ms", next((metrics[key] for key in timing_keys if key in metrics), None))
    fields = (
        ("Peak deformation", "max_displacement_m", 1000, " mm"),
        ("Final deformation", "final_displacement_m", 1000, " mm"),
        ("Peak penetration", "max_penetration_m", 1000, " mm"),
        (timing_label, "sim_frame_ms", 1, " ms"),
    )
    if "max_sampled_pusher_penetration_m" in metrics:
        fields = (
            ("Peak principal strain", "max_principal_strain", 100, "%"),
            ("Sampled penetration", "max_sampled_pusher_penetration_m", 1000, " mm"),
            ("Wrist attachment gap", "max_wrist_gap_m", 1000, " mm"),
            (timing_label, "sim_frame_ms", 1, " ms"),
        )
        if metrics.get("actuation_enabled") and metrics.get("per_finger"):
            metrics["max_finger_hold_m"] = max(
                row["hold_max_distal_vertex_displacement_m"] for row in metrics["per_finger"]
            )
            fields = (
                fields[0],
                ("Peak finger hold travel", "max_finger_hold_m", 1000, " mm"),
                *fields[2:],
            )
    rows = "".join(
        f"<div><dt>{label}</dt><dd>{number(metrics.get(key), scale, suffix)}</dd></div>"
        for label, key, scale, suffix in fields
    )
    return f'<dl class="metrics">{rows}</dl>'


def hand_pod_study(root):
    audit = load(root / "hand_pod_audit.json")
    if not audit:
        return ""
    rows = "".join(
        f"<tr><td>{item['modes']}</td><td>{item['heldout_rms_m'] * 1000:.3f} mm</td>"
        f"<td>{item['heldout_max_m'] * 1000:.3f} mm</td></tr>"
        for item in audit["pose_validation"]
    )
    dynamics = load(root / "hand_dynamics_audit.json")
    dynamic_note = ""
    if dynamics:
        dynamic_note = (
            "<p><b>Contact experiment: WIP.</b> A private 24-mode hand runs contact, wrist carry and drop. "
            f"Contact response is {number(dynamics.get('contact_response_m'), 1000, ' mm')}, but "
            f"{esc(dynamics.get('drop_self_intersections', 'unmeasured'))} intersection pairs at impact "
            "and excessive local strain reject this trajectory. An intersection-free native rest pose "
            "alone does not fix unrestricted linear combinations. "
            '<a href="hand_dynamics_audit.json">Dynamics audit</a>.</p>'
        )
        shell = dynamics.get("projected_shell_trial")
        if shell:
            dynamic_note += (
                "<p><b>Strain-derived stiffness variant:</b> "
                f"{esc(shell['max_self_intersections'])} self-intersections and "
                f"{esc(shell['normal_reversals'])} reversed faces. "
                "This improves geometry, but impact area distortion and "
                f"{number(shell['max_floor_penetration_m'], 1000, ' mm')} ground penetration "
                "still fail acceptance.</p>"
            )
        low_drop = dynamics.get("lower_drop_trial")
        if low_drop:
            dynamic_note += (
                "<p><b>Lower-drop follow-up: still WIP.</b> "
                f"Contact produces {number(low_drop['contact_response_m'], 1000, ' mm')} "
                "deformation and recovery without intersections. Firmer ground contact brings penetration to "
                f"{number(low_drop['max_floor_penetration_m'], 1000, ' mm')}, but impact reaches "
                f"{number(low_drop['drop_max_edge_strain'], 100, '%')} local edge strain and "
                f"{number(low_drop['drop_min_area_ratio'])} minimum area ratio. "
                "The no-pusher drop control fails the same geometry checks.</p>"
            )
        bounded = dynamics.get("bounded_shell_trial")
        if bounded and bounded.get("accepted") is True:
            dynamic_note = (
                "<p><b>Bounded shell follow-up: checks pass.</b> "
                "An explicit 10 MPa material choice and 2 mm longer physical pusher stroke give "
                f"{number(bounded['contact_response_m'], 1000, ' mm')} contact deformation. "
                "All-substep impact checks measure "
                f"{number(bounded['substep_max_edge_strain'], 100, '%')} maximum edge strain, "
                f"{number(bounded['substep_min_area_ratio'])} minimum area ratio and "
                f"{number(bounded['max_floor_penetration_m'], 1000, ' mm')} penetration. "
                "The matched no-pusher control, 16 CPU tests and exact state recovery pass. "
                "This is not a validated glove or anatomical material; geometry remains private "
                "pending redistribution clearance. Four substeps, 20 iterations; no default change.</p>"
                "<details><summary>Earlier failed trials</summary>" + dynamic_note + "</details>"
            )
    return f"""<details><summary>NIMBLE pose-to-POD investigation</summary>
    <p>Native NIMBLE skin, {audit["vertices"]:,} vertices. {audit["training_poses"]} training poses and
    {audit["heldout_poses"]} held-out poses. These are reconstruction errors, not validated contact dynamics.</p>
    <table class="parameters"><thead><tr><th>Modes</th><th>RMS error</th><th>Maximum error</th></tr></thead><tbody>{rows}</tbody></table>
    <p>Mass, stiffness and damping still need physical choices; linear combinations require strain and contact checks.
    Source assets and derived meshes are retained privately pending clarification of model-data redistribution terms.</p>{dynamic_note}
    <p><a href="https://github.com/reyuwei/NIMBLE_model">NIMBLE source</a> /
    <a href="hand_pod_audit.json">POD audit</a> /
    <a href="nimble-hand-scalars/index.html">Contact and impact report</a> /
    <a href="source/HAND_MODELS.md">Assessment and licensing notes</a></p></details>"""


def model_metadata(root, slug, meta, render):
    """Resolve display notes only for the scene revision they describe."""
    meta = meta | load(root / slug / "gallery_view.json")
    annotation = load(root / "presentation.json").get(slug, {})
    if annotation.get("scene_sha256") and annotation["scene_sha256"] == render.get("inputs_sha256", {}).get(
        "scene.json"
    ):
        meta |= annotation["metadata"]
    return meta


def detail_caption(folder, metadata):
    """Keep strain-only replays distinct from alternate camera views."""
    if folder == "strain":
        label = "Principal surface strain"
        description = "Same saved motion and camera, colored by principal surface strain; not calibrated stress."
    else:
        label = "Wrapping end view" if folder == "wrap-side" else "Detail view"
        description = "Same saved simulation, alternate camera."
    return (
        metadata.get("detail_label", label),
        metadata.get("view_description", metadata.get("view", description)),
    )


def section(root, slug, video_assets=None):
    directory = root / slug
    meta, metrics, render = (load(directory / name) for name in ("scene.json", "metrics.json", "rtx.json"))
    meta = model_metadata(root, slug, meta, render)
    display = presentation_metadata(slug, meta)
    performance_data = load(root / "performance/summary.json")
    performance = next(
        (row for row in performance_data.get("objects", []) if row["slug"] == slug and not row.get("pending")),
        None,
    )
    display_metrics = dict(metrics)
    timing_label = "Optimized sim frame"
    revised_model = bool(metrics.get("model_revision"))
    candidate = performance_data.get("selected_candidates", {}).get(slug, {})
    display_metrics["sim_frame_ms"] = metrics.get("optimized_sim_frame_ms")
    if candidate.get("candidate") and not revised_model:
        display_metrics["sim_frame_ms"] = candidate["ms"]
    performance_note = ""
    if revised_model and performance:
        performance_note = (
            "<p>Geometry or actuation has changed. "
            '<a href="performance/index.html">Earlier optimization measurements</a> '
            "describe the archived model, not this revision.</p>"
        )
    elif candidate.get("candidate") and slug == "spatula":
        performance_note = (
            "<p>The optimized spatula timing is historical; a fresh negative-control run "
            "exceeded the penetration bound. "
            '<a href="performance/reproduction-v2/index.html">Reproduction check</a>.</p>'
        )
    title = display.get("title", slug.replace("-", " ").title())
    ready = render_current(directory, render)
    passed = metrics.get("test_passed") is True and metrics.get("negative_control_passed") is True
    status = display.get("status", "In progress") if ready else "In progress"
    review = load(root / "review.json").get(slug, {})
    if ready and review.get("status"):
        status = review["status"]
    if review.get("blocking"):
        status = review["status"]
        passed = False
    high_strain = metrics.get("principal_strain_screen_passed") is False
    if ready and high_strain:
        status = "High-strain ROM"
    badge = "pass" if ready and passed and not high_strain and "wip" not in status.lower() else "wip"
    if ready:
        replay = load(directory / "replay/rtx.json")
        has_replay = replay_current(directory / "replay", replay)
        video_folder = f"{slug}/replay" if has_replay else slug
        video_name = video_path(slug, "replay" if has_replay else "")
        if video_assets is not None:
            video_assets[video_name] = f"{video_folder}/rtx.mp4"
        version = (replay if has_replay else render).get("video_sha256", "")[:16]
        media = (
            f'<video controls playsinline preload="none" poster="{slug}/poster.jpg?v={version}" '
            f'aria-label="{esc(title)} simulation"><source src="{video_name}?v={version}" type="video/mp4"></video>'
            f'<p class="downloads"><a href="{video_name}?v={version}" '
            f'download="{Path(video_name).name}">Download MP4</a></p>'
        )
        if has_replay:
            media += (
                '<p class="replay-note"><b>Principal surface strain:</b> The second half repeats the exact '
                "saved motion and camera with a fixed strain scale. Colors show deformation, not calibrated stress. "
                f'<a href="{slug}/replay/rtx.json">Replay audit</a> / '
                f'<a href="{slug}/strain/strain.npz">Strain field</a>.</p>'
            )
    else:
        media = '<div class="pending">Simulation and validation in progress</div>'
    detail_media = ""
    for folder in ("detail", "profile/detail", "lift", "lift/detail", "strain", "wrap-side"):
        detail_audit = load(directory / folder / "rtx.json")
        if not ready or not render_current(directory / folder, detail_audit):
            continue
        if folder in ("detail", "profile/detail", "wrap-side") and detail_audit["inputs_sha256"].get(
            "scene.npz"
        ) != render["inputs_sha256"].get("scene.npz"):
            continue
        if folder.startswith("lift"):
            detail_metrics = load(directory / folder / "metrics.json")
            if not all(detail_metrics.get(key) is True for key in ("test_passed", "negative_control_passed")):
                continue
        detail_meta = load(directory / folder / "scene.json") | load(directory / folder / "gallery_view.json")
        detail_version = detail_audit.get("video_sha256", "")[:16]
        detail_video = video_path(slug, folder)
        if video_assets is not None:
            video_assets[detail_video] = f"{slug}/{folder}/rtx.mp4"
        detail_label, view_description = map(esc, detail_caption(folder, detail_meta))
        extra_links = "".join(
            f' / <a href="{slug}/{folder}/{name}">{label}</a>'
            for name, label in (("strain.npz", "Strain field"), ("STRAIN_VIEW.md", "Definition and checks"))
            if (directory / folder / name).is_file()
        )
        detail_media += (
            f'<details class="detail-video"><summary>{detail_label}</summary>'
            f'<video controls playsinline preload="none" aria-label="{detail_label}" '
            f'poster="{slug}/{folder}/poster.jpg?v={detail_version}">'
            f'<source src="{detail_video}?v={detail_version}" type="video/mp4"></video>'
            f'<p class="downloads"><a href="{detail_video}?v={detail_version}" '
            f'download="{Path(detail_video).name}">Download MP4</a></p>'
            f'<p>{view_description} <a href="{slug}/{folder}/rtx.json">Render audit</a> / '
            f'<a href="{slug}/{folder}/metrics.json">Sequence metrics</a>{extra_links}.</p></details>'
        )
    limitations = "".join(f"<li>{esc(item)}</li>" for item in meta.get("limitations", []))
    if high_strain:
        limitations = (
            "<li><b>High-strain ROM:</b> Contact and recovery checks pass, but the reconstructed surface "
            "exceeds the principal-strain screening threshold. This is an illustrative contact example, "
            "not a material validation.</li>" + limitations
        )
    limitations += "".join(f"<li><b>Review:</b> {esc(item)}</li>" for item in review.get("findings", []))
    parameters = meta.get("parameters", {})
    param_rows = "".join(
        f"<tr><th>{esc(key.replace('_', ' '))}</th><td>{esc(value)}</td></tr>" for key, value in parameters.items()
    )
    checks = "Passed" if passed else "Pending / not accepted"
    control = "Passed" if metrics.get("negative_control_passed") is True else "See object-specific notes"
    source_links = []
    for path in meta.get("source_files", []):
        name = Path(path).name
        if (directory / "source" / name).is_file():
            folder = f"{slug}/source"
        elif (directory / name).is_file():
            folder = slug
        else:
            folder = "source"
        source_links.append(f'<a href="{esc(folder)}/{esc(name)}">{esc(name)}</a>')
    source_links = " ".join(source_links)
    references = " / ".join(
        f'<a href="{esc(item["url"])}">{esc(item["title"])}</a>'
        for item in meta.get("references", [])
        if item["url"].startswith("https://")
    )
    reference_note = (
        f"<p><b>Shape references:</b> {references}. Original procedural geometry; no product scan.</p>"
        if references
        else ""
    )
    normal_files = load(directory / "normal_manifest.json").get("files_sha256")
    detail_label = (
        "Historical report notes"
        if normal_files is not None and "index.html" not in normal_files
        else "Detailed report"
    )
    downloads = " ".join(
        f'<a href="{slug}/{name}">{label}</a>'
        for name, label in (
            ("index.html", detail_label),
            ("metrics.json", "Metrics"),
            ("scene.json", "Model"),
            ("rtx.json", "Render audit"),
            ("scene.npz", "Saved surfaces"),
            ("scene.usdc", "USD"),
            ("VALIDATION.md", "Historical validation notes" if slug == "banana" else "Validation notes"),
            ("ATTRIBUTION.md", "Asset attribution"),
            (
                "PUBLICATION.md",
                "Historical publication notes"
                if slug in ("hand", "glasses-case")
                else "Publication and reproduction notes",
            ),
            ("REPRODUCE.md", "Current reproduction workflow"),
            ("reproduction.zip", "Portable source bundle"),
            ("paired_results.json", "Paired contact response"),
            (
                "static_quadratic_diagnostic.json",
                "Historical static energy diagnostic" if slug == "banana" else "Static energy diagnostic",
            ),
            ("surface_strain_audit.json", "Independent surface audit"),
            ("geometry_audit.json", "Render embedding geometry audit"),
            ("render_embedding.npz", "Render-to-physics mapping"),
            ("debug/README.md", "Free-flight diagnostic"),
            ("dynamic-validation.md", "Dynamic validation"),
            ("PERFORMANCE.md", "Performance investigation"),
            ("SCRUB_VALIDATION.md", "Scrubbing checks"),
            ("CAP_DYNAMICS_HANDOFF.md", "Carry, release and impact checks"),
            ("pair_validation.json", "Paired controls"),
            ("independent_audit.json", "Independent geometry audit"),
            ("profile/HANDOFF.md", "Profile camera and visibility audit"),
            ("lift/VALIDATION.md", "Held lift and impact checks"),
            ("lift/simulation_HANDOFF.md", "Held-lift source and failed comparison"),
            ("lift/impact_sensitivity.json", "61 mm versus rejected 70 mm lift"),
            ("lift/media_validation.json", "Held-lift video validation"),
            ("nonlinear-feasibility/index.html", "Nonlinear peel feasibility"),
            ("full-order-snapshots/index.html", "Contact-aware peel snapshots"),
            ("pod-path/index.html", "Ordered POD peeling: WIP dynamics and videos"),
            ("nonlinear-dynamics/index.html", "Nonlinear dynamics diagnostic"),
            ("nonlinear-dynamics/nested-enrichment/index.html", "Nested modes and actual peel motion"),
            ("nonlinear-dynamics/grip-motion/index.html", "Grip motion and repeatability diagnostic"),
            ("nonlinear-dynamics/cohesive-release/index.html", "Gradual seam release and timestep diagnostic"),
            ("nonlinear-dynamics/trial-damage/index.html", "Trial damage: iteration and work diagnostics"),
            ("nonlinear-dynamics/final-residual/index.html", "Final-state residual and timestep diagnostics"),
            ("nonlinear-dynamics/surface-sampling/index.html", "Surface sampling and contact geometry"),
            ("nonlinear-dynamics/contact-capacity/index.html", "Contact capacity correction and controls"),
        )
        if (directory / name).exists()
    )
    reproduction = (
        f'<p>Use the <a href="{slug}/REPRODUCE.md">current reproduction workflow</a> '
        "for explicit input paths, runtime versions and fresh-output safeguards.</p>"
        if (directory / "REPRODUCE.md").is_file()
        else f"<pre><code>{esc(meta.get('reproduce', 'Pending'))}</code></pre>"
    )
    details = f"""<details><summary>Model, checks and reproducibility</summary>
      <p><b>Driven:</b> {esc(meta.get("driven", "Not yet recorded"))}</p>
      <p><b>Model:</b> {esc(meta.get("model", "Not yet recorded"))}</p>
      <p><b>Modes:</b> {esc(meta.get("mode_count", "Pending"))}. <b>Simulation checks:</b> {checks}.
      <b>Negative control:</b> {control}.</p>
      <table class="parameters"><tbody>{param_rows}</tbody></table>
      <p class="downloads">{downloads}</p><p class="downloads">{source_links}</p>
      {reference_note}
      {reproduction}
      <pre class="raw"><code>{esc(json.dumps(compact_metrics(metrics), indent=2))}</code></pre>
    </details>"""
    latest = ""
    if slug == "banana" and (directory / "pod-path/index.html").is_file():
        if badge == "pass":
            latest = (
                '<p><a href="banana/index.html">Validated ordered-POD peeling report</a> / '
                '<a href="banana/pod-path/index.html">Earlier WIP and failed progression checks</a>.</p>'
            )
        else:
            latest = (
                '<p><a href="banana/pod-path/index.html">Latest WIP: ordered POD peeling videos and audit</a>. '
                "The passive path produces progressive backward folding; two progression checks still fail.</p>"
            )
    return f"""<section id="{slug}" class="object">
      <div class="section-heading"><h2>{esc(title)}</h2><span class="badge {badge}">{esc(status)}</span></div>
      <p class="description">{esc(display.get("description", ""))}</p>
      {latest}{media}{metric_table(display_metrics, timing_label)}{performance_note}{detail_media}
      <ul class="limitations">{limitations}</ul>{details}
    </section>"""


def capacity_summary(root):
    review = load(root / "review.json")
    entries = [
        v["contact_capacity_review"] for name, v in review.items() if name in SLUGS and "contact_capacity_review" in v
    ]
    if not entries:
        return ""
    verified = sum(v["status"] in ("covered", "replayed", "rerun_verified") for v in entries)
    pending = len(entries) - verified
    link = "capacity-reruns/index.html"
    if not (root / link).is_file():
        link = "banana/nonlinear-dynamics/contact-capacity/index.html"
    banana_note = (
        "Banana has separate physical acceptance failures." if review.get("banana", {}).get("blocking") else ""
    )
    return (
        f"<p><b>Contact-capacity verification:</b> {verified}/{len(entries)} audited object studies have complete evidence; "
        f"{pending} remain pending. Missing evidence is not a confirmed simulation failure. "
        f'<a href="{link}">Counts, controlled reruns and original audit</a>. '
        f"{banana_note}</p>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    video_assets = {}
    sections = "".join(section(root, slug, video_assets) for slug in SLUGS)
    publish_videos(root, video_assets)
    nav = "".join(f'<a href="#{slug}">{esc(slug.replace("-", " ").title())}</a>' for slug in SLUGS)
    complete = sum(render_current(root / slug, load(root / slug / "rtx.json")) for slug in SLUGS)
    replays = sum(replay_current(root / slug / "replay", load(root / slug / "replay/rtx.json")) for slug in SLUGS)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Everyday Objects | Reduced Elastic ROMs</title>
    <style>
    *{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:16px}
    body{margin:0;background:#f5f7f7;color:#202629;font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}
    a{color:#086a79;text-underline-offset:3px}a:hover{color:#004650}
    header,main,footer{max-width:1440px;margin:auto;padding:28px 36px}
    header{padding-top:40px;padding-bottom:24px}header .eyebrow{font-size:12px;font-weight:650;text-transform:uppercase;color:#526267;margin:0 0 7px}
    h1{font-size:32px;font-weight:600;line-height:1.2;margin:0 0 14px}h2{font-size:21px;font-weight:600;line-height:1.25;margin:0}
    p{margin:8px 0}header>p{max-width:980px;color:#45545a}.meta{font-size:12px;color:#637277}
    nav{display:flex;flex-wrap:wrap;gap:8px 20px;border-top:1px solid #d6dfe0;border-bottom:1px solid #d6dfe0;margin-top:24px;padding:13px 0;font-size:13px}
    .summary{padding:14px 0 0;border-bottom:1px solid #d6dfe0;margin-bottom:20px}.summary p{max-width:1100px;padding-bottom:14px}
    main{padding-top:0}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));column-gap:32px;row-gap:40px}
    .object{min-width:0;border-top:2px solid #becdce;padding-top:18px}.section-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap}
    .badge{font-size:11px;line-height:1.4;font-weight:600;border-radius:999px;padding:4px 9px;white-space:normal;max-width:100%}
    .pass{background:#deeee5;color:#245d40}.wip{background:#fff0cc;color:#72500c}
    .description{min-height:48px;color:#4a595f;font-size:14px;overflow-wrap:anywhere}
    video,.pending{display:block;width:100%;aspect-ratio:4/3;background:#e3e8e9;border-radius:4px;object-fit:contain;margin:15px 0 0}
    .pending{display:flex;align-items:center;justify-content:center;color:#6b777c}
    .metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin:0;border-bottom:1px solid #d6dfe0}
    .metrics>div{padding:13px 8px 13px 0;min-width:0}.metrics dt{font-size:10px;color:#5e6d73}.metrics dd{margin:2px 0 0;font-size:17px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
    .limitations{padding-left:18px;font-size:12px;color:#59696e;margin:13px 0}.limitations li{margin:4px 0}
    details{font-size:12px;color:#48565c;border-top:1px solid #d6dfe0;padding-top:9px}summary{cursor:pointer;font-weight:600;color:#31555c}
    details p{overflow-wrap:anywhere}pre{overflow:auto;background:#eaf0f0;border-radius:4px;padding:12px;font:11px/1.55 ui-monospace,monospace;max-height:300px}
    .parameters{border-collapse:collapse;width:100%;margin:14px 0}.parameters th,.parameters td{padding:4px 8px;border-bottom:1px solid #dbe3e3;text-align:left;vertical-align:top;overflow-wrap:anywhere}.parameters th{font-weight:500;width:48%}
    .downloads{display:flex;gap:8px 16px;flex-wrap:wrap}footer{border-top:1px solid #ccd7d9;font-size:12px;color:#617176;margin-top:25px}
    @media(max-width:800px){header,main,footer{padding-left:18px;padding-right:18px}.gallery{grid-template-columns:1fr;gap:30px}h1{font-size:27px}.description{min-height:0}.metrics dd{font-size:16px}nav{gap:6px 16px}}
    @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
    </style></head><body>
    <header><p class="eyebrow">Newton / reduced-elastic links / ovrtx</p><h1>Everyday Objects</h1>
    <p>__TOTAL__ studies of flexible objects and contact. Reduced elastic coordinates and VBD dynamics;
    path-traced videos are recovered from saved simulation states.</p>
    <p class="meta">__COUNT__/__TOTAL__ rendered studies &nbsp;|&nbsp; __REPLAYS__/__TOTAL__ normal-then-strain replays &nbsp;|&nbsp; Updated __BUILT__</p><nav aria-label="Objects">__NAV__</nav></header>
    <main><div class="summary"><p><b>Scope.</b> Procedural, uncalibrated prototypes.
    Only stated actuators and supports are prescribed; deformation is solved, with no visual amplification.
    Each study lists its physical assumptions and limits.</p>
    __CAPACITY__</div>
    <p>Timings below are the selected optimized CUDA-graph simulation measurements,
    excluding rendering and diagnostic recording. Videos retain their audited simulation settings.
    <a href="performance/index.html">Benchmark details and validation</a>.</p>
    <div class="gallery">__SECTIONS__</div></main>
    <footer><p>Simulation source base: <a href="https://github.com/mmacklin/newton/tree/8228bf6f690fff4dbb359d5de6447b09cb77c1e7">consolidated reduced-elastic branch</a>.
    <a href="performance/REFINEMENT.md">Current refinement and measurement notes</a> /
    <a href="everyday-rom-source.zip">Initial source bundle (historical)</a> / <a href="source/README.md">Initial reproduction guide</a> /
    <a href="source/CONTRACT.md">Simulation contract</a> / <a href="source/research.md">Work log</a> /
    <a href="source/final_report.md">Research phase summary and remaining gaps</a> /
    <a href="evaluation.json">Artifact validation</a> / <a href="source/render_scene.py">ovrtx renderer</a> /
    <a href="source/MASS_SCALING.md">Isolated mass-scaling experiment</a> /
    <a href="source/FREE_FRAME_INERTIA.md">Free-frame inertial consistency</a>.</p>
    <p>Penetration and recovery definitions are object-specific; inspect each metrics file. Simulation timings exclude rendering and capture.</p></footer>
    <script>document.addEventListener('play',e=>{if(e.target.tagName==='VIDEO')document.querySelectorAll('video').forEach(v=>{if(v!==e.target)v.pause()})},true)</script>
    </body></html>"""
    page = (
        page.replace("__COUNT__", str(complete))
        .replace("__REPLAYS__", str(replays))
        .replace("__TOTAL__", str(len(SLUGS)))
        .replace("__BUILT__", built)
        .replace("__NAV__", nav)
        .replace("__SECTIONS__", sections)
        .replace("__CAPACITY__", capacity_summary(root))
    )
    (root / "index.html").write_text(page)
    print(f"Wrote {root / 'index.html'} ({complete}/{len(SLUGS)} rendered)")


if __name__ == "__main__":
    main()
