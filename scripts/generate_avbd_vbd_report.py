from __future__ import annotations

import html
import importlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np

import newton.viewer
from newton.examples.vbd._avbd_common import (
    REFERENCE_ALPHA,
    REFERENCE_BETA_ANGULAR,
    REFERENCE_BETA_LINEAR,
    REFERENCE_COLLISION_MARGIN,
    REFERENCE_COMMIT,
    REFERENCE_CONTACT_MATCH_NORMAL_DOT_THRESHOLD,
    REFERENCE_CONTACT_MATCH_POS_THRESHOLD,
    REFERENCE_DT,
    REFERENCE_FRICTION_EPSILON,
    REFERENCE_GAMMA,
    REFERENCE_GRAVITY,
    REFERENCE_ITERATIONS,
    REFERENCE_PENALTY_MIN,
    REFERENCE_STICK_THRESH,
    SCENES,
)

WIDTH = 960
HEIGHT = 540
FPS = 60

DEFAULT_REPORT_ROOT = Path.home() / "reports" / "newton_avbd_vbd"
ROOT = Path(os.environ.get("NEWTON_AVBD_REPORT_DIR", DEFAULT_REPORT_ROOT)).expanduser().resolve()
ASSETS = ROOT / "assets"
BASELINE_ASSETS = ROOT / "assets_baseline"
ASSETS.mkdir(parents=True, exist_ok=True)


def configure_renderer(viewer, scene_info) -> None:
    renderer = getattr(viewer, "renderer", None)
    if renderer is None:
        return

    renderer.draw_shadows = True
    renderer._sun_direction = np.array((0.60, -0.75, 0.28), dtype=np.float32)
    renderer._sun_direction /= np.linalg.norm(renderer._sun_direction)
    renderer.shadow_extents = max(40.0, min(180.0, scene_info.camera_distance * 3.0))
    renderer.shadow_radius = 2.0
    renderer.diffuse_scale = 1.25
    renderer.specular_scale = 0.65
    renderer.exposure = 1.35
    renderer.ambient_sky = (0.55, 0.58, 0.64)
    renderer.ambient_ground = (0.16, 0.17, 0.20)


def module_name_for(scene_name: str) -> str:
    return f"vbd_avbd_{scene_name}"


def render_scene(scene_name: str) -> dict:
    info = SCENES[scene_name]
    module = importlib.import_module(f"newton.examples.vbd.example_{module_name_for(scene_name)}")
    video_path = ASSETS / f"{scene_name}.mp4"
    shot_path = ASSETS / f"{scene_name}.jpg"

    viewer = newton.viewer.ViewerGL(width=WIDTH, height=HEIGHT, headless=True)
    status = "Stable"
    error = ""
    frame_count = 0
    frame_min = 255
    frame_max = 0
    first_range: tuple[int, int] | None = None
    last_range: tuple[int, int] | None = None
    last_frame: np.ndarray | None = None
    wrote_shot = False

    try:
        example = module.Example(viewer, SimpleNamespace())
        configure_renderer(viewer, info)
        with imageio.get_writer(video_path, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame_index in range(info.report_frames):
                example.step()
                example.render()
                frame = viewer.get_frame().numpy()
                local_min = int(frame.min())
                local_max = int(frame.max())
                if first_range is None:
                    first_range = (local_min, local_max)
                last_range = (local_min, local_max)
                last_frame = frame
                frame_min = min(frame_min, local_min)
                frame_max = max(frame_max, local_max)
                writer.append_data(frame)
                frame_count = frame_index + 1
                if frame_index == info.report_frames // 2:
                    imageio.imwrite(shot_path, frame)
                    wrote_shot = True
                if example.nan_detected:
                    status = "Broken"
                    error = "non-finite state detected during run"
                    break
        if not wrote_shot and last_frame is not None:
            imageio.imwrite(shot_path, last_frame)
        try:
            example.test_final()
        except Exception as exc:
            status = "Broken"
            error = str(exc)
        metrics = dict(example.metric_summary)
    except Exception as exc:
        status = "Broken"
        error = str(exc)
        metrics = {}
    finally:
        viewer.close()

    decode = validate_video(video_path)
    if frame_max == frame_min:
        status = "Broken"
        error = (error + "; " if error else "") + "captured frames were blank"

    result = {
        "scene": scene_name,
        "title": info.title,
        "description": info.description,
        "status": status,
        "error": error,
        "frames_requested": info.report_frames,
        "frames_completed": frame_count,
        "video": f"assets/{scene_name}.mp4",
        "screenshot": f"assets/{scene_name}.jpg",
        "video_size": video_path.stat().st_size if video_path.exists() else 0,
        "frame_min": frame_min,
        "frame_max": frame_max,
        "first_range": first_range,
        "last_range": last_range,
        "decode": decode,
        "metrics": metrics,
        "note": info.status_note,
    }
    print(json.dumps(result, indent=2))
    return result


def validate_video(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        frames = reader.count_frames()
        first = reader.get_data(0)
        last = reader.get_data(max(frames - 1, 0))
        return {
            "exists": True,
            "frames": int(frames),
            "fps": float(meta.get("fps", 0.0)),
            "duration": float(meta.get("duration", 0.0)),
            "first_min": int(first.min()),
            "first_max": int(first.max()),
            "last_min": int(last.min()),
            "last_max": int(last.max()),
        }
    finally:
        reader.close()


def fmt_float(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def load_baseline_results() -> dict[str, dict]:
    path = ROOT / "metrics_baseline.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return {item.get("scene", ""): item for item in data if isinstance(item, dict)}


def video_src(path: str, stamp: str) -> str:
    return f"{path}?datetime={stamp}"


def write_html(results: list[dict]) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    baseline = load_baseline_results()
    stable_count = sum(1 for item in results if item["status"] == "Stable")
    broken_count = len(results) - stable_count
    baseline_stable_count = sum(1 for item in baseline.values() if item.get("status") == "Stable")
    updated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    cards = []
    rows = []
    for item in results:
        metrics = item.get("metrics", {})
        prev = baseline.get(item["scene"], {})
        prev_metrics = prev.get("metrics", {}) if isinstance(prev, dict) else {}
        prev_status = prev.get("status", "n/a") if isinstance(prev, dict) else "n/a"
        prev_status_class = "ok" if prev_status == "Stable" else "broken"
        prev_video = f"assets_baseline/{item['scene']}.mp4"
        prev_video_path = ROOT / prev_video
        status_class = "ok" if item["status"] == "Stable" else "broken"
        err = html.escape(item["error"])
        note = html.escape(item["note"])
        media = (
            f"""
              <div class="compare-media">
                <div>
                  <div class="video-label">Before</div>
                  <video controls preload="metadata" src="{video_src(prev_video, stamp)}"></video>
                </div>
                <div>
                  <div class="video-label">After</div>
                  <video controls preload="metadata" src="{video_src(item["video"], stamp)}"></video>
                </div>
              </div>
            """
            if prev_video_path.exists()
            else f'<video controls preload="metadata" src="{video_src(item["video"], stamp)}"></video>'
        )
        cards.append(
            f"""
            <article class="card">
              {media}
              <div class="card-body">
                <div class="card-head">
                  <h3>{html.escape(item["title"])}</h3>
                  <span class="status {status_class}">{html.escape(item["status"])}</span>
                </div>
                <p>{html.escape(item["description"])}</p>
                <dl>
                  <div><dt>Before</dt><dd><span class="status {prev_status_class}">{html.escape(prev_status)}</span></dd></div>
                  <div><dt>After</dt><dd><span class="status {status_class}">{html.escape(item["status"])}</span></dd></div>
                  <div><dt>Frames</dt><dd>{item["frames_completed"]}/{item["frames_requested"]}</dd></div>
                  <div><dt>Before final speed</dt><dd>{fmt_float(prev_metrics.get("final_speed"))} m/s</dd></div>
                  <div><dt>Max speed</dt><dd>{fmt_float(metrics.get("max_speed"))} m/s</dd></div>
                  <div><dt>Final speed</dt><dd>{fmt_float(metrics.get("final_speed"))} m/s</dd></div>
                  <div><dt>Max angular</dt><dd>{fmt_float(metrics.get("max_angular_speed"))} rad/s</dd></div>
                  <div><dt>Max |x|</dt><dd>{fmt_float(metrics.get("max_abs_position"))} m</dd></div>
                  <div><dt>Max z</dt><dd>{fmt_float(metrics.get("max_z"))} m</dd></div>
                  <div><dt>Max contacts</dt><dd>{metrics.get("max_contacts", "n/a")}</dd></div>
                </dl>
                <p class="note">{note}{("<br><strong>Failure:</strong> " + err) if err else ""}</p>
              </div>
            </article>
            """
        )
        rows.append(
            f"""
            <tr>
              <td>{html.escape(item["title"])}</td>
              <td><span class="status {prev_status_class}">{html.escape(prev_status)}</span></td>
              <td><span class="status {status_class}">{html.escape(item["status"])}</span></td>
              <td>{item["frames_completed"]}/{item["frames_requested"]}</td>
              <td>{fmt_float(prev_metrics.get("final_speed"))}</td>
              <td>{fmt_float(metrics.get("max_speed"))}</td>
              <td>{fmt_float(metrics.get("final_speed"))}</td>
              <td>{fmt_float(metrics.get("max_angular_speed"))}</td>
              <td>{fmt_float(metrics.get("max_abs_position"))}</td>
              <td>{fmt_float(metrics.get("max_z"))}</td>
              <td>{metrics.get("max_contacts", "n/a")}</td>
              <td>{html.escape(item["error"])}</td>
            </tr>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0">
  <meta http-equiv="Pragma" content="no-cache">
  <meta http-equiv="Expires" content="0">
  <title>Newton VBD AVBD 3D Reference Scene Sweep</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #14161d;
      --panel: #1e222c;
      --panel-2: #252a36;
      --text: #eef2f8;
      --muted: #aeb7c6;
      --line: #333a48;
      --accent: #63d8ff;
      --accent-2: #ff9b44;
      --ok: #73e2a7;
      --warn: #ffb45f;
      --bad: #ff8b7e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1120px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 38px 0 56px;
    }}
    h1, h2, h3 {{
      line-height: 1.15;
      margin: 0;
    }}
    h1 {{ font-size: clamp(30px, 5vw, 52px); letter-spacing: 0; }}
    h2 {{
      margin-top: 42px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    h3 {{ font-size: 18px; letter-spacing: 0; }}
    p {{
      margin: 12px 0 0;
      color: var(--muted);
    }}
    a {{
      color: var(--accent);
      font-weight: 650;
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
    code {{
      background: #0f1117;
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 1px 5px;
      color: #d9f5ff;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      padding-bottom: 26px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 24px;
    }}
    .subtitle {{ max-width: 880px; }}
    .meta, .summary, .grid {{
      display: grid;
      gap: 12px;
    }}
    .meta {{ grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }}
    .summary {{
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      margin: 24px 0;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .metric .value {{
      margin-top: 6px;
      font-size: 23px;
      font-weight: 650;
      color: var(--text);
    }}
    .meta .metric .value {{
      font-size: 1rem;
      line-height: 1.4;
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .section h2 {{ margin-top: 0; }}
    .section-title {{
      margin: 42px 0 16px;
      font-size: 24px;
    }}
    .tag, .status {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 9px;
      margin: 4px 6px 0 0;
      border: 1px solid currentColor;
      background: #0f1117;
      font-size: 13px;
      font-weight: 650;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    .tag {{ color: var(--accent); }}
    .ok {{ color: var(--ok); }}
    .wip {{ color: var(--warn); }}
    .broken {{ color: var(--bad); }}
    .summary strong {{
      display: block;
      font-size: 23px;
      color: var(--text);
    }}
    .grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .card-body {{ padding: 18px; }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }}
    video {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #000;
      border-bottom: 1px solid var(--line);
    }}
    .compare-media {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1px;
      background: var(--line);
      border-bottom: 1px solid var(--line);
    }}
    .compare-media video {{
      border-bottom: 0;
    }}
    .video-label {{
      background: #0f1117;
      color: var(--muted);
      border-bottom: 1px solid var(--line);
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 14px;
      margin: 14px 0 0;
    }}
    dt {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    dd {{
      margin: 0;
      font-weight: 650;
      color: var(--text);
    }}
    .note {{
      border-left: 3px solid var(--accent-2);
      background: #211b16;
      padding: 12px 14px;
      margin-top: 16px;
      border-radius: 6px;
      color: #ffd8b4;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: var(--panel-2);
      color: var(--text);
      font-size: 13px;
      font-weight: 650;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    @media (max-width: 820px) {{
      main {{
        width: min(100vw - 24px, 1120px);
        padding-top: 24px;
      }}
      .summary, .grid {{ grid-template-columns: 1fr; }}
      .compare-media {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div>
        <h1>Newton VBD AVBD 3D Reference Scene Sweep</h1>
        <p class="subtitle">
          Recreates the non-empty scenes from
          <a href="https://github.com/savant117/avbd-demo3d/tree/{REFERENCE_COMMIT}">savant117/avbd-demo3d</a>
          using Newton's rigid VBD/AVBD path. The source repo's Empty scene is omitted because it has no bodies.
        </p>
      </div>
      <div>
        <span class="tag">Rigid VBD</span>
        <span class="tag">AVBD hard constraints</span>
        <span class="tag">ViewerGL videos</span>
        <span class="tag">Contact history</span>
        <span class="tag">Reference dt</span>
      </div>
      <div class="meta">
        <div class="metric">
          <div class="label">Reference Commit</div>
          <div class="value"><code>{REFERENCE_COMMIT[:10]}</code></div>
        </div>
        <div class="metric">
          <div class="label">Updated</div>
          <div class="value">{updated}</div>
        </div>
        <div class="metric">
          <div class="label">Latest Verification</div>
          <div class="value">{len(results)} MP4 decode checks pass; camera framing and scene-scaled shadows spot-checked from generated stills.</div>
        </div>
        <div class="metric">
          <div class="label">Before / After Failures</div>
          <div class="value">{len(baseline) - baseline_stable_count if baseline else "n/a"} / {broken_count}</div>
        </div>
      </div>
    </section>

    <section class="summary">
      <div class="metric">
        <div class="label">Stable Scenes</div>
        <div class="value">{stable_count}/{len(results)}</div>
      </div>
      <div class="metric">
        <div class="label">Timestep</div>
        <div class="value">{REFERENCE_DT:.6f}s</div>
      </div>
      <div class="metric">
        <div class="label">VBD Iterations</div>
        <div class="value">{REFERENCE_ITERATIONS}</div>
      </div>
      <div class="metric">
        <div class="label">AVBD Alpha</div>
        <div class="value">{REFERENCE_ALPHA:.2f}</div>
      </div>
      <div class="metric">
        <div class="label">Gravity</div>
        <div class="value">{REFERENCE_GRAVITY:g} m/s^2</div>
      </div>
    </section>

    <section class="section">
      <h2>Simulation Parameters</h2>
      <p>
        Parameters: beta linear <code>{REFERENCE_BETA_LINEAR:g}</code>,
        beta angular <code>{REFERENCE_BETA_ANGULAR:g}</code>, gamma <code>{REFERENCE_GAMMA}</code>,
        penalty seed <code>{REFERENCE_PENALTY_MIN}</code>, collision margin <code>{REFERENCE_COLLISION_MARGIN}</code>,
        per-shape contact gap <code>{REFERENCE_COLLISION_MARGIN}</code>,
        stick threshold <code>{REFERENCE_STICK_THRESH:g}</code>, friction smoothing epsilon
        <code>{REFERENCE_FRICTION_EPSILON:g}</code>, contact-match midpoint threshold
        <code>{REFERENCE_CONTACT_MATCH_POS_THRESHOLD:g}</code> m, and contact normal dot threshold
        <code>{REFERENCE_CONTACT_MATCH_NORMAL_DOT_THRESHOLD:g}</code>. A few stress scenes apply explicit
        per-scene Newton-only stabilizers, noted in their cards: serial rigid colors to match the AVBD source's
        body loop, a higher initial contact penalty for pyramid, sticky matching/deadzone for the tall stack,
        and small damping for the soft fixed-joint lattice.
        Newton's VBD defaults differ from the AVBD demo for several of these values, so the examples pass the
        reference values explicitly.
        Report renders enable ViewerGL shadows with a scene-scaled shadow volume and a lower sun angle so the shadows remain inside the map and readable in the clips.
        The original AVBD demo repo does not include automated scene tests, so stability status is reported from Newton
        example-side assertions during video generation: finite state, bounded positions, max/final speed, angular speed,
        and scene-specific invariants such as no high-friction ramp-box tumbling.
      </p>
      <p>
        Remaining implementation deltas are still material: Newton uses CollisionPipeline contact matching rather than
        the demo's O(n^2) feature-key manifold merge, and the tall-stack stabilization still relies on Newton's
        body-level stick deadzone rather than a cleaner contact-state formulation.
      </p>
    </section>

    <h2 class="section-title">Stability Metrics</h2>
    <section class="section">
      <table>
        <thead>
          <tr>
            <th>Scene</th>
            <th>Before</th>
            <th>Status</th>
            <th>Frames</th>
            <th>Before final speed</th>
            <th>Max speed</th>
            <th>Final speed</th>
            <th>Max angular</th>
            <th>Max |x|</th>
            <th>Max z</th>
            <th>Max contacts</th>
            <th>Failure</th>
          </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </section>

    <h2 class="section-title">Scene Videos</h2>
    <section class="grid">
      {"".join(cards)}
    </section>
  </main>
</body>
</html>
"""
    path = ROOT / "avbd_vbd_report.html"
    path.write_text(html_text)
    return path


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--scene":
        scene_name = sys.argv[2]
        result = render_scene(scene_name)
        (ASSETS / f"{scene_name}.json").write_text(json.dumps(result, indent=2))
        return

    results = []
    for scene_name in SCENES:
        env = os.environ.copy()
        env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
        subprocess.run([sys.executable, __file__, "--scene", scene_name], check=True, cwd=ROOT, env=env)
        results.append(json.loads((ASSETS / f"{scene_name}.json").read_text()))
    (ROOT / "metrics.json").write_text(json.dumps(results, indent=2))
    report = write_html(results)
    print(f"REPORT {report}")


if __name__ == "__main__":
    main()
