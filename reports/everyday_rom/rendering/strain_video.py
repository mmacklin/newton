# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify and concatenate an existing normal video with a strain-only replay.

The plan command is CPU-only and writes no video. Concat is explicit, requires
an unchanged verified plan, and audits every decoded output frame. No normal
phase is rendered again. Lossless reencoding is used when stream copy is not
compatible, so the decoded normal appearance can still be checked exactly.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

if __package__:
    from .strain_adapters import digest, load_arrays
    from .strain_replay import array_digest, validate_scene, verify, write_json
    from .surface_strain import field, jet
else:
    from strain_adapters import digest, load_arrays
    from strain_replay import array_digest, validate_scene, verify, write_json
    from surface_strain import field, jet

COPY_FIELDS = ("codec_name", "profile", "width", "height", "pix_fmt", "r_frame_rate", "time_base", "extradata_hash")


def probe_video(path: Path) -> dict:
    """Fully decode timestamps and reject missing, duplicate, or dropped frames."""
    fields = "codec_type,codec_name,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_read_frames,extradata_hash"
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_data_hash",
        "sha256",
        "-show_entries",
        f"stream={fields}:frame=media_type,best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    raw = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    if len(raw["streams"]) != 1 or raw["streams"][0]["codec_type"] != "video":
        raise ValueError("Replay clips must contain exactly one video stream and no audio")
    stream = raw["streams"][0]
    fps = Fraction(stream["avg_frame_rate"])
    if fps <= 0 or fps != Fraction(stream["r_frame_rate"]):
        raise ValueError("Variable or invalid video frame rate")
    times = np.array([float(item["best_effort_timestamp_time"]) for item in raw["frames"]])
    frames = int(stream["nb_read_frames"])
    expected = np.arange(frames) / float(fps)
    if frames < 2 or len(times) != frames or not np.allclose(times, expected, rtol=0, atol=2e-6):
        raise ValueError("Video PTS are not exact zero-based uniform saved frames")
    duration = float(stream["duration"])
    if abs(duration - frames / float(fps)) > 2e-6:
        raise ValueError("Video duration does not match its frame count and FPS")
    return {
        "stream": stream,
        "frames": frames,
        "fps": float(fps),
        "duration_s": duration,
        "timestamps_sha256": array_digest(times),
        "uniform_pts": True,
    }


def decoded_frame_hashes(path: Path) -> list[str]:
    """Hash the RGB24 appearance of every decoded frame, without reencoding."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "0",
        "-f",
        "framehash",
        "-hash",
        "sha256",
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    hashes = []
    for line in result.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        columns = [value.strip() for value in line.split(",")]
        if len(columns) != 6 or len(columns[-1]) != 64:
            raise ValueError("Invalid ffmpeg framehash output")
        hashes.append(columns[-1])
    if len(hashes) < 2:
        raise ValueError("Decoded video contains too few frames")
    return hashes


def validate_video_pair(normal: dict, strain: dict, frames: int, fps: float) -> str:
    """Return a compatible concat mode, never silently changing size or FPS."""
    for video in (normal, strain):
        if video["frames"] != frames or video["fps"] != fps or not video["uniform_pts"]:
            raise ValueError("Video frame count or FPS differs from the saved source")
        if abs(video["duration_s"] - frames / fps) > 2e-6:
            raise ValueError("Video duration differs from the saved source")
    a, b = normal["stream"], strain["stream"]
    if any(a[key] != b[key] for key in ("width", "height", "pix_fmt")):
        raise ValueError("Both clips must have identical resolution and pixel format")
    if all(a.get(key) is not None and a.get(key) == b.get(key) for key in COPY_FIELDS):
        return "copy"
    if a["pix_fmt"] not in ("yuv420p", "yuv422p", "yuv444p"):
        raise ValueError("Incompatible codec headers and unsupported lossless H.264 pixel format")
    return "lossless"


def _render_audit(directory: Path, colored: bool) -> tuple[dict, dict]:
    audit = json.loads((directory / "rtx.json").read_text())
    required = {"scene.npz", "scene.json", "metrics.json"}
    if not required <= set(audit["inputs_sha256"]):
        raise ValueError("Render audit does not bind all source inputs")
    checked = {str(directory / "rtx.json"): digest(directory / "rtx.json")}
    for name, expected in audit["inputs_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Stale rendered source: {directory / name}")
        checked[str(directory / name)] = expected
    video = directory / "rtx.mp4"
    if digest(video) != audit["video_sha256"]:
        raise ValueError("Video hash differs from the renderer audit")
    checked[str(video)] = audit["video_sha256"]
    if not 0 <= audit["max_renderer_vertex_error_m"] <= 2e-7:
        raise ValueError("Renderer geometry readback did not match source")
    if colored and not 0 <= audit.get("max_renderer_color_error", float("inf")) <= 2e-7:
        raise ValueError("Renderer lacks valid every-frame color readback")
    return audit, checked


def _compare_scenes(prepared: Path, candidate: Path) -> tuple[dict, dict, dict, dict]:
    manifest = json.loads((prepared / "manifest.json").read_text())
    normal_meta = json.loads((prepared / "normal_scene.json").read_text())
    meta = json.loads((candidate / "scene.json").read_text())
    arrays = load_arrays(candidate / "scene.npz")
    count = validate_scene(meta, arrays)
    if count != manifest["schedule"]["frames_per_phase"] or meta["fps"] != manifest["schedule"]["fps"]:
        raise ValueError("Candidate strain sequence differs from saved frame count/FPS")
    if meta["camera"] != manifest["camera"]:
        raise ValueError("Candidate strain camera differs from the original camera")
    for key, expected in manifest["original_arrays_sha256"].items():
        if key not in arrays or array_digest(arrays[key]) != expected:
            raise ValueError(f"Candidate changed original source array: {key}")
    if len(meta["meshes"]) != len(normal_meta["meshes"]):
        raise ValueError("Candidate mesh count differs")
    selected = {item["mesh_index"] for item in manifest["surfaces"]}
    for i, item in enumerate(meta["meshes"]):
        actual, expected = copy.deepcopy(item), copy.deepcopy(normal_meta["meshes"][i])
        if i in selected:
            if "vertex_colors" not in actual:
                raise ValueError("Candidate lacks selected strain colors")
            for key in ("vertex_colors", "roughness", "metallic"):
                actual.pop(key, None)
                expected.pop(key, None)
        if actual != expected:
            raise ValueError("Candidate changed mesh mapping or an uncolored material")
    if "color_legend" not in meta:
        raise ValueError("Candidate lacks a quantitative strain legend")
    return manifest, normal_meta, meta, arrays


def verify_headphone_reuse(prepared: Path, candidate: Path) -> dict:
    """Verify accepted local-state headphone strain, not just a metadata claim."""
    manifest, _, meta, arrays = _compare_scenes(prepared, candidate)
    if manifest["object"] != "headphones" or len(manifest["surfaces"]) != 1:
        raise ValueError("Legacy accepted-strain reuse is specific to the headphones")
    scale = manifest["scale"]["max"]
    metrics = json.loads((candidate / "metrics.json").read_text())["strain_view"]
    if metrics["color_scale"] != [0.0, scale] or scale != 0.03 or metrics["units"] != "dimensionless":
        raise ValueError("Accepted headphone video must use the exact fixed 0-3% scale")
    source = Path(manifest["source"])
    for name, expected in metrics["source_hashes"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Legacy headphone source hash mismatch: {name}")
    state, audit = load_arrays(source / "original_states.npz"), load_arrays(candidate / "strain.npz")
    rest, phi = state["runtime_vertex_local"], state["runtime_vertex_phi"]
    summary = manifest["surfaces"][0]
    item = meta["meshes"][summary["mesh_index"]]
    if item["name"] != "elastic_headband":
        raise ValueError("Legacy headphone strain colors assigned to wrong surface")
    faces = arrays[item["faces"]]
    np.testing.assert_array_equal(rest, audit["rest_vertices"])
    np.testing.assert_array_equal(faces, audit["faces"])
    np.testing.assert_array_equal(state["times"], audit["times"])
    np.testing.assert_array_equal(arrays["times"], audit["times"])
    if (
        array_digest(rest) != summary["reference_vertices_sha256"]
        or array_digest(faces) != summary["reference_faces_sha256"]
    ):
        raise ValueError("Legacy headphone material reference differs")
    start = int(state["q_start"])
    world_error = 0.0
    for frame, q in enumerate(state["joint_q"][:, start : start + phi.shape[1]]):
        local = rest + np.einsum("nmc,m->nc", phi, q)
        principal, vertices = field(rest, faces, local)
        np.testing.assert_array_equal(principal, audit["principal"][frame])
        np.testing.assert_array_equal(vertices, audit["vertex_magnitude"][frame])
        np.testing.assert_array_equal(jet(vertices / scale), arrays[item["vertex_colors"]][frame])
        world_principal, _ = field(rest, faces, arrays[item["vertices"]][frame])
        world_error = max(world_error, float(np.abs(world_principal - principal).max()))
    if world_error > 2e-4:
        raise ValueError("Legacy local-state strain disagrees with exact saved world geometry")
    return {
        "accepted_headphones_reused": True,
        "frames": len(state["times"]),
        "scale": [0.0, scale],
        "every_principal_value_and_color_recomputed": True,
        "max_world_local_strain_error": world_error,
        "note": "Exact same simulation and geometry. Accepted colors use local-state strain; new world-coordinate fields differ only by recorded export precision.",
    }


def plan_video(prepared: Path, candidate: Path, output: Path, accepted_headphones: bool = False) -> dict:
    """Verify render provenance and both full decodes, without producing video."""
    prepared, candidate, output = Path(prepared).resolve(), Path(candidate).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Video plan output must be a fresh owned directory")
    verify(prepared)
    manifest, normal_meta, candidate_meta, candidate_arrays = _compare_scenes(prepared, candidate)
    source = Path(manifest["source"])
    if source in output.parents or candidate in output.parents:
        raise ValueError("Do not write plans inside original video bundles")
    if json.loads((source / "scene.json").read_text()) != normal_meta:
        raise ValueError("Normal source camera/material metadata changed")
    normal_audit, checked = _render_audit(source, False)
    strain_audit, hashes = _render_audit(candidate, True)
    checked.update(hashes)
    checked.update(manifest["inputs_sha256"])
    checked[str(prepared / "manifest.json")] = digest(prepared / "manifest.json")
    for name, value in manifest["outputs_sha256"].items():
        checked[str(prepared / name)] = value
    if accepted_headphones:
        field_check = verify_headphone_reuse(prepared, candidate)
        checked[str(candidate / "strain.npz")] = digest(candidate / "strain.npz")
    else:
        verify(candidate)
        other = json.loads((candidate / "manifest.json").read_text())
        if other["scale"] != manifest["scale"] or other["original_arrays_sha256"] != manifest["original_arrays_sha256"]:
            raise ValueError("Strain video uses a different scale or source trajectory")
        prepared_arrays = load_arrays(prepared / "scene.npz")
        for summary in manifest["surfaces"]:
            item = candidate_meta["meshes"][summary["mesh_index"]]
            np.testing.assert_array_equal(
                candidate_arrays[item["vertex_colors"]], prepared_arrays[summary["color_array"]]
            )
        checked[str(candidate / "manifest.json")] = digest(candidate / "manifest.json")
        field_check = {"every_field_recomputed": True, "identical_colors": True}
    frames, fps = manifest["schedule"]["frames_per_phase"], manifest["schedule"]["fps"]
    normal_video, strain_video = source / "rtx.mp4", candidate / "rtx.mp4"
    normal_probe, strain_probe = probe_video(normal_video), probe_video(strain_video)
    mode = validate_video_pair(normal_probe, strain_probe, frames, fps)
    for audit, probe in ((normal_audit, normal_probe), (strain_audit, strain_probe)):
        if (
            audit["frames"] != frames
            or audit["fps"] != fps
            or audit["resolution"] != [probe["stream"]["width"], probe["stream"]["height"]]
        ):
            raise ValueError("Render audit and decoded video dimensions/sampling differ")
    normal_frames, strain_frames = decoded_frame_hashes(normal_video), decoded_frame_hashes(strain_video)
    if len(normal_frames) != frames or len(strain_frames) != frames:
        raise ValueError("Full video decode frame count mismatch")
    for path, value in checked.items():
        if digest(Path(path)) != value:
            raise ValueError(f"Input changed during video reuse verification: {path}")
    output.mkdir(parents=True)
    write_json(
        output / "decoded-frames.json", {"normal_rgb24_sha256": normal_frames, "strain_rgb24_sha256": strain_frames}
    )
    plan = {
        "schema_version": 1,
        "object": manifest["object"],
        "status": "verified_ready_for_concat",
        "normal_video": str(normal_video),
        "strain_video": str(strain_video),
        "prepared": str(prepared),
        "inputs_sha256": checked,
        "normal_probe": normal_probe,
        "strain_probe": strain_probe,
        "recommended_mode": mode,
        "scale": manifest["scale"],
        "camera": manifest["camera"],
        "schedule": manifest["schedule"],
        "field_verification": field_check,
        "decoded_frames_sha256": digest(output / "decoded-frames.json"),
        "decoded_appearance": "Every RGB24 frame hashed; concat must reproduce the normal half exactly, and also the strain half",
        "expected_duration_s": 2 * frames / fps,
        "new_video_written": False,
        "normal_rerender_required": False,
        "strain_rerender_required": False,
        "labels": "NORMAL first, then STRAIN with its existing quantitative legend; normal pixels are not overlaid or modified",
    }
    write_json(output / "video-plan.json", plan)
    return plan


def concatenate(plan_directory: Path, mode: str | None = None) -> dict:
    """Concatenate a verified plan, then require exact full decoded appearance."""
    directory = Path(plan_directory).resolve()
    plan = json.loads((directory / "video-plan.json").read_text())
    if plan.get("schema_version") != 1 or plan.get("status") != "verified_ready_for_concat":
        raise ValueError("A verified video reuse plan is required")
    for path, expected in plan["inputs_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Reuse plan input changed: {path}")
    if digest(directory / "decoded-frames.json") != plan["decoded_frames_sha256"]:
        raise ValueError("Decoded source-frame audit changed")
    frames, fps = plan["schedule"]["frames_per_phase"], plan["schedule"]["fps"]
    available = validate_video_pair(plan["normal_probe"], plan["strain_probe"], frames, fps)
    mode = mode or available
    if mode not in ("copy", "lossless") or (mode == "copy" and available != "copy"):
        raise ValueError("Requested concat codec mode is incompatible")
    final, partial = directory / "rtx.mp4", directory / "rtx.partial.mp4"
    if final.exists() or partial.exists():
        raise ValueError("Video output already exists; use a fresh verified plan directory")
    if mode == "copy":
        lines = ["ffconcat version 1.0"]
        for key in ("normal_video", "strain_video"):
            path = plan[key]
            if "\n" in path or "\r" in path:
                raise ValueError("Invalid newline in concat input path")
            escaped = path.replace("'", "'\\''")
            lines.extend([f"file '{escaped}'", f"duration {frames / fps:.17g}"])
        listing = directory / "inputs.ffconcat"
        listing.write_text("\n".join(lines) + "\n")
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            str(partial),
        ]
    else:
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-i",
            plan["normal_video"],
            "-i",
            plan["strain_video"],
            "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-qp",
            "0",
            "-preset",
            "medium",
            "-pix_fmt",
            plan["normal_probe"]["stream"]["pix_fmt"],
            "-vsync",
            "0",
            "-movflags",
            "+faststart",
            str(partial),
        ]
    subprocess.run(command, check=True)
    probe = probe_video(partial)
    if probe["frames"] != 2 * frames or probe["fps"] != fps or abs(probe["duration_s"] - 2 * frames / fps) > 2e-6:
        raise ValueError("Concatenated duration/frame count/FPS failed; partial video is not publishable")
    decoded = decoded_frame_hashes(partial)
    original = json.loads((directory / "decoded-frames.json").read_text())
    if decoded[:frames] != original["normal_rgb24_sha256"] or decoded[frames:] != original["strain_rgb24_sha256"]:
        raise ValueError("Decoded appearance changed; partial video is not publishable")
    partial.rename(final)
    audit = {
        "schema_version": 1,
        "verified": True,
        "mode": mode,
        "ffmpeg_command": command,
        "plan_sha256": digest(directory / "video-plan.json"),
        "video_sha256": digest(final),
        "probe": probe,
        "normal_decoded_frames_identical": frames,
        "strain_decoded_frames_identical": frames,
        "geometry_and_camera_replay_verified": True,
        "normal_rerendered": False,
        "duration_error_s": abs(probe["duration_s"] - 2 * frames / fps),
    }
    write_json(directory / "concat-audit.json", audit)
    write_json(directory / "publication.json", publication_entry(directory))
    return audit


def publication_entry(directory: Path) -> dict:
    """Format a verified primary replay entry without claiming new physics."""
    directory = Path(directory)
    plan = json.loads((directory / "video-plan.json").read_text())
    audit = json.loads((directory / "concat-audit.json").read_text())
    if not audit.get("verified") or audit["plan_sha256"] != digest(directory / "video-plan.json"):
        raise ValueError("Publication requires a verified unchanged concat audit")
    if audit["video_sha256"] != digest(directory / "rtx.mp4"):
        raise ValueError("Publication video hash differs from the decoded audit")
    frames, fps = plan["schedule"]["frames_per_phase"], plan["schedule"]["fps"]
    return {
        "schema_version": 1,
        "slug": plan["object"],
        "primary_video": f"{plan['object']}/replay/rtx.mp4",
        "staged_video": str((directory / "rtx.mp4").resolve()),
        "video_sha256": audit["video_sha256"],
        "cache_key": audit["video_sha256"][:16],
        "frames": 2 * frames,
        "fps": fps,
        "duration_s": audit["probe"]["duration_s"],
        "phase_boundary_s": frames / fps,
        "phases": ["NORMAL", "STRAIN"],
        "strain_scale": plan["scale"],
        "camera": plan["camera"],
        "audit_file": "concat-audit.json",
        "plan_file": "video-plan.json",
        "normal_decoded_frames_identical": audit["normal_decoded_frames_identical"],
        "strain_decoded_frames_identical": audit["strain_decoded_frames_identical"],
        "trajectory_provenance": "Exact saved input states, no resimulation; physical validation remains as recorded in metrics.json",
        "performance_provenance": "Separate from simulation timing; this replay does not establish which runtime produced the saved states",
        "normal_render_reused": True,
        "strain_render_reused": bool(plan["field_verification"].get("accepted_headphones_reused")),
        "new_physics_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Verify existing clips and write publication-ready manifests only")
    plan.add_argument("prepared", type=Path)
    plan.add_argument("strain_video_directory", type=Path)
    plan.add_argument("output", type=Path)
    plan.add_argument("--accepted-headphones", action="store_true")
    concat = sub.add_parser("concat", help="Explicit CPU concat of an unchanged verified plan")
    concat.add_argument("plan_directory", type=Path)
    concat.add_argument("--mode", choices=("copy", "lossless"))
    args = parser.parse_args()
    result = (
        plan_video(args.prepared, args.strain_video_directory, args.output, args.accepted_headphones)
        if args.command == "plan"
        else concatenate(args.plan_directory, args.mode)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
