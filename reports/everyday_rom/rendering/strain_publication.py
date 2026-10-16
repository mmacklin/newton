# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Stage sealed normal scenes and package portable, separately audited replays."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

if __package__:
    from .strain_adapters import adapt, digest, load_arrays
    from .strain_replay import verify, write_json
    from .strain_video import _render_audit, decoded_frame_hashes, publication_entry
    from .surface_strain import field, jet, transform_points
else:
    from strain_adapters import adapt, digest, load_arrays
    from strain_replay import verify, write_json
    from strain_video import _render_audit, decoded_frame_hashes, publication_entry
    from surface_strain import field, jet, transform_points


def stage_normal(source: Path, target: Path) -> dict:
    """Copy every sealed source file, retaining bytes and recording ownership."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists() or source == target or source in target.parents:
        raise ValueError("Normal staging requires a new directory outside the sealed source")
    seal_checks = {}
    for seal_name, key in (
        ("manifest.json", None),
        ("integration_manifest.json", "sha256"),
        ("source_manifest.json", "files"),
    ):
        if not (source / seal_name).is_file():
            continue
        seal = json.loads((source / seal_name).read_text())
        hashes = seal if key is None else seal[key]
        for name, expected in hashes.items():
            path = (source / name).resolve()
            if not path.is_relative_to(source) or digest(path) != expected:
                raise ValueError(f"Sealed source hash mismatch: {seal_name}: {name}")
        seal_checks[seal_name] = {"sha256": digest(source / seal_name), "verified_files": len(hashes)}
    if (source / "READY.json").is_file():
        ready = json.loads((source / "READY.json").read_text())
        if ready.get("integration_manifest_sha256") != digest(source / "integration_manifest.json"):
            raise ValueError("READY does not match the integration seal")
    files = {str(path.relative_to(source)): digest(path) for path in source.rglob("*") if path.is_file()}
    shutil.copytree(source, target)
    for name, expected in files.items():
        if digest(source / name) != expected or digest(target / name) != expected:
            raise ValueError(f"Sealed source changed during copy: {name}")
    record = {
        "schema_version": 1,
        "sealed_source": str(source),
        "files_sha256": files,
        "copied_files": len(files),
        "new_physics_run": False,
        "source_mutated": False,
        "verified_seals": seal_checks,
    }
    write_json(target / "normal_manifest.json", record)
    return record


def verify_normal(directory: Path) -> dict:
    directory = Path(directory)
    record = json.loads((directory / "normal_manifest.json").read_text())
    for name, expected in record["files_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Staged sealed normal input changed: {name}")
    return record


def package_replay(plan_directory: Path, target: Path) -> dict:
    """Create object/{normal,strain,replay} evidence without touching sources.

    Root normal files retain their original names and bytes. replay/scene.npz
    is explicitly a schedule, not geometry or a claim of one combined USD.
    """
    plan_directory, target = Path(plan_directory).resolve(), Path(target).resolve()
    if target.exists():
        raise ValueError("Publication package target already exists")
    entry = publication_entry(plan_directory)
    plan = json.loads((plan_directory / "video-plan.json").read_text())
    normal, strain = Path(plan["normal_video"]).parent, Path(plan["strain_video"]).parent
    for path, expected in plan["inputs_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Video plan source changed: {path}")
    target.mkdir(parents=True)
    copied = []
    source_renders = []
    for phase, source, destination, prefix in (
        ("NORMAL", normal, target, ".."),
        ("STRAIN", strain, target / "strain", "../strain"),
    ):
        destination.mkdir(exist_ok=True)
        audit = json.loads((source / "rtx.json").read_text())
        names = set(audit["inputs_sha256"]) | {"rtx.mp4", "rtx.json", "scene.usdc"}
        names |= {
            name
            for name in ("strain.npz", "normal_manifest.json", "manifest.json", "normal_scene.json", "poster.jpg")
            if (source / name).is_file()
        }
        if phase == "NORMAL" and (source / "normal_manifest.json").exists():
            names.update(verify_normal(source)["files_sha256"])
        if phase == "STRAIN" and (source / "manifest.json").exists():
            names.update(json.loads((source / "manifest.json").read_text())["outputs_sha256"])
        # Include state/basis evidence consumed by the adapter, preserving names.
        for input_path in plan["inputs_sha256"]:
            path = Path(input_path)
            if path.is_relative_to(source) and len(path.relative_to(source).parts) == 1:
                names.add(path.name)
        if phase == "STRAIN" and (source / "source").is_dir():
            names.update(str(path.relative_to(source)) for path in (source / "source").rglob("*") if path.is_file())
        for name in sorted(names):
            dest = destination / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, dest)
            copied.append((str(dest.relative_to(target)), digest(dest)))
        if digest(destination / "scene.usdc") != audit["usd_sha256"]:
            raise ValueError("Source USD hash differs from its original render audit")
        source_renders.append(
            {
                "phase": phase,
                "audit": f"{prefix}/rtx.json",
                "usd": f"{prefix}/scene.usdc",
                "video": f"{prefix}/rtx.mp4",
                "scene": f"{prefix}/scene.json",
                "arrays": f"{prefix}/scene.npz",
                "frames": audit["frames"],
                "audit_sha256": digest(destination / "rtx.json"),
                "usd_sha256": audit["usd_sha256"],
                "video_sha256": audit["video_sha256"],
                "max_renderer_vertex_error_m": audit["max_renderer_vertex_error_m"],
                "max_renderer_color_error": audit.get("max_renderer_color_error"),
            }
        )
    replay = target / "replay"
    replay.mkdir()
    for name in ("rtx.mp4", "video-plan.json", "concat-audit.json", "decoded-frames.json"):
        shutil.copy2(plan_directory / name, replay / name)
    count, fps = plan["schedule"]["frames_per_phase"], plan["schedule"]["fps"]
    with np.load(normal / "scene.npz", allow_pickle=False) as arrays:
        times = arrays["times"].copy()
    np.savez_compressed(
        replay / "scene.npz",
        times=np.arange(2 * count, dtype=float) / fps,
        source_times=np.tile(times, 2),
        source_frame=np.tile(np.arange(count, dtype=np.int32), 2),
        phase=np.repeat(np.array([0, 1], dtype=np.int8), count),
    )
    write_json(
        replay / "scene.json",
        {
            "schema_version": 1,
            "kind": "normal_then_strain_assembly",
            "slug": plan["object"],
            "frames": 2 * count,
            "fps": fps,
            "camera": plan["camera"],
            "source_renders": source_renders,
            "strain_scale": plan["scale"],
            "schedule": plan["schedule"],
            "note": "Schedule only; geometry and colors are in the two referenced source scenes. No combined USD render.",
        },
    )
    normal_metrics = json.loads((normal / "metrics.json").read_text())
    metrics = {
        "assembly_verified": True,
        "resimulated": False,
        "native_source_test_passed": normal_metrics.get("test_passed"),
        "native_source_negative_control_passed": normal_metrics.get("negative_control_passed"),
        "performance_provenance": entry["performance_provenance"],
    }
    if plan["object"] == "cup-lid" and normal_metrics.get("principal_strain_screen_passed") is False:
        if plan["scale"]["max"] < 0.5:
            raise ValueError("High-strain cup requires an unclipped complete-skin scale of at least 50 percent")
        metrics["warning"] = {
            "severity": "warning",
            "material_validation_passed": False,
            "label": "High-strain illustrative ROM; independent 15% principal-strain screen fails",
            "detail": "Original contact/recovery/edge gates pass. Midsurface strain is about 18.13%; complete-skin strain about 47.95%. No calibrated stress/material validation. Entire skin remains visible on a fixed >=50% scale.",
        }
        entry["warning"] = metrics["warning"]
    write_json(replay / "metrics.json", metrics)
    inputs = {
        name: digest(replay / name)
        for name in (
            "scene.json",
            "scene.npz",
            "metrics.json",
            "concat-audit.json",
            "video-plan.json",
            "decoded-frames.json",
        )
    }
    inputs.update({"../" + name: value for name, value in copied})
    audit = {
        "schema_version": 1,
        "artifact_type": "normal_then_strain_concat",
        "renderer": "ffmpeg",
        "frames": 2 * count,
        "fps": fps,
        "video_sha256": digest(replay / "rtx.mp4"),
        "inputs_sha256": inputs,
        "source_renders": source_renders,
        "assembly_audit": "concat-audit.json",
        "assembly_audit_sha256": digest(replay / "concat-audit.json"),
        "max_renderer_vertex_error_m": max(item["max_renderer_vertex_error_m"] for item in source_renders),
        "vertex_error_scope": "Maximum across the two source render audits; no combined USD render",
        "normal_decoded_frames_identical": count,
        "strain_decoded_frames_identical": count,
        "native_trajectory_reused": True,
        "new_physics_run": False,
    }
    write_json(replay / "rtx.json", audit)
    entry["staged_video"] = str(replay / "rtx.mp4")
    entry["staged_object_directory"] = str(target)
    entry["primary_audit"] = f"{plan['object']}/replay/rtx.json"
    entry["staged_strain_directory"] = str(target / "strain")
    write_json(replay / "publication.json", entry)
    verify_package(target)
    return entry


def verify_package(directory: Path, full_decode: bool = False) -> dict:
    """Validate the portable assembly schema without original absolute paths."""
    directory = Path(directory).resolve()
    replay = directory / "replay"
    audit = json.loads((replay / "rtx.json").read_text())
    if audit.get("artifact_type") != "normal_then_strain_concat":
        raise ValueError("Not an independently audited normal-then-strain assembly")
    required = {
        "scene.json",
        "scene.npz",
        "metrics.json",
        "concat-audit.json",
        "video-plan.json",
        "decoded-frames.json",
        "../scene.json",
        "../scene.npz",
        "../metrics.json",
        "../rtx.json",
        "../rtx.mp4",
        "../scene.usdc",
        "../strain/scene.json",
        "../strain/scene.npz",
        "../strain/metrics.json",
        "../strain/rtx.json",
        "../strain/rtx.mp4",
        "../strain/scene.usdc",
        "../strain/strain.npz",
    }
    if not required <= set(audit["inputs_sha256"]):
        raise ValueError("Replay audit omits required evidence hashes")
    for name, expected in audit["inputs_sha256"].items():
        path = (replay / name).resolve()
        if not path.is_relative_to(directory) or digest(path) != expected:
            raise ValueError(f"Replay package input mismatch: {name}")
    if digest(replay / "rtx.mp4") != audit["video_sha256"]:
        raise ValueError("Replay video mismatch")
    assembly = json.loads((replay / audit["assembly_audit"]).read_text())
    if (
        audit["assembly_audit"] != "concat-audit.json"
        or digest(replay / "concat-audit.json") != audit["assembly_audit_sha256"]
    ):
        raise ValueError("Replay assembly audit reference mismatch")
    if not assembly["verified"] or digest(replay / "video-plan.json") != assembly["plan_sha256"]:
        raise ValueError("Replay assembly proof mismatch")
    if assembly["video_sha256"] != audit["video_sha256"] or assembly["probe"]["frames"] != audit["frames"]:
        raise ValueError("Replay assembly does not bind the final video")
    normal, strain = load_arrays(directory / "scene.npz"), load_arrays(directory / "strain/scene.npz")
    for name, value in normal.items():
        np.testing.assert_array_equal(value, strain[name])
    nmeta = json.loads((directory / "scene.json").read_text())
    smeta = json.loads((directory / "strain/scene.json").read_text())
    meta = json.loads((replay / "scene.json").read_text())
    if nmeta["camera"] != smeta["camera"] or meta["camera"] != nmeta["camera"]:
        raise ValueError("Replay source camera mismatch")
    count, fps = len(normal["times"]), nmeta["fps"]
    if audit["frames"] != 2 * count or audit["fps"] != fps or smeta["fps"] != fps:
        raise ValueError("Replay source frame count/FPS mismatch")
    if (
        assembly["probe"]["fps"] != fps
        or not assembly["probe"]["uniform_pts"]
        or abs(assembly["probe"]["duration_s"] - 2 * count / fps) > 2e-6
    ):
        raise ValueError("Replay duration/FPS proof mismatch")
    for key in ("normal_decoded_frames_identical", "strain_decoded_frames_identical"):
        if audit[key] != count or assembly[key] != count:
            raise ValueError("Replay does not prove every decoded frame")
    mapping = load_arrays(replay / "scene.npz")
    np.testing.assert_array_equal(mapping["times"], np.arange(2 * count) / fps)
    np.testing.assert_array_equal(mapping["source_times"], np.tile(normal["times"], 2))
    np.testing.assert_array_equal(mapping["source_frame"], np.tile(np.arange(count), 2))
    np.testing.assert_array_equal(mapping["phase"], np.repeat([0, 1], count))
    if [source["phase"] for source in audit["source_renders"]] != ["NORMAL", "STRAIN"]:
        raise ValueError("Replay must reference normal and strain source audits in order")
    for source in audit["source_renders"]:
        prefix = ".." if source["phase"] == "NORMAL" else "../strain"
        if any(
            source[key] != f"{prefix}/{name}"
            for key, name in (("audit", "rtx.json"), ("video", "rtx.mp4"), ("usd", "scene.usdc"))
        ):
            raise ValueError("Source render paths do not match their phases")
        source_audit = json.loads((replay / source["audit"]).read_text())
        source_directory = (replay / source["audit"]).parent
        _render_audit(source_directory, source["phase"] == "STRAIN")
        if source_audit["frames"] != count or source_audit["fps"] != fps:
            raise ValueError("Original render audit sampling differs from replay")
        if digest(replay / source["audit"]) != source["audit_sha256"]:
            raise ValueError("Original render audit reference changed")
        if source["max_renderer_vertex_error_m"] != source_audit["max_renderer_vertex_error_m"]:
            raise ValueError("Source vertex error summary changed")
        for key, file in (("video_sha256", "video"), ("usd_sha256", "usd")):
            if digest(replay / source[file]) != source_audit[key]:
                raise ValueError("A source render no longer matches its original audit")
    expected_error = max(source["max_renderer_vertex_error_m"] for source in audit["source_renders"])
    if audit["max_renderer_vertex_error_m"] != expected_error:
        raise ValueError("Replay vertex error differs from the source audits")
    strain_directory = directory / "strain"
    if (strain_directory / "manifest.json").is_file():
        verify(strain_directory, check_sources=False)
    elif meta["slug"] == "headphones":
        # The accepted legacy view uses the same saved state in local coordinates.
        # Recompute it exactly; its world-coordinate export has float32 roundoff.
        refs, _, _ = adapt(directory, "headphones", nmeta, normal)
        reference = refs[0]
        raw = load_arrays(strain_directory / "strain.npz")
        scale = json.loads((strain_directory / "metrics.json").read_text())["strain_view"]["color_scale"]
        if scale != [0.0, 0.03] or meta["strain_scale"]["max"] != 0.03:
            raise ValueError("Legacy headphone scale changed")
        np.testing.assert_array_equal(reference.rest, raw["rest_vertices"])
        np.testing.assert_array_equal(reference.faces, raw["faces"])
        np.testing.assert_array_equal(normal["times"], raw["times"])
        selected = [item for item in smeta["meshes"] if "vertex_colors" in item]
        if len(selected) != 1 or selected[0]["name"] != reference.mesh:
            raise ValueError("Legacy headphone color assignment changed")
        item = selected[0]
        for frame, local in enumerate(reference.local):
            principal, vertices = field(reference.rest, reference.faces, local)
            np.testing.assert_array_equal(principal, raw["principal"][frame])
            np.testing.assert_array_equal(vertices, raw["vertex_magnitude"][frame])
            np.testing.assert_array_equal(jet(vertices / 0.03), strain[item["vertex_colors"]][frame])
            world = normal[item["vertices"]][frame]
            recovered = transform_points(local, reference.poses[frame])
            np.testing.assert_allclose(recovered, world, atol=2e-6, rtol=0)
    else:
        raise ValueError("Missing portable quantitative strain evidence")
    if full_decode:
        expected = json.loads((replay / "decoded-frames.json").read_text())
        actual = decoded_frame_hashes(replay / "rtx.mp4")
        if actual != expected["normal_rgb24_sha256"] + expected["strain_rgb24_sha256"]:
            raise ValueError("Replay decoded appearance mismatch")
    return {
        "verified": True,
        "portable": True,
        "frames": 2 * count,
        "full_decode": full_decode,
        "all_fields_recomputed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage-normal")
    stage.add_argument("source", type=Path)
    stage.add_argument("target", type=Path)
    package = sub.add_parser("package")
    package.add_argument("plan", type=Path)
    package.add_argument("target", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("target", type=Path)
    verify.add_argument("--full-decode", action="store_true")
    args = parser.parse_args()
    if args.command == "stage-normal":
        result = stage_normal(args.source, args.target)
    elif args.command == "package":
        result = package_replay(args.plan, args.target)
    else:
        result = verify_package(args.target, args.full_decode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
