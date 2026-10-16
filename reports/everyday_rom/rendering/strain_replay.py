# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Export exact NORMAL THEN STRAIN replays; no simulation or video generation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

if __package__:
    from .strain_adapters import ALIASES, OBJECTS, adapt, digest, load_arrays
    from .surface_strain import components, jet, rest_metric, stretches, transform_points, validate_mesh, vertex_average
else:
    from strain_adapters import ALIASES, OBJECTS, adapt, digest, load_arrays
    from surface_strain import components, jet, rest_metric, stretches, transform_points, validate_mesh, vertex_average

SCHEMA_VERSION = 1
DEFINITION = "sqrt(eigvalsh(F.T @ F)) - 1; F = current_edges @ inverse_rest_tangent_chart"
DISPLAY = "Rest-area weighted vertex average of max(abs(epsilon_1), abs(epsilon_2))"
LIMITATION = "Surface kinematics only; not displacement, bending curvature, Green strain, or calibrated 3D stress."


def write_json(path: Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def array_digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(json.dumps([array.dtype.str, array.shape]).encode("ascii"))
    hasher.update(memoryview(array).cast("B"))
    return hasher.hexdigest()


def validate_scene(meta: dict, arrays: dict) -> int:
    """Validate static or moving exports, including every color frame."""
    times = arrays["times"]
    fps = float(meta["fps"])
    if not np.isfinite(fps) or fps <= 0 or times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("Need positive fps and at least two finite simulation times")
    if not np.allclose(np.diff(times), 1 / fps, rtol=0.001, atol=1e-6):
        raise ValueError("Scene timestamps do not match fps")
    if not meta["meshes"]:
        raise ValueError("Scene has no meshes")
    names = [item["name"] for item in meta["meshes"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate mesh names")
    for item in meta["meshes"]:
        points, faces = arrays[item["vertices"]], arrays[item["faces"]]
        if points.ndim not in (2, 3) or not np.isfinite(points).all():
            raise ValueError(f"Invalid saved geometry: {item['name']}")
        validate_mesh(points[0] if points.ndim == 3 else points, faces)
        if points.ndim == 3 and len(points) != len(times):
            raise ValueError(f"Frame count mismatch: {item['name']}")
        if "vertex_colors" in item:
            colors = arrays[item["vertex_colors"]]
            if colors.shape != (len(times), points.shape[-2], 3):
                raise ValueError("Vertex colors must match every frame and vertex")
            if not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 1)):
                raise ValueError("Invalid RGB colors")
    camera = meta["camera"]
    for key in ("position", "target"):
        if np.shape(camera[key]) != (3,) or not np.isfinite(camera[key]).all():
            raise ValueError("Invalid camera")
    if not np.linalg.norm(np.asarray(camera["position"]) - camera["target"]) > 0:
        raise ValueError("Camera coincides with target")
    if not 0 < float(camera.get("fov", 38)) < 180:
        raise ValueError("Invalid camera field of view")
    if set(camera.get("track_meshes", [])) - set(names):
        raise ValueError("Camera tracks an unknown mesh")
    reference = camera.get("reference_frame", 0)
    if not isinstance(reference, int) or not 0 <= reference < len(times):
        raise ValueError("Camera reference frame outside sequence")
    return len(times)


def legend(path: Path, scale: float) -> None:
    """Write a quantitative fixed-scale legend, including clipping semantics."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (720, 142), (245, 247, 247))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    draw.text((16, 8), "Principal surface engineering strain magnitude", font=font, fill=(25, 30, 34))
    colors = (255 * jet(np.linspace(0, 1, 656))).astype(np.uint8)
    image.paste(Image.fromarray(np.repeat(colors[None], 28, axis=0)), (30, 38))
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        x = 30 + int(fraction * 656)
        label = f"{100 * scale * fraction:.3g}%"
        draw.text((x, 72), label, font=font, fill=(25, 30, 34), anchor="mt")
    draw.text(
        (16, 108),
        f"Fixed 0-{100 * scale:.4g}% | values above scale clipped in color only",
        font=font,
        fill=(25, 30, 34),
    )
    image.save(path)


def prepare(
    source: Path,
    output: Path,
    object_name: str,
    scale_max: float,
    reference_manifest: Path | None = None,
    combined: bool = False,
) -> dict:
    """Export a colored view and auditable exact replay of a saved trajectory.

    Args:
        source: Existing immutable scene bundle directory.
        output: New owned output directory; existing directories are rejected.
        object_name: One of the ten gallery objects or its documented alias.
        scale_max: Fixed dimensionless strain maximum for every colored part.
        reference_manifest: Optional worker-provided material-rest references.
        combined: Also export a complete normal-then-strain renderer scene.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("Output must be a fresh directory outside the source bundle")
    if not np.isfinite(scale_max) or scale_max <= 0:
        raise ValueError("A positive fixed dimensionless --scale-max is required")
    object_name = ALIASES.get(object_name, object_name)
    input_hashes = {str(source / name): digest(source / name) for name in ("scene.json", "scene.npz", "metrics.json")}
    normal = json.loads((source / "scene.json").read_text())
    metrics = json.loads((source / "metrics.json").read_text())
    arrays = load_arrays(source / "scene.npz")
    count = validate_scene(normal, arrays)
    original_hashes = {key: array_digest(value) for key, value in arrays.items()}
    references, uncolored, hashes = adapt(source, object_name, normal, arrays, reference_manifest)
    input_hashes.update(hashes)
    colored = copy.deepcopy(normal)
    by_name = {item["name"]: (i, item) for i, item in enumerate(colored["meshes"])}
    audit = {"times": arrays["times"].copy()}
    summaries = []
    for reference in references:
        i, item = by_name[reference.mesh]
        points, faces, rest = arrays[item["vertices"]], arrays[item["faces"]], np.asarray(reference.rest)
        if rest.shape != points.shape[-2:] or not np.array_equal(reference.faces, faces):
            raise ValueError(f"Material reference does not match scene topology: {reference.mesh}")
        inverse, area = rest_metric(rest, faces)
        vertex_average(np.zeros(len(faces)), faces, area, len(rest))
        mid_principal = None
        if reference.midsurface_rest is not None:
            mid_inverse, mid_area = rest_metric(reference.midsurface_rest, reference.midsurface_faces)
            mid_principal = np.empty((count, len(reference.midsurface_faces), 2), dtype=np.float64)
        if (reference.local is None) != (reference.poses is None):
            raise ValueError("State recovery requires local positions and body poses")
        if reference.local is not None:
            if reference.local.shape != (count, len(rest), 3) or reference.poses.shape != (count, 7):
                raise ValueError(f"Independent recovery frame or vertex mismatch: {reference.mesh}")
        principal = np.empty((count, len(faces), 2), dtype=np.float64)
        vertices = np.empty((count, len(rest)), dtype=np.float64)
        recovery_error, local_error = 0.0, 0.0
        for frame in range(count):
            world = points[frame] if points.ndim == 3 else points
            p = stretches(world, faces, inverse) - 1
            if reference.local is not None:
                local = reference.local[frame]
                recovered = transform_points(local, reference.poses[frame])
                error = float(np.max(np.abs(recovered - world)))
                recovery_error = max(recovery_error, error)
                if not np.isfinite(error) or error > 2e-6:
                    raise ValueError(f"Saved-state recovery mismatch: {reference.mesh}, frame {frame}, {error:g} m")
                local_error = max(local_error, float(np.max(np.abs(p - (stretches(local, faces, inverse) - 1)))))
            if mid_principal is not None:
                # This translated top layer has the same metric as the recovered
                # midsurface because the exporter shares its displacement field.
                mid_principal[frame] = (
                    stretches(world[reference.midsurface_vertex_ids], reference.midsurface_faces, mid_inverse) - 1
                )
            principal[frame] = p
            vertices[frame] = vertex_average(np.max(np.abs(p), axis=1), faces, area, len(rest))
        if not np.isfinite(principal).all() or not np.isfinite(vertices).all():
            raise ValueError("Nonfinite computed strain")
        key = f"strain_rgb_{i}"
        if key in arrays:
            raise ValueError(f"Reserved output array already exists: {key}")
        arrays[key] = jet(vertices / scale_max)
        item["vertex_colors"] = key
        vertex_component, face_component = components(faces, len(rest))
        magnitude = np.max(np.abs(principal), axis=2)
        clipped = vertices > scale_max
        for suffix, value in {
            "rest": rest,
            "faces": faces,
            "rest_area": area,
            "principal": principal,
            "triangle_magnitude": magnitude,
            "vertex_magnitude": vertices,
            "vertex_component": vertex_component,
            "face_component": face_component,
            "clipped": clipped,
        }.items():
            audit[f"mesh_{i}_{suffix}"] = value
        summaries.append(
            {
                "mesh": reference.mesh,
                "mesh_index": i,
                "color_array": key,
                "audit_prefix": f"mesh_{i}",
                "reference": reference.provenance,
                "reference_vertices_sha256": array_digest(rest),
                "reference_faces_sha256": array_digest(faces),
                "vertices": len(rest),
                "triangles": len(faces),
                "components": int(vertex_component.max()) + 1,
                "component_vertex_counts": np.bincount(vertex_component).tolist(),
                "component_triangle_counts": np.bincount(face_component).tolist(),
                "all_triangles_included": True,
                "all_vertices_referenced": True,
                "independent_state_recovery_checked": reference.local is not None,
                "max_recovery_error_m": recovery_error if reference.local is not None else None,
                "max_world_local_strain_error": local_error if reference.local is not None else None,
                "principal_min": float(principal.min()),
                "principal_max": float(principal.max()),
                "max_triangle_magnitude": float(magnitude.max()),
                "max_vertex_magnitude": float(vertices.max()),
                "peak_frame": int(np.unravel_index(magnitude.argmax(), magnitude.shape)[0]),
                "clipped_vertex_samples": int(clipped.sum()),
                "clipped_vertex_fraction": float(clipped.mean()),
                "max_clipped_triangle_area_fraction": float(((magnitude > scale_max) @ area / area.sum()).max()),
                "collapsed_triangle_samples": int(np.count_nonzero(principal[:, :, 0] <= -1 + 1e-10)),
            }
        )
        if mid_principal is not None:
            for suffix, value in {
                "rest": reference.midsurface_rest,
                "faces": reference.midsurface_faces,
                "vertex_ids": reference.midsurface_vertex_ids,
                "principal": mid_principal,
                "rest_area": mid_area,
            }.items():
                audit[f"mesh_{i}_midsurface_{suffix}"] = value
            summaries[-1]["midsurface"] = {
                "max_triangle_magnitude": float(np.abs(mid_principal).max()),
                "principal_min": float(mid_principal.min()),
                "principal_max": float(mid_principal.max()),
                "triangles": len(reference.midsurface_faces),
                "note": "Complete skin is colored. Midsurface is reported separately; top/bottom share a linear field, and side-wall shear is not calibrated shell strain.",
            }
    # Check again after reading/processing: workers must not mutate inputs mid-export.
    for path, expected in input_hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Source changed during strain export: {path}")
    scale = {
        "min": 0.0,
        "max": float(scale_max),
        "units": "dimensionless",
        "colormap": "classicaljet",
        "label": f"Principal surface engineering strain magnitude, 0-{100 * scale_max:.4g}%",
        "normalization": "Fixed per object for every mesh and frame",
        "clipping": "Display only; raw values retained",
    }
    schedule = {
        "frames_per_phase": count,
        "total_frames": 2 * count,
        "fps": normal["fps"],
        "segments": [
            {"phase": "NORMAL", "start": 0, "stop": count},
            {"phase": "STRAIN", "start": count, "stop": 2 * count},
        ],
        "frame_mapping": "source_frame = playback_frame % frames_per_phase",
        "camera_identical": True,
        "geometry_identical": True,
        "resimulated": False,
    }
    colored.update(
        color_legend="strain-legend.png",
        strain_view={"scale": scale, "definition": DEFINITION, "display": DISPLAY},
        replay={"phase": "STRAIN", "frames_per_phase": count},
    )
    validate_scene(colored, arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "object": object_name,
        "source": str(source),
        "implementation_base_commit": "8228bf6f690fff4dbb359d5de6447b09cb77c1e7",
        "definition": DEFINITION,
        "display": DISPLAY,
        "limitation": LIMITATION,
        "scale": scale,
        "schedule": schedule,
        "camera": normal["camera"],
        "original_arrays_sha256": original_hashes,
        "inputs_sha256": input_hashes,
        "surfaces": summaries,
        "uncolored": uncolored,
        "physics_validation": {key: metrics.get(key) for key in ("test_passed", "negative_control_passed")},
        "rendered": False,
        "video_status": "Not rendered; this export contains saved-surface strain evidence only",
        "implementation_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("surface_strain.py", "strain_adapters.py", "strain_replay.py")
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "scene.npz", **arrays)
    np.savez_compressed(output / "strain.npz", **audit)
    write_json(output / "scene.json", colored)
    write_json(output / "normal_scene.json", normal)
    write_json(output / "metrics.json", metrics)
    legend(output / "strain-legend.png", scale_max)
    (output / "source").mkdir()
    for name in manifest["implementation_sha256"]:
        shutil.copy2(Path(__file__).with_name(name), output / "source" / name)
    manifest["outputs_sha256"] = {
        name: digest(output / name)
        for name in ("scene.json", "scene.npz", "strain.npz", "normal_scene.json", "metrics.json", "strain-legend.png")
    }
    write_json(output / "manifest.json", manifest)
    if combined:
        export_combined(output)
    return manifest


def verify(output: Path, check_sources: bool = True) -> dict:
    """Recompute every field and check all geometry, color, and input hashes."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported strain replay manifest")
    for name, expected in manifest["outputs_sha256"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Output hash mismatch: {name}")
    if check_sources:
        for path, expected in manifest["inputs_sha256"].items():
            if digest(Path(path)) != expected:
                raise ValueError(f"Source hash mismatch: {path}")
    if (output / "source").exists():
        for name, expected in manifest["implementation_sha256"].items():
            if digest(output / "source" / name) != expected:
                raise ValueError(f"Implementation snapshot hash mismatch: {name}")
    arrays, audit = load_arrays(output / "scene.npz"), load_arrays(output / "strain.npz")
    meta = json.loads((output / "scene.json").read_text())
    normal = json.loads((output / "normal_scene.json").read_text())
    count = validate_scene(meta, arrays)
    if count != manifest["schedule"]["frames_per_phase"] or meta["camera"] != normal["camera"]:
        raise ValueError("Replay camera/frame count changed")
    for key, expected in manifest["original_arrays_sha256"].items():
        if array_digest(arrays[key]) != expected:
            raise ValueError(f"Original scene array changed: {key}")
    if not np.array_equal(arrays["times"], audit["times"]):
        raise ValueError("Strain sample times changed")
    changed = {surface["mesh_index"]: surface for surface in manifest["surfaces"]}
    if len(meta["meshes"]) != len(normal["meshes"]):
        raise ValueError("Mesh count changed")
    for i, item in enumerate(meta["meshes"]):
        expected = copy.deepcopy(normal["meshes"][i])
        if i not in changed:
            if item != expected:
                raise ValueError("Uncolored material changed")
            continue
        summary = changed[i]
        expected["vertex_colors"] = summary["color_array"]
        if item != expected or item["name"] != summary["mesh"]:
            raise ValueError("Incorrect mesh color assignment")
        prefix = summary["audit_prefix"]
        rest, faces = audit[prefix + "_rest"], audit[prefix + "_faces"]
        if (
            array_digest(rest) != summary["reference_vertices_sha256"]
            or array_digest(faces) != summary["reference_faces_sha256"]
        ):
            raise ValueError("Material reference changed")
        np.testing.assert_array_equal(faces, arrays[item["faces"]])
        inverse, area = rest_metric(rest, faces)
        np.testing.assert_array_equal(area, audit[prefix + "_rest_area"])
        vc, fc = components(faces, len(rest))
        np.testing.assert_array_equal(vc, audit[prefix + "_vertex_component"])
        np.testing.assert_array_equal(fc, audit[prefix + "_face_component"])
        points = arrays[item["vertices"]]
        mid_prefix = prefix + "_midsurface"
        if "midsurface" in summary:
            mid_inverse, mid_area = rest_metric(audit[mid_prefix + "_rest"], audit[mid_prefix + "_faces"])
            np.testing.assert_array_equal(mid_area, audit[mid_prefix + "_rest_area"])
        for frame in range(count):
            principal = stretches(points[frame] if points.ndim == 3 else points, faces, inverse) - 1
            magnitude = np.max(np.abs(principal), axis=1)
            vertices = vertex_average(magnitude, faces, area, len(rest))
            np.testing.assert_array_equal(principal, audit[prefix + "_principal"][frame])
            np.testing.assert_array_equal(magnitude, audit[prefix + "_triangle_magnitude"][frame])
            np.testing.assert_array_equal(vertices, audit[prefix + "_vertex_magnitude"][frame])
            np.testing.assert_array_equal(vertices > manifest["scale"]["max"], audit[prefix + "_clipped"][frame])
            np.testing.assert_array_equal(
                jet(vertices / manifest["scale"]["max"]), arrays[item["vertex_colors"]][frame]
            )
            if "midsurface" in summary:
                world = points[frame] if points.ndim == 3 else points
                mid_principal = (
                    stretches(world[audit[mid_prefix + "_vertex_ids"]], audit[mid_prefix + "_faces"], mid_inverse) - 1
                )
                np.testing.assert_array_equal(mid_principal, audit[mid_prefix + "_principal"][frame])
    return {
        "verified": True,
        "frames": count,
        "colored_meshes": len(changed),
        "all_fields_recomputed": True,
        "source_hashes_checked": check_sources,
    }


def export_combined(output: Path, *, check_sources: bool = True) -> Path:
    """Write a renderer scene with identical normal and strain geometry halves."""
    output = Path(output)
    verify(output, check_sources=check_sources)
    target = output / "replay"
    if target.exists():
        raise ValueError("Replay output already exists")
    manifest = json.loads((output / "manifest.json").read_text())
    meta = json.loads((output / "normal_scene.json").read_text())
    strain = load_arrays(output / "scene.npz")
    count = len(strain["times"])
    arrays = {
        "times": np.arange(2 * count, dtype=np.float64) / meta["fps"],
        "source_times": np.tile(strain["times"], 2),
        "source_frame": np.tile(np.arange(count, dtype=np.int32), 2),
        "phase": np.repeat(np.array([0, 1], dtype=np.int8), count),
    }
    colored = {item["mesh_index"]: item for item in manifest["surfaces"]}
    for i, item in enumerate(meta["meshes"]):
        points, faces = strain[item["vertices"]], strain[item["faces"]]
        arrays[item["vertices"]] = np.concatenate((points, points)) if points.ndim == 3 else points
        arrays[item["faces"]] = faces
        if i in colored:
            normal = (
                strain[item["vertex_colors"]]
                if "vertex_colors" in item
                else np.broadcast_to(
                    np.asarray(item.get("color", [0.6, 0.6, 0.6]), dtype=np.float32), (count, points.shape[-2], 3)
                )
            )
            key = colored[i]["color_array"]
            arrays[key] = np.concatenate((normal, strain[key]))
            item["vertex_colors"] = key
        elif "vertex_colors" in item:
            values = strain[item["vertex_colors"]]
            arrays[item["vertex_colors"]] = np.concatenate((values, values))
    meta.update(
        color_legend="strain-legend.png",
        strain_view={"scale": manifest["scale"], "definition": DEFINITION, "display": DISPLAY},
        replay={"phase": "NORMAL_THEN_STRAIN", "frames_per_phase": count, "source_times": "source_times"},
    )
    validate_scene(meta, arrays)
    target.mkdir()
    np.savez_compressed(target / "scene.npz", **arrays)
    write_json(target / "scene.json", meta)
    shutil.copy2(output / "metrics.json", target / "metrics.json")
    shutil.copy2(output / "strain-legend.png", target / "strain-legend.png")
    replay = {
        "schema_version": 1,
        "parent_manifest_sha256": digest(output / "manifest.json"),
        "schedule": manifest["schedule"],
        "inputs_sha256": {
            name: digest(target / name) for name in ("scene.npz", "scene.json", "metrics.json", "strain-legend.png")
        },
    }
    write_json(target / "replay_manifest.json", replay)
    verify_combined(output)
    return target


def verify_combined(output: Path) -> dict:
    """Check both replay halves, including source time, camera, and materials."""
    output = Path(output)
    target = output / "replay"
    replay = json.loads((target / "replay_manifest.json").read_text())
    if replay["parent_manifest_sha256"] != digest(output / "manifest.json"):
        raise ValueError("Stale parent replay manifest")
    for name, expected in replay["inputs_sha256"].items():
        if digest(target / name) != expected:
            raise ValueError(f"Combined scene changed: {name}")
    source = load_arrays(output / "scene.npz")
    arrays = load_arrays(target / "scene.npz")
    meta = json.loads((target / "scene.json").read_text())
    normal = json.loads((output / "normal_scene.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    count = len(source["times"])
    if validate_scene(meta, arrays) != 2 * count or meta["camera"] != normal["camera"]:
        raise ValueError("Combined scene camera or frame count mismatch")
    np.testing.assert_array_equal(arrays["source_times"], np.tile(source["times"], 2))
    np.testing.assert_array_equal(arrays["source_frame"], np.tile(np.arange(count), 2))
    np.testing.assert_array_equal(arrays["phase"], np.repeat([0, 1], count))
    colored = {item["mesh_index"]: item for item in manifest["surfaces"]}
    for i, item in enumerate(meta["meshes"]):
        points = source[item["vertices"]]
        expected = np.concatenate((points, points)) if points.ndim == 3 else points
        np.testing.assert_array_equal(arrays[item["vertices"]], expected)
        np.testing.assert_array_equal(arrays[item["faces"]], source[item["faces"]])
        material = copy.deepcopy(normal["meshes"][i])
        if i in colored:
            key = colored[i]["color_array"]
            expected_normal = (
                source[material["vertex_colors"]]
                if "vertex_colors" in material
                else np.broadcast_to(
                    np.asarray(material.get("color", [0.6, 0.6, 0.6]), dtype=np.float32), (count, points.shape[-2], 3)
                )
            )
            np.testing.assert_array_equal(arrays[key][:count], expected_normal)
            np.testing.assert_array_equal(arrays[key][count:], source[key])
            material["vertex_colors"] = key
        elif "vertex_colors" in item:
            expected_colors = source[item["vertex_colors"]]
            np.testing.assert_array_equal(
                arrays[item["vertex_colors"]], np.concatenate((expected_colors, expected_colors))
            )
        if item != material:
            raise ValueError("Replay changed material settings")
    return {"verified": True, "normal_then_strain": True, "frames": 2 * count, "identical_geometry_halves": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("prepare")
    export.add_argument("source", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--object", required=True, choices=(*OBJECTS, *ALIASES))
    export.add_argument("--scale-max", type=float, required=True, help="Fixed dimensionless strain (0.03 means 3%%)")
    export.add_argument("--references", type=Path)
    export.add_argument("--combined", action="store_true")
    check = sub.add_parser("verify")
    check.add_argument("output", type=Path)
    check.add_argument("--offline-bundle", action="store_true", help="Verify output without reopening original inputs")
    combined = sub.add_parser("combine-scene")
    combined.add_argument("output", type=Path)
    combined.add_argument(
        "--offline-bundle", action="store_true", help="Verify output without reopening original inputs"
    )
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.output, args.object, args.scale_max, args.references, args.combined)
        print(json.dumps({key: result[key] for key in ("object", "scale", "schedule", "surfaces")}, indent=2))
    elif args.command == "verify":
        result = verify(args.output, not args.offline_bundle)
        if (args.output / "replay").exists():
            result["combined"] = verify_combined(args.output)
        print(json.dumps(result, indent=2))
    else:
        print(export_combined(args.output, check_sources=not args.offline_bundle))


if __name__ == "__main__":
    main()
