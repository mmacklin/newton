# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render prepared normal-then-strain scenes with exact USD color readback.

Derived from the accepted headphone strain renderer. Rendering is an explicit
separate command; CPU preparation never invokes this module's main function.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

if __package__:
    from .strain_publication import verify_normal
    from .strain_replay import verify, verify_combined
else:
    from strain_publication import verify_normal
    from strain_replay import verify, verify_combined

os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")
PRODUCT = "/Render/Camera"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_scene(meta, arrays):
    """Reject invalid meshes or inconsistent time sampling before invoking RTX."""
    times = arrays["times"]
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("Need at least two finite simulation times")
    np.testing.assert_allclose(np.diff(times), 1 / meta["fps"], rtol=0.001, atol=1e-6)
    if not meta["meshes"]:
        raise ValueError("Scene has no meshes")
    moving = False
    for item in meta["meshes"]:
        points, faces = arrays[item["vertices"]], arrays[item["faces"]]
        if points.ndim not in (2, 3) or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"Invalid points: {item['name']}")
        if points.ndim == 3:
            if len(points) != len(times):
                raise ValueError(f"Frame count mismatch: {item['name']}")
            moving |= float(np.max(np.ptp(points, axis=0))) > 1e-5
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
            raise ValueError(f"Invalid triangles: {item['name']}")
        if faces.min() < 0 or faces.max() >= points.shape[-2]:
            raise ValueError(f"Invalid vertex index: {item['name']}")
        if "vertex_colors" in item:
            colors = arrays[item["vertex_colors"]]
            if colors.shape != (len(times), points.shape[-2], 3):
                raise ValueError("Vertex colors must match every saved frame and vertex")
            if not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 1)):
                raise ValueError("Invalid vertex colors")
    if not moving:
        raise ValueError("All saved geometry is stationary")
    return len(times)


def normals(points, faces):
    tri = points[faces]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if len(points) == 8 and len(faces) == 12:
        unit = face / np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-15)
        return np.repeat(unit, 3, axis=0).astype(np.float32)
    result = np.zeros_like(points)
    for corner in range(3):
        np.add.at(result, faces[:, corner], face)
    return (result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-15)).astype(np.float32)


def look_at(position, target):
    from pxr import Gf

    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()


def camera_path(meta, arrays):
    """Translate a detail camera with saved mesh centroids, leaving geometry unchanged."""
    camera = meta["camera"]
    count = len(arrays["times"])
    positions = np.broadcast_to(np.asarray(camera["position"], dtype=float), (count, 3)).copy()
    targets = np.broadcast_to(np.asarray(camera["target"], dtype=float), (count, 3)).copy()
    names = camera.get("track_meshes", [])
    if names:
        meshes = {item["name"]: item for item in meta["meshes"]}
        centers = []
        for name in names:
            if name not in meshes:
                raise ValueError(f"Unknown camera tracking mesh: {name}")
            points = arrays[meshes[name]["vertices"]]
            center = points.mean(axis=-2)
            centers.append(np.broadcast_to(center, (count, 3)))
        center = np.mean(centers, axis=0)
        reference = camera.get("reference_frame", 0)
        if not isinstance(reference, int) or not 0 <= reference < count:
            raise ValueError("Camera reference frame outside saved trajectory")
        offset = center - center[reference]
        positions += offset
        targets += offset
    return positions, targets


def material(stage, name, item):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, f"/World/Materials/{name}")
    shader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse.Set(Gf.Vec3f(*item.get("color", [0.6, 0.6, 0.6])))
    if "vertex_colors" in item:
        reader = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/ColorReader")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        diffuse.ConnectToSource(reader.ConnectableAPI(), "result")
    for key, default in (("roughness", 0.4), ("metallic", 0.0), ("opacity", 1.0)):
        shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(float(item.get(key, default)))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def mesh(stage, path, points, faces, mat):
    from pxr import UsdGeom, UsdShade, Vt

    obj = UsdGeom.Mesh.Define(stage, path)
    obj.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    obj.GetFaceVertexCountsAttr().Set([3] * len(faces))
    obj.GetFaceVertexIndicesAttr().Set(faces.ravel().tolist())
    obj.GetSubdivisionSchemeAttr().Set("none")
    obj.GetDoubleSidedAttr().Set(True)
    obj.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals(points, faces)))
    obj.SetNormalsInterpolation("faceVarying" if len(points) == 8 and len(faces) == 12 else "vertex")
    UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)
    return obj


def create_stage(args, meta, arrays):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdRender, Vt

    count = validate_scene(meta, arrays)
    scene_path = args.run / ("scene_preview.usdc" if args.frame is not None else "scene.usdc")
    stage = Usd.Stage.CreateNew(str(scene_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetTimeCodesPerSecond(meta["fps"])
    stage.SetFramesPerSecond(meta["fps"])
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(count - 1)
    geometry = {}
    for i, item in enumerate(meta["meshes"]):
        points = arrays[item["vertices"]]
        faces = arrays[item["faces"]]
        path = f"/World/Object_{i}"
        mat = material(stage, f"Material_{i}", item)
        obj = mesh(stage, path, points[0] if points.ndim == 3 else points, faces, mat)
        if "vertex_colors" in item:
            colors = arrays[item["vertex_colors"]]
            color_attr = obj.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
            color_attr.Set(Vt.Vec3fArray.FromNumpy(colors[0]))
            for frame in [args.frame] if args.frame is not None else range(count):
                color_attr.Set(Vt.Vec3fArray.FromNumpy(colors[frame]), frame)
        if points.ndim == 3:
            indices = [args.frame] if args.frame is not None else range(count)
            for frame in indices:
                p = points[frame]
                obj.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)), frame)
                obj.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals(p, faces)), frame)
        geometry[path] = points

    camera = meta["camera"]
    target = np.array(camera["target"], dtype=float)
    distance = float(np.linalg.norm(np.array(camera["position"]) - target))
    scale = max(distance / 0.5, 0.1)
    floor_z = float(meta.get("floor_z", 0)) - 0.001 * scale
    away = target - np.array(camera["position"], dtype=float)
    away[2] = 0
    away /= np.linalg.norm(away)
    right = np.array([away[1], -away[0], 0])
    profile = [(-20 * scale, floor_z), (3 * scale, floor_z)]
    for angle in np.linspace(0, math.pi / 2, 49)[1:]:
        profile.append((scale * (3 + 2 * math.sin(angle)), floor_z + 2 * scale * (1 - math.cos(angle))))
    profile.append((5 * scale, floor_z + 20 * scale))
    floor = np.array([target + u * right + v * away for v, z in profile for u in (-20 * scale, 20 * scale)])
    floor[:, 2] = np.repeat([z for v, z in profile], 2)
    floor_faces = np.array(
        [tri for i in range(len(profile) - 1) for tri in ((2 * i, 2 * i + 1, 2 * i + 3), (2 * i, 2 * i + 3, 2 * i + 2))]
    )
    ground = material(stage, "Ground", {"color": [0.63, 0.66, 0.68], "roughness": 0.8})
    mesh(stage, "/World/Ground", floor, floor_faces, ground)
    dome = UsdLux.DomeLight.Define(stage, "/World/Ambient")
    dome.GetIntensityAttr().Set(220)
    dome.GetColorAttr().Set(Gf.Vec3f(0.91, 0.95, 1.0))
    for name, offset, intensity, size, color in (
        ("Key", (0.12, -0.30, 0.50), 3000, (0.36, 0.44), (1.0, 0.96, 0.91)),
        ("Fill", (-0.4, -0.1, 0.25), 1100, (0.3, 0.4), (0.86, 0.94, 1.0)),
        ("Rim", (0.18, 0.30, 0.36), 3500, (0.24, 0.35), (1.0, 1.0, 1.0)),
    ):
        light = UsdLux.RectLight.Define(stage, f"/World/{name}")
        light.GetIntensityAttr().Set(intensity)
        light.GetColorAttr().Set(Gf.Vec3f(*color))
        light.GetWidthAttr().Set(size[0] * scale)
        light.GetHeightAttr().Set(size[1] * scale)
        light.AddTransformOp().Set(look_at(target + scale * np.array(offset), target))
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.GetVerticalApertureAttr().Set(20.955)
    cam.GetHorizontalApertureAttr().Set(20.955 * args.width / args.height)
    cam.GetFocalLengthAttr().Set(20.955 / (2 * math.tan(math.radians(camera.get("fov", 38)) / 2)))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.0001, max(10, 10 * scale)))
    camera_op = cam.AddTransformOp()
    camera_op.Set(look_at(camera["position"], camera["target"]))
    if camera.get("track_meshes"):
        positions, targets = camera_path(meta, arrays)
        for frame in [args.frame] if args.frame is not None else range(count):
            camera_op.Set(look_at(positions[frame], targets[frame]), frame)
    product = UsdRender.Product.Define(stage, PRODUCT)
    product.GetCameraRel().SetTargets([cam.GetPath()])
    product.GetResolutionAttr().Set(Gf.Vec2i(args.width, args.height))
    var = UsdRender.Var.Define(stage, "/Render/LdrColor")
    var.GetSourceNameAttr().Set("LdrColor")
    product.GetOrderedVarsRel().SetTargets([var.GetPath()])
    for name, dtype, value in (
        ("rendermode", Sdf.ValueTypeNames.Token, "PathTracing"),
        ("pt:samplesPerPixel", Sdf.ValueTypeNames.Int, args.samples),
        ("pt:denoising:enabled", Sdf.ValueTypeNames.Bool, True),
        ("dlss:frameGeneration", Sdf.ValueTypeNames.Bool, False),
        ("rt:fractionalOpacity", Sdf.ValueTypeNames.Bool, True),
    ):
        product.GetPrim().CreateAttribute(f"omni:rtx:{name}", dtype).Set(value)
    stage.GetRootLayer().Save()
    return geometry, count


def legend_image(run, meta, width, height):
    """Return the compact quantitative legend; total overlay height is <=144 px."""
    from PIL import Image

    with Image.open(run / meta["color_legend"]) as source:
        legend = source.convert("RGB")
    legend.thumbnail((max(1, min(width - 32, 720)), max(1, min(height // 5, 96))))
    return legend


def legend_origin(legend, width, height, anchor="top-left"):
    """Place a compact legend in a verified empty image corner."""
    return (
        16 if anchor.endswith("left") else width - 16 - legend.width,
        48 if anchor.startswith("top") else height - 16 - legend.height,
    )


def label_origin(label_width, legend_height, width, height, anchor="top-left"):
    """Keep the phase label beside the verified legend corner."""
    return (
        16 if anchor.endswith("left") else width - 16 - label_width,
        12 if anchor.startswith("top") else height - 16 - legend_height - 36,
    )


def annotate_frame(rgb, run, meta, arrays, frame, legend_anchor="top-left"):
    """Label both phases and original simulation time without changing geometry."""
    from PIL import Image, ImageDraw, ImageFont

    overlay = Image.fromarray(rgb)
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    replay = meta.get("replay", {})
    normal = replay.get("phase") == "NORMAL_THEN_STRAIN" and frame < replay["frames_per_phase"]
    phase = "NORMAL" if normal else "STRAIN"
    times = arrays[replay.get("source_times", "times")]
    label = f"{phase} | t = {times[frame]:.2f} s"
    legend = legend_image(run, meta, rgb.shape[1], rgb.shape[0]) if "color_legend" in meta else None
    label_width = draw.textbbox((0, 0), label, font=font)[2]
    x, y = label_origin(label_width, legend.height if legend else 96, rgb.shape[1], rgb.shape[0], legend_anchor)
    bounds = draw.textbbox((x, y), label, font=font)
    draw.rectangle((x - 8, y - 6, bounds[2] + 8, bounds[3] + 6), fill=(245, 247, 247))
    draw.text((x, y), label, font=font, fill=(25, 30, 34))
    if not normal and "color_legend" in meta:
        overlay.paste(legend, legend_origin(legend, rgb.shape[1], rgb.shape[0], legend_anchor))
    return np.asarray(overlay)


def load_verified_scene(args):
    """Check saved inputs and final-video gates before loading USD or ovrtx."""
    check_sources = not args.offline_bundle
    if args.normal:
        verify_normal(args.run)
    elif (args.run / "replay_manifest.json").exists():
        verify(args.run.parent, check_sources=check_sources)
        verify_combined(args.run.parent)
    elif (args.run / "manifest.json").exists():
        verify(args.run, check_sources=check_sources)
    else:
        raise ValueError("Prepare and verify a strain replay bundle before rendering")
    meta = json.loads((args.run / "scene.json").read_text())
    metrics = json.loads((args.run / "metrics.json").read_text())
    if args.frame is None and not metrics.get("fixture_only"):
        if metrics.get("test_passed") is not True or metrics.get("negative_control_passed") is not True:
            raise ValueError("Refusing a final video before simulation and negative-control checks pass")
    arrays = dict(np.load(args.run / "scene.npz", allow_pickle=False))
    count = validate_scene(meta, arrays)
    if args.frame is not None and not 0 <= args.frame < count:
        raise ValueError("Preview frame outside saved sequence")
    if not args.normal and args.frame is None:
        overlay_check = json.loads((args.run / "overlay-audit.json").read_text())
        if (
            not overlay_check["verified_no_projected_overlap"]
            or overlay_check["style"] != "compact"
            or overlay_check["anchor"] != args.legend_anchor
            or overlay_check["resolution"] != [args.width, args.height]
        ):
            raise ValueError("A matching clear compact-overlay audit is required before final strain rendering")
        for name, expected in overlay_check["inputs_sha256"].items():
            if digest(args.run / name) != expected:
                raise ValueError("Overlay projection audit is stale")
    return meta, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--normal", action="store_true", help="Render an owned exact copy of a sealed normal scene")
    parser.add_argument(
        "--offline-bundle", action="store_true", help="Verify relocated strain output without reopening original inputs"
    )
    parser.add_argument(
        "--legend-anchor", choices=("top-left", "top-right", "bottom-left", "bottom-right"), default="top-left"
    )
    args = parser.parse_args(argv)
    args.run = args.run.resolve()
    if args.width <= 0 or args.height <= 0 or args.samples <= 0:
        parser.error("Width, height and sample count must be positive")
    if args.frame is None and (args.width % 2 or args.height % 2):
        parser.error("Full yuv420p video requires even dimensions")
    meta, arrays = load_verified_scene(args)
    renderer_source = args.run / "renderer-source.py"
    shutil.copy2(Path(__file__), renderer_source)
    renderer_sha256 = digest(renderer_source)
    geometry, count = create_stage(args, meta, arrays)
    # USD must initialize first to avoid duplicate plugin registration by ovrtx.
    import ovrtx
    from PIL import Image

    renderer = ovrtx.Renderer(ovrtx.RendererConfig(log_level="warn", log_file_path=str(args.run / "rtx.log")))
    renderer.open_usd(str(args.run / ("scene_preview.usdc" if args.frame is not None else "scene.usdc")))
    writer = None
    if args.frame is None:
        writer = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{args.width}x{args.height}",
                "-framerate",
                str(meta["fps"]),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.run / "rtx.mp4"),
            ],
            stdin=subprocess.PIPE,
        )
    start = time.perf_counter()
    changes, errors, color_errors = [], [], []
    previous = None
    selected = [args.frame] if args.frame is not None else range(count)
    poster_frame = int(meta.get("poster_frame", count // 2))
    try:
        for frame in selected:
            renderer.update_from_usd_time(frame / meta["fps"])
            products = renderer.step(render_products={PRODUCT}, delta_time=1 / meta["fps"])
            with products[PRODUCT].frames[0].render_vars["LdrColor"].map(device=ovrtx.Device.CPU) as mapped:
                rgb = np.asarray(np.from_dlpack(mapped)[..., :3]).copy()
            if rgb.shape != (args.height, args.width, 3) or rgb.std() < 5:
                raise AssertionError("Blank or invalid render")
            actual = renderer.read_array_attribute("points", list(geometry))
            for path, points in geometry.items():
                expected = points[frame] if points.ndim == 3 else points
                rendered = np.from_dlpack(actual[path]).reshape(-1, 3)
                error = float(np.max(np.abs(rendered - expected)))
                if error > 2e-7:
                    raise AssertionError(f"Stale geometry: {path}, frame={frame}, error={error}")
                errors.append(error)
            colored = {
                f"/World/Object_{i}": arrays[item["vertex_colors"]]
                for i, item in enumerate(meta["meshes"])
                if "vertex_colors" in item
            }
            if colored:
                actual_colors = renderer.read_array_attribute("primvars:displayColor", list(colored))
                for path, expected_colors in colored.items():
                    actual_color = np.from_dlpack(actual_colors[path]).reshape(-1, 3)
                    error = float(np.max(np.abs(actual_color - expected_colors[frame])))
                    if error > 2e-7:
                        raise AssertionError(f"Stale vertex colors: {path}, frame={frame}, error={error}")
                    color_errors.append(error)
            if previous is not None:
                changes.append(float(np.mean(np.abs(rgb.astype(float) - previous))))
            previous = rgb.astype(float)
            if not args.normal:
                rgb = annotate_frame(rgb, args.run, meta, arrays, frame, args.legend_anchor)
            if writer:
                writer.stdin.write(rgb.tobytes())
            if args.frame is not None or frame in (0, poster_frame, count - 1):
                Image.fromarray(rgb).save(args.run / f"rtx_{frame:04d}.png")
            if frame == poster_frame or args.frame is not None:
                Image.fromarray(rgb).save(args.run / "poster.jpg", quality=94)
            if frame % 60 == 0:
                print(
                    f"{meta.get('slug', args.run.name)}: {frame}/{count}, {time.perf_counter() - start:.1f}s",
                    flush=True,
                )
    finally:
        if writer:
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError("Video encoding failed")
    if args.frame is None:
        if max(changes, default=0) < 0.005:
            raise AssertionError("No visible motion in video")
        audit = {
            "renderer": "ovrtx",
            "version": importlib.metadata.version("ovrtx"),
            "mode": "PathTracing",
            "samples_per_pixel": args.samples,
            "denoising": True,
            "resolution": [args.width, args.height],
            "frames": count,
            "fps": meta["fps"],
            "render_seconds": time.perf_counter() - start,
            "max_renderer_vertex_error_m": max(errors),
            "max_renderer_color_error": max(color_errors, default=0),
            "max_pixel_change": max(changes),
            "inputs_sha256": {
                p: digest(args.run / p)
                for p in (
                    "scene.npz",
                    "scene.json",
                    "metrics.json",
                    "renderer-source.py",
                    *(["overlay-audit.json"] if not args.normal else []),
                    *([meta["color_legend"]] if "color_legend" in meta else []),
                )
            },
            "video_sha256": digest(args.run / "rtx.mp4"),
            "usd_sha256": digest(args.run / "scene.usdc"),
            "renderer_script_sha256": renderer_sha256,
            "source_verification": "staged_normal_hashes"
            if args.normal
            else ("relocated_bundle_only" if args.offline_bundle else "bundle_and_original_inputs"),
            "replay": meta.get("replay", {"phase": "NORMAL"}),
            "strain_view": meta.get("strain_view"),
            "overlay": None
            if args.normal
            else {
                "style": "compact",
                "legend_anchor": args.legend_anchor,
                "max_legend_height_px": 96,
                "projection_audit": "overlay-audit.json",
            },
            "note": "Exact saved world-space surfaces; no resimulation, interpolation or deformation amplification.",
        }
        (args.run / "rtx.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
