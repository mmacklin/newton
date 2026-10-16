# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU projected-triangle checks for quantitative video overlay occlusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .render_strain_replay import camera_path, label_origin, legend_image, legend_origin
    from .strain_adapters import digest, load_arrays
    from .strain_replay import write_json
else:
    from render_strain_replay import camera_path, label_origin, legend_image, legend_origin
    from strain_adapters import digest, load_arrays
    from strain_replay import write_json


def intersects_rectangle(triangles, rectangle):
    """Return exact convex triangle/rectangle intersection via separating axes."""
    lo, hi = np.asarray(rectangle[:2]), np.asarray(rectangle[2:])
    keep = np.all(triangles.max(axis=1) >= lo, axis=1) & np.all(triangles.min(axis=1) <= hi, axis=1)
    indices = np.flatnonzero(keep)
    tri = triangles[indices]
    center, half = (lo + hi) / 2, (hi - lo) / 2
    for edge in range(3):
        delta = tri[:, (edge + 1) % 3] - tri[:, edge]
        axis = np.stack((-delta[:, 1], delta[:, 0]), axis=1)
        values = np.einsum("ntc,nc->nt", tri - center, axis)
        radius = np.abs(axis) @ half
        keep[indices] &= (values.min(axis=1) <= radius) & (values.max(axis=1) >= -radius)
    return keep


def camera_triangles(points, faces, position, target, width, height, fov):
    """Project triangles using the renderer's vertical FOV and Z-up camera."""
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    camera = (points - position) @ np.stack((right, up, forward), axis=1)
    triangles = camera[faces]
    near = 1e-4
    front = np.all(triangles[:, :, 2] >= near, axis=1)
    clipped = [triangles[front]]
    for triangle in triangles[~front & np.any(triangles[:, :, 2] >= near, axis=1)]:
        polygon = []
        for i, b in enumerate(triangle):
            a = triangle[i - 1]
            if (a[2] >= near) != (b[2] >= near):
                polygon.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
            if b[2] >= near:
                polygon.append(b)
        for i in range(1, len(polygon) - 1):
            clipped.append(np.asarray([[polygon[0], polygon[i], polygon[i + 1]]]))
    triangles = np.concatenate(clipped)
    focal = height / (2 * np.tan(np.deg2rad(fov) / 2))
    projected = triangles[:, :, :2] / triangles[:, :, 2:3] * focal
    projected[:, :, 0] += width / 2
    projected[:, :, 1] = height / 2 - projected[:, :, 1]
    return projected


def audit_overlay(directory: Path, style="compact", width=1280, height=960, anchor="top-left"):
    """Audit all saved frames; floor meshes are background, not object occlusion."""
    from PIL import Image, ImageDraw, ImageFont

    directory = Path(directory)
    meta = json.loads((directory / "scene.json").read_text())
    arrays = load_arrays(directory / "scene.npz")
    positions, targets = camera_path(meta, arrays)
    if style == "headphones-legacy":
        rectangles = [[24, 24, 450, 121], [24, 130, 190, 154]]
    else:
        with Image.open(directory / meta["color_legend"]) as source:
            legend = source.copy()
        if style == "compact":
            legend = legend_image(directory, meta, width, height)
        elif style == "original":
            legend.thumbnail((min(width - 32, 720), height // 5))
        else:
            raise ValueError("Unknown overlay style")
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        label_rectangles = []
        for t in arrays["times"]:
            label = f"STRAIN | t = {t:.2f} s"
            label_width = draw.textbbox((0, 0), label, font=font)[2]
            x, y = label_origin(label_width, legend.height, width, height, anchor)
            bounds = draw.textbbox((x, y), label, font=font)
            label_rectangles.append([x - 8, y - 6, bounds[2] + 8, bounds[3] + 6])
        bounds = np.asarray(label_rectangles)
        x, y = legend_origin(legend, width, height, anchor)
        rectangles = [
            [*bounds[:, :2].min(axis=0).tolist(), *bounds[:, 2:].max(axis=0).tolist()],
            [x, y, x + legend.width, y + legend.height],
        ]
    overlaps, ignored = {}, []
    for item in meta["meshes"]:
        if "floor" in item["name"].lower():
            ignored.append(item["name"])
            continue
        points, faces = arrays[item["vertices"]], arrays[item["faces"]]
        frames, counts = [], []
        for frame in range(len(arrays["times"])):
            projected = camera_triangles(
                points[frame] if points.ndim == 3 else points,
                faces,
                positions[frame],
                targets[frame],
                width,
                height,
                meta["camera"].get("fov", 38),
            )
            hits = np.zeros(len(projected), dtype=bool)
            for rect in rectangles:
                hits |= intersects_rectangle(projected, rect)
            if hits.any():
                frames.append(frame)
                counts.append(int(hits.sum()))
        if frames:
            overlaps[item["name"]] = {"frames": frames, "max_intersecting_triangles": max(counts)}
    return {
        "verified_no_projected_overlap": not overlaps,
        "frames": len(arrays["times"]),
        "style": style,
        "anchor": anchor,
        "resolution": [width, height],
        "overlay_rectangles_px": rectangles,
        "overlaps": overlaps,
        "ignored_background": ignored,
        "method": "Every source triangle in every saved frame; exact perspective camera, near-plane clipping, triangle/rectangle separating axes. Conservative: hidden/back-facing object triangles also count.",
        "inputs_sha256": {
            name: digest(directory / name)
            for name in ("scene.json", "scene.npz", *([meta["color_legend"]] if "color_legend" in meta else []))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--style", choices=("compact", "original", "headphones-legacy"), default="compact")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--anchor", choices=("top-left", "top-right", "bottom-left", "bottom-right"), default="top-left"
    )
    args = parser.parse_args()
    result = audit_overlay(args.directory, args.style, args.width, args.height, args.anchor)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
