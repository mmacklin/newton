# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Audit every saved triangle relative to the first exported surface frame.

This is an independent trajectory check, not a replacement for material-rest
strain metrics. A deformed first frame will change the reference metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rest_metric(points, faces):
    triangles = np.asarray(points, dtype=float)[faces]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    length = np.linalg.norm(a, axis=1)
    area2 = np.linalg.norm(np.cross(a, b), axis=1)
    scale = length * np.linalg.norm(b, axis=1)
    valid = (scale > 0) & (area2 > 1e-10 * scale)
    if not valid.all():
        raise ValueError(f"{np.count_nonzero(~valid)} degenerate reference triangles")
    projection = np.einsum("ij,ij->i", a, b) / length
    height = area2 / length
    inverse = np.zeros((len(faces), 2, 2))
    inverse[:, 0, 0] = 1 / length
    inverse[:, 0, 1] = -projection / (length * height)
    inverse[:, 1, 1] = 1 / height
    return inverse, area2 / 2


def stretches(points, faces, inverse):
    triangles = np.asarray(points, dtype=float)[faces]
    edges = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1)
    deformation = edges @ inverse
    metric = np.swapaxes(deformation, -1, -2) @ deformation
    return np.sqrt(np.maximum(np.linalg.eigvalsh(metric), 0))


def audit(directory):
    meta = json.loads((directory / "scene.json").read_text())
    results = {}
    with np.load(directory / "scene.npz", allow_pickle=False) as arrays:
        for mesh in meta["meshes"]:
            vertices, faces = arrays[mesh["vertices"]], arrays[mesh["faces"]]
            if vertices.ndim == 2:
                vertices = vertices[None]
            if not np.isfinite(vertices).all():
                raise ValueError(f"Nonfinite vertices: {mesh['name']}")
            inverse, area = rest_metric(vertices[0], faces)
            result = {
                "frames": len(vertices),
                "triangles": len(faces),
                "min_stretch": 1.0,
                "max_stretch": 1.0,
                "max_area_fraction_over_10pct": 0.0,
                "worst_frame": 0,
                "worst_triangle": 0,
            }
            worst = 0.0
            for frame, points in enumerate(vertices):
                principal = stretches(points, faces, inverse)
                deviation = np.max(np.abs(principal - 1), axis=1)
                result["min_stretch"] = min(result["min_stretch"], float(principal.min()))
                result["max_stretch"] = max(result["max_stretch"], float(principal.max()))
                result["max_area_fraction_over_10pct"] = max(
                    result["max_area_fraction_over_10pct"], float(area[deviation > 0.1].sum() / area.sum())
                )
                if deviation.max() > worst:
                    worst = float(deviation.max())
                    result.update(worst_frame=frame, worst_triangle=int(deviation.argmax()))
            results[mesh["name"]] = result
    with (directory / "scene.npz").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "reference": "First exported frame per mesh, not necessarily stress-free material rest",
        "scene_sha256": digest,
        "all_triangles_included": True,
        "max_stretch_deviation": max(max(1 - x["min_stretch"], x["max_stretch"] - 1) for x in results.values()),
        "meshes": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    (args.directory / "surface_strain_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
