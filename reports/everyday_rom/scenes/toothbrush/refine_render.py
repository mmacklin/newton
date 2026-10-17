# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU-only, single-application brush refinement from its physical basis.

The two subdivision functions are extracted unchanged into local subdivision.py.
This adapter consumes a fresh local simulator export, never saved gallery states.
Quaternion conversion
preserves the float32 operation order of Warp 1.17's host builtin using NumPy;
no Warp import, device arrays, kernels, simulation, or CUDA probe is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from subdivision import refined_plastic


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def fresh_output(path, inputs):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Output must not contain symlink components")
    for input_path in inputs:
        source = Path(input_path).resolve()
        if path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("Output must not overlap inputs")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output must be absent or empty")
    return path


def transform_points(points, pose):
    # Warp 1.17 native/quat.h quat_rotate, applied to the three unit columns.
    # Keeping its float32 evaluation order preserves the frozen export bits.
    x, y, z, w = np.asarray(pose[3:], dtype=np.float32)
    two, one = np.float32(2), np.float32(1)
    c = two * w * w - one
    columns = []
    for a, b, d in np.eye(3, dtype=np.float32):
        dot = two * (x * a + y * b + z * d)
        columns.append(
            [
                a * c + x * dot + (y * d - z * b) * w * two,
                b * c + y * dot + (z * a - x * d) * w * two,
                d * c + z * dot + (x * b - y * a) * w * two,
            ]
        )
    rotation = np.asarray(columns, dtype=float).T
    return (points @ rotation.T + pose[:3]).astype(np.float32)


def validate_physical(root):
    root = Path(root)
    meta = json.loads((root / "scene.json").read_text())
    if meta.get("render_refinement"):
        raise ValueError("Refuse double refinement: scene already has render_refinement")
    metrics = json.loads((root / "metrics.json").read_text())
    if not (metrics.get("test_passed") and metrics.get("negative_control_passed")):
        raise ValueError("Require a fresh positive/control export passing its original simulation gates")
    # Fresh settling coordinates can differ from an archived run. Bind the
    # producer instead of requiring the old run's complete basis container.
    producers = ["toothbrush.py", "brush_glove_geometry.py", "field_coverage.py", "capacity_audit.py"]
    if meta.get("rigid_fusion"):
        producers += ["fused_toothbrush.py", "subdivision.py"]
        if metrics.get("solver_implementation") != "public generic SolverVBD":
            raise ValueError("Require a fresh public generic fused producer, not a private timed export")
    for name in producers:
        if metrics.get("source_sha256", {}).get(name) != digest(Path(__file__).with_name(name)):
            raise ValueError(f"Fresh simulation source differs: {name}")
    with np.load(root / "basis.npz", allow_pickle=False) as basis:
        for i, count, name in ((0, 880, "ivory_handle"), (3, 560, "ivory_plastic_head")):
            if meta["meshes"][i]["name"] != name:
                raise ValueError("Unexpected plastic mesh ordering")
            if len(basis[f"faces_{i}"]) != count:
                raise ValueError("Not the accepted coarse topology")
    return meta


def refine(physical, output):
    physical, assets = Path(physical), Path(__file__).resolve().parent
    output = fresh_output(output, (physical, assets / "subdivision.py", assets / "presentation.json", Path(__file__)))
    meta = validate_physical(physical)
    expected_basis_sha256 = digest(physical / "basis.npz")
    with np.load(physical / "basis.npz", allow_pickle=False) as data:
        basis = dict(data)
    with np.load(physical / "state.npz", allow_pickle=False) as data:
        states = dict(data)
    with np.load(physical / "scene.npz", allow_pickle=False) as data:
        original = dict(data)
    np.testing.assert_array_equal(states["times"], original["times"])
    np.testing.assert_allclose(states["times"], np.arange(1, 451) / 60, rtol=0, atol=1e-12)
    for i, mesh in enumerate(meta["meshes"]):
        np.testing.assert_array_equal(original[mesh["faces"]], basis[f"faces_{i}"])
    refined = refined_plastic(basis, meta)
    arrays, embedding = dict(original), {}
    render_basis = {
        key: basis[key]
        for i in range(len(meta["meshes"]))
        for key in (f"rest_vertices_{i}", f"faces_{i}", f"phi_{i}")
        if key in basis
    }
    checks = []
    for i, item in refined.items():
        mesh = meta["meshes"][i]
        rest, faces, weights = item["points"], item["faces"], item["weights"]
        render_basis[f"rest_vertices_{i}"] = rest
        render_basis[f"faces_{i}"] = faces
        if f"phi_{i}" in basis:
            phi = (weights @ basis[f"phi_{i}"].reshape(len(item["original_points"]), -1)).reshape(len(rest), -1, 3)
            np.testing.assert_array_equal(phi, 0)
            render_basis[f"phi_{i}"] = phi.astype(np.float32)
        embedding[f"weights_{i}"] = weights
        frames, error = [], 0.0
        for frame in range(450):
            world = transform_points(rest, states["body_q"][frame, mesh["body_index"]])
            error = max(error, float(np.abs(world - weights @ original[mesh["vertices"]][frame]).max()))
            frames.append(world)
        if error >= 5e-8:
            raise ValueError(f"Physical-to-render embedding failed: {mesh['name']}: {error}")
        arrays[mesh["vertices"]] = np.asarray(frames, dtype=np.float32)
        arrays[mesh["faces"]] = faces
        checks.append({"mesh": mesh["name"], "frames": 450, "embedding_max_error_m": error})
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((physical / "metrics.json").read_text())
    audit = {
        "render_only": True,
        "physics_rerun_by_refinement": False,
        "applications": 1,
        "physical_basis_sha256": expected_basis_sha256,
        "physical_inputs_sha256": {
            name: digest(physical / name) for name in ("basis.npz", "state.npz", "scene.npz", "scene.json")
        },
        "all_frame_checks": checks,
        "unchanged_mesh_arrays_bit_identical": True,
        "subdivision_source_sha256": digest(assets / "subdivision.py"),
        "fresh_source_hashes": metrics["source_sha256"],
    }
    presentation = json.loads((assets / "presentation.json").read_text())
    meta.update(camera=presentation["camera"], poster_frame=presentation["poster_frame"], render_refinement=audit)
    meta["reproduce"] = "See companion README.md; refine only a fresh coarse simulator export"
    metrics["render_refinement"] = audit
    np.savez_compressed(output / "scene.npz", **arrays)
    np.savez_compressed(output / "basis.npz", **render_basis)
    np.savez_compressed(output / "render_embedding.npz", **embedding)
    shutil.copy2(physical / "state.npz", output / "state.npz")
    (output / "physics").mkdir()
    for name in ("basis.npz", "scene.json", "metrics.json"):
        shutil.copy2(physical / name, output / "physics" / name)
    write_json(output / "scene.json", meta)
    write_json(output / "metrics.json", metrics)
    write_json(output / "geometry_audit.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physical", type=Path, required=True, help="Fresh 450-frame positive/control simulation output"
    )
    parser.add_argument("--output", type=Path, required=True, help="New, separate smooth-render output directory")
    args = parser.parse_args()
    print(json.dumps(refine(args.physical, args.output), indent=2))
