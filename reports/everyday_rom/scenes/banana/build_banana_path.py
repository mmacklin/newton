# SPDX-License-Identifier: Apache-2.0
"""Fit ordered whole-peel snapshots; no simulation or animation state writes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ordered_pod import fit_ordered_pod


def mass_weights(rest, faces, density=900.0, thickness=0.0016):
    # Average the two skins; rim triangles must not add a second shell mass.
    n = 65 * 17
    mid = (rest[:n] + rest[n:]) / 2
    skin = faces[np.all(faces < n, axis=1)]
    a, b, c = mid[skin].transpose(1, 0, 2)
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    mass = np.zeros(n)
    np.add.at(mass, skin.ravel(), np.repeat(density * thickness * area / 3, 3))
    return np.tile(mass / 2, 2)


def principal_strain(rest, current, faces):
    a, b, c = rest[faces].transpose(1, 0, 2)
    u, v = b - a, c - a
    length = np.linalg.norm(u, axis=1)
    x = np.einsum("nc,nc->n", u, v) / length
    height = np.sqrt(np.maximum(0, np.einsum("nc,nc->n", v, v) - x * x))
    inv = np.zeros((len(faces), 2, 2))
    inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1] = 1 / length, -x / (length * height), 1 / height
    a, b, c = current[faces].transpose(1, 0, 2)
    deformation = np.stack((b - a, c - a), axis=2) @ inv
    stretches = np.linalg.svd(deformation, compute_uv=False)
    return float(np.max(np.abs(stretches - 1)))


def load_overrides(manifest):
    overrides = {}
    if manifest is None:
        return overrides
    for record in json.loads(manifest.read_text()):
        strip, front = int(record["strip"]), float(record["front"])
        key = (strip, round(front, 3))
        if strip not in range(3) or key in overrides:
            raise ValueError("Invalid or duplicate snapshot override")
        path = manifest.parent / record["snapshot"]
        with np.load(path, allow_pickle=False) as data:
            overrides[key] = {
                "arrays": {k: data[k].copy() for k in ("rest", "faces", "vertices")},
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return overrides


def apply_overrides(strip, rest, faces, cases, shapes, overrides):
    """Replace offline poses only, after checking their material correspondence."""
    used = []
    shapes = shapes.copy()
    for index, case in enumerate(cases, start=1):
        key = (strip, round(case["front"], 3))
        if key not in overrides:
            continue
        record = overrides[key]
        data = record["arrays"]
        if not np.array_equal(data["rest"], rest) or not np.array_equal(data["faces"], faces):
            raise ValueError("Snapshot override changes the rest mesh or topology")
        vertices = data["vertices"]
        if vertices.shape != rest.shape or not np.isfinite(vertices).all():
            raise ValueError("Invalid snapshot geometry")
        if principal_strain(rest, vertices, faces) >= 0.15:
            raise ValueError("Snapshot override exceeds the existing strain bound")
        shapes[index] = vertices
        used.append({"strip": strip, "front": case["front"], "path": record["path"], "sha256": record["sha256"]})
    return shapes, used


def build(output, modes=32, override_manifest=None, *, sources):
    if len(sources) != 3:
        raise ValueError("Provide one training directory for each of the three strips")
    arrays, audit = {}, []
    overrides = load_overrides(override_manifest)
    used_keys = set()
    for strip, source in enumerate(map(Path, sources)):
        with np.load(source / "snapshots.npz", allow_pickle=False) as data:
            rest = data[f"strip_{strip}_rest"].astype(float)
            faces = data[f"strip_{strip}_faces"].copy()
            cases = sorted(
                (v for v in json.loads((source / "metrics.json").read_text())["cases"] if v["strip"] == strip),
                key=lambda v: -v["front"],
            )[12:]
            shapes = np.array([rest, *[data[f"strip_{strip}_front_{v['front']:.3f}_vertices"] for v in cases]])
        shapes, used = apply_overrides(strip, rest, faces, cases, shapes, overrides)
        used_keys.update((v["strip"], round(v["front"], 3)) for v in used)
        mass = mass_weights(rest, faces)
        fit = fit_ordered_pod(rest, shapes, mass, np.linspace(0, 1, len(shapes)), modes)
        lengths = np.linalg.norm(np.diff(fit.coefficients, axis=0), axis=1)
        progress = np.r_[0, np.cumsum(lengths)]
        progress /= progress[-1]
        peak = 0.0
        for segment in range(len(shapes) - 1):
            for fraction in np.linspace(0, 1, 11):
                q = (1 - fraction) * fit.coefficients[segment] + fraction * fit.coefficients[segment + 1]
                peak = max(peak, principal_strain(rest, fit.recover(q), faces))
        root = ((rest[:1105] + rest[1105:]) / 2).reshape(65, 17, 3)[0, 8]
        arrays.update(
            {
                f"rest_{strip}": rest - root,
                f"faces_{strip}": faces,
                f"phi_{strip}": fit.phi,
                f"coefficients_{strip}": fit.coefficients,
                f"progress_{strip}": progress,
                f"mass_{strip}": mass,
                f"origin_{strip}": root,
                f"front_{strip}": np.array([1.0, *[v["front"] for v in cases]]),
            }
        )
        audit.append(
            {
                "strip": strip,
                "modes": modes,
                "knots": len(shapes),
                "mass_kg": float(mass.sum()),
                "reconstruction_error_m": fit.maximum_reconstruction_error,
                "retained_energy": fit.retained_energy,
                "interpolated_principal_strain": peak,
                "source": str(source),
                "source_sha256": hashlib.sha256((source / "snapshots.npz").read_bytes()).hexdigest(),
                "snapshot_overrides": used,
                "geometry_passed": peak < 0.15,
            }
        )
    if used_keys != set(overrides):
        raise ValueError("Snapshot override did not match an included training knot")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "ordered_rom.npz", **arrays)
    (output / "training_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if not all(v["geometry_passed"] for v in audit):
        raise AssertionError("Ordered/interpolated training geometry fails the existing strain bound")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument(
        "--sources",
        type=Path,
        nargs=3,
        required=True,
        help="Training directories for strips 0, 1, 2; not needed to run the bundled ROM",
    )
    args = parser.parse_args()
    build(args.output, args.modes, args.overrides, sources=args.sources)
