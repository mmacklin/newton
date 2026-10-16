"""Independent persisted-state and complete-surface audit for the C1 candidate."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_surface_strain import rest_metric, stretches
from clamped_geometry import NS, NX, SHELL_CENTER, continuity, curvature, material_orientation
from contact_capacity import validate_trace
from peel_case_geometry import surface_strain_metrics, transform


def validate_root(result):
    assert result["root_gap_max_m"] < 1e-6, result
    assert result["root_tangent_error_max_degrees"] < 0.01, result
    assert result["root_normal_error_max_degrees"] < 0.01, result


def triangle_radius_min(triangles):
    """Exact minimum distance to origin on each closed triangle, including edges."""
    a, b, c = np.moveaxis(triangles, 1, 0)
    e, f = b - a, c - a
    ee, ff, ef = (np.sum(x * y, axis=1) for x, y in [(e, e), (f, f), (e, f)])
    ae, af = np.sum(a * e, axis=1), np.sum(a * f, axis=1)
    determinant = ee * ff - ef * ef
    valid = determinant > 1e-25
    denominator = np.where(valid, determinant, 1.0)
    u = (ef * af - ff * ae) / denominator
    v = (ef * ae - ee * af) / denominator
    interior = valid & (u >= 0) & (v >= 0) & (u + v <= 1)
    best = np.where(interior, np.linalg.norm(a + u[:, None] * e + v[:, None] * f, axis=1), np.inf)
    for p, q in [(a, b), (b, c), (c, a)]:
        d = q - p
        t = np.clip(-np.sum(p * d, axis=1) / np.maximum(np.sum(d * d, axis=1), 1e-30), 0, 1)
        best = np.minimum(best, np.linalg.norm(p + t[:, None] * d, axis=1))
    return best


def polygon_crossings(points):
    a = points
    b = np.roll(a, -1, axis=0)
    d = b - a
    relative = a[None] - a[:, None]

    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    denominator = cross(d[:, None], d[None])
    divisor = np.where(np.abs(denominator) > 1e-20, denominator, 1.0)
    t = cross(relative, d[None]) / divisor
    u = cross(relative, d[:, None]) / divisor
    crossings = (np.abs(denominator) > 1e-20) & (t > 1e-5) & (t < 1 - 1e-5) & (u > 1e-5) & (u < 1 - 1e-5)
    return int(np.count_nonzero(np.triu(crossings, 1)))


def inspect(directory):
    directory = Path(directory)
    with np.load(directory / "scene.npz") as data:
        scene = dict(data)
    with np.load(directory / "basis.npz") as data:
        basis = dict(data)
    with np.load(directory / "state.npz") as data:
        state = dict(data)
    metrics = json.loads((directory / "metrics.json").read_text())
    metadata = json.loads((directory / "scene.json").read_text())
    records = []
    strain_by_mesh = {}
    worst_recovery = 0.0
    independent_stretch = 0.0
    for i, mesh in enumerate(metadata["meshes"]):
        rest, phi, faces = basis[f"rest_{i}"], basis[f"phi_{i}"], scene[f"faces_{i}"]
        body, start = int(basis[f"body_{i}"]), int(basis[f"q_start_{i}"])
        inverse, _ = rest_metric(rest, faces)
        for frame in range(len(state["times"])):
            q = state["joint_q"][frame, start : start + phi.shape[1]]
            local = rest + np.einsum("nmc,m->nc", phi, q)
            recovered = transform(local, state["body_q"][frame, body]) if body >= 0 else local
            saved = scene[f"vertices_{i}"]
            target = saved[frame] if saved.ndim == 3 else saved
            worst_recovery = max(worst_recovery, float(np.abs(recovered - target).max()))
            if phi.shape[1]:
                values = surface_strain_metrics(rest, local, faces)
                old = strain_by_mesh.get(mesh["name"], values)
                strain_by_mesh[mesh["name"]] = {k: max(v, old[k]) for k, v in values.items()}
                measured = metrics["records"][frame]["strain_by_mesh"][mesh["name"]]
                for key in values:
                    np.testing.assert_allclose(values[key], measured[key], rtol=2e-6, atol=1e-8)
                independent_stretch = max(
                    independent_stretch, float(np.abs(stretches(local, faces, inverse) - 1).max())
                )
    flap = scene["vertices_3"]
    faces = scene["faces_3"]
    # The two fixed material strips are the shared sewn attachment, not free flap.
    free_faces = np.all(faces % (NX * NS) % NS >= 2, axis=1)
    for frame, points in enumerate(flap):
        value = continuity(scene["vertices_0"][frame], points)
        value.update(curvature(points), **material_orientation(points))
        skins = points.reshape(2, NX, NS, 3)
        cross_section = np.concatenate((skins[0, NX // 2, :, 1:], skins[1, NX // 2, ::-1, 1:]))
        value["self_intersections"] = polygon_crossings(cross_section)
        value["extrusion_error_m"] = float(np.ptp(skins[:, :, :, 1:], axis=1).max())
        value["material_x_error_m"] = float(np.abs(points[:, 0] - basis["rest_3"][:, 0]).max())
        triangle = points[faces]
        radius = triangle_radius_min((triangle - SHELL_CENTER) / [0.067, 0.019, 0.034])
        value["former_penetration_bound_m"] = max(0.0, 1 - float(radius.min())) * 0.067
        cap_depth = 0.0
        for sign in [-1, 1]:
            cap_center = SHELL_CENTER + np.array([sign * 0.074, 0, 0])
            radius = triangle_radius_min((triangle[free_faces] - cap_center) / [0.007, 0.020, 0.035])
            cap_depth = max(cap_depth, max(0.0, 1 - float(radius.min())) * 0.035)
        value["free_flap_endcap_penetration_bound_m"] = cap_depth
        # Norm convexity is insufficient for a minimum: check entire triangles.
        local = triangle - SHELL_CENTER
        radius = triangle_radius_min(local[:, :, [1, 2]] / [0.020, 0.035])
        theta = np.mod(np.arctan2(local[:, :, 1] / 0.020, local[:, :, 2] / 0.035), 2 * np.pi)
        covered = np.any((theta >= np.pi / 2) & (theta <= 2 * np.pi - 0.68), axis=1) & free_faces
        value["pouch_ellipse_penetration_bound_m"] = max(0.0, 1 - float(radius[covered].min(initial=1.0))) * 0.035
        records.append(value)
    result = {
        k: max(r[k] for r in records)
        for k in records[0]
        if k not in ["material_director_min_cosine", "minimum_thickness_m", "curvature_min_width_degrees"]
    }
    result.update(
        material_director_min_cosine=min(r["material_director_min_cosine"] for r in records),
        independent_principal_strain_max=independent_stretch,
        strain_by_mesh=strain_by_mesh,
        exact_surface_recovery_max_m=worst_recovery,
        frames=len(records),
        open_curvature_degrees=float(
            np.mean([r["curvature_min_width_degrees"] for i, r in enumerate(records) if state["times"][i] < 0.6])
        ),
        closed_curvature_degrees=float(
            np.mean([r["curvature_min_width_degrees"] for i, r in enumerate(records) if 3.6 < state["times"][i] < 4.2])
        ),
        scene_sha256=hashlib.sha256((directory / "scene.npz").read_bytes()).hexdigest(),
        root_contract="Both exact fixed rear frames, zero modal displacement on two root strips across all 37 width samples; first material finite-difference tangent and skin normal are invariant for arbitrary q, including every solve substep.",
        geometry_scope="Every saved triangle; exact triangle minimum distance to enclosing analytic former/endcap solids. First two sewn root strips excluded only from endcap/pouch clearance, never strain, recovery, director, tangent or seam checks.",
        material_orientation_gate="Finite-rotation material director, positive signed thickness, simple extruded cross-section. Rest-normal reversal count retained as a diagnostic, not treated as inversion.",
    )
    with np.load(directory / "contact_trace.npz") as trace:
        steps = metrics["config"]["substeps"]
        result["capacity"] = validate_trace(trace, len(records) * steps, 16384, 4096, 1 / (60 * steps))
    checks = {
        "root": result["root_gap_max_m"] < 1e-6
        and result["root_tangent_error_max_degrees"] < 0.01
        and result["root_normal_error_max_degrees"] < 0.01,
        "strain": independent_stretch < 0.15 and max(v["max_abs_area_strain"] for v in strain_by_mesh.values()) < 0.15,
        "orientation": result["material_inverted_cells"] == 0 and result["material_director_min_cosine"] > 0.9,
        "self_intersection": result["self_intersections"] == 0
        and result["extrusion_error_m"] < 1e-6
        and result["material_x_error_m"] < 1e-7,
        "recovery": worst_recovery < 1e-7,
        "former_clearance": result["former_penetration_bound_m"] < 0.001,
        "endcap_clearance": result["free_flap_endcap_penetration_bound_m"] < 0.001,
        "wall_clearance": result["pouch_ellipse_penetration_bound_m"] < 0.001,
        "contact_penetration": metrics["max_penetration_m"] < 0.001,
    }
    result["checks"] = checks
    result["passed"] = all(checks.values())
    (directory / "c1_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    assert result["passed"], checks
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    inspect(args.directory)
