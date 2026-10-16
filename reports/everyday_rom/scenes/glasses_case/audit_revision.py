"""Retain independent saved-state, full-thickness, and matched-control gates."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_clamped import inspect, triangle_radius_min
from clamped_geometry import NS, NX, SHELL_CENTER, root_skin_ids
from opening_coverage import opening_coverage
from test_peel_case import check_mesh

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
ROOT = parser.parse_args().root


def load(path):
    with np.load(path) as data:
        return dict(data)


results = {}
for name in ("candidate", "no-tip", "no-close"):
    directory = ROOT / name
    audit = inspect(directory)
    scene, basis = load(directory / "scene.npz"), load(directory / "basis.npz")
    meta = json.loads((directory / "scene.json").read_text())
    for i in range(len(meta["meshes"])):
        check_mesh(basis[f"rest_{i}"], scene[f"faces_{i}"])
    np.testing.assert_array_equal(basis["phi_3"][root_skin_ids()], 0)
    ids = [skin * NX * 65 + ix * 65 + col for skin in range(2) for ix in range(NX) for col in (0, 1)]
    np.testing.assert_array_equal(basis["phi_0"][ids], 0)
    faces = scene["faces_3"]
    free = np.all(faces % (NX * NS) % NS >= 2, axis=1)
    maximum = 0.0
    min_volume = np.inf
    for frame, points in enumerate(scene["vertices_3"]):
        wall = scene["vertices_0"][frame] - SHELL_CENTER
        enclosing_radius = float(np.linalg.norm(wall[:, 1:] / [0.020, 0.035], axis=1).max())
        triangles = points[faces] - SHELL_CENTER
        radius = triangle_radius_min(triangles[:, :, [1, 2]] / [0.020, 0.035])
        theta = np.mod(np.arctan2(triangles[:, :, 1] / 0.020, triangles[:, :, 2] / 0.035), 2 * np.pi)
        covered = np.any((theta >= np.pi / 2) & (theta <= 2 * np.pi - 0.68), axis=1) & free
        depth = max(0.0, enclosing_radius - float(radius[covered].min(initial=enclosing_radius))) * 0.035
        maximum = max(maximum, depth)
        min_volume = min(
            min_volume,
            float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6),
        )
    assert maximum < 0.001 and min_volume > 0
    results[name] = {
        "passed": True,
        "full_thickness_pouch_penetration_upper_bound_m": maximum,
        "minimum_flap_volume_m3": min_volume,
        "rest_topology_meshes": len(meta["meshes"]),
        "audit": audit,
    }
    if name == "candidate":
        times = load(directory / "state.npz")["times"]
        coverage = [opening_coverage(points) for i, points in enumerate(scene["vertices_3"]) if 3.6 < times[i] < 4.2]
        assert coverage and all(r["opening_uncovered_sections"] == 0 for r in coverage)
        results[name]["closed_coverage"] = {
            "minimum_covered_fraction": min(r["opening_covered_fraction"] for r in coverage),
            "minimum_width_margin_m": min(r["opening_min_width_margin_m"] for r in coverage),
            "frames": len(coverage),
            "sections_per_frame": coverage[0]["opening_sections"],
        }

positive, detached, fixed = [load(ROOT / name / "state.npz") for name in ("candidate", "no-tip", "no-close")]
basis = load(ROOT / "candidate/basis.npz")
np.testing.assert_array_equal(positive["body_q"][:, 3], detached["body_q"][:, 3])
for state in (positive, detached, fixed):
    for body in (0, 1, 2):
        np.testing.assert_array_equal(
            state["body_q"][:, body], np.broadcast_to(state["body_q"][0, body], state["body_q"][:, body].shape)
        )
start = int(basis["q_start_3"])
phi = basis["phi_3"]
q = positive["joint_q"][:, start : start + phi.shape[1]]
qfixed = fixed["joint_q"][:, start : start + phi.shape[1]]
hold = (positive["times"] > 3.6) & (positive["times"] < 4.2)
motion = np.einsum("nmc,tm->tnc", phi, q)
extent = float(np.linalg.norm(np.ptp(motion[hold], axis=0), axis=1).max())
residual = float(np.linalg.norm(np.einsum("nmc,m->nc", phi, q[-1] - qfixed[-1]), axis=1).max())
assert extent < 0.0005 and residual < 0.001 and np.ptp(q, axis=0).max() > 1e-5
a, b, c = [results[name]["audit"] for name in ("candidate", "no-tip", "no-close")]
change = a["closed_curvature_degrees"] - a["open_curvature_degrees"]
detached_change = b["closed_curvature_degrees"] - b["open_curvature_degrees"]
assert change > 140 and abs(detached_change) < 5 and c["closed_curvature_degrees"] < 5
results["paired"] = {
    "passed": True,
    "positive_curvature_change_degrees": change,
    "detached_curvature_change_degrees": detached_change,
    "stationary_curvature_degrees": c["closed_curvature_degrees"],
    "hold_surface_peak_to_peak_m": extent,
    "unload_residual_m": residual,
    "actuator_poses_identical": True,
}
(ROOT / "audits.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results["paired"], indent=2))
