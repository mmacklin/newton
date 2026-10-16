# SPDX-License-Identifier: Apache-2.0
"""CPU geometry/schema audits; simulation assertions live in each Example."""

from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

import numpy as np
from banana import peel_progression
from peel_case_geometry import (
    banana_flesh,
    banana_strip_snapshot,
    banana_whole_strip,
    box,
    case_flap,
    case_shell,
    ellipsoid,
    glasses_mesh,
    initial_case_clearance,
    make_basis,
    pouch_clearance,
    pouch_endcaps,
    pouch_flap,
    pouch_shell,
    skin_embedded_modes,
    strip_geometry_metrics,
    surface_strain_metrics,
    transform,
    tube,
)


def check_mesh(points, faces):
    assert np.isfinite(points).all()
    assert faces.dtype.kind in "iu"
    assert faces.min() >= 0 and faces.max() < len(points)
    triangles = points[faces]
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    assert area2.min() > 1e-12, f"Degenerate face: {area2.min()}"
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), f"Not closed: {np.unique(counts, return_counts=True)}"
    volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    assert volume > 0, f"Inverted closed mesh: {volume}"


def audit_export(directory):
    """Recover every frame from persisted float32 states and sampled bases."""
    directory = Path(directory)
    meta = json.loads((directory / "scene.json").read_text())
    scene = np.load(directory / "scene.npz", allow_pickle=False)
    basis = np.load(directory / "basis.npz", allow_pickle=False)
    states = np.load(directory / "state.npz", allow_pickle=False)
    metrics = json.loads((directory / "metrics.json").read_text())
    np.testing.assert_allclose(np.diff(scene["times"]), 1 / meta["fps"], atol=1e-6)
    worst = 0.0
    strain_by_mesh = {}
    for i, item in enumerate(meta["meshes"]):
        body, start = int(basis[f"body_{i}"]), int(basis[f"q_start_{i}"])
        rest, phi = basis[f"rest_{i}"], basis[f"phi_{i}"]
        exported = scene[item["vertices"]]
        check_mesh(rest, scene[item["faces"]])
        for frame in range(len(scene["times"])):
            q = states["joint_q"][frame, start : start + phi.shape[1]]
            local = rest + np.einsum("nmc,m->nc", phi, q)
            if meta["slug"] == "glasses-case" and phi.shape[1] > 0:
                strain = surface_strain_metrics(rest, local, scene[item["faces"]])
                previous = strain_by_mesh.get(item["name"], strain)
                strain_by_mesh[item["name"]] = {key: max(value, previous[key]) for key, value in strain.items()}
            recovered = transform(local, states["body_q"][frame, body]) if body >= 0 else local
            expected = exported[frame] if exported.ndim == 3 else exported
            worst = max(worst, float(np.abs(recovered - expected).max()))
    assert worst < 1e-7, worst
    mass_error, stiffness_error = 0.0, 0.0
    for key in basis.files:
        if not key.endswith("_sample_phi") or "raw_" in key:
            continue
        prefix = key.removesuffix("sample_phi")
        phi = basis[key]
        gram = np.einsum("n,nic,njc->ij", basis[prefix + "sample_mass"], phi, phi, dtype=np.float64)
        mass_error = max(mass_error, float(np.abs(gram - np.eye(len(gram))).max()))
        t = basis[prefix + "coordinate_transform"]
        k = t.T @ basis[prefix + "raw_stiffness_matrix"] @ t
        stiffness_error = max(
            stiffness_error, float(np.linalg.norm(k - np.diag(basis[prefix + "mode_stiffness"])) / np.linalg.norm(k))
        )
    assert mass_error < 2e-6 and stiffness_error < 2e-6
    assert 0 <= meta["poster_frame"] < len(scene["times"])
    result = {
        "surface_recovery_all_frames_passed": True,
        "surface_recovery_max_abs_error_m": worst,
        "meshes": len(meta["meshes"]),
        "frames": len(scene["times"]),
        "mass_identity_max_error": mass_error,
        "stiffness_diagonalization_relative_error": stiffness_error,
    }
    if meta["slug"] == "glasses-case":
        recorded = metrics.get("strain_by_mesh", {})
        assert set(strain_by_mesh) == set(recorded), "Strain coverage omits deforming meshes"
        for name, measured in strain_by_mesh.items():
            for key, value in measured.items():
                np.testing.assert_allclose(value, recorded[name][key], rtol=2e-6, atol=1e-8)
            assert (
                max(measured[key] for key in ("max_abs_edge_strain", "max_abs_area_strain", "max_abs_principal_strain"))
                < 0.15
            )
            assert measured["reversed_surface_triangles"] == 0
        result.update(
            strain_coverage_passed=True, deforming_mesh_count=len(strain_by_mesh), strain_by_mesh=strain_by_mesh
        )
    (directory / "export_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def static_banana_diagnostic(directory, modes=32):
    """Solve the same quadratic energy to numerical convergence on CPU.

    The rigid frame is held at the bottom-joint rest frame. Retained attachments
    and the tip pin are included. Contact, inertia, and gravity are omitted to
    isolate the linear elastic energy; these are not simulation trajectories.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays, results = {}, []
    attachment_rows = np.repeat([8, 16, 24, 32, 40, 48, 56], 3)
    attachment_cols = np.tile([2, 8, 14], 7)
    for strip in range(3):
        p, faces, basis, origin, _, _, _ = banana_whole_strip(strip, modes)
        phi = basis.sample_phi.astype(np.float64)
        stiffness = basis.mode_stiffness.astype(np.float64)
        arrays[f"strip_{strip}_rest"] = p + origin
        arrays[f"strip_{strip}_faces"] = faces
        arrays[f"strip_{strip}_phi"] = phi
        arrays[f"strip_{strip}_stiffness"] = stiffness
        for front in (0.85, 0.65, 0.40, 0.06):
            points, _, mid, anchors = banana_strip_snapshot(strip, front)
            target = np.concatenate((points, mid[attachment_rows, attachment_cols], anchors)) - origin
            displacement = target - basis.sample_points
            fit_q = np.einsum("n,nmc,nc->m", basis.sample_mass, phi, displacement)
            retained = np.flatnonzero(attachment_rows / 64 < front) + len(p)
            samples = np.concatenate((retained, [len(phi) - 2, len(phi) - 1]))
            a = phi[samples].transpose(0, 2, 1).reshape(-1, modes)
            b = displacement[samples].ravel()
            # Exact least-energy equality solve, with explicit infeasibility diagnostics.
            c = a / np.sqrt(stiffness)[None]
            left, singular, right = np.linalg.svd(c, full_matrices=False)
            rank = int(np.count_nonzero(singular > singular[0] * 1e-11))
            row_space = right[:rank]
            y = row_space.T @ ((left[:, :rank].T @ b) / singular[:rank])
            exact_q = y / np.sqrt(stiffness)
            stationarity = np.linalg.norm(y - row_space.T @ (row_space @ y)) / max(np.linalg.norm(y), 1e-20)
            # Finite joint penalties match the example's nominal linear stiffness.
            penalty = 40000.0
            hessian = np.diag(stiffness) + penalty * a.T @ a
            rhs = penalty * a.T @ b
            finite_q = np.linalg.solve(hessian, rhs)
            finite_residual = np.linalg.norm(hessian @ finite_q - rhs) / max(np.linalg.norm(rhs), 1e-20)
            key = f"strip_{strip}_front_{front:.2f}"
            arrays[key + "constraint_matrix"] = a
            arrays[key + "constraint_target"] = b
            arrays[key + "training"] = points
            result = {
                "strip": strip,
                "front_s": front,
                "constraint_rank": int(rank),
                "constraint_rows": len(a),
                "constraint_smallest_retained_singular_value": float(singular[rank - 1]) if rank else 0,
                "training_geometry": strip_geometry_metrics(p, points - origin),
                "exact_stationarity_relative_residual": float(stationarity),
                "finite_penalty_relative_residual": float(finite_residual),
            }
            for name, q in (("POD_fit", fit_q), ("equality_minimum", exact_q), ("penalty_minimum", finite_q)):
                recovered = p + np.einsum("nmc,m->nc", phi[: len(p)], q)
                arrays[key + name + "_vertices"] = recovered + origin
                arrays[key + name + "_q"] = q
                errors = (a @ q - b).reshape(-1, 3)
                result[name] = {
                    **strip_geometry_metrics(p, recovered),
                    "elastic_energy_J": float(0.5 * np.sum(stiffness * q**2)),
                    "joint_penalty_energy_J": float(0.5 * penalty * np.sum(errors**2)),
                    "maximum_attachment_error_m": float(np.linalg.norm(errors, axis=1).max()),
                    "tip_error_m": float(np.linalg.norm(errors[-1])),
                }
            results.append(result)
    report = {
        "scope": "CPU static joint-only subproblem with fixed bottom frame; excludes contacts, gravity and inertia. The equality minimum and finite-penalty minimum are diagnostics, never assigned to Newton states.",
        "mode_count_per_strip": modes,
        "stationarity_definition": "Equality stationarity is the gradient projected onto the constraint nullspace in stiffness-whitened coordinates; equality feasibility is reported separately as attachment error.",
        "max_finite_penalty_relative_residual": max(r["finite_penalty_relative_residual"] for r in results),
        "max_equality_stationarity_relative_residual": max(r["exact_stationarity_relative_residual"] for r in results),
        "max_penalty_minimum_edge_strain": max(r["penalty_minimum"]["max_abs_edge_strain"] for r in results),
        "max_penalty_minimum_inverted_cells": max(r["penalty_minimum"]["inverted_surface_cells"] for r in results),
        "cases": results,
    }
    (directory / "static_quadratic_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(directory / "static_quadratic_diagnostic.npz", **arrays)
    assert report["max_finite_penalty_relative_residual"] < 1e-10
    assert report["max_equality_stationarity_relative_residual"] < 1e-7
    return report


class TestGeometry(unittest.TestCase):
    def test_all_triangle_surface_strain(self):
        triangle = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        offset = np.asarray([[0.2, 0.3, 0.2], [0.3, 0.2, -0.1]], dtype=np.float32)
        omega = np.array([0.3, 0.5, 0.7])
        trial = np.stack((np.tile([1, 2, 3], (3, 1)), np.cross(omega, triangle)), axis=1)
        embedded = skin_embedded_modes(offset, triangle, np.asarray([[0, 1, 2]], dtype=np.int32), trial)
        np.testing.assert_allclose(embedded[:, 0], np.tile([1, 2, 3], (2, 1)), atol=1e-7)
        np.testing.assert_allclose(embedded[:, 1], np.cross(omega, offset), atol=1e-7)
        points, faces = box((0.01, 0.02, 0.03))
        metrics = surface_strain_metrics(points, points * 1.04, faces)
        self.assertAlmostEqual(metrics["max_abs_edge_strain"], 0.04, places=6)
        self.assertAlmostEqual(metrics["max_abs_area_strain"], 1.04**2 - 1, places=6)
        self.assertAlmostEqual(metrics["max_abs_principal_strain"], 0.04, places=6)
        self.assertEqual(metrics["reversed_surface_triangles"], 0)
        anisotropic = surface_strain_metrics(points, points * [0.90, 1.08, 1.0], faces)
        self.assertAlmostEqual(anisotropic["max_principal_compression"], 0.10, places=6)
        self.assertAlmostEqual(anisotropic["max_principal_extension"], 0.08, places=6)
        reflected = points.copy()
        reflected[:, 0] *= -1
        self.assertGreater(surface_strain_metrics(points, reflected, faces)["reversed_surface_triangles"], 0)

    def test_primitives(self):
        for mesh in (
            box((0.1, 0.2, 0.3)),
            ellipsoid((0.03, 0.02, 0.01)),
            tube([[0, 0, 0], [0.01, 0, 0.02], [0.02, 0, 0.03]], 0.002),
        ):
            check_mesh(*mesh)

    def test_case_geometry(self):
        for mesh in (case_shell()[0], case_flap()[0], glasses_mesh()):
            check_mesh(*mesh)
        p, _ = case_shell()[0]
        np.testing.assert_allclose(np.ptp(p, axis=0)[:2], [0.166, 0.068], atol=0.003)

    def test_initial_case_clearance_regression(self):
        old = initial_case_clearance(0.063)
        self.assertGreater(old["initial_flap_vertices_inside_shell"], 0)
        self.assertFalse(old["initial_clearance_passed"])
        new = initial_case_clearance()
        self.assertEqual(new["initial_flap_vertices_inside_shell"], 0)
        self.assertTrue(new["initial_clearance_passed"])

    def test_rounded_pouch_geometry_and_clearance(self):
        for mesh in [pouch_shell()[0], pouch_flap()[0], *pouch_endcaps()]:
            check_mesh(*mesh)
        points, _ = pouch_shell()[0]
        extents = np.ptp(points, axis=0)
        self.assertGreater(float(extents[2]), 0.060)
        self.assertLess(float(extents[1]), 0.045)
        audit = pouch_clearance()
        self.assertTrue(audit["initial_clearance_passed"], audit)

    def test_coupled_trial_modes_preserve_mass_and_energy(self):
        rng = np.random.default_rng(12)
        points = rng.normal(size=(100, 3)) * 0.02
        phi = rng.normal(size=(100, 3, 3))
        phi[:, 1] += phi[:, 0] * 1.8
        basis = make_basis(points, phi, 0.025, [25, 50, 75])
        raw = basis.raw_mass_matrix
        normalized_raw = raw / np.sqrt(np.outer(np.diag(raw), np.diag(raw)))
        self.assertGreater(float(np.abs(normalized_raw - np.eye(3)).max()), 0.5)
        normalized = np.einsum("n,nic,njc->ij", basis.sample_mass, basis.sample_phi, basis.sample_phi)
        np.testing.assert_allclose(normalized, np.eye(3), atol=2e-6)
        t = basis.coordinate_transform
        np.testing.assert_allclose(
            t.T @ basis.raw_stiffness_matrix @ t, np.diag(basis.mode_stiffness), rtol=1e-6, atol=1e-5
        )
        q = np.array([0.001, -0.002, 0.003])
        np.testing.assert_allclose(
            np.einsum("nmc,m->nc", phi, t @ q), np.einsum("nmc,m->nc", basis.sample_phi, q), rtol=1e-6, atol=1e-8
        )
        self.assertAlmostEqual(
            float((t @ q) @ basis.raw_stiffness_matrix @ (t @ q)), float(np.sum(basis.mode_stiffness * q**2)), places=8
        )

    def test_whole_banana_geometry_and_basis(self):
        check_mesh(*banana_flesh())
        for strip in range(3):
            p, f, basis, origin, anchors, mass, _ = banana_whole_strip(strip)
            check_mesh(p, f)
            self.assertGreater(mass, 0)
            self.assertTrue(np.all(basis.mode_stiffness > 0))
            np.testing.assert_allclose(basis.sample_phi[-2], 0, atol=1e-8)
            np.testing.assert_allclose(basis.sample_points[-2:], anchors, atol=1e-8)
            gram = np.einsum("n,nic,njc->ij", basis.sample_mass, basis.sample_phi, basis.sample_phi, dtype=np.float64)
            np.testing.assert_allclose(gram, np.eye(24), atol=2e-6)
            self.assertGreater(basis.training_energy_fraction, 0.999)
            self.assertLess(basis.training_reconstruction_max_abs_error_m, 0.004)
            moved, _, _, _ = banana_strip_snapshot(strip, 0.06)
            self.assertTrue(np.isfinite(moved).all())
            self.assertEqual(strip_geometry_metrics(p, moved - origin)["inverted_surface_cells"], 0)

    def test_geometry_audit_detects_inversion_and_is_objective(self):
        p, _, _, _ = banana_strip_snapshot(0)
        theta = 2.4
        rotation = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        rigid = p @ rotation.T + np.array([0.1, 0.2, 0.3])
        audit = strip_geometry_metrics(p, rigid)
        self.assertLess(audit["max_abs_edge_strain"], 2e-5)
        self.assertEqual(audit["inverted_surface_cells"], 0)
        n = 65 * 17
        flipped = np.concatenate((p[n:], p[:n]))
        self.assertGreater(strip_geometry_metrics(p, flipped)["inverted_surface_cells"], 0)

    def test_progression_rejects_all_at_once(self):
        def records(progressive):
            result = []
            for t in np.arange(420) / 60:
                thresholds = 0.7 + np.arange(6, -1, -1) * 0.45 if progressive else np.full(7, 0.7)
                distance = np.tile(np.where(t > thresholds, 0.02, 0.0006), (3, 1))
                result.append(
                    {
                        "time": float(t),
                        "band_median_fruit_distance_m": distance,
                        "band_max_fruit_distance_m": distance,
                        "peel_front_s": float(1 - np.count_nonzero(distance[0] > 0.004) / 7),
                    }
                )
            return result

        self.assertTrue(peel_progression(records(True))["progressive_peel_passed"])
        self.assertFalse(peel_progression(records(False))["progressive_peel_passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--static-banana", type=Path)
    args, rest = parser.parse_known_args()
    if args.static_banana:
        result = static_banana_diagnostic(args.static_banana)
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    elif args.audit:
        print(json.dumps(audit_export(args.audit), indent=2))
    else:
        unittest.main(argv=[__file__, *rest])
