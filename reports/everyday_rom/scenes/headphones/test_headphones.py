# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU geometry checks and independent saved-state reconstruction audit."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from headphone_geometry import (
    CROWN_Z,
    CUP_X,
    CUP_Z,
    RADIUS,
    box,
    build_headband,
    capsule,
    cushion,
    ellipsoid,
    transform_points,
)

import newton


def assert_closed_mesh(vertices, faces):
    """Check finite, nondegenerate, consistently wound watertight geometry."""
    assert np.isfinite(vertices).all()
    assert faces.min() >= 0 and faces.max() < len(vertices)
    tri = vertices[faces].astype(float)
    assert np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).min() > 1e-12
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert np.all(counts == 2), "Mesh has an open or nonmanifold edge"
    signed = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    _, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    assert np.all(np.bincount(inverse, weights=signed) == 0), "Inconsistent face winding"
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    assert volume > 0, "Inward mesh"


class GeometryTests(unittest.TestCase):
    def test_closed_surfaces(self):
        """Reject holes, degenerate faces, and reversed procedural surfaces."""
        band_v, band_f, _, _ = build_headband()
        for vertices, faces in (
            (band_v, band_f),
            cushion(-1),
            cushion(1),
            ellipsoid((0.018, 0.034, 0.047)),
            capsule(0.0017, 0.006),
            box((0.008, 0.038, 0.032)),
        ):
            assert_closed_mesh(vertices, faces)

    def test_mass_and_inertia(self):
        """Match modal, rigid, and coupling inertia to the same sample masses."""
        _, _, basis, audit = build_headband()
        weights = basis.sample_mass.astype(float)
        phi = basis.sample_phi.astype(float)
        gram = np.einsum("n,nic,njc->ij", weights, phi, phi)
        np.testing.assert_allclose(gram, np.eye(2), atol=2e-6)
        np.testing.assert_allclose(basis.mode_mass, np.diag(gram), atol=2e-6)
        np.testing.assert_allclose(
            basis.mode_stiffness * np.array(audit["raw_modal_mass"]), audit["tip_stiffness_N_per_m"], rtol=1e-6
        )
        np.testing.assert_allclose(basis.mode_coupling_linear, np.einsum("n,nic->ic", weights, phi), atol=1e-7)
        self.assertAlmostEqual(float(weights.sum()), audit["mass_kg"], places=8)
        com = (weights[:, None] * basis.sample_points).sum(axis=0) / weights.sum()
        np.testing.assert_allclose(com, audit["com_m"], atol=1e-8)
        x = basis.sample_points - com
        inertia = (weights * (x * x).sum(axis=1)).sum() * np.eye(3) - np.einsum("n,ni,nj->ij", weights, x, x)
        np.testing.assert_allclose(inertia, audit["inertia_kg_m2"], atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(inertia).min(), 0)

    def test_crown_and_pivots(self):
        """Keep the crown fixed and sample both joint endpoints explicitly."""
        _, _, basis, _ = build_headband()
        crown = np.abs(basis.sample_points[:, 0]) < 1e-8
        np.testing.assert_allclose(basis.sample_phi[crown], 0, atol=1e-6)
        np.testing.assert_allclose(basis.sample_psi[crown], 0, atol=1e-6)
        for side in (-1, 1):
            endpoint = np.array([side * RADIUS, 0, -RADIUS])
            self.assertLess(np.linalg.norm(basis.sample_points - endpoint, axis=1).min(), 1e-7)

    def test_initial_clearance(self):
        """Start the actuators clear of the padded earcups and the floor."""
        for side in (-1, 1):
            v, _ = cushion(side)
            v += (side * CUP_X, 0, CUP_Z)
            self.assertGreater(float(np.min(side * v[:, 0])) - (0.040 + 0.008), 0.030)
            self.assertGreater(v[:, 2].min(), 0.18)
        self.assertAlmostEqual(CROWN_Z - RADIUS, CUP_Z + 0.051)

    def test_public_api_and_actuation(self):
        """Restrict imports to public Newton and prohibit direct state assignment."""
        source = Path(__file__).with_name("headphones.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith("newton._src"))
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.startswith("newton._src") for alias in node.names))
        self.assertNotIn("joint_q.assign", source)
        self.assertNotIn("joint_qd.assign", source)
        self.assertNotIn("body_q.assign", source)


def rotation_matrix(quat):
    """Independently recover an xyzw quaternion's rotation matrix."""
    x, y, z, w = np.asarray(quat, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def audit_export(directory):
    """Reconstruct every surface from original states, without render transforms."""
    directory = Path(directory)
    metadata = json.loads((directory / "scene.json").read_text())
    error = 0.0
    audited = 0
    bit_exact = True
    with (
        np.load(directory / "original_states.npz", allow_pickle=False) as states,
        np.load(directory / "scene.npz", allow_pickle=False) as scene,
    ):
        np.testing.assert_array_equal(scene["times"], states["times"])
        for key in ("body_q", "body_qd", "joint_q", "joint_qd", "sample_phi", "sample_psi"):
            assert np.isfinite(states[key]).all(), key
        phi = states["runtime_vertex_phi"]
        indices = states["runtime_vertex_sample"]
        np.testing.assert_array_equal(phi, states["sample_phi"][indices])
        np.testing.assert_array_equal(states["runtime_vertex_local"], states["sample_points"][indices])
        gram = np.einsum(
            "n,nic,njc->ij",
            states["sample_mass"].astype(float),
            states["sample_phi"].astype(float),
            states["sample_phi"].astype(float),
        )
        np.testing.assert_allclose(gram, np.eye(2), atol=2e-6)
        q0 = int(states["q_start"])
        for i, mesh in enumerate(metadata["meshes"]):
            rest = states[f"rest_vertices_{i}"]
            faces = scene[mesh["faces"]]
            assert_closed_mesh(rest, faces)
            np.testing.assert_array_equal(faces, states[f"faces_{i}"])
            body = int(states[f"body_{i}"])
            assert body == mesh["body"]
            actual = scene[mesh["vertices"]]
            if body < 0:
                np.testing.assert_array_equal(actual, rest)
                audited += len(rest)
                continue
            assert actual.shape == (len(states["times"]), len(rest), 3)
            for frame, body_q in enumerate(states["body_q"]):
                local = rest
                if mesh["elastic"]:
                    # Match float32 displacement accumulation in Newton's saved basis.
                    displacement = (phi * states["joint_q"][frame, q0 : q0 + 2][None, :, None]).sum(axis=1)
                    local = rest + displacement
                rotation = rotation_matrix(body_q[body, 3:])
                expected = (local.astype(float) @ rotation.T + body_q[body, :3]).astype(np.float32)
                error = max(error, float(np.max(np.abs(expected - actual[frame]))))
                assert error <= 6e-8, f"Mesh recovery mismatch {error:g} m"
                bit_exact = bit_exact and np.array_equal(transform_points(local, body_q[body]), actual[frame])
                audited += len(rest)
        assert error <= 6e-8, f"Mesh recovery mismatch {error:g} m"
        cups = states["cup_bodies"]
        dynamic = np.r_[states["band_body"].reshape(-1), cups, states["cable_bodies"], states["jack_body"].reshape(-1)]
        dynamic = dynamic[dynamic >= 0]
        assert np.all(states["body_mass"][dynamic] > 0)
        assert np.all((states["body_flags"][dynamic] & int(newton.BodyFlags.KINEMATIC)) == 0)
        support_joint = int(states["support_joint"])
        assert states["joint_parent"][support_joint] == states["support_body"]
        assert states["joint_child"][support_joint] == states["band_body"]
        assert states["joint_type"][support_joint] == int(newton.JointType.FIXED)
        enabled = states["support_enabled"]
        assert enabled[0] and not enabled[-1]
        assert np.count_nonzero(np.diff(enabled.astype(int))) == 1
        release = states["times"][~enabled][0]
        assert 11.19 < release < 11.24
        for joint, cup in zip(states["cup_joints"], cups, strict=True):
            assert states["joint_parent"][joint] == states["band_body"]
            assert states["joint_child"][joint] == cup
        jack = int(states["jack_body"])
        if jack >= 0:
            assert np.any((states["joint_child"] == jack) & np.isin(states["joint_parent"], states["cable_bodies"]))
            assert np.linalg.norm(states["body_q"][-1, jack, :3] - states["body_q"][0, jack, :3]) > 1e-4
    return {
        "passed": bool(error <= 6e-8 and bit_exact),
        "bit_exact_export_recovery": bool(bit_exact),
        "max_vertex_error_m": error,
        "audited_vertex_positions": audited,
        "mesh_count": len(metadata["meshes"]),
        "method": "all frames, independent quaternion-matrix reconstruction from original Newton states and exact sampled runtime basis",
        "tolerance_m": 6e-8,
    }


def audit_rejects_corruption(directory):
    """Require the independent source proof to reject a 1 mm vertex edit."""
    directory = Path(directory).resolve()
    with tempfile.TemporaryDirectory(prefix="audit-negative-", dir=directory) as temporary:
        target = Path(temporary)
        (target / "original_states.npz").symlink_to(directory / "original_states.npz")
        shutil.copy2(directory / "scene.json", target / "scene.json")
        with np.load(directory / "scene.npz", allow_pickle=False) as scene:
            arrays = {k: scene[k] for k in scene.files}
        arrays["vertices_0"][0, 0, 0] += 0.001
        np.savez_compressed(target / "scene.npz", **arrays)
        try:
            audit_export(target)
        except AssertionError:
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path)
    args, remaining = parser.parse_known_args()
    if args.audit:
        result = audit_export(args.audit)
        result["corrupted_vertex_rejected"] = audit_rejects_corruption(args.audit)
        assert result["passed"] and result["corrupted_vertex_rejected"]
        print(json.dumps(result, indent=2))
    else:
        unittest.main(argv=[__file__, *remaining])
