# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Procedural headphone surfaces and a crown-clamped, mass-orthogonal ROM."""

from __future__ import annotations

import numpy as np

import newton

RADIUS = 0.108
CROWN_Z = 0.400
CUP_Z = CROWN_Z - RADIUS - 0.051
CUP_X = RADIUS


def orient(vertices, faces):
    """Return outward-oriented, float32 vertices and int32 triangles."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    tri = vertices[faces].astype(np.float64)
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    if volume < 0:
        faces = faces[:, [0, 2, 1]]
    return vertices, faces


def ring_surface(rings, *, closed=False):
    """Triangulate rings, with nondegenerate end caps for an open sweep."""
    rings = np.asarray(rings)
    n, k, _ = rings.shape
    vertices = list(rings.reshape(-1, 3))
    faces = []
    for j in range(n if closed else n - 1):
        for i in range(k):
            a, b = j * k + i, j * k + (i + 1) % k
            c, d = ((j + 1) % n) * k + (i + 1) % k, ((j + 1) % n) * k + i
            faces.extend(((a, b, c), (a, c, d)))
    if not closed:
        for j in (0, n - 1):
            center = len(vertices)
            vertices.append(rings[j].mean(axis=0))
            for i in range(k):
                a, b = j * k + i, j * k + (i + 1) % k
                faces.append((center, b, a) if j == 0 else (center, a, b))
    return orient(vertices, faces)


def ellipsoid(radii, *, center=(0, 0, 0), segments=48, rings=20):
    """Build an ellipsoid without coincident polar vertices [m]."""
    angle = np.arange(segments) * 2 * np.pi / segments
    vertices = [[0, 0, -1]]
    for latitude in np.linspace(-np.pi / 2, np.pi / 2, rings + 1)[1:-1]:
        vertices.extend(
            np.column_stack(
                (
                    np.cos(latitude) * np.cos(angle),
                    np.cos(latitude) * np.sin(angle),
                    np.full(segments, np.sin(latitude)),
                )
            )
        )
    vertices.append([0, 0, 1])
    faces = []
    for i in range(segments):
        faces.append((0, 1 + (i + 1) % segments, 1 + i))
    for j in range(rings - 2):
        for i in range(segments):
            a, b = 1 + j * segments + i, 1 + j * segments + (i + 1) % segments
            faces.extend(((a, b, b + segments), (a, b + segments, a + segments)))
    for i in range(segments):
        a = 1 + (rings - 2) * segments + i
        b = 1 + (rings - 2) * segments + (i + 1) % segments
        faces.append((a, b, len(vertices) - 1))
    return orient(np.asarray(vertices) * np.asarray(radii) + center, faces)


def cushion(side):
    """Build a closed oval ear cushion with a genuine open center [m]."""
    a = np.arange(56) * 2 * np.pi / 56
    b = np.arange(16) * 2 * np.pi / 16
    rings = []
    for theta in a:
        rings.append(
            np.column_stack(
                (
                    np.full(len(b), -side * 0.015) + 0.006 * np.sin(b),
                    (0.026 + 0.006 * np.cos(b)) * np.cos(theta),
                    (0.038 + 0.006 * np.cos(b)) * np.sin(theta),
                )
            )
        )
    return ring_surface(rings, closed=True)


def box(half_extents, *, center=(0, 0, 0)):
    """Build a box matching the collision primitive [m]."""
    vertices = np.array(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
    )
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    return orient(vertices * half_extents + center, faces)


def capsule(radius, half_height, *, segments=16, hemisphere_rings=5):
    """Build the exact capsule surface sampled along local Z [m]."""
    # Split the equatorial ring, so the cylinder has its full authored length.
    a = np.arange(segments) * 2 * np.pi / segments
    latitudes = np.r_[
        np.linspace(-np.pi / 2, 0, hemisphere_rings + 1)[1:], np.linspace(0, np.pi / 2, hemisphere_rings + 1)[:-1]
    ]
    rings = []
    for j, latitude in enumerate(latitudes):
        z = (-half_height if j < hemisphere_rings else half_height) + radius * np.sin(latitude)
        rings.append(
            np.column_stack(
                (radius * np.cos(latitude) * np.cos(a), radius * np.cos(latitude) * np.sin(a), np.full(segments, z))
            )
        )
    vertices, faces = ring_surface(rings)
    vertices[-2, 2] = -half_height - radius
    vertices[-1, 2] = half_height + radius
    return vertices, faces


def transform_points(vertices, pose):
    """Recover world vertices from an xyzw Newton body pose [m]."""
    v = np.asarray(vertices, dtype=np.float64)
    q = np.asarray(pose[3:], dtype=np.float64)
    uv = np.cross(q[:3], v)
    return (v + 2 * (q[3] * uv + np.cross(q[:3], uv)) + pose[:3]).astype(np.float32)


def headband_fields(points):
    """Sample two curved-beam tip-load fields and cross-section rotations."""
    p = np.asarray(points, dtype=float)
    theta = np.arctan2(np.abs(p[:, 0]), p[:, 2] + RADIUS)
    sign = np.sign(p[:, 0])
    center = np.column_stack((sign * RADIUS * np.sin(theta), np.zeros(len(p)), RADIUS * (np.cos(theta) - 1)))
    beta = -sign * 4 * np.sin(theta) / (np.pi * RADIUS)
    offset = p - center
    displacement = np.column_stack(
        (sign * (2 * theta - np.sin(2 * theta)) / np.pi, np.zeros(len(p)), 2 * np.sin(theta) ** 2 / np.pi)
    )
    displacement[:, 0] += beta * offset[:, 2]
    displacement[:, 2] -= beta * offset[:, 0]
    phi = np.zeros((len(p), 2, 3))
    psi = np.zeros_like(phi)
    for mode, side in enumerate((-1, 1)):
        mask = sign == side
        phi[mask, mode] = displacement[mask]
        psi[mask, mode, 1] = beta[mask]
    return phi, psi


def build_headband(*, mass=0.065, tip_stiffness=240.0, damping_ratio=1.2):
    """Build a clamped arch with consistent sampled inertia and normalized modes."""
    theta = np.linspace(-np.pi / 2, np.pi / 2, 81)
    a = np.arange(16) * 2 * np.pi / 16
    rings = []
    for t in theta:
        radial = RADIUS + 0.0028 * np.cos(a)
        rings.append(np.column_stack((radial * np.sin(t), 0.014 * np.sin(a), radial * np.cos(t) - RADIUS)))
    vertices, faces = ring_surface(rings)
    endpoints = np.array([[-RADIUS, 0, -RADIUS], [RADIUS, 0, -RADIUS], [0, 0, 0]])
    samples = np.vstack((vertices, endpoints)).astype(np.float32)
    # Arc-length trapezoidal mass lumping; cap/attachment samples have zero mass.
    station_mass = np.ones(len(theta))
    station_mass[[0, -1]] = 0.5
    station_mass *= mass / station_mass.sum()
    weights = np.r_[np.repeat(station_mass / len(a), len(a)), np.zeros(5)]
    raw, rotations = headband_fields(samples)
    gram = np.einsum("n,nic,njc->ij", weights, raw, raw)
    scales = 1 / np.sqrt(np.diag(gram))
    phi = raw * scales[None, :, None]
    psi = rotations * scales[None, :, None]
    stiffness = tip_stiffness * scales**2
    basis = newton.ModalBasis(
        sample_points=samples,
        sample_phi=phi,
        sample_psi=psi,
        sample_mass=weights,
        mode_stiffness=stiffness,
        mode_damping=2 * damping_ratio * np.sqrt(stiffness),
        label="crown_clamped_two_side_bending",
    )
    com = np.sum(weights[:, None] * samples, axis=0) / mass
    centered = samples - com
    inertia = np.einsum("n,n->", weights, np.sum(centered**2, axis=1)) * np.eye(3)
    inertia -= np.einsum("n,ni,nj->ij", weights, centered, centered)
    normalized_gram = np.einsum("n,nic,njc->ij", basis.sample_mass, basis.sample_phi, basis.sample_phi)
    audit = {
        "mass_kg": float(weights.sum()),
        "mass_gram": normalized_gram.tolist(),
        "mass_orthogonality_error": float(np.abs(normalized_gram - np.eye(2)).max()),
        "tip_stiffness_N_per_m": tip_stiffness,
        "mode_stiffness": stiffness.tolist(),
        "mode_mass": basis.mode_mass.tolist(),
        "com_m": com.tolist(),
        "inertia_kg_m2": inertia.tolist(),
        "raw_modal_mass": np.diag(gram).tolist(),
        "sample_mass_model": "arc-length quadrature, uniform cross-section surface samples",
    }
    return vertices, faces, basis, audit
