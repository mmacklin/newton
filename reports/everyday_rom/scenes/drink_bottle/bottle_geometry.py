# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Procedural hollow bottles and mass-orthogonal, localized wall modes."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

import newton


@dataclass(frozen=True)
class BottleSpec:
    slug: str
    title: str
    thickness: float
    density: float
    squeeze_z: float
    fixed_z: float
    neck_z: float
    frequency_hz: float
    color: tuple[float, float, float]
    collar_color: tuple[float, float, float]
    tip_color: tuple[float, float, float]


SPECS = {
    "drink-bottle": BottleSpec(
        "drink-bottle",
        "Vented sport drink bottle",
        0.0008,
        950.0,
        0.095,
        0.017,
        0.178,
        18.0,
        (0.045, 0.59, 0.64),
        (0.035, 0.17, 0.26),
        (0.15, 0.19, 0.21),
    ),
    "mustard-bottle": BottleSpec(
        "mustard-bottle",
        "Empty mustard squeeze bottle",
        0.0007,
        930.0,
        0.076,
        0.014,
        0.143,
        14.0,
        (0.98, 0.69, 0.025),
        (0.76, 0.045, 0.025),
        (1.0, 0.75, 0.045),
    ),
}


def _smooth_profile(z, knots, values):
    i = np.clip(np.searchsorted(knots, z, side="right") - 1, 0, len(knots) - 2)
    t = (z - np.asarray(knots)[i]) / (np.asarray(knots)[i + 1] - np.asarray(knots)[i])
    t = np.clip(t, 0, 1)
    t = t * t * (3 - 2 * t)
    values = np.asarray(values)
    return values[i] * (1 - t) + values[i + 1] * t


def build_shell(spec: BottleSpec, segments: int = 64):
    """Return a watertight material shell with a genuinely open central vent.

    Wall thickness is a radial geometric offset, not a nonlinear shell law.
    The bottom is closed; the top annulus joins inner and outer walls without
    capping the cavity. Units are metres.
    """
    if spec.slug == "drink-bottle":
        knots = [0, 0.006, 0.016, 0.04, 0.07, 0.095, 0.13, 0.156, 0.168, 0.190, 0.194, 0.208, 0.212, 0.215, 0.229]
        rx = [
            0.030,
            0.034,
            0.0355,
            0.036,
            0.0347,
            0.0338,
            0.0347,
            0.036,
            0.034,
            0.017,
            0.016,
            0.016,
            0.013,
            0.008,
            0.008,
        ]
        ry = rx
        collar, nozzle = 0.192, 0.213
    else:
        knots = [0, 0.005, 0.015, 0.035, 0.07, 0.105, 0.125, 0.137, 0.150, 0.154, 0.161, 0.164, 0.177, 0.196, 0.203]
        rx = [0.029, 0.034, 0.038, 0.040, 0.040, 0.039, 0.036, 0.031, 0.018, 0.017, 0.017, 0.013, 0.008, 0.0038, 0.0035]
        ry = [0.021, 0.025, 0.028, 0.029, 0.029, 0.028, 0.027, 0.024, 0.018, 0.017, 0.017, 0.013, 0.008, 0.0038, 0.0035]
        collar, nozzle = 0.151, 0.163
    z = np.unique(np.r_[np.linspace(0, knots[-1], 72), knots])
    theta = np.arange(segments) * (2 * np.pi / segments)
    c, s = np.cos(theta), np.sin(theta)
    a, b = _smooth_profile(z, knots, rx), _smooth_profile(z, knots, ry)
    exponent = np.ones(len(z))
    if spec.slug == "mustard-bottle":
        exponent = 0.82 + 0.18 * np.clip((z - 0.125) / 0.025, 0, 1)
    cx = np.sign(c)[None, :] * np.abs(c)[None, :] ** exponent[:, None]
    sy = np.sign(s)[None, :] * np.abs(s)[None, :] ** exponent[:, None]
    rib = np.zeros((len(z), segments))
    if spec.slug == "drink-bottle":
        for center in [0.025, 0.035, 0.045, 0.137, 0.147, 0.157]:
            rib += 0.0009 * np.exp(-(((z[:, None] - center) / 0.0019) ** 2))
        grip = np.clip(np.sin(np.pi * np.clip((z - 0.05) / 0.08, 0, 1)), 0, 1) ** 2
        rib += 0.00024 * grip[:, None] * np.cos(16 * theta)[None, :]
    else:
        for center in [0.018, 0.128, 0.156, 0.159]:
            rib += 0.0005 * np.exp(-(((z[:, None] - center) / 0.0015) ** 2))
        rib += 0.00025 * ((z > 0.151) & (z < 0.163))[:, None] * np.cos(32 * theta)[None, :]
    outer = np.stack(
        [(a[:, None] + rib) * cx, (b[:, None] + rib) * sy, np.broadcast_to(z[:, None], rib.shape)], axis=-1
    )
    inner = outer.copy()
    inner[:, :, 0] -= spec.thickness * cx
    inner[:, :, 1] -= spec.thickness * sy
    inner[0, :, 2] = spec.thickness
    count = len(z) * segments
    vertices = np.concatenate([outer.reshape(-1, 3), inner.reshape(-1, 3), [[0, 0, 0], [0, 0, spec.thickness]]])
    faces = []
    for j in range(len(z) - 1):
        for i in range(segments):
            a0, b0 = j * segments + i, j * segments + (i + 1) % segments
            a1, b1 = a0 + segments, b0 + segments
            faces.extend(
                [(a0, b0, b1), (a0, b1, a1), (a0 + count, b1 + count, b0 + count), (a0 + count, a1 + count, b1 + count)]
            )
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend([(2 * count, j, i), (2 * count + 1, count + i, count + j)])
        a0, b0 = count - segments + i, count - segments + j
        faces.extend([(a0, b0, b0 + count), (a0, b0 + count, a0 + count)])
    faces = np.asarray(faces, dtype=np.int32)
    # Whole ring bands keep neighboring triangles of each quad together.
    # This is a material-only split; collision and recovery vertices stay exact.
    ring_min_z = vertices[faces, 2].min(axis=1)
    materials = np.zeros(len(faces), dtype=np.int32)
    materials[ring_min_z >= z[np.argmin(np.abs(z - collar))]] = 1
    materials[ring_min_z >= z[np.argmin(np.abs(z - nozzle))]] = 2
    return {
        "vertices": vertices.astype(np.float32),
        "faces": faces,
        "materials": materials,
        "ring_count": len(z),
        "segments": segments,
        "theta": np.r_[np.tile(theta, len(z)), np.tile(theta, len(z)), 0, 0],
        "outer_count": count,
    }


def cavity_volume(vertices, shell):
    """Estimate empty cavity volume [m^3] by polygon sections, closed at the vent."""
    n, rings, segments = shell["outer_count"], shell["ring_count"], shell["segments"]
    p = np.asarray(vertices)[n : 2 * n].reshape(rings, segments, 3)
    area = 0.5 * np.sum(
        p[:, :, 0] * np.roll(p[:, :, 1], -1, axis=1) - p[:, :, 1] * np.roll(p[:, :, 0], -1, axis=1), axis=1
    )
    dz = np.diff(p[:, :, 2].mean(axis=1))
    return float(np.sum(0.5 * (area[:-1] + area[1:]) * dz))


def signed_volume(vertices, faces):
    """Signed volume enclosed by the oriented material skin [m^3]."""
    p = np.asarray(vertices, dtype=np.float64)[faces]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6)


def ovalization_fields(shell, spec):
    """Four compensated ring-bending fields, before mass orthogonalization.

    The circular seed has u_r=-cos(2 theta), u_t=sin(2 theta)/2.
    Projecting its ring-edge derivatives removes first-order circumferential
    extension on the actual noncircular mid-wall polygons. An axial correction
    cancels the cylindrical seed's first-order circumferential/axial shear.
    This remains a linear approximation; finite triangle stretches are audited.
    """
    rings, segments, n = shell["ring_count"], shell["segments"], shell["outer_count"]
    vertices = shell["vertices"].astype(float)
    mid = 0.5 * (vertices[:n] + vertices[n : 2 * n]).reshape(rings, segments, 3)
    z = vertices[:n].reshape(rings, segments, 3)[:, 0, 2]
    theta = np.arange(segments) * 2 * np.pi / segments
    radial = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    tangent = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    length = spec.neck_z - spec.fixed_z
    t = np.clip((z - spec.fixed_z) / length, 0, 1)
    envelope = np.sin(np.pi * t) ** 2
    derivative = np.pi / length * np.sin(2 * np.pi * t)
    fixed = (t == 0) | (t == 1)
    envelope[fixed], derivative[fixed] = 0, 0
    profiles = ((envelope, derivative), (envelope * (2 * t - 1), derivative * (2 * t - 1) + 2 * envelope / length))
    radius = float(np.linalg.norm(mid[np.argmin(abs(z - spec.squeeze_z)), :, :2], axis=1).mean())
    raw_ring = np.zeros((rings, segments, 4, 3))
    for ring in range(rings):
        edge = np.roll(mid[ring, :, :2], -1, axis=0) - mid[ring, :, :2]
        unit = edge / np.linalg.norm(edge, axis=1)[:, None]
        constraints = np.zeros((segments, 2 * segments))
        for i in range(segments):
            j = (i + 1) % segments
            constraints[i, 2 * i : 2 * i + 2] = -unit[i]
            constraints[i, 2 * j : 2 * j + 2] = unit[i]
        for phase in range(2):
            angle = 2 * theta - phase * np.pi / 2
            seed = -np.cos(angle)[:, None] * radial + 0.5 * np.sin(angle)[:, None] * tangent
            correction = constraints.T @ np.linalg.solve(constraints @ constraints.T, constraints @ seed.ravel())
            bending = seed - correction.reshape(segments, 2)
            for j, (profile, gradient) in enumerate(profiles):
                mode = 2 * j + phase
                raw_ring[ring, :, mode, :2] = profile[ring] * bending
                raw_ring[ring, :, mode, 2] = radius * gradient[ring] * np.cos(angle) / 4
    raw = np.zeros((len(vertices), 4, 3))
    raw[:n] = raw_ring.reshape(n, 4, 3)
    raw[n : 2 * n] = raw[:n]
    return raw


def triangle_reference(vertices, faces):
    """Return inverse reference tangent maps and reference triangle areas [m^2]."""
    tri = np.asarray(vertices, dtype=float)[faces]
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    length = np.linalg.norm(a, axis=1)
    projection = np.einsum("ij,ij->i", a, b) / length
    height = np.linalg.norm(np.cross(a, b), axis=1) / length
    if np.any(height <= 0):
        raise ValueError("degenerate reference triangle")
    inverse = np.zeros((len(faces), 2, 2))
    inverse[:, 0, 0] = 1 / length
    inverse[:, 0, 1] = -projection / (length * height)
    inverse[:, 1, 1] = 1 / height
    return inverse, 0.5 * length * height


def triangle_stretches(vertices, faces, reference):
    """Singular values of the 3x2 surface deformation gradient, all triangles."""
    inverse, _ = reference
    tri = np.asarray(vertices, dtype=float)[faces]
    deformed = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]], axis=-1)
    gradient = deformed @ inverse
    gram = np.swapaxes(gradient, 1, 2) @ gradient
    return np.sqrt(np.maximum(np.linalg.eigvalsh(gram), 0))


def build_bottle_rom(spec: BottleSpec, damping_ratio: float = 0.20):
    shell = build_shell(spec)
    v, f = shell["vertices"], shell["faces"]
    raw = ovalization_fields(shell, spec)
    count = raw.shape[1]
    tri = v[f].astype(float)
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    sample_mass = np.zeros(len(v))
    # Both skins are sampled, so each receives half the thin-wall areal mass.
    np.add.at(sample_mass, f.ravel(), np.repeat(area * spec.thickness * spec.density / 6, 3))
    center = np.average(v.astype(float), weights=sample_mass, axis=0)
    centered = v - center
    inertia = np.eye(3) * np.sum(sample_mass * np.sum(centered**2, axis=1))
    inertia -= np.einsum("v,vi,vj->ij", sample_mass, centered, centered)
    p = raw.transpose(0, 2, 1).reshape(-1, count)
    weights = np.repeat(sample_mass, 3)
    gram = p.T @ (weights[:, None] * p)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues.min() <= 0:
        raise ValueError("localized modes are not independent")
    p = p @ (eigenvectors / np.sqrt(eigenvalues)[None, :])
    scale = np.linalg.norm(p.reshape(-1, 3, count), axis=1).max(axis=0)
    p /= scale[None, :]
    mass = 1 / scale**2
    phi = p.reshape(-1, 3, count).transpose(0, 2, 1).astype(np.float32)
    stiffness = (2 * np.pi * spec.frequency_hz) ** 2 * mass
    damping = 2 * damping_ratio * np.sqrt(mass * stiffness)
    basis = newton.ModalBasis(
        sample_points=v,
        sample_phi=phi,
        sample_mass=sample_mass,
        mode_mass=mass,
        mode_stiffness=stiffness,
        mode_damping=damping,
        label=f"{spec.slug}_four_compensated_ovalization_modes",
    )
    audit = {
        "method": "Four analytic ovalization fields with polygon-ring tangent compensation and axial shear correction, mass-orthogonalized; not FEM modes",
        "mode_count": count,
        "thickness_m": spec.thickness,
        "assumed_density_kg_m3": spec.density,
        "wall_mass_kg": float(sample_mass.sum()),
        "center_of_mass_local_m": center.tolist(),
        "inertia_about_com_kg_m2": inertia.tolist(),
        "mass_model": "Thin-wall surface quadrature, shared by rigid inertia and floating-frame/modal coupling",
        "effective_mode_frequency_hz": spec.frequency_hz,
        "damping_ratio": damping_ratio,
        "stiffness_model": "K_i = (2*pi*f)^2 M_i; phenomenological, not a calibrated constitutive law",
        "mode_mass_kg": mass.tolist(),
        "mode_stiffness_N_m": stiffness.tolist(),
        "neutral_cavity_volume_ml": cavity_volume(v, shell) * 1e6,
        "neutral_material_skin_volume_ml": signed_volume(v, f) * 1e6,
        "mass_orthogonality_max_error_kg": float(np.max(np.abs(p.T @ (weights[:, None] * p) - np.diag(mass)))),
        "locally_rigid_base_below_z_m": spec.fixed_z,
        "locally_rigid_neck_above_z_m": spec.neck_z,
        "surface_vertices": len(v),
        "triangles": len(f),
    }
    return shell, basis, audit


def capsule_mesh(radius: float, half_height: float, segments: int = 48):
    """Closed Z-axis capsule matching Newton's analytic contact shape [m]."""
    points = [[0, 0, -half_height - radius]]
    rings = []
    for angle in np.linspace(-np.pi / 2, np.pi / 2, 25)[1:-1]:
        center = -half_height if angle < 0 else half_height
        # Two equators keep the straight cylindrical middle exact.
        centers = (-half_height, half_height) if abs(angle) < 1e-10 else (center,)
        for center in centers:
            ring = []
            for theta in np.arange(segments) * 2 * np.pi / segments:
                ring.append(len(points))
                points.append(
                    [
                        radius * np.cos(angle) * np.cos(theta),
                        radius * np.cos(angle) * np.sin(theta),
                        center + radius * np.sin(angle),
                    ]
                )
            rings.append(ring)
    top = len(points)
    points.append([0, 0, half_height + radius])
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend([(0, rings[0][j], rings[0][i]), (top, rings[-1][i], rings[-1][j])])
    for lower, upper in pairwise(rings):
        for i in range(segments):
            j = (i + 1) % segments
            faces.extend([(lower[i], lower[j], upper[j]), (lower[i], upper[j], upper[i])])
    return np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def box_mesh(half_extents):
    h = np.asarray(half_extents)
    vertices = (
        np.asarray(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            dtype=np.float32,
        )
        * h
    )
    faces = np.asarray(
        [
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
        ],
        dtype=np.int32,
    )
    return vertices.astype(np.float32), faces
