# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Procedural meshes and shared state-faithful capture for two bounded ROM demos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp
from capacity_audit import CapacityAudit

import newton
import newton.solvers


@dataclass
class Config:
    substeps: int = 4
    iterations: int = 16
    contact_enabled: bool = True
    contact_ke: float = 3.0e3
    contact_kd: float = 0.0
    friction: float = 0.10
    contact_gap: float = 0.00015
    elastic_relaxation: float = 0.3


def shape_config(cfg: Config, collision: bool = True):
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=cfg.contact_ke,
        kd=cfg.contact_kd,
        mu=cfg.friction,
        margin=0.0,
        gap=cfg.contact_gap,
        has_shape_collision=collision,
        has_particle_collision=False,
    )


def join_meshes(parts):
    points, faces, offset = [], [], 0
    for p, f in parts:
        points.append(p)
        faces.append(f + offset)
        offset += len(p)
    return np.concatenate(points).astype(np.float32), np.concatenate(faces).astype(np.int32)


def loft(centers, radii, segments=24, axis=1, cap=True):
    """Closed elliptical ring loft with deterministic outward winding."""
    centers, radii = np.asarray(centers), np.asarray(radii)
    transverse = [k for k in range(3) if k != axis]
    a = np.arange(segments) * (2 * np.pi / segments)
    vertices = np.repeat(centers[:, None, :], segments, axis=1)
    vertices[:, :, transverse[0]] += radii[:, 0, None] * np.cos(a)
    vertices[:, :, transverse[1]] += radii[:, 1, None] * np.sin(a)
    vertices = vertices.reshape(-1, 3).tolist()
    faces = []
    for i in range(len(centers) - 1):
        for j in range(segments):
            a0, b0 = i * segments + j, i * segments + (j + 1) % segments
            faces.extend([(a0, b0, a0 + segments), (b0, b0 + segments, a0 + segments)])
    if cap:
        vertices.extend([centers[0].tolist(), centers[-1].tolist()])
        n = len(centers) * segments
        for j in range(segments):
            faces.extend(
                [
                    (n, (j + 1) % segments, j),
                    (n + 1, (len(centers) - 1) * segments + j, (len(centers) - 1) * segments + (j + 1) % segments),
                ]
            )
    p, f = np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)
    if axis == 1:
        f = f[:, ::-1]
    return p, f


def ellipsoid(radii, center=(0, 0, 0), rings=24, segments=48):
    """Smooth closed ellipsoid without duplicate pole vertices."""
    p = [(0, 0, radii[2]), (0, 0, -radii[2])]
    for i in range(1, rings):
        t = np.pi * i / rings
        for j in range(segments):
            a = 2 * np.pi * j / segments
            p.append((radii[0] * np.sin(t) * np.cos(a), radii[1] * np.sin(t) * np.sin(a), radii[2] * np.cos(t)))
    f = []
    for j in range(segments):
        k = (j + 1) % segments
        f.extend([(0, 2 + j, 2 + k), (1, 2 + (rings - 2) * segments + k, 2 + (rings - 2) * segments + j)])
    for i in range(rings - 2):
        for j in range(segments):
            a, b = 2 + i * segments + j, 2 + i * segments + (j + 1) % segments
            f.extend([(a, a + segments, b), (b, a + segments, b + segments)])
    return np.asarray(p, dtype=np.float32) + np.asarray(center, dtype=np.float32), np.asarray(f, dtype=np.int32)


def ellipse_ring(center, radii, tube, segments=96, sides=12):
    p, f = [], []
    for i in range(segments):
        a = 2 * np.pi * i / segments
        for j in range(sides):
            b = 2 * np.pi * j / sides
            p.append(
                (
                    center[0] + (radii[0] + tube * np.cos(b)) * np.cos(a),
                    center[1] + tube * np.sin(b),
                    center[2] + (radii[1] + tube * np.cos(b)) * np.sin(a),
                )
            )
    for i in range(segments):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = ((i + 1) % segments) * sides + j, ((i + 1) % segments) * sides + (j + 1) % sides
            f.extend([(a, b, c), (b, d, c)])
    return np.asarray(p, dtype=np.float32), np.asarray(f, dtype=np.int32)


def capsule_y(radius, half_length):
    """Rounded Y-axis tool surface; tiny closed pole disks avoid zero-area faces."""
    a = np.linspace(-np.pi / 2 + 0.015, 0, 13)
    b = np.linspace(0, np.pi / 2 - 0.015, 13)
    y = np.r_[-half_length + radius * np.sin(a), half_length + radius * np.sin(b)]
    r = radius * np.cos(np.r_[a, b])
    return loft(np.column_stack((np.zeros_like(y), y, np.zeros_like(y))), np.column_stack((r, r)), segments=48)


def surface_mass(vertices, faces, total_mass):
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    weights = np.zeros(len(vertices))
    np.add.at(weights, faces.ravel(), np.repeat(area / 3, 3))
    return weights * (total_mass / weights.sum())


def sampled_mass_properties(points, weights):
    """Mass, COM and unregularized inertia from one consistent lumped distribution."""
    points, weights = np.asarray(points, dtype=float), np.asarray(weights, dtype=float)
    mass = float(weights.sum())
    com = np.average(points, axis=0, weights=weights)
    centered = points - com
    inertia = np.eye(3) * np.sum(weights * np.sum(centered**2, axis=1))
    inertia -= np.einsum("v,vi,vj->ij", weights, centered, centered)
    return mass, com, inertia


def coupled_mass_audit(mass, com, inertia, mode_mass, linear, angular):
    """Audit the complete COM-frame mass block, including Newton's acceptance test."""
    frame = np.zeros((6, 6))
    frame[:3, :3] = mass * np.eye(3)
    frame[3:, 3:] = inertia
    cross = np.concatenate([np.asarray(linear).T, -(np.asarray(angular) + np.cross(com, linear)).T])
    matrix = np.block([[frame, cross], [cross.T, np.diag(mode_mass)]])
    eigenvalues = np.linalg.eigvalsh(matrix)
    scaled = matrix / np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))
    normalized = np.linalg.eigvalsh(scaled)
    threshold = 1e-8 * max(float(np.abs(np.diag(matrix)).max()), 1.0)
    np.linalg.cholesky(matrix)
    return {
        "dimension": len(matrix),
        "raw_min_eigenvalue": float(eigenvalues[0]),
        "raw_max_eigenvalue": float(eigenvalues[-1]),
        "raw_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "normalized_min_eigenvalue": float(normalized[0]),
        "normalized_condition_number": float(normalized[-1] / normalized[0]),
        "runtime_spd_threshold": threshold,
        "runtime_spd_passed": bool(eigenvalues[0] > threshold),
        "cholesky_passed": True,
        "normalization": "D^-1/2 M D^-1/2, D=diag(M); raw condition is coordinate-unit dependent",
    }


def brush_tuft(height=0.012):
    """Seven grouped nylon strands; two Euler-Bernoulli tip-normalized modes."""
    parts = []
    strand_xy = [(0.0, 0.0)] + [(0.00055 * np.cos(a), 0.00055 * np.sin(a)) for a in np.arange(6) * np.pi / 3]
    for dx, dy in strand_xy:
        z = np.linspace(0, height, 9)
        centers = np.column_stack((np.full_like(z, dx), np.full_like(z, dy), z))
        parts.append(loft(centers, np.tile([0.00019, 0.00019], (len(z), 1)), segments=8, axis=2))
    p, f = join_meshes(parts)
    s = p[:, 2] / height
    bend = s * s * (3 - s) / 2
    slope = (3 * s - 1.5 * s * s) / height
    phi = np.zeros((len(p), 2, 3), dtype=np.float32)
    phi[:, 0, 0], phi[:, 1, 1] = bend, bend
    # The section rotates about the tuft center; all seven strands share a beam.
    phi[:, 0, 2], phi[:, 1, 2] = -p[:, 0] * slope, -p[:, 1] * slope
    density, young = 1150.0, 1.3e9
    mass = 7 * density * np.pi * 0.00019**2 * height
    weights = surface_mass(p, f, mass)
    modal_mass = np.einsum("v,vmc,vmc->m", weights, phi, phi)
    stiffness = np.full(2, 7 * 3 * young * np.pi * 0.00019**4 / (4 * height**3))
    basis = newton.ModalBasis(
        sample_points=p,
        sample_phi=phi,
        sample_mass=weights,
        mode_mass=modal_mass,
        mode_stiffness=stiffness,
        mode_damping=0.9 * np.sqrt(modal_mass * stiffness),
        label="grouped_brush_beam",
    )
    return p, f, basis


def brush_handle(split_head=False):
    y = np.array(
        [
            -0.160,
            -0.158,
            -0.151,
            -0.134,
            -0.115,
            -0.090,
            -0.067,
            -0.049,
            -0.033,
            -0.022,
            -0.014,
            -0.009,
            -0.004,
            0.008,
            0.023,
            0.030,
            0.033,
        ]
    )
    rx = np.array(
        [
            0.001,
            0.004,
            0.007,
            0.008,
            0.0075,
            0.0065,
            0.0055,
            0.0044,
            0.0033,
            0.0029,
            0.0035,
            0.0055,
            0.0075,
            0.008,
            0.0075,
            0.0048,
            0.001,
        ]
    )
    rz = np.array(
        [
            0.001,
            0.002,
            0.003,
            0.0032,
            0.0030,
            0.0028,
            0.0027,
            0.0025,
            0.0025,
            0.0025,
            0.0025,
            0.0028,
            0.0032,
            0.0035,
            0.0032,
            0.0026,
            0.001,
        ]
    )
    centers = np.column_stack((np.zeros_like(y), y, 0.010 + 0.0008 * np.cos((y + 0.04) * 22)))
    handle = loft(centers, np.column_stack((rx, rz)), segments=40)
    grip = ellipsoid((0.0064, 0.043, 0.0015), center=(0, -0.105, 0.0132), rings=16, segments=48)
    ridges = [
        ellipsoid((0.0055, 0.00065, 0.0005), center=(0, y0, 0.0145), rings=8, segments=24)
        for y0 in np.linspace(-0.136, -0.076, 12)
    ]
    head = ellipsoid((0.0073, 0.020, 0.0010), center=(0, 0.010, 0.0140), rings=16, segments=48)
    if split_head:
        cut = 10
        handle = loft(centers[: cut + 1], np.column_stack((rx[: cut + 1], rz[: cut + 1])), segments=40)
        plastic_head = loft(centers[cut:], np.column_stack((rx[cut:], rz[cut:])), segments=40)
        return handle, grip, join_meshes(ridges), head, plastic_head, centers[cut].astype(np.float32)
    return handle, grip, join_meshes(ridges), head


def brush_head_rom(plastic_head, tuft_plate, root):
    """Physical 2 g zero-Phi head plate and twenty unchanged two-mode tuft fields."""
    parts = [
        {"name": "ivory_plastic_head", "mesh": plastic_head, "color": (0.86, 0.92, 0.91)},
        {"name": "teal_tuft_plate", "mesh": tuft_plate, "color": (0.05, 0.46, 0.43)},
    ]
    plate_p, plate_f = join_meshes([plastic_head, tuft_plate])
    weights = [surface_mass(plate_p, plate_f, 0.002).astype(np.float32)]
    mode_mass, stiffness, damping = [], [], []
    tuft_bases = []
    for row, y in enumerate(np.linspace(-0.002, 0.024, 5)):
        for col, x in enumerate([-0.0045, -0.0015, 0.0015, 0.0045]):
            height = 0.0118 + (0.0004 if row in (0, 4) else 0.0) + (0.0003 if col in (0, 3) else 0)
            p, f, basis = brush_tuft(height)
            origin = np.array([x, y, 0.0145], dtype=np.float32)
            parts.append(
                {
                    "name": f"tuft_{row}_{col}",
                    "mesh": (p + origin, f),
                    "color": (0.04, 0.55, 0.55) if col in (0, 3) else (0.80, 0.89, 0.87),
                    "tip_mask": p[:, 2] > height - 1e-6,
                }
            )
            tuft_bases.append(basis)
            weights.append(basis.sample_mass.copy())
            mode_mass.extend(basis.mode_mass)
            stiffness.extend(basis.mode_stiffness)
            damping.extend(basis.mode_damping)
    p, f = join_meshes([part["mesh"] for part in parts])
    p -= root
    phi = np.zeros((len(p), 40, 3), dtype=np.float32)
    start = 0
    for index, part in enumerate(parts):
        count = len(part["mesh"][0])
        part["start"], part["count"] = start, count
        if index >= 2:
            phi[start : start + count, 2 * (index - 2) : 2 * (index - 1)] = tuft_bases[index - 2].sample_phi
            part["tip_indices"] = start + np.flatnonzero(part["tip_mask"])
        start += count
    basis = newton.ModalBasis(
        sample_points=p,
        sample_phi=phi,
        sample_mass=np.concatenate(weights),
        mode_mass=mode_mass,
        mode_stiffness=stiffness,
        mode_damping=damping,
        label="physical_plastic_head_with_20_independent_tuft_fields",
    )
    return p, f, basis, parts


def glove_surface():
    """One closed rounded glove envelope with five distinct fingers.

    A planar triangulation fills a concave silhouette, with rounded thickness
    determined by distance to its edge. SciPy is already in Newton's examples
    extras; no additional dependency is introduced. This is not a scanned hand.
    """
    outline = np.array(
        [
            [-0.032, -0.060],
            [0.032, -0.060],
            [0.035, -0.030],
            [0.042, 0.012],
            [0.048, 0.055],
            [0.054, 0.117],
            [0.052, 0.126],
            [0.046, 0.130],
            [0.040, 0.126],
            [0.034, 0.080],
            [0.031, 0.151],
            [0.028, 0.163],
            [0.020, 0.168],
            [0.013, 0.163],
            [0.010, 0.091],
            [0.006, 0.176],
            [0.002, 0.188],
            [-0.006, 0.192],
            [-0.014, 0.186],
            [-0.017, 0.094],
            [-0.023, 0.159],
            [-0.028, 0.169],
            [-0.036, 0.171],
            [-0.043, 0.165],
            [-0.045, 0.079],
            [-0.051, 0.048],
            [-0.073, 0.095],
            [-0.081, 0.101],
            [-0.089, 0.098],
            [-0.093, 0.090],
            [-0.088, 0.080],
            [-0.061, 0.016],
            [-0.043, -0.008],
        ],
        dtype=float,
    )
    # Centripetal-style corner rounding using quadratic Bezier arcs, avoiding
    # the overshoot that can close narrow finger gaps with cubic interpolation.
    curve = []
    for i, b in enumerate(outline):
        a, c = outline[(i - 1) % len(outline)], outline[(i + 1) % len(outline)]
        start, end = 0.18 * a + 0.82 * b, 0.82 * b + 0.18 * c
        for s in np.linspace(0, 1, 5, endpoint=False):
            curve.append((1 - s) ** 2 * start + 2 * s * (1 - s) * b + s * s * end)
        for s in (0.0, 0.33, 0.66):
            curve.append((1 - s) * end + s * (0.18 * b + 0.82 * c))
    curve = np.asarray(curve)
    dense = []
    for a, b in zip(curve, np.roll(curve, -1, axis=0), strict=True):
        for s in np.linspace(0, 1, max(1, int(np.ceil(np.linalg.norm(b - a) / 0.002))), endpoint=False):
            dense.append((1 - s) * a + s * b)
    curve = np.asarray(dense)
    edge_a, edge_b = curve, np.roll(curve, -1, axis=0)
    d = edge_b - edge_a

    def inside(points):
        x, y = points[:, 0, None], points[:, 1, None]
        crosses = (edge_a[:, 1] > y) != (edge_b[:, 1] > y)
        denominator = edge_b[:, 1] - edge_a[:, 1]
        denominator = np.where(np.abs(denominator) > 1e-14, denominator, 1.0)
        hit = x < edge_a[:, 0] + (y - edge_a[:, 1]) * (edge_b[:, 0] - edge_a[:, 0]) / denominator
        return (np.sum(crosses & hit, axis=1) % 2) == 1

    gx, gy = np.meshgrid(np.arange(-0.095, 0.057, 0.0025), np.arange(-0.061, 0.195, 0.0025))
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    grid = grid[inside(grid)]
    t = np.clip(np.sum((grid[:, None] - edge_a[None]) * d[None], axis=2) / np.sum(d * d, axis=1), 0, 1)
    distance = np.sqrt(np.min(np.sum((grid[:, None] - edge_a[None] - t[:, :, None] * d[None]) ** 2, axis=2), axis=1))
    xy = np.concatenate([curve, grid[distance > 0.001]])
    from scipy.spatial import Delaunay

    # Insert samples on interior diagonals joining silhouette points. Otherwise
    # both skins pinch to zero thickness along those diagonals near fingertips.
    for _ in range(5):
        top = Delaunay(xy).simplices.copy()
        top = top[inside(xy[top].mean(axis=1))]
        e1, e2 = xy[top[:, 1]] - xy[top[:, 0]], xy[top[:, 2]] - xy[top[:, 0]]
        area2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        top = top[area2 > 1e-13]
        edges = np.sort(np.concatenate([top[:, [0, 1]], top[:, [1, 2]], top[:, [2, 0]]]), axis=1)
        edges, counts = np.unique(edges, return_counts=True, axis=0)
        boundary = np.zeros(len(xy), dtype=bool)
        boundary[np.unique(edges[counts == 1])] = True
        diagonals = edges[(counts == 2) & boundary[edges].all(axis=1)]
        if not len(diagonals):
            break
        xy = np.concatenate([xy, xy[diagonals].mean(axis=1)])
    else:
        raise ValueError("Could not eliminate zero-thickness glove diagonals")
    # Require exact silhouette coverage, not merely plausible-looking triangles.
    polygon_area = 0.5 * np.sum(edge_a[:, 0] * edge_b[:, 1] - edge_b[:, 0] * edge_a[:, 1])
    e1, e2 = xy[top[:, 1]] - xy[top[:, 0]], xy[top[:, 2]] - xy[top[:, 0]]
    signed = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    np.testing.assert_allclose(np.sum(signed) / 2, polygon_area, atol=1e-10)
    top[signed < 0] = top[signed < 0, ::-1]
    t = np.clip(np.sum((xy[:, None] - edge_a[None]) * d[None], axis=2) / np.sum(d * d, axis=1), 0, 1)
    distance = np.sqrt(np.min(np.sum((xy[:, None] - edge_a[None] - t[:, :, None] * d[None]) ** 2, axis=2), axis=1))
    half = 0.0060 + 0.0030 * np.exp(-(((xy[:, 1] - 0.025) / 0.063) ** 2))
    z = half * np.sqrt(np.maximum(0, 1 - (1 - np.clip(distance / 0.008, 0, 1)) ** 2))
    n = len(xy)
    # Share the outline vertices between top and bottom, not duplicate seams.
    z[boundary] = 0.0
    vertices = np.column_stack((xy, z)).tolist() + np.column_stack((xy[~boundary], -z[~boundary])).tolist()
    p = np.asarray(vertices, dtype=np.float32)
    normal = np.cross(p[top[:, 1]] - p[top[:, 0]], p[top[:, 2]] - p[top[:, 0]])[:, 2]
    top[normal < 0] = top[normal < 0, ::-1]
    mapping = np.arange(n)
    mapping[~boundary] = n + np.arange(np.count_nonzero(~boundary))
    bottom = mapping[top][:, ::-1]
    return p, np.concatenate([top, bottom]).astype(np.int32)


def glove_rom():
    p, f = glove_surface()
    clamp = np.clip((p[:, 1] + 0.055) / 0.035, 0, 1)
    clamp = clamp * clamp * (3 - 2 * clamp)
    fields = []
    for x, y, width in [(-0.018, 0.028, 0.027), (0.018, 0.045, 0.026), (-0.010, 0.078, 0.025), (0.020, 0.102, 0.025)]:
        fields.append(np.exp(-((p[:, 0] - x) ** 2 + (p[:, 1] - y) ** 2) / (2 * width**2)) * clamp)
    for x, y in [(-0.082, 0.090), (-0.034, 0.165), (-0.006, 0.183), (0.022, 0.160), (0.047, 0.123)]:
        fields.append(np.exp(-((p[:, 0] - x) ** 2 / 0.016**2 + (p[:, 1] - y) ** 2 / 0.048**2) / 2) * clamp)
    raw = np.zeros((len(p), len(fields), 3))
    raw[:, :, 2] = -np.asarray(fields).T
    weights = surface_mass(p, f, 0.024)
    gram = np.einsum("v,vmc,vnc->mn", weights, raw, raw)
    values, vectors = np.linalg.eigh(gram)
    phi = np.einsum("vmc,mn->vnc", raw, vectors / np.sqrt(values)[None])
    scale = np.linalg.norm(phi, axis=2).max(axis=0)
    phi /= scale[None, :, None]
    masses = 1 / scale**2
    stiffness = masses * (2 * np.pi * 12) ** 2
    damping = 1.0 * np.sqrt(masses * stiffness)
    basis = newton.ModalBasis(
        sample_points=p,
        sample_phi=phi,
        sample_mass=weights,
        mode_mass=masses,
        mode_stiffness=stiffness,
        mode_damping=damping,
        label="glove_local_mass_orthogonal_linear_ROM",
    )
    return p, f, basis


def transform_points(points, pose):
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)
    return (points @ rotation.T + pose[:3]).astype(np.float32)


def gravity_equilibrium(basis, gravity):
    """Static linear equilibrium K q = S g in a fixed, identity-oriented frame."""
    force = np.asarray(basis.mode_coupling_linear, dtype=float) @ np.asarray(gravity, dtype=float)
    return (force / np.asarray(basis.mode_stiffness, dtype=float)).astype(np.float32)


@wp.func
def smooth(t: float, start: float, duration: float):
    s = wp.clamp((t - start) / duration, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@wp.func
def actuator_position(t: float, kind: int):
    if kind == 0:
        x = 0.043 - 0.0332 * smooth(t, 0.3, 1.0) + 0.0332 * smooth(t, 2.2, 0.6)
        x = x - 0.086 * smooth(t, 3.0, 0.6) + 0.0332 * smooth(t, 4.1, 0.7) - 0.0332 * smooth(t, 5.6, 0.6)
        z = 0.027 + 0.035 * smooth(t, 2.8, 0.2) * (1.0 - smooth(t, 3.65, 0.35))
        return wp.vec3(x, 0.011, z)
    press = smooth(t, 0.45, 1.1) * (1.0 - smooth(t, 4.9, 1.0))
    y = 0.035 + 0.078 * smooth(t, 2.0, 2.0)
    x = 0.010 + 0.006 * smooth(t, 2.0, 2.0)
    return wp.vec3(x, y, 0.111 - 0.024 * press)


@wp.kernel
def drive_actuator(
    clock: wp.array[float],
    dt: float,
    body: int,
    kind: int,
    pose0: wp.array[wp.transform],
    vel0: wp.array[wp.spatial_vector],
    pose1: wp.array[wp.transform],
    vel1: wp.array[wp.spatial_vector],
):
    t = clock[0]
    p = actuator_position(t, kind)
    nxt = actuator_position(t + dt, kind)
    pose = wp.transform(p, wp.quat_identity())
    velocity = wp.spatial_vector((nxt - p) / dt, wp.vec3(0.0))
    pose0[body] = pose
    pose1[body] = pose
    vel0[body] = velocity
    vel1[body] = velocity


@wp.kernel
def advance_clock(clock: wp.array[float], dt: float):
    clock[0] = clock[0] + dt


@wp.kernel
def record_contact_peak(count: wp.array[int], peak: wp.array[int]):
    peak[0] = wp.max(peak[0], count[0])


class CapturedExample:
    """Common graph stepping and exact model-sample export, not a runtime patch."""

    def begin(self, viewer, cfg, kind):
        self.viewer, self.cfg, self.kind = viewer, cfg, kind
        self.sim_time, self.dt = 0.0, 1 / (60 * cfg.substeps)
        self.meshes, self.elastic, self.records, self.saved, self.timings = [], [], [], [], []
        self.gravity = (0, 0, -9.81)
        self.builder = newton.ModelBuilder(gravity=self.gravity, up_axis="Z")
        self.contact_capacity = 16384

    def add_visual(self, name, mesh, color, body=-1, roughness=0.4, metallic=0.0):
        p, f = mesh
        shape = self.builder.add_shape_mesh(
            body, mesh=newton.Mesh(p, f.ravel()), cfg=shape_config(self.cfg, False), color=color, label=name
        )
        self.meshes.append(
            {
                "name": name,
                "p": p,
                "f": f,
                "color": color,
                "body": body,
                "shape": shape,
                "roughness": roughness,
                "metallic": metallic,
                "elastic": None,
            }
        )

    def add_elastic(self, name, mesh, basis, origin, color, dynamic=False):
        p, f = mesh
        mass = float(basis.sample_mass.sum())
        inertia = max(mass * float(np.max(np.sum(p * p, axis=1))) / 3, 1e-10)
        com = np.zeros(3)
        tensor = np.eye(3) * inertia
        if dynamic:
            mass, com, tensor = sampled_mass_properties(p, basis.sample_mass)
        body = self.builder.add_body_elastic(
            xform=wp.transform(wp.vec3(*origin), wp.quat_identity()),
            mass=mass,
            inertia=wp.mat33(*tensor.ravel().tolist()),
            com=wp.vec3(*com),
            modal_basis=basis,
            is_kinematic=not dynamic,
            lock_inertia=True,
            label=name,
        )
        shape = self.builder.add_shape_mesh(
            body,
            mesh=newton.Mesh(p, f.ravel()),
            cfg=shape_config(self.cfg, dynamic or self.cfg.contact_enabled),
            color=color,
            label=name,
        )
        item = {
            "body": body,
            "shape": shape,
            "basis": basis,
            "p": p,
            "f": f,
            "name": name,
            "source_sample_mass": basis.sample_mass[: len(p)].copy(),
            "source_phi": basis.sample_phi[: len(p)].copy(),
        }
        self.elastic.append(item)
        self.meshes.append(
            {
                "name": name,
                "p": p,
                "f": f,
                "color": color,
                "body": body,
                "shape": shape,
                "roughness": 0.42,
                "metallic": 0.0,
                "elastic": item,
            }
        )
        return body

    def add_sphere_actuator(self, radius, color, half_length=0.0):
        self.radius = radius
        self.tool_half_length = half_length
        p = actuator_position(0.0, self.kind)
        self.actuator = self.builder.add_body(
            xform=wp.transform(p, wp.quat_identity()), is_kinematic=True, label="prescribed_visible_contact_tool"
        )
        if half_length:
            self.tool_shape = self.builder.add_shape_capsule(
                self.actuator,
                radius=radius,
                half_height=half_length,
                xform=wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2)),
                cfg=shape_config(self.cfg),
                color=color,
            )
            mesh = capsule_y(radius, half_length)
        else:
            self.tool_shape = self.builder.add_shape_sphere(
                self.actuator, radius=radius, cfg=shape_config(self.cfg), color=color
            )
            mesh = ellipsoid((radius, radius, radius))
        self.add_visual(
            "rounded_contact_tool",
            mesh,
            color,
            body=self.actuator,
            roughness=0.22,
            metallic=0.12 if self.kind else 0.0,
        )

    def finish(self):
        self.builder.color()
        self.model = self.builder.finalize()
        starts = self.model.elastic_shape_vertex_start.numpy()
        shapes = self.model.elastic_shape_shape.numpy()
        phi = self.model.elastic_shape_vertex_phi.numpy().reshape(-1, self.model.elastic_max_mode_count, 3)
        local = self.model.elastic_shape_vertex_local.numpy()
        for item in self.elastic:
            eid = int(self.model.body_elastic_index.numpy()[item["body"]])
            owner = int(self.model.elastic_joint.numpy()[eid])
            item["q_start"] = int(self.model.joint_q_start.numpy()[owner]) + 7
            shape_id = int(np.flatnonzero(shapes == item["shape"])[0])
            start = int(starts[shape_id])
            count = len(item["p"])
            item["phi"] = phi[start : start + count, : item["basis"].mode_count].copy()
            np.testing.assert_allclose(local[start : start + count], item["p"], atol=1e-8)
            np.testing.assert_allclose(item["phi"], item["basis"].sample_phi[:count], atol=1e-6)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.equilibrium_joint_q = self.state_0.joint_q.numpy().copy()
        self.initial_gravity_sag = 0.0
        for item in self.elastic:
            equilibrium = gravity_equilibrium(item["basis"], self.gravity)
            item["equilibrium_q"] = equilibrium
            item["equilibrium_disp"] = np.einsum("vmc,m->vc", item["phi"], equilibrium)
            self.initial_gravity_sag = max(
                self.initial_gravity_sag, float(np.linalg.norm(item["equilibrium_disp"], axis=1).max())
            )
            start = item["q_start"]
            self.equilibrium_joint_q[start : start + len(equilibrium)] = equilibrium
        # This is an initial condition, never a per-step modal target. Gravity
        # remains enabled in both driven and no-contact simulations.
        self.state_0.joint_q.assign(self.equilibrium_joint_q)
        self.state_1.joint_q.assign(self.equilibrium_joint_q)
        self.analytic_equilibrium_joint_q = self.equilibrium_joint_q.copy()
        self.pipeline = newton.CollisionPipeline(
            self.model, reduce_contacts=False, rigid_contact_max=self.contact_capacity
        )
        self.control, self.contacts = self.model.control(), self.pipeline.contacts()
        self.contact_peak = wp.zeros(1, dtype=int, device=self.model.device)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.cfg.iterations,
            rigid_body_contact_buffer_size=self.contact_capacity,
            rigid_contact_k_start=self.cfg.contact_ke,
            rigid_compliant_alm=False,
            rigid_contact_hard=False,
            friction_epsilon=1e-4,
            elastic_body_relaxation=self.cfg.elastic_relaxation,
            elastic_solver_metrics=True,
            **self.solver_options(),
        )
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.capacity_audit = CapacityAudit(self.model, self.contacts, self.solver, self.clock, self.dt)
        self.setup_audits()
        self.graph = None
        start = time.perf_counter()
        if self.model.device.is_cuda:
            if self.cfg.substeps % 2:
                raise ValueError("Graph replay requires an even number of substeps")
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        self.capture_seconds = time.perf_counter() - start
        # Let the actual time-discrete Newton solver reach its unloaded state.
        # Reset only the actuator clock during preroll; modal coordinates remain
        # solver outputs, including any offset from the analytic initial seed.
        settle_start = time.perf_counter()
        for _ in range(60):
            self.clock.assign(np.zeros(1, dtype=np.float32))
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        wp.synchronize()
        self.settle_seconds = time.perf_counter() - settle_start
        self.clock.assign(np.zeros(1, dtype=np.float32))
        self.equilibrium_joint_q = self.state_0.joint_q.numpy().copy()
        self.initial_gravity_sag = 0.0
        for item in self.elastic:
            start, count = item["q_start"], item["basis"].mode_count
            item["analytic_equilibrium_q"] = item["equilibrium_q"].copy()
            item["equilibrium_q"] = self.equilibrium_joint_q[start : start + count].copy()
            item["equilibrium_disp"] = np.einsum("vmc,m->vc", item["phi"], item["equilibrium_q"])
            self.initial_gravity_sag = max(
                self.initial_gravity_sag, float(np.linalg.norm(item["equilibrium_disp"], axis=1).max())
            )
        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.show_elastic_strain = False

    def simulate(self):
        for _ in range(self.cfg.substeps):
            wp.launch(
                drive_actuator,
                1,
                inputs=[
                    self.clock,
                    self.dt,
                    self.actuator,
                    self.kind,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_q,
                    self.state_1.body_qd,
                ],
            )
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            wp.launch(record_contact_peak, 1, inputs=[self.contacts.rigid_contact_count, self.contact_peak])
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_clock, 1, inputs=[self.clock, self.dt])

    def solver_options(self):
        return {}

    def step(self):
        start = time.perf_counter()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        wp.synchronize()
        self.timings.append(time.perf_counter() - start)
        self.sim_time += 1 / 60
        self.update_metrics()

    def update_metrics(self):
        peak = int(self.contact_peak.numpy()[0])
        assert peak <= self.contacts.rigid_contact_max, (
            f"Contact buffer overflow: {peak} > {self.contacts.rigid_contact_max}"
        )
        poses, q = self.state_0.body_q.numpy(), self.state_0.joint_q.numpy()
        qd, body_qd = self.state_0.joint_qd.numpy(), self.state_0.body_qd.numpy()
        assert all(np.isfinite(a).all() for a in (poses, q, qd, body_qd)), "Non-finite Newton state"
        world, maxima, total_maxima, displacements = [], [], [], []
        for item in self.elastic:
            amplitudes = q[item["q_start"] : item["q_start"] + item["basis"].mode_count]
            disp = np.einsum("vmc,m->vc", item["phi"], amplitudes)
            p = transform_points(item["p"] + disp, poses[item["body"]])
            tri = p[item["f"]]
            world.extend(
                [
                    p,
                    tri.mean(axis=1),
                    (tri[:, 0] + tri[:, 1]) / 2,
                    (tri[:, 1] + tri[:, 2]) / 2,
                    (tri[:, 2] + tri[:, 0]) / 2,
                ]
            )
            relative = disp - item["equilibrium_disp"]
            total_maxima.append(float(np.linalg.norm(disp, axis=1).max()))
            maxima.append(float(np.linalg.norm(relative, axis=1).max()))
            displacements.append(relative)
        samples = np.concatenate(world)
        tool_local = samples - poses[self.actuator, :3]
        tool_rotation = np.asarray(wp.quat_to_matrix(wp.quat(*poses[self.actuator, 3:])), dtype=float).reshape(3, 3)
        tool_local = tool_local @ tool_rotation
        if self.tool_half_length:
            tool_local[:, 1] -= np.clip(tool_local[:, 1], -self.tool_half_length, self.tool_half_length)
        sdf = np.linalg.norm(tool_local, axis=1) - self.radius
        metrics = self.solver.elastic_mode_solve_metrics()
        ratio = metrics["solve_residual_norm"] / np.maximum(metrics["initial_residual_norm"], 1e-12)
        frame = {
            "time": self.sim_time,
            "contacts": int(self.contacts.rigid_contact_count.numpy()[0]),
            "max_deformation_m": max(maxima),
            "max_total_displacement_m": max(total_maxima),
            "penetration_m": max(0.0, -float(sdf.min())),
            "per_body_deformation_m": maxima,
            "modal_solve_residual_ratio": float(ratio.max()),
            "modal_update_m": float(metrics["update_max"].max()),
            "actuator_position_m": poses[self.actuator, :3].tolist(),
        }
        self.object_metrics(frame, displacements)
        self.records.append(frame)
        self.saved.append(
            {
                "body_q": poses,
                "joint_q": q,
                "body_qd": body_qd,
                "joint_qd": qd,
                "joint_enabled": self.model.joint_enabled.numpy().copy(),
            }
        )

    def object_metrics(self, frame, displacements):
        pass

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

    def summary(self):
        release = [r for r in self.records if r["time"] > 6.8]
        result = {
            "config": asdict(self.cfg),
            "gravity_m_s2": list(self.gravity),
            "initial_gravity_sag_m": self.initial_gravity_sag,
            "displacement_reference": "Newton-equilibrated gravity-loaded baseline after 1 s settling from analytic K q = S g; total rest-relative displacement is reported separately",
            "equilibration_simulated_seconds": 1.0,
            "equilibration_wall_seconds": self.settle_seconds,
            "device": str(self.model.device),
            "graph_enabled": self.graph is not None,
            "frames": len(self.records),
            "duration_s": self.sim_time,
            "capture_seconds": self.capture_seconds,
            "elastic_body_count": len(self.elastic),
            "mode_count": sum(i["basis"].mode_count for i in self.elastic),
            "max_deformation_m": max(r["max_deformation_m"] for r in self.records),
            "max_total_displacement_m": max(r["max_total_displacement_m"] for r in self.records),
            "final_total_displacement_m": self.records[-1]["max_total_displacement_m"],
            "max_penetration_m": max(r["penetration_m"] for r in self.records),
            "max_contacts": max(r["contacts"] for r in self.records),
            "contact_capacity": self.contacts.rigid_contact_max,
            "max_substep_contacts_including_settling": int(self.contact_peak.numpy()[0]),
            "contact_buffer_overflow": bool(self.contact_peak.numpy()[0] > self.contacts.rigid_contact_max),
            "contact_frames": sum(r["contacts"] > 0 for r in self.records),
            "final_deformation_m": self.records[-1]["max_deformation_m"],
            "release_max_deformation_m": max((r["max_deformation_m"] for r in release), default=None),
            "max_modal_solve_residual_ratio": max(r["modal_solve_residual_ratio"] for r in self.records),
            "max_modal_update_m": max(r["modal_update_m"] for r in self.records),
            "warm_graph_mean_ms": 1000 * float(np.mean(self.timings[10:])),
            "warm_graph_p95_ms": 1000 * float(np.percentile(self.timings[10:], 95)),
            "timing_scope": "Synchronized CUDA graph simulation only; excludes CPU geometry, readback, metrics and export; includes solver metrics kernels",
            "records": self.records,
        }
        return result

    def test_final(self):
        summary = self.summary()
        summary.pop("records")
        assert len(self.records) >= 420, "Run at least seven seconds to test recovery"
        assert summary["max_modal_solve_residual_ratio"] < 0.05, summary
        if self.kind == 1:
            assert 0.002 < summary["initial_gravity_sag_m"] < 0.004, summary
        if not self.cfg.contact_enabled:
            assert summary["max_deformation_m"] < 1e-5, summary
            assert summary["max_contacts"] == 0, summary
            return
        lower, upper = (0.001, 0.006) if self.kind == 0 else (0.004, 0.016)
        assert lower < summary["max_deformation_m"] < upper, summary
        assert summary["max_penetration_m"] < (0.00065 if self.kind == 0 else 0.0010), summary
        assert summary["contact_frames"] > 70, summary
        assert summary["release_max_deformation_m"] < 0.0002, summary
        self.test_object(summary)

    def test_object(self, summary):
        pass

    def validate_negative_control(self, negative):
        assert negative.summary()["max_deformation_m"] < 0.1 * self.summary()["max_deformation_m"]

    def export(self, output, status, negative=None, test_passed=False, negative_control_passed=False):
        assert status != "validated prototype" or (test_passed and negative_control_passed)
        output.mkdir(parents=True, exist_ok=True)
        self.capacity_audit.export(output, (60 + len(self.records)) * self.cfg.substeps)
        self.field_audit.export(output)
        if negative is not None:
            negative.capacity_audit.export(
                output / "control_audit", (60 + len(negative.records)) * negative.cfg.substeps
            )
            negative.field_audit.export(output / "control_audit")
        times = np.arange(1, len(self.saved) + 1, dtype=np.float64) / 60
        arrays = {"times": times}
        descriptions = []
        basis_arrays = {
            f"model_{name}": getattr(self.model, name).numpy()
            for name in (
                "body_mass",
                "body_inertia",
                "body_com",
                "body_flags",
                "joint_parent",
                "joint_child",
                "joint_type",
                "joint_X_p",
                "joint_X_c",
                "shape_material_ke",
                "shape_material_kd",
                "shape_material_mu",
            )
        }
        basis_arrays["solver_elastic_implicit_mass_coupling"] = self.solver.elastic_implicit_mass_coupling.numpy()
        for index, mesh in enumerate(self.meshes):
            p, f, body, item = mesh["p"], mesh["f"], mesh["body"], mesh["elastic"]
            selection = slice(mesh.get("elastic_vertex_start", 0), mesh.get("elastic_vertex_end"))
            vertices = []
            if body < 0:
                vertices = p
            else:
                for state in self.saved:
                    local = p
                    if item is not None:
                        amplitudes = state["joint_q"][item["q_start"] : item["q_start"] + item["basis"].mode_count]
                        local = p + np.einsum("vmc,m->vc", item["phi"][selection], amplitudes)
                    vertices.append(transform_points(local, state["body_q"][body]))
            arrays[f"vertices_{index}"] = np.asarray(vertices, dtype=np.float32)
            arrays[f"faces_{index}"] = f.astype(np.int32)
            descriptions.append(
                {
                    "name": mesh["name"],
                    "vertices": f"vertices_{index}",
                    "faces": f"faces_{index}",
                    "color": list(mesh["color"]),
                    "roughness": mesh["roughness"],
                    "metallic": mesh["metallic"],
                    "body_index": body,
                    "q_start": None if item is None else item["q_start"],
                    "elastic_vertex_start": mesh.get("elastic_vertex_start", 0) if item is not None else None,
                }
            )
            basis_arrays[f"rest_vertices_{index}"] = p
            basis_arrays[f"faces_{index}"] = f
            if item is not None:
                for key, value in {
                    "phi": item["phi"][selection],
                    "source_phi": item["source_phi"][selection],
                    "sample_mass": item["source_sample_mass"][selection],
                    "mode_mass": item["basis"].mode_mass,
                    "mode_stiffness": item["basis"].mode_stiffness,
                    "mode_damping": item["basis"].mode_damping,
                    "mode_coupling_linear": item["basis"].mode_coupling_linear,
                    "mode_coupling_angular": item["basis"].mode_coupling_angular,
                    "equilibrium_q": item["equilibrium_q"],
                    "analytic_equilibrium_q": item["analytic_equilibrium_q"],
                }.items():
                    basis_arrays[f"{key}_{index}"] = value
        scene = dict(self.scene_metadata())
        scene.update(
            slug=self.slug,
            status=status,
            fps=60,
            meshes=descriptions,
            poster_frame=int(np.argmax([r["max_deformation_m"] for r in self.records])),
            mode_count=sum(i["basis"].mode_count for i in self.elastic),
            parameters=asdict(self.cfg),
            gravity_m_s2=list(self.gravity),
            contact_tool={
                "shape": "capsule_y" if self.tool_half_length else "sphere",
                "radius_m": self.radius,
                "half_length_m": self.tool_half_length,
            },
            initial_condition="1 s Newton settling from analytic gravity equilibrium K q = S g, tool parked clear; no per-step modal actuation",
            source_files=[
                f"{self.slug}.py",
                "brush_glove_geometry.py",
                "test_brush_glove.py",
                "field_coverage.py",
                "capacity_audit.py",
            ],
            reproduce=reproduce(self.slug),
        )
        summary = self.summary()
        summary["test_status"] = status
        summary["test_passed"] = test_passed
        summary["negative_control_passed"] = negative_control_passed
        summary["max_displacement_m"] = summary["max_deformation_m"]
        summary["final_displacement_m"] = summary["final_deformation_m"]
        if negative is not None:
            summary["negative_control"] = {k: v for k, v in negative.summary().items() if k != "records"}
            summary["negative_control"]["interaction"] = (
                "Tool-to-bristle collisions disabled; floor, joints, gravity and gripper trajectory retained"
                if getattr(self, "dynamic_assembly", False)
                else "Elastic-to-rigid shape collision disabled; identical actuator trajectory, gravity, initial state and modal parameters"
            )
            summary["negative_control"]["deformation_ratio"] = negative.summary()["max_deformation_m"] / max(
                summary["max_deformation_m"], 1e-12
            )
            if getattr(self, "dynamic_assembly", False):
                summary["negative_control"]["contact_window_deformation_ratio"] = (
                    negative.summary()["contact_window_deformation_m"] / summary["contact_window_deformation_m"]
                )
            differences = []
            for driven, control in zip(self.saved, negative.saved, strict=True):
                np.testing.assert_array_equal(driven["body_q"][self.actuator], control["body_q"][self.actuator])
                frame_max = 0.0
                for item in self.elastic:
                    start, count = item["q_start"], item["basis"].mode_count
                    delta = driven["joint_q"][start : start + count] - control["joint_q"][start : start + count]
                    frame_max = max(
                        frame_max, float(np.linalg.norm(np.einsum("vmc,m->vc", item["phi"], delta), axis=1).max())
                    )
                differences.append(frame_max)
            summary["negative_control"]["matched_trajectory_max_difference_m"] = max(differences)
            summary["negative_control"]["matched_trajectory_final_difference_m"] = differences[-1]
            if status == "validated prototype":
                assert max(differences) > (0.001 if self.kind == 0 else 0.004)
                assert differences[-1] < 0.0002
            np.savez_compressed(
                output / "negative_state.npz",
                times=times,
                equilibrium_joint_q=negative.equilibrium_joint_q,
                **{key: np.asarray([s[key] for s in negative.saved]) for key in self.saved[0]},
            )
        np.savez_compressed(output / "scene.npz", **arrays)
        np.savez_compressed(output / "basis.npz", **basis_arrays)
        np.savez_compressed(
            output / "state.npz",
            times=times,
            equilibrium_joint_q=self.equilibrium_joint_q,
            analytic_equilibrium_joint_q=self.analytic_equilibrium_joint_q,
            **{key: np.asarray([s[key] for s in self.saved]) for key in self.saved[0]},
        )
        (output / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
        source = output / "source"
        source.mkdir(exist_ok=True)
        summary["source_sha256"] = {}
        for name in scene["source_files"]:
            path = Path(__file__).parent / name
            shutil.copyfile(path, source / name)
            summary["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


def reproduce(slug):
    helpers = Path(__file__).resolve().parent
    root = Path(newton.__file__).resolve().parents[1]
    return (
        "flock -w 1800 /tmp/newton-everyday-rom-gpu.lock timeout 8m env "
        f'PYTHONPATH="{root}" OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 '
        f'uv run --project "{root}" python "{helpers}/{slug}.py" '
        f'--device cuda:0 --negative-control --output "output/{slug}"'
    )


def run_cli(example_class):
    parser = argparse.ArgumentParser(description=example_class.__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=450)
    parser.add_argument("--output", type=Path, default=Path("output") / example_class.slug)
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--no-contact", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=getattr(example_class, "default_iterations", 16))
    args = parser.parse_args()
    wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", "/tmp/newton-everyday-brush-glove-warp")
    wp.init()
    wp.set_device(args.device)
    cfg = Config(substeps=args.substeps, iterations=args.iterations, contact_enabled=not args.no_contact)
    example = example_class(None, cfg)
    negative = None
    positive_passed, negative_passed = False, False
    status = "WIP: untested"
    try:
        for i in range(args.frames):
            example.step()
            if (i + 1) % 60 == 0:
                r = example.records[-1]
                print(
                    f"{example.slug} frame={i + 1} contacts={r['contacts']} deformation_mm={r['max_deformation_m'] * 1000:.3f} penetration_mm={r['penetration_m'] * 1000:.3f}",
                    flush=True,
                )
        if not args.no_test:
            example.test_final()
            positive_passed = True
            status = "tested prototype: negative control not run"
        if args.negative_control:
            negative = example_class(
                None, Config(substeps=args.substeps, iterations=args.iterations, contact_enabled=False)
            )
            for _ in range(args.frames):
                negative.step()
            negative.test_final()
            example.validate_negative_control(negative)
            # The positive deformation guard must reject this negative control.
            negative.cfg.contact_enabled = True
            rejected = False
            try:
                negative.test_final()
            except AssertionError:
                rejected = True
            finally:
                negative.cfg.contact_enabled = False
            assert rejected, "Positive physical test incorrectly accepted the negative control"
            negative_passed = True
        if positive_passed and negative_passed:
            status = "validated prototype"
    except Exception:
        status = "WIP: validation failed"
        raise
    finally:
        if example.records:
            example.export(args.output, status, negative, positive_passed, negative_passed)
            print(json.dumps({k: v for k, v in example.summary().items() if k != "records"}, indent=2), flush=True)
