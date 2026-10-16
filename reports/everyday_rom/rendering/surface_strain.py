# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU principal surface engineering strain, independent of rigid motion.

The rest chart, eigenstretch calculation, area averaging, and classical jet
map reuse the accepted headphone_strain.py/audit_surface_strain.py method.
No constitutive law, displacement proxy, or per-frame normalization is used.
"""

from __future__ import annotations

import numpy as np


def validate_mesh(points: np.ndarray, faces: np.ndarray) -> None:
    """Validate a finite triangle mesh without welding disconnected vertices."""
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected finite nonempty (N,3) vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in "iu":
        raise ValueError("Expected nonempty integer (T,3) triangles")
    if faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("Triangle vertex index out of range")


def rest_metric(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return inverse orthonormal material charts and reference areas [m^2]."""
    points, faces = np.asarray(points, dtype=np.float64), np.asarray(faces)
    validate_mesh(points, faces)
    triangles = points[faces]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    length = np.linalg.norm(a, axis=1)
    area2 = np.linalg.norm(np.cross(a, b), axis=1)
    scale = length * np.linalg.norm(b, axis=1)
    valid = (scale > 0) & (area2 > 1e-10 * scale) & np.isfinite(area2)
    if not valid.all():
        raise ValueError(f"{np.count_nonzero(~valid)} degenerate reference triangles")
    projection = np.einsum("ij,ij->i", a, b) / length
    height = area2 / length
    inverse = np.zeros((len(faces), 2, 2))
    inverse[:, 0, 0] = 1 / length
    inverse[:, 0, 1] = -projection / (length * height)
    inverse[:, 1, 1] = 1 / height
    return inverse, area2 / 2


def stretches(points: np.ndarray, faces: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """Return ascending principal stretches from the corotational metric."""
    triangles = np.asarray(points, dtype=np.float64)[faces]
    if not np.isfinite(triangles).all():
        raise ValueError("Nonfinite current surface")
    edges = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1)
    gradient = edges @ inverse
    metric = np.swapaxes(gradient, -1, -2) @ gradient
    return np.sqrt(np.maximum(np.linalg.eigvalsh(metric), 0))


def vertex_average(values: np.ndarray, faces: np.ndarray, area: np.ndarray, count: int) -> np.ndarray:
    """Average triangle scalars using rest areas, never crossing vertex IDs."""
    total, weight = np.zeros(count), np.zeros(count)
    for corner in range(3):
        np.add.at(total, faces[:, corner], values * area)
        np.add.at(weight, faces[:, corner], area)
    if np.any(weight <= 0):
        raise ValueError("Unreferenced vertices cannot receive a strain value")
    return total / weight


def field(rest: np.ndarray, faces: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return signed triangle principal strains and vertex magnitudes [1]."""
    if np.shape(current) != np.shape(rest):
        raise ValueError("Current and material-rest vertex ordering must match")
    inverse, area = rest_metric(rest, faces)
    principal = stretches(current, faces, inverse) - 1
    return principal, vertex_average(np.max(np.abs(principal), axis=1), faces, area, len(rest))


def jet(values: np.ndarray) -> np.ndarray:
    """Return classical jet RGB for normalized values, clipping only display."""
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Cannot color nonfinite strain")
    x = np.clip(x, 0, 1)
    return np.stack([np.clip(1.5 - np.abs(4 * x - shift), 0, 1) for shift in (3, 2, 1)], axis=-1).astype(np.float32)


def components(faces: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Label topologically disconnected parts without geometric welding."""
    parent = np.arange(count)

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for triangle in faces:
        a = root(triangle[0])
        for b in triangle[1:]:
            parent[root(b)] = a
    _, vertex = np.unique([root(i) for i in range(count)], return_inverse=True)
    return vertex.astype(np.int32), vertex[faces[:, 0]].astype(np.int32)


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply Newton xyzw rigid poses without changing the supplied quaternion."""
    pose = np.asarray(pose)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Expected one finite [position, xyzw] pose")
    if abs(np.linalg.norm(pose[3:]) - 1) > 1e-5:
        raise ValueError("Saved rigid quaternion is not unit length")
    q = pose[3:6]
    t = 2 * np.cross(q, points)
    return points + pose[6] * t + np.cross(q, t) + pose[:3]
