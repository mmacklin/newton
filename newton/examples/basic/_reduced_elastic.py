# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BOX_RENDER_MAX_EDGE_LENGTH = 0.01


def _segment_count(extent: float, edge: float) -> int:
    return max(1, int(np.ceil(float(extent) / edge - 1.0e-5)))


def box_surface_mesh(
    length: float,
    half_width_y: float,
    half_width_z: float,
    *,
    max_edge_length: float = BOX_RENDER_MAX_EDGE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the subdivided box mesh used by reduced elastic beam examples."""
    half_length = 0.5 * length
    edge = max(float(max_edge_length), 1.0e-6)
    nx = _segment_count(length, edge)
    ny = _segment_count(2.0 * half_width_y, edge)
    nz = _segment_count(2.0 * half_width_z, edge)
    vertices: list[tuple[float, float, float]] = []
    indices: list[int] = []
    grid: dict[tuple[int, int, int], int] = {}

    for i in range(nx + 1):
        x = -half_length + length * (float(i) / float(nx))
        for j in range(ny + 1):
            y = -half_width_y + 2.0 * half_width_y * (float(j) / float(ny))
            for k in range(nz + 1):
                if i not in (0, nx) and j not in (0, ny) and k not in (0, nz):
                    continue
                z = -half_width_z + 2.0 * half_width_z * (float(k) / float(nz))
                grid[(i, j, k)] = len(vertices)
                vertices.append((x, y, z))

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        indices.extend([a, b, c, a, c, d])

    for j in range(ny):
        for k in range(nz):
            add_quad(grid[(0, j, k)], grid[(0, j, k + 1)], grid[(0, j + 1, k + 1)], grid[(0, j + 1, k)])
            add_quad(
                grid[(nx, j, k)],
                grid[(nx, j + 1, k)],
                grid[(nx, j + 1, k + 1)],
                grid[(nx, j, k + 1)],
            )

    for i in range(nx):
        for k in range(nz):
            add_quad(grid[(i, 0, k)], grid[(i + 1, 0, k)], grid[(i + 1, 0, k + 1)], grid[(i, 0, k + 1)])
            add_quad(
                grid[(i, ny, k)],
                grid[(i, ny, k + 1)],
                grid[(i + 1, ny, k + 1)],
                grid[(i + 1, ny, k)],
            )

    for i in range(nx):
        for j in range(ny):
            add_quad(grid[(i, j, 0)], grid[(i, j + 1, 0)], grid[(i + 1, j + 1, 0)], grid[(i + 1, j, 0)])
            add_quad(
                grid[(i, j, nz)],
                grid[(i + 1, j, nz)],
                grid[(i + 1, j + 1, nz)],
                grid[(i, j + 1, nz)],
            )

    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32)


def beam_render_sample_points(
    length: float,
    half_width_y: float,
    half_width_z: float,
    *,
    extra_points: Sequence[Sequence[float]] = (),
    max_edge_length: float = BOX_RENDER_MAX_EDGE_LENGTH,
    diagnostic_count: int = 33,
) -> np.ndarray:
    """Return modal sample points covering rendered beam vertices and overlays."""
    surface_vertices, _ = box_surface_mesh(length, half_width_y, half_width_z, max_edge_length=max_edge_length)
    diagnostic_points = np.column_stack(
        (
            np.linspace(-0.5 * length, 0.5 * length, diagnostic_count, dtype=np.float32),
            np.zeros(diagnostic_count, dtype=np.float32),
            np.full(diagnostic_count, half_width_z + 0.045, dtype=np.float32),
        )
    )
    point_sets = [surface_vertices, diagnostic_points]
    if extra_points:
        point_sets.append(np.asarray(extra_points, dtype=np.float32).reshape((-1, 3)))
    return np.vstack(point_sets).astype(np.float32, copy=False)
