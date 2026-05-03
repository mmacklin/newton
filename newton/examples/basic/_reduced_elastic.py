# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


BOX_RENDER_SEGMENTS = 128


def box_surface_mesh(
    length: float, half_width_y: float, half_width_z: float, segments: int = BOX_RENDER_SEGMENTS
) -> tuple[np.ndarray, np.ndarray]:
    """Return the x-subdivided box mesh used by reduced elastic beam examples."""
    half_length = 0.5 * length
    vertices: list[tuple[float, float, float]] = []
    indices: list[int] = []

    for i in range(segments + 1):
        x = -half_length + length * float(i) / float(segments)
        vertices.extend(
            [
                (x, -half_width_y, -half_width_z),
                (x, half_width_y, -half_width_z),
                (x, half_width_y, half_width_z),
                (x, -half_width_y, half_width_z),
            ]
        )

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        indices.extend([a, b, c, a, c, d])

    for i in range(segments):
        base = 4 * i
        nxt = base + 4
        add_quad(base + 0, base + 1, nxt + 1, nxt + 0)
        add_quad(base + 1, base + 2, nxt + 2, nxt + 1)
        add_quad(base + 2, base + 3, nxt + 3, nxt + 2)
        add_quad(base + 3, base + 0, nxt + 0, nxt + 3)

    add_quad(0, 3, 2, 1)
    end = 4 * segments
    add_quad(end + 0, end + 1, end + 2, end + 3)

    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32)


def beam_render_sample_points(
    length: float,
    half_width_y: float,
    half_width_z: float,
    *,
    extra_points: Sequence[Sequence[float]] = (),
    segments: int = BOX_RENDER_SEGMENTS,
    diagnostic_count: int = 33,
) -> np.ndarray:
    """Return modal sample points covering rendered beam vertices and overlays."""
    surface_vertices, _ = box_surface_mesh(length, half_width_y, half_width_z, segments=segments)
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
