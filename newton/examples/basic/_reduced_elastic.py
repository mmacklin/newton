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


def finite_torsion_mode_fields(points: np.ndarray, length: float, tip_twist: float) -> tuple[np.ndarray, np.ndarray]:
    """Return circumferential and radial finite-twist displacement fields."""
    points = np.asarray(points, dtype=np.float32)
    s = np.clip(points[:, 0] + 0.5 * length, 0.0, length)
    theta = float(tip_twist) * s / float(length)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    y = points[:, 1]
    z = points[:, 2]
    twist = np.zeros_like(points, dtype=np.float32)
    twist[:, 1] = -z * sin_theta
    twist[:, 2] = y * sin_theta

    radial = np.zeros_like(points, dtype=np.float32)
    radial[:, 1] = y * (cos_theta - 1.0)
    radial[:, 2] = z * (cos_theta - 1.0)
    return twist, radial


def finite_torsion_displacement(points: np.ndarray, length: float, tip_twist: float) -> np.ndarray:
    """Return exact finite-rotation twist displacements for beam-local points."""
    twist, radial = finite_torsion_mode_fields(points, length, tip_twist)
    return twist + radial


def mesh_volume(vertices: np.ndarray, indices: np.ndarray) -> float:
    """Return the absolute volume enclosed by a closed triangle mesh."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = vertices[np.asarray(indices, dtype=np.int32).reshape((-1, 3))]
    signed = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
    return abs(float(np.sum(signed) / 6.0))


def elastic_shape_deformed_vertices(model, state, elastic_shape: int = 0) -> np.ndarray:
    """Return body-local deformed render vertices for one reduced elastic shape."""
    shape_count = int(getattr(model, "elastic_shape_count", 0))
    if elastic_shape < 0 or elastic_shape >= shape_count:
        raise IndexError(f"elastic_shape index {elastic_shape} out of range for {shape_count} shapes")

    vertex_start = model.elastic_shape_vertex_start.numpy()
    vertex_count = model.elastic_shape_vertex_count.numpy()
    elastic_shape_body = model.elastic_shape_body.numpy()
    body_elastic_index = model.body_elastic_index.numpy()
    elastic_mode_count = model.elastic_mode_count.numpy()
    elastic_joint = model.elastic_joint.numpy()
    joint_q_start = model.joint_q_start.numpy()

    start = int(vertex_start[elastic_shape])
    count = int(vertex_count[elastic_shape])
    body = int(elastic_shape_body[elastic_shape])
    elastic_index = int(body_elastic_index[body])
    mode_count = int(elastic_mode_count[elastic_index])
    q_start = int(joint_q_start[int(elastic_joint[elastic_index])]) + 7

    vertices = np.array(model.elastic_shape_vertex_local.numpy()[start : start + count], dtype=np.float64)
    if mode_count == 0:
        return vertices

    max_modes = int(model.elastic_max_mode_count)
    phi = model.elastic_shape_vertex_phi.numpy().reshape((-1, max_modes, 3))[start : start + count, :mode_count]
    q = state.joint_q.numpy()[q_start : q_start + mode_count]
    vertices += np.einsum("vmc,m->vc", phi, q)
    return vertices


def elastic_shape_volume_ratio(model, state, elastic_shape: int = 0) -> float:
    """Return deformed/rest volume for one reduced elastic render shape."""
    vertex_start = model.elastic_shape_vertex_start.numpy()
    vertex_count = model.elastic_shape_vertex_count.numpy()
    index_start = model.elastic_shape_index_start.numpy()
    index_count = model.elastic_shape_index_count.numpy()

    v_start = int(vertex_start[elastic_shape])
    v_count = int(vertex_count[elastic_shape])
    i_start = int(index_start[elastic_shape])
    i_count = int(index_count[elastic_shape])

    rest_vertices = np.array(model.elastic_shape_vertex_local.numpy()[v_start : v_start + v_count], dtype=np.float64)
    indices = model.elastic_shape_indices.numpy()[i_start : i_start + i_count]
    rest_volume = mesh_volume(rest_vertices, indices)
    if rest_volume <= 0.0:
        raise ValueError("elastic shape rest mesh has non-positive volume")

    deformed_vertices = elastic_shape_deformed_vertices(model, state, elastic_shape)
    return mesh_volume(deformed_vertices, indices) / rest_volume
