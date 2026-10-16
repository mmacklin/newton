# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Published two-level Loop embedding; extracted unchanged for CPU-only use."""

import numpy as np


def subdivision_embedding(points, faces, pinned, levels=2):
    """Return a Loop surface and nonnegative affine weights on the source mesh.

    The neck seam stays fixed. New seam edges are exact linear subdivisions.
    This is a render-only volume embedding, not extra contact or mass samples.
    """
    source = np.asarray(points, dtype=np.float64)
    weights = np.eye(len(source))
    faces = np.asarray(faces, dtype=np.int32)
    pinned = np.asarray(pinned, dtype=bool)
    for _ in range(levels):
        n = len(weights)
        directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
        if not np.all(counts == 2):
            raise ValueError("Expected a closed two-manifold collision mesh")
        opposites = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
        opposite = opposites[np.argsort(inverse, kind="stable")].reshape(-1, 2)
        adjacent = [[] for _ in range(n)]
        for a, b in edges:
            adjacent[a].append(b)
            adjacent[b].append(a)
        updated = np.empty((n + len(edges), weights.shape[1]), dtype=np.float64)
        for i, neighbors in enumerate(adjacent):
            k = len(neighbors)
            beta = 3 / 16 if k == 3 else 3 / (8 * k)
            updated[i] = (
                weights[i] if pinned[i] else (1 - k * beta) * weights[i] + beta * weights[neighbors].sum(axis=0)
            )
        edge_pinned = pinned[edges].all(axis=1)
        for i, (a, b) in enumerate(edges):
            if edge_pinned[i]:
                updated[n + i] = (weights[a] + weights[b]) / 2
            else:
                c, d = opposite[i]
                updated[n + i] = 3 / 8 * (weights[a] + weights[b]) + 1 / 8 * (weights[c] + weights[d])
        ab, bc, ca = inverse.reshape(3, -1) + n
        a, b, c = faces.T
        faces = np.concatenate(
            [
                np.column_stack([a, ab, ca]),
                np.column_stack([b, bc, ab]),
                np.column_stack([c, ca, bc]),
                np.column_stack([ab, bc, ca]),
            ]
        ).astype(np.int32)
        weights = updated
        pinned = np.r_[pinned, edge_pinned]
    return (weights @ source).astype(np.float32), faces, weights


def refined_plastic(basis, scene):
    result = {}
    for i, mesh in enumerate(scene["meshes"]):
        if mesh["name"] not in ("ivory_handle", "ivory_plastic_head"):
            continue
        p, f = basis[f"rest_vertices_{i}"], basis[f"faces_{i}"]
        seam = p[:, 1].max() if mesh["name"] == "ivory_handle" else p[:, 1].min()
        pinned = np.isclose(p[:, 1], seam, atol=1e-8)
        refined, faces, weights = subdivision_embedding(p, f, pinned)
        result[i] = {"points": refined, "faces": faces, "weights": weights, "original_points": p, "original_faces": f}
    return result
