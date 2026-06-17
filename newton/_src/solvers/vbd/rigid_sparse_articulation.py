# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host-side sparse articulation layout for the VBD rigid solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class RigidArticulationSparseLayout:
    """Static block-sparse articulation layout for rigid VBD solves."""

    articulation_body_offsets: wp.array
    articulation_joint_offsets: wp.array
    articulation_bodies: wp.array
    articulation_joints: wp.array
    articulation_block_row_offsets: wp.array
    articulation_block_cols: wp.array
    articulation_diag_slots: wp.array
    body_articulation_sparse: wp.array
    body_articulation_local: wp.array
    articulation_count: int
    articulation_body_count: int
    block_count: int


def _symbolic_cholesky_pattern(body_count: int, edges: set[tuple[int, int]]) -> list[list[int]]:
    rows = [{i} for i in range(body_count)]
    for a, b in edges:
        row = max(a, b)
        col = min(a, b)
        rows[row].add(col)

    for k in range(body_count):
        later = [i for i in range(k + 1, body_count) if k in rows[i]]
        for i_index, i in enumerate(later):
            for j in later[:i_index]:
                rows[max(i, j)].add(min(i, j))

    return [sorted(row) for row in rows]


def build_rigid_articulation_sparse_layout(
    model, device: wp.context.Devicelike
) -> RigidArticulationSparseLayout | None:
    """Build the static articulation sparse layout from Newton articulation ranges."""

    if model.body_count == 0:
        return None

    with wp.ScopedDevice("cpu"):
        articulation_start = np.asarray(model.articulation_start.to("cpu").numpy(), dtype=np.int32)
        joint_parent = np.asarray(model.joint_parent.to("cpu").numpy(), dtype=np.int32)
        joint_child = np.asarray(model.joint_child.to("cpu").numpy(), dtype=np.int32)

    articulation_bodies_host: list[int] = []
    articulation_joints_host: list[int] = []
    articulation_body_offsets_host = [0]
    articulation_joint_offsets_host = [0]
    block_row_offsets_host = [0]
    block_cols_host: list[int] = []
    diag_slots_host: list[int] = []

    body_articulation_sparse_host = np.full((model.body_count,), -1, dtype=np.int32)
    body_articulation_local_host = np.full((model.body_count,), -1, dtype=np.int32)
    body_seen = np.zeros((model.body_count,), dtype=bool)

    articulation_groups: list[tuple[list[int], list[int]]] = []
    for articulation_id in range(model.articulation_count):
        joint_start = int(articulation_start[articulation_id])
        joint_end = int(articulation_start[articulation_id + 1])
        joints = list(range(joint_start, joint_end))

        bodies: list[int] = []
        for joint_idx in joints:
            parent = int(joint_parent[joint_idx])
            child = int(joint_child[joint_idx])
            if parent >= 0 and parent not in bodies:
                bodies.append(parent)
            if child >= 0 and child not in bodies:
                bodies.append(child)

        if bodies:
            articulation_groups.append((bodies, joints))
            for body in bodies:
                body_seen[body] = True

    for body in range(model.body_count):
        if not body_seen[body]:
            articulation_groups.append(([body], []))

    for articulation_sparse, (bodies, joints) in enumerate(articulation_groups):
        local_index = {body: i for i, body in enumerate(bodies)}
        edges: set[tuple[int, int]] = set()
        for joint_idx in joints:
            parent = int(joint_parent[joint_idx])
            child = int(joint_child[joint_idx])
            if parent >= 0 and child >= 0 and parent in local_index and child in local_index:
                edges.add((local_index[parent], local_index[child]))

        pattern = _symbolic_cholesky_pattern(len(bodies), edges)

        articulation_bodies_host.extend(bodies)
        articulation_joints_host.extend(joints)
        articulation_body_offsets_host.append(len(articulation_bodies_host))
        articulation_joint_offsets_host.append(len(articulation_joints_host))

        for local_body, body in enumerate(bodies):
            body_articulation_sparse_host[body] = articulation_sparse
            body_articulation_local_host[body] = local_body

        for local_row, row_cols in enumerate(pattern):
            row_start = len(block_cols_host)
            block_cols_host.extend(row_cols)
            diag_slots_host.append(row_start + row_cols.index(local_row))
            block_row_offsets_host.append(len(block_cols_host))

    articulation_bodies_np = np.asarray(articulation_bodies_host, dtype=np.int32)
    articulation_joints_np = np.asarray(articulation_joints_host, dtype=np.int32)
    articulation_body_offsets_np = np.asarray(articulation_body_offsets_host, dtype=np.int32)
    articulation_joint_offsets_np = np.asarray(articulation_joint_offsets_host, dtype=np.int32)
    block_row_offsets_np = np.asarray(block_row_offsets_host, dtype=np.int32)
    block_cols_np = np.asarray(block_cols_host, dtype=np.int32)
    diag_slots_np = np.asarray(diag_slots_host, dtype=np.int32)

    return RigidArticulationSparseLayout(
        articulation_body_offsets=wp.array(articulation_body_offsets_np, dtype=wp.int32, device=device),
        articulation_joint_offsets=wp.array(articulation_joint_offsets_np, dtype=wp.int32, device=device),
        articulation_bodies=wp.array(articulation_bodies_np, dtype=wp.int32, device=device),
        articulation_joints=wp.array(articulation_joints_np, dtype=wp.int32, device=device),
        articulation_block_row_offsets=wp.array(block_row_offsets_np, dtype=wp.int32, device=device),
        articulation_block_cols=wp.array(block_cols_np, dtype=wp.int32, device=device),
        articulation_diag_slots=wp.array(diag_slots_np, dtype=wp.int32, device=device),
        body_articulation_sparse=wp.array(body_articulation_sparse_host, dtype=wp.int32, device=device),
        body_articulation_local=wp.array(body_articulation_local_host, dtype=wp.int32, device=device),
        articulation_count=len(articulation_groups),
        articulation_body_count=len(articulation_bodies_host),
        block_count=len(block_cols_host),
    )
