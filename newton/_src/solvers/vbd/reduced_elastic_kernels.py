# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from newton._src.sim import JointType


@wp.func
def _elastic_endpoint_xform(
    joint_index: int,
    body_index: int,
    is_parent_side: bool,
    xform_rest: wp.transform,
    body_elastic_index: wp.array(dtype=wp.int32),
    elastic_joint: wp.array(dtype=wp.int32),
    elastic_mode_count: wp.array(dtype=wp.int32),
    joint_q: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_parent_elastic_endpoint: wp.array(dtype=wp.int32),
    joint_child_elastic_endpoint: wp.array(dtype=wp.int32),
    elastic_endpoint_phi: wp.array(dtype=wp.vec3),
    elastic_max_mode_count: int,
):
    if body_index < 0:
        return xform_rest

    elastic_index = body_elastic_index[body_index]
    if elastic_index < 0:
        return xform_rest

    endpoint = joint_child_elastic_endpoint[joint_index]
    if is_parent_side:
        endpoint = joint_parent_elastic_endpoint[joint_index]
    if endpoint < 0:
        return xform_rest

    p = wp.transform_get_translation(xform_rest)
    q = wp.transform_get_rotation(xform_rest)
    owner_joint = elastic_joint[elastic_index]
    q_start = joint_q_start[owner_joint] + 7
    mode_count = elastic_mode_count[elastic_index]

    for mode in range(elastic_max_mode_count):
        if mode < mode_count:
            p = p + elastic_endpoint_phi[endpoint * elastic_max_mode_count + mode] * joint_q[q_start + mode]

    return wp.transform(p, q)


@wp.kernel
def copy_elastic_joint_frame_to_body(
    elastic_body: wp.array(dtype=wp.int32),
    elastic_joint: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    elastic_index = wp.tid()
    body = elastic_body[elastic_index]
    joint = elastic_joint[elastic_index]
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]

    body_q[body] = wp.transform(
        wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
        wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]),
    )
    body_qd[body] = wp.spatial_vector(
        wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),
        wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]),
    )


@wp.kernel
def integrate_elastic_modes_implicit(
    dt: float,
    elastic_joint: wp.array(dtype=wp.int32),
    elastic_mode_start: wp.array(dtype=wp.int32),
    elastic_mode_count: wp.array(dtype=wp.int32),
    elastic_mode_mass: wp.array(dtype=float),
    elastic_mode_stiffness: wp.array(dtype=float),
    elastic_mode_damping: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_f: wp.array(dtype=float),
    joint_q_in: wp.array(dtype=float),
    joint_qd_in: wp.array(dtype=float),
    joint_q_out: wp.array(dtype=float),
    joint_qd_out: wp.array(dtype=float),
):
    elastic_index = wp.tid()
    joint = elastic_joint[elastic_index]
    q_start = joint_q_start[joint] + 7
    qd_start = joint_qd_start[joint] + 6
    mode_start = elastic_mode_start[elastic_index]
    mode_count = elastic_mode_count[elastic_index]

    for i in range(mode_count):
        mode = mode_start + i
        q_idx = q_start + i
        qd_idx = qd_start + i

        q = joint_q_in[q_idx]
        v = joint_qd_in[qd_idx]
        force = joint_f[qd_idx]
        mass = elastic_mode_mass[mode]
        stiffness = elastic_mode_stiffness[mode]
        damping = elastic_mode_damping[mode]

        denom = mass + dt * damping + dt * dt * stiffness
        if denom <= 0.0:
            joint_q_out[q_idx] = q
            joint_qd_out[qd_idx] = 0.0
        else:
            v_new = (mass * v + dt * (force - stiffness * q)) / denom
            q_new = q + dt * v_new
            joint_q_out[q_idx] = q_new
            joint_qd_out[qd_idx] = v_new


@wp.kernel
def copy_elastic_modes(
    elastic_joint: wp.array(dtype=wp.int32),
    elastic_mode_count: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_q_src: wp.array(dtype=float),
    joint_qd_src: wp.array(dtype=float),
    joint_q_dst: wp.array(dtype=float),
    joint_qd_dst: wp.array(dtype=float),
):
    elastic_index = wp.tid()
    joint = elastic_joint[elastic_index]
    q_start = joint_q_start[joint] + 7
    qd_start = joint_qd_start[joint] + 6
    mode_count = elastic_mode_count[elastic_index]

    for i in range(mode_count):
        joint_q_dst[q_start + i] = joint_q_src[q_start + i]
        joint_qd_dst[qd_start + i] = joint_qd_src[qd_start + i]


@wp.kernel
def solve_elastic_modes_from_joint_constraints(
    dt: float,
    elastic_body: wp.array(dtype=wp.int32),
    elastic_joint: wp.array(dtype=wp.int32),
    elastic_mode_start: wp.array(dtype=wp.int32),
    elastic_mode_count: wp.array(dtype=wp.int32),
    elastic_mode_mass: wp.array(dtype=float),
    elastic_mode_stiffness: wp.array(dtype=float),
    elastic_mode_damping: wp.array(dtype=float),
    elastic_endpoint_count: int,
    elastic_endpoint_joint: wp.array(dtype=wp.int32),
    elastic_endpoint_side: wp.array(dtype=wp.int32),
    elastic_endpoint_body: wp.array(dtype=wp.int32),
    elastic_endpoint_phi: wp.array(dtype=wp.vec3),
    elastic_max_mode_count: int,
    body_elastic_index: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_constraint_start: wp.array(dtype=wp.int32),
    joint_penalty_k: wp.array(dtype=float),
    joint_parent_elastic_endpoint: wp.array(dtype=wp.int32),
    joint_child_elastic_endpoint: wp.array(dtype=wp.int32),
    joint_f: wp.array(dtype=float),
    joint_q_prev: wp.array(dtype=float),
    joint_qd_prev: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
):
    elastic_index = wp.tid()
    body = elastic_body[elastic_index]
    owner_joint = elastic_joint[elastic_index]
    q_start = joint_q_start[owner_joint] + 7
    qd_start = joint_qd_start[owner_joint] + 6
    mode_start = elastic_mode_start[elastic_index]
    mode_count = elastic_mode_count[elastic_index]

    inv_dt = 1.0 / dt
    inv_dt_sq = inv_dt * inv_dt

    for mode in range(mode_count):
        mode_data = mode_start + mode
        q_idx = q_start + mode
        qd_idx = qd_start + mode

        q = joint_q[q_idx]
        q_prev = joint_q_prev[q_idx]
        v_prev = joint_qd_prev[qd_idx]
        mass = elastic_mode_mass[mode_data]
        stiffness = elastic_mode_stiffness[mode_data]
        damping = elastic_mode_damping[mode_data]
        force = joint_f[qd_idx]

        h = stiffness
        grad = stiffness * q - force
        if mass > 0.0:
            h = h + mass * inv_dt_sq
            grad = grad + mass * inv_dt_sq * (q - q_prev - dt * v_prev)
        if damping > 0.0:
            h = h + damping * inv_dt
            grad = grad + damping * inv_dt * (q - q_prev)

        for endpoint in range(elastic_endpoint_count):
            if elastic_endpoint_body[endpoint] != body:
                continue

            joint = elastic_endpoint_joint[endpoint]
            if not joint_enabled[joint]:
                continue

            jt = joint_type[joint]
            if jt == JointType.ELASTIC or jt == JointType.FREE or jt == JointType.DISTANCE or jt == JointType.CABLE:
                continue

            c_start = joint_constraint_start[joint]
            k = joint_penalty_k[c_start]
            if k <= 0.0:
                continue

            parent = joint_parent[joint]
            child = joint_child[joint]
            X_pj = _elastic_endpoint_xform(
                joint,
                parent,
                True,
                joint_X_p[joint],
                body_elastic_index,
                elastic_joint,
                elastic_mode_count,
                joint_q,
                joint_q_start,
                joint_parent_elastic_endpoint,
                joint_child_elastic_endpoint,
                elastic_endpoint_phi,
                elastic_max_mode_count,
            )
            X_cj = _elastic_endpoint_xform(
                joint,
                child,
                False,
                joint_X_c[joint],
                body_elastic_index,
                elastic_joint,
                elastic_mode_count,
                joint_q,
                joint_q_start,
                joint_parent_elastic_endpoint,
                joint_child_elastic_endpoint,
                elastic_endpoint_phi,
                elastic_max_mode_count,
            )

            if parent >= 0:
                X_wp = body_q[parent] * X_pj
            else:
                X_wp = X_pj
            X_wc = body_q[child] * X_cj

            x_p = wp.transform_get_translation(X_wp)
            x_c = wp.transform_get_translation(X_wc)
            C = x_c - x_p

            P = wp.identity(3, float)
            if jt == JointType.PRISMATIC:
                axis = joint_axis[joint_qd_start[joint]]
                axis_w = wp.normalize(wp.quat_rotate(wp.transform_get_rotation(X_wp), axis))
                P = P - wp.outer(axis_w, axis_w)

            side = elastic_endpoint_side[endpoint]
            phi_local = elastic_endpoint_phi[endpoint * elastic_max_mode_count + mode]
            dC = wp.quat_rotate(wp.transform_get_rotation(body_q[body]), phi_local)
            if side == 0:
                dC = -dC

            PC = P * C
            PdC = P * dC
            grad = grad + k * wp.dot(PC, PdC)
            h = h + k * wp.dot(PdC, PdC)

        if h > 0.0:
            q_new = q - grad / h
            joint_q[q_idx] = q_new
            joint_qd[qd_idx] = (q_new - q_prev) * inv_dt


@wp.kernel
def copy_body_frame_to_elastic_joint(
    elastic_body: wp.array(dtype=wp.int32),
    elastic_joint: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
):
    elastic_index = wp.tid()
    body = elastic_body[elastic_index]
    joint = elastic_joint[elastic_index]
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]

    X_wb = body_q[body]
    p = wp.transform_get_translation(X_wb)
    q = wp.transform_get_rotation(X_wb)
    v = body_qd[body]
    lin = wp.spatial_top(v)
    ang = wp.spatial_bottom(v)

    joint_q[q_start + 0] = p[0]
    joint_q[q_start + 1] = p[1]
    joint_q[q_start + 2] = p[2]
    joint_q[q_start + 3] = q[0]
    joint_q[q_start + 4] = q[1]
    joint_q[q_start + 5] = q[2]
    joint_q[q_start + 6] = q[3]

    joint_qd[qd_start + 0] = lin[0]
    joint_qd[qd_start + 1] = lin[1]
    joint_qd[qd_start + 2] = lin[2]
    joint_qd[qd_start + 3] = ang[0]
    joint_qd[qd_start + 4] = ang[1]
    joint_qd[qd_start + 5] = ang[2]
