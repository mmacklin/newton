# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from newton._src.sim import JointType
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    _SMALL_LENGTH_EPS,
    evaluate_contact_point_world,
    evaluate_rigid_contact_from_world_points,
)

wp.set_module_options({"enable_backward": False})

_NUM_ELASTIC_CONTACT_THREADS_PER_BODY = 32
"""Threads per reduced elastic body for modal contact accumulation."""


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


@wp.func
def _accumulate_elastic_contact_modes(
    dt: float,
    contact_idx: int,
    body: int,
    elastic_index: int,
    body_R: wp.mat33,
    body_R_T: wp.mat33,
    elastic_joint: wp.array(dtype=wp.int32),
    elastic_mode_count: wp.array(dtype=wp.int32),
    elastic_max_mode_count: int,
    body_elastic_index: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=wp.int32),
    rigid_contact_shape0: wp.array(dtype=wp.int32),
    rigid_contact_shape1: wp.array(dtype=wp.int32),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_margin0: wp.array(dtype=float),
    rigid_contact_margin1: wp.array(dtype=float),
    rigid_contact_elastic_sample0: wp.array(dtype=wp.int32),
    rigid_contact_elastic_sample1: wp.array(dtype=wp.int32),
    contact_material_ke: wp.array(dtype=float),
    contact_material_kd: wp.array(dtype=float),
    contact_material_mu: wp.array(dtype=float),
    friction_epsilon: float,
    joint_q_start: wp.array(dtype=wp.int32),
    elastic_shape_vertex_local: wp.array(dtype=wp.vec3),
    elastic_shape_vertex_phi: wp.array(dtype=wp.vec3),
    joint_q_prev: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
    elastic_mode_block_grad: wp.array(dtype=float),
    elastic_mode_block_matrix: wp.array(dtype=float),
):
    s0 = rigid_contact_shape0[contact_idx]
    s1 = rigid_contact_shape1[contact_idx]
    if s0 < 0 or s1 < 0:
        return

    b0 = shape_body[s0]
    b1 = shape_body[s1]
    elastic_sample0 = rigid_contact_elastic_sample0[contact_idx]
    elastic_sample1 = rigid_contact_elastic_sample1[contact_idx]

    elastic_body = -1
    elastic_sample = -1
    use_side0 = False
    if elastic_sample0 >= 0:
        elastic_body = b0
        elastic_sample = elastic_sample0
        use_side0 = True
    elif elastic_sample1 >= 0:
        elastic_body = b1
        elastic_sample = elastic_sample1

    if elastic_body != body or elastic_sample < 0:
        return

    cp0_local = rigid_contact_point0[contact_idx]
    cp1_local = rigid_contact_point1[contact_idx]
    cp0_world, cp0_world_prev = evaluate_contact_point_world(
        b0,
        cp0_local,
        elastic_sample0,
        body_q,
        body_q_prev,
        body_elastic_index,
        elastic_joint,
        elastic_mode_count,
        joint_q,
        joint_q_prev,
        joint_q_start,
        elastic_shape_vertex_local,
        elastic_shape_vertex_phi,
        elastic_max_mode_count,
    )
    cp1_world, cp1_world_prev = evaluate_contact_point_world(
        b1,
        cp1_local,
        elastic_sample1,
        body_q,
        body_q_prev,
        body_elastic_index,
        elastic_joint,
        elastic_mode_count,
        joint_q,
        joint_q_prev,
        joint_q_start,
        elastic_shape_vertex_local,
        elastic_shape_vertex_phi,
        elastic_max_mode_count,
    )

    contact_normal = rigid_contact_normal[contact_idx]
    thickness = rigid_contact_margin0[contact_idx] + rigid_contact_margin1[contact_idx]
    dist = wp.dot(contact_normal, cp1_world - cp0_world)
    penetration = thickness - dist
    if penetration <= _SMALL_LENGTH_EPS:
        return

    (
        force_0,
        _torque_0,
        h_ll_0,
        _h_al_0,
        _h_aa_0,
        force_1,
        _torque_1,
        h_ll_1,
        _h_al_1,
        _h_aa_1,
    ) = evaluate_rigid_contact_from_world_points(
        b0,
        b1,
        body_q,
        body_com,
        cp0_world,
        cp1_world,
        cp0_world_prev,
        cp1_world_prev,
        contact_normal,
        penetration,
        contact_material_ke[contact_idx],
        contact_material_kd[contact_idx],
        contact_material_mu[contact_idx],
        friction_epsilon,
        dt,
    )

    elastic_force = force_1
    elastic_h = h_ll_1
    if use_side0:
        elastic_force = force_0
        elastic_h = h_ll_0

    elastic_force_local = body_R_T * elastic_force
    elastic_h_local = body_R_T * (elastic_h * body_R)
    mode_count = elastic_mode_count[elastic_index]
    max_modes = elastic_max_mode_count
    block_vec_start = elastic_index * max_modes
    block_mat_start = elastic_index * max_modes * max_modes

    for i in range(mode_count):
        phi_i_local = elastic_shape_vertex_phi[elastic_sample * max_modes + i]
        wp.atomic_add(
            elastic_mode_block_grad,
            block_vec_start + i,
            -wp.dot(elastic_force_local, phi_i_local),
        )

        for j in range(i, mode_count):
            phi_j_local = elastic_shape_vertex_phi[elastic_sample * max_modes + j]
            h_ij = wp.dot(phi_i_local, elastic_h_local * phi_j_local)
            mat_idx = block_mat_start + i * max_modes + j
            wp.atomic_add(elastic_mode_block_matrix, mat_idx, h_ij)


@wp.kernel
def assemble_elastic_joints(
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
    body_q_prev: wp.array(dtype=wp.transform),
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_constraint_start: wp.array(dtype=wp.int32),
    joint_penalty_k: wp.array(dtype=float),
    joint_penalty_kd: wp.array(dtype=float),
    joint_parent_elastic_endpoint: wp.array(dtype=wp.int32),
    joint_child_elastic_endpoint: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_f: wp.array(dtype=float),
    joint_q_prev: wp.array(dtype=float),
    joint_qd_prev: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
    elastic_mode_block_grad: wp.array(dtype=float),
    elastic_mode_block_delta: wp.array(dtype=float),
    elastic_mode_block_matrix: wp.array(dtype=float),
):
    elastic_index = wp.tid()
    body = elastic_body[elastic_index]
    owner_joint = elastic_joint[elastic_index]
    q_start = joint_q_start[owner_joint] + 7
    qd_start = joint_qd_start[owner_joint] + 6
    mode_start = elastic_mode_start[elastic_index]
    mode_count = elastic_mode_count[elastic_index]
    max_modes = elastic_max_mode_count

    inv_dt = 1.0 / dt
    inv_dt_sq = inv_dt * inv_dt
    body_rot = wp.transform_get_rotation(body_q[body])
    body_R = wp.quat_to_matrix(body_rot)
    body_R_T = wp.transpose(body_R)
    block_vec_start = elastic_index * max_modes
    block_mat_start = elastic_index * max_modes * max_modes

    for i in range(max_modes):
        elastic_mode_block_grad[block_vec_start + i] = 0.0
        elastic_mode_block_delta[block_vec_start + i] = 0.0
        for j in range(max_modes):
            elastic_mode_block_matrix[block_mat_start + i * max_modes + j] = 0.0

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

        elastic_mode_block_grad[block_vec_start + mode] = grad
        elastic_mode_block_matrix[block_mat_start + mode * max_modes + mode] = h

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
        kd = joint_penalty_kd[c_start]

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
        X_pj_prev = _elastic_endpoint_xform(
            joint,
            parent,
            True,
            joint_X_p[joint],
            body_elastic_index,
            elastic_joint,
            elastic_mode_count,
            joint_q_prev,
            joint_q_start,
            joint_parent_elastic_endpoint,
            joint_child_elastic_endpoint,
            elastic_endpoint_phi,
            elastic_max_mode_count,
        )
        X_cj_prev = _elastic_endpoint_xform(
            joint,
            child,
            False,
            joint_X_c[joint],
            body_elastic_index,
            elastic_joint,
            elastic_mode_count,
            joint_q_prev,
            joint_q_start,
            joint_parent_elastic_endpoint,
            joint_child_elastic_endpoint,
            elastic_endpoint_phi,
            elastic_max_mode_count,
        )

        if parent >= 0:
            X_wp = body_q[parent] * X_pj
            X_wp_prev = body_q_prev[parent] * X_pj_prev
        else:
            X_wp = X_pj
            X_wp_prev = X_pj_prev
        X_wc = body_q[child] * X_cj
        X_wc_prev = body_q_prev[child] * X_cj_prev

        x_p = wp.transform_get_translation(X_wp)
        x_c = wp.transform_get_translation(X_wc)
        C = x_c - x_p
        x_p_prev = wp.transform_get_translation(X_wp_prev)
        x_c_prev = wp.transform_get_translation(X_wc_prev)
        C_prev = x_c_prev - x_p_prev

        P = wp.identity(3, float)
        if jt == JointType.PRISMATIC:
            axis = joint_axis[joint_qd_start[joint]]
            axis_w = wp.normalize(wp.quat_rotate(wp.transform_get_rotation(X_wp), axis))
            P = P - wp.outer(axis_w, axis_w)

        PC = P * C
        joint_force = k * PC
        h_scale = k
        if kd > 0.0:
            dC_dt = (C - C_prev) * inv_dt
            joint_force = joint_force + (kd * k) * (P * dC_dt)
            h_scale = h_scale + (kd * inv_dt) * k

        side = elastic_endpoint_side[endpoint]
        joint_force_local = body_R_T * joint_force
        P_local = body_R_T * (P * body_R)
        for i in range(mode_count):
            phi_i_local = elastic_endpoint_phi[endpoint * elastic_max_mode_count + i]
            if side == 0:
                phi_i_local = -phi_i_local
            PdC_i = P_local * phi_i_local
            elastic_mode_block_grad[block_vec_start + i] = elastic_mode_block_grad[block_vec_start + i] + wp.dot(
                joint_force_local, PdC_i
            )

            for j in range(i, mode_count):
                phi_j_local = elastic_endpoint_phi[endpoint * elastic_max_mode_count + j]
                if side == 0:
                    phi_j_local = -phi_j_local
                PdC_j = P_local * phi_j_local
                mat_idx = block_mat_start + i * max_modes + j
                h_ij = h_scale * wp.dot(PdC_i, PdC_j)
                elastic_mode_block_matrix[mat_idx] = elastic_mode_block_matrix[mat_idx] + h_ij


@wp.kernel
def assemble_elastic_contacts(
    dt: float,
    elastic_body: wp.array(dtype=wp.int32),
    elastic_joint: wp.array(dtype=wp.int32),
    elastic_mode_count: wp.array(dtype=wp.int32),
    elastic_max_mode_count: int,
    body_elastic_index: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=wp.int32),
    rigid_contact_max: int,
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=wp.int32),
    rigid_contact_shape1: wp.array(dtype=wp.int32),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_margin0: wp.array(dtype=float),
    rigid_contact_margin1: wp.array(dtype=float),
    rigid_contact_elastic_sample0: wp.array(dtype=wp.int32),
    rigid_contact_elastic_sample1: wp.array(dtype=wp.int32),
    contact_material_ke: wp.array(dtype=float),
    contact_material_kd: wp.array(dtype=float),
    contact_material_mu: wp.array(dtype=float),
    friction_epsilon: float,
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array(dtype=wp.int32),
    body_contact_indices: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    elastic_shape_vertex_local: wp.array(dtype=wp.vec3),
    elastic_shape_vertex_phi: wp.array(dtype=wp.vec3),
    joint_q_prev: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
    elastic_mode_block_grad: wp.array(dtype=float),
    elastic_mode_block_matrix: wp.array(dtype=float),
):
    tid = wp.tid()
    elastic_index = tid // _NUM_ELASTIC_CONTACT_THREADS_PER_BODY
    thread_id_within_body = tid % _NUM_ELASTIC_CONTACT_THREADS_PER_BODY

    body = elastic_body[elastic_index]
    body_rot = wp.transform_get_rotation(body_q[body])
    body_R = wp.quat_to_matrix(body_rot)
    body_R_T = wp.transpose(body_R)

    contact_limit = rigid_contact_count[0]
    if contact_limit > rigid_contact_max:
        contact_limit = rigid_contact_max

    # Use the rigid per-body list when it is complete; otherwise stride the
    # full contact range so contact overflow never truncates reduced forces.
    use_contact_list = False
    contact_count = wp.int32(0)
    if body_contact_buffer_pre_alloc > 0:
        contact_count = body_contact_counts[body]
        use_contact_list = contact_count <= body_contact_buffer_pre_alloc

    if use_contact_list:
        contact_i = thread_id_within_body
        while contact_i < contact_count:
            contact_idx = body_contact_indices[body * body_contact_buffer_pre_alloc + contact_i]
            if contact_idx < contact_limit:
                _accumulate_elastic_contact_modes(
                    dt,
                    contact_idx,
                    body,
                    elastic_index,
                    body_R,
                    body_R_T,
                    elastic_joint,
                    elastic_mode_count,
                    elastic_max_mode_count,
                    body_elastic_index,
                    body_q,
                    body_q_prev,
                    body_com,
                    shape_body,
                    rigid_contact_shape0,
                    rigid_contact_shape1,
                    rigid_contact_point0,
                    rigid_contact_point1,
                    rigid_contact_normal,
                    rigid_contact_margin0,
                    rigid_contact_margin1,
                    rigid_contact_elastic_sample0,
                    rigid_contact_elastic_sample1,
                    contact_material_ke,
                    contact_material_kd,
                    contact_material_mu,
                    friction_epsilon,
                    joint_q_start,
                    elastic_shape_vertex_local,
                    elastic_shape_vertex_phi,
                    joint_q_prev,
                    joint_q,
                    elastic_mode_block_grad,
                    elastic_mode_block_matrix,
                )

            contact_i += _NUM_ELASTIC_CONTACT_THREADS_PER_BODY
    else:
        contact_idx = thread_id_within_body
        while contact_idx < contact_limit:
            _accumulate_elastic_contact_modes(
                dt,
                contact_idx,
                body,
                elastic_index,
                body_R,
                body_R_T,
                elastic_joint,
                elastic_mode_count,
                elastic_max_mode_count,
                body_elastic_index,
                body_q,
                body_q_prev,
                body_com,
                shape_body,
                rigid_contact_shape0,
                rigid_contact_shape1,
                rigid_contact_point0,
                rigid_contact_point1,
                rigid_contact_normal,
                rigid_contact_margin0,
                rigid_contact_margin1,
                rigid_contact_elastic_sample0,
                rigid_contact_elastic_sample1,
                contact_material_ke,
                contact_material_kd,
                contact_material_mu,
                friction_epsilon,
                joint_q_start,
                elastic_shape_vertex_local,
                elastic_shape_vertex_phi,
                joint_q_prev,
                joint_q,
                elastic_mode_block_grad,
                elastic_mode_block_matrix,
            )

            contact_idx += _NUM_ELASTIC_CONTACT_THREADS_PER_BODY


@wp.kernel
def solve_elastic_body(
    dt: float,
    elastic_joint: wp.array(dtype=wp.int32),
    elastic_mode_count: wp.array(dtype=wp.int32),
    elastic_max_mode_count: int,
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_q_prev: wp.array(dtype=float),
    elastic_mode_block_grad: wp.array(dtype=float),
    elastic_mode_block_delta: wp.array(dtype=float),
    elastic_mode_block_matrix: wp.array(dtype=float),
    elastic_mode_block_initial_residual_norm: wp.array(dtype=float),
    elastic_mode_block_solve_residual_norm: wp.array(dtype=float),
    elastic_mode_block_applied_residual_norm: wp.array(dtype=float),
    elastic_mode_block_update_norm: wp.array(dtype=float),
    elastic_mode_block_update_max: wp.array(dtype=float),
    elastic_mode_relaxation: float,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
):
    elastic_index = wp.tid()
    owner_joint = elastic_joint[elastic_index]
    q_start = joint_q_start[owner_joint] + 7
    qd_start = joint_qd_start[owner_joint] + 6
    mode_count = elastic_mode_count[elastic_index]
    max_modes = elastic_max_mode_count

    inv_dt = 1.0 / dt
    block_vec_start = elastic_index * max_modes
    block_mat_start = elastic_index * max_modes * max_modes

    for i in range(mode_count):
        for j in range(i + 1, mode_count):
            elastic_mode_block_matrix[block_mat_start + j * max_modes + i] = elastic_mode_block_matrix[
                block_mat_start + i * max_modes + j
            ]

    # Small dense solve by Gauss-Seidel on the reduced modal block.
    for _sweep in range(max_modes * 2):
        for i in range(mode_count):
            diag = elastic_mode_block_matrix[block_mat_start + i * max_modes + i]
            if diag <= 0.0:
                continue

            rhs = -elastic_mode_block_grad[block_vec_start + i]
            for j in range(mode_count):
                if j != i:
                    rhs = (
                        rhs
                        - elastic_mode_block_matrix[block_mat_start + i * max_modes + j]
                        * elastic_mode_block_delta[block_vec_start + j]
                    )
            elastic_mode_block_delta[block_vec_start + i] = rhs / diag

    initial_residual_sq = float(0.0)
    solve_residual_sq = float(0.0)
    applied_residual_sq = float(0.0)
    update_norm_sq = float(0.0)
    update_max = float(0.0)
    for i in range(mode_count):
        grad_i = elastic_mode_block_grad[block_vec_start + i]
        solve_residual_i = grad_i
        applied_residual_i = grad_i
        for j in range(mode_count):
            h_ij = elastic_mode_block_matrix[block_mat_start + i * max_modes + j]
            delta_j = elastic_mode_block_delta[block_vec_start + j]
            solve_residual_i = solve_residual_i + h_ij * delta_j
            applied_residual_i = applied_residual_i + h_ij * (elastic_mode_relaxation * delta_j)

        applied_delta_i = elastic_mode_relaxation * elastic_mode_block_delta[block_vec_start + i]
        initial_residual_sq = initial_residual_sq + grad_i * grad_i
        solve_residual_sq = solve_residual_sq + solve_residual_i * solve_residual_i
        applied_residual_sq = applied_residual_sq + applied_residual_i * applied_residual_i
        update_norm_sq = update_norm_sq + applied_delta_i * applied_delta_i
        update_max = wp.max(update_max, wp.abs(applied_delta_i))

    elastic_mode_block_initial_residual_norm[elastic_index] = wp.sqrt(initial_residual_sq)
    elastic_mode_block_solve_residual_norm[elastic_index] = wp.sqrt(solve_residual_sq)
    elastic_mode_block_applied_residual_norm[elastic_index] = wp.sqrt(applied_residual_sq)
    elastic_mode_block_update_norm[elastic_index] = wp.sqrt(update_norm_sq)
    elastic_mode_block_update_max[elastic_index] = update_max

    for mode in range(mode_count):
        q_idx = q_start + mode
        qd_idx = qd_start + mode
        q_prev = joint_q_prev[q_idx]
        q_old = joint_q[q_idx]
        q_solved = q_old + elastic_mode_block_delta[block_vec_start + mode]
        q_new = q_old + elastic_mode_relaxation * (q_solved - q_old)
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
