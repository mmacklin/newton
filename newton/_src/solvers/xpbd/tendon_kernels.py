# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for tendon (cable joint) simulation in the XPBD solver.

Implements the Cable Joints method [Müller et al. SCA 2018] extended with
capstan friction.  All operations are designed for GPU-parallel execution
and are differentiable through Warp autodiff.
"""

import warp as wp

from ...sim.tendon import TendonLinkType


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@wp.func
def tangent_point_circle(
    p: wp.vec3,
    center: wp.vec3,
    radius: float,
    plane_normal: wp.vec3,
    orientation: int,
) -> wp.vec3:
    """Compute the tangent point on a circle from an external point.

    The source point *p* is projected into the cable plane defined by
    *plane_normal* through *center* before computing the tangent, so *p*
    need not lie in the plane.  This matches the original Cable Joints
    paper which works in each cylinder's own profile frame.

    Algorithm 3 from Müller et al. 2018, adapted to 3D cable planes.
    """
    d = center - p
    # project into the cable plane — use in-plane distance for tangent angle
    d_proj = d - wp.dot(d, plane_normal) * plane_normal
    dist_in_plane = wp.length(d_proj)
    if dist_in_plane <= radius:
        if dist_in_plane < 1.0e-8:
            fallback = wp.vec3(1.0, 0.0, 0.0) - wp.dot(wp.vec3(1.0, 0.0, 0.0), plane_normal) * plane_normal
            return center + wp.normalize(fallback) * radius
        return center - wp.normalize(d_proj) * radius

    u = d_proj / dist_in_plane
    v = wp.cross(plane_normal, u)

    phi = wp.asin(wp.min(radius / dist_in_plane, 1.0))

    if orientation > 0:
        angle = -1.5707963 - phi  # -pi/2 - phi
    else:
        angle = 1.5707963 + phi  # +pi/2 + phi

    return center + radius * (wp.cos(angle) * u + wp.sin(angle) * v)


@wp.func
def tangent_circle_circle(
    center_a: wp.vec3,
    radius_a: float,
    orient_a: int,
    center_b: wp.vec3,
    radius_b: float,
    orient_b: int,
    plane_normal: wp.vec3,
) -> wp.vec3:
    """Compute the two tangent points between two circles.

    Returns the tangent point on circle A (first 3 floats) and circle B
    (second 3 floats) packed into two vec3 outputs via the return and
    an output parameter.

    Algorithm 4 from Müller et al. 2018.
    """
    d = center_b - center_a
    dist = wp.length(d)

    # build local frame in cable plane
    d_in_plane = d - wp.dot(d, plane_normal) * plane_normal
    d_in_plane_len = wp.length(d_in_plane)
    if d_in_plane_len < 1.0e-8:
        d_in_plane = wp.vec3(1.0, 0.0, 0.0) - wp.dot(wp.vec3(1.0, 0.0, 0.0), plane_normal) * plane_normal
        d_in_plane_len = wp.length(d_in_plane)
    u = d_in_plane / d_in_plane_len
    v = wp.cross(plane_normal, u)

    dx = wp.dot(d, u)
    dy = wp.dot(d, v)
    alpha = wp.atan2(dy, dx)

    # virtual radius depends on whether orientations match
    same_orient = (orient_a * orient_b) > 0
    if same_orient:
        r_virtual = radius_b - radius_a
    else:
        r_virtual = radius_a + radius_b

    r_abs = wp.abs(r_virtual)
    if dist < r_abs + 1.0e-8:
        # circles overlapping or too close — degenerate
        mid = (center_a + center_b) * 0.5
        return mid

    phi = wp.asin(wp.min(r_abs / dist, 1.0))
    sign_r = 1.0
    if r_virtual < 0.0:
        sign_r = -1.0

    if orient_a > 0:
        angle_a = alpha - 1.5707963 - phi * sign_r
    else:
        angle_a = alpha + 1.5707963 + phi * sign_r

    return center_a + radius_a * (wp.cos(angle_a) * u + wp.sin(angle_a) * v)


@wp.func
def signed_arc_length(
    old_pt: wp.vec3,
    new_pt: wp.vec3,
    center: wp.vec3,
    radius: float,
    plane_normal: wp.vec3,
    orientation: int,
) -> float:
    """Compute signed arc length from old_pt to new_pt on a circle.

    Positive if the arc goes in the direction of *orientation*.
    """
    r_old = old_pt - center
    r_new = new_pt - center
    cross_val = wp.dot(wp.cross(r_old, r_new), plane_normal)
    dot_val = wp.dot(r_old, r_new)
    angle = wp.atan2(cross_val, dot_val)
    return angle * radius * float(orientation)


@wp.func
def wrap_angle(
    pt_left: wp.vec3,
    pt_right: wp.vec3,
    center: wp.vec3,
    plane_normal: wp.vec3,
) -> float:
    """Compute the unsigned wrap angle of cable around a body between two
    attachment points."""
    r_l = pt_left - center
    r_r = pt_right - center
    cross_val = wp.dot(wp.cross(r_l, r_r), plane_normal)
    dot_val = wp.dot(r_l, r_r)
    return wp.abs(wp.atan2(cross_val, dot_val))


# ---------------------------------------------------------------------------
# Tendon solve kernels
# ---------------------------------------------------------------------------

@wp.kernel
def update_tendon_attachments(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    tendon_start: wp.array(dtype=int),
    tendon_link_body: wp.array(dtype=int),
    tendon_link_type: wp.array(dtype=int),
    tendon_link_radius: wp.array(dtype=float),
    tendon_link_orientation: wp.array(dtype=int),
    tendon_link_offset: wp.array(dtype=wp.vec3),
    tendon_link_axis: wp.array(dtype=wp.vec3),
    seg_rest_length: wp.array(dtype=float),
    seg_attachment_l: wp.array(dtype=wp.vec3),
    seg_attachment_r: wp.array(dtype=wp.vec3),
    seg_lambda: wp.array(dtype=float),
    seg_link_l: wp.array(dtype=int),
):
    """Phase 1: Update tangent points and rest lengths for each segment.

    Launched with dim = tendon_segment_count.
    seg_link_l[seg] gives the left link index; right link is seg_link_l[seg]+1.
    """
    seg = wp.tid()
    link_l = seg_link_l[seg]
    link_r = link_l + 1

    body_l = tendon_link_body[link_l]
    body_r = tendon_link_body[link_r]
    type_l = tendon_link_type[link_l]
    type_r = tendon_link_type[link_r]
    radius_l = tendon_link_radius[link_l]
    radius_r = tendon_link_radius[link_r]
    orient_l = tendon_link_orientation[link_l]
    orient_r = tendon_link_orientation[link_r]
    offset_l = tendon_link_offset[link_l]
    offset_r = tendon_link_offset[link_r]
    axis_l = tendon_link_axis[link_l]
    axis_r = tendon_link_axis[link_r]

    pose_l = body_q[body_l]
    pose_r = body_q[body_r]

    # world-space cable plane center and normal for each link
    center_l = wp.transform_point(pose_l, offset_l)
    center_r = wp.transform_point(pose_r, offset_r)
    normal_l = wp.transform_vector(pose_l, axis_l)
    normal_r = wp.transform_vector(pose_r, axis_r)

    # previous attachment points
    old_al = seg_attachment_l[seg]
    old_ar = seg_attachment_r[seg]

    # compute new tangent points based on link types
    new_al = old_al
    new_ar = old_ar

    both_rolling = (type_l == int(TendonLinkType.ROLLING)) and (type_r == int(TendonLinkType.ROLLING))

    if both_rolling and radius_l > 0.0 and radius_r > 0.0:
        # Per-cylinder iterative tangent computation: each tangent is
        # computed in its own cylinder's wrapping plane, then we iterate
        # until the pair converges.  This correctly handles non-coplanar
        # cylinders (e.g. right-angle drives) where averaging normals
        # would pull tangent points off the actual wrapping circles.
        new_al = center_l
        new_ar = center_r
        for _iter in range(4):
            new_ar = tangent_point_circle(new_al, center_r, radius_r, normal_r, orient_r)
            new_al = tangent_point_circle(new_ar, center_l, radius_l, normal_l, -orient_l)

    elif type_l == int(TendonLinkType.ROLLING) and radius_l > 0.0:
        # left is circle, right is a point — departure tangent in left's plane
        new_ar = center_r
        new_al = tangent_point_circle(center_r, center_l, radius_l, normal_l, -orient_l)

    elif type_r == int(TendonLinkType.ROLLING) and radius_r > 0.0:
        # right is circle, left is a point — arrival tangent in right's plane
        new_al = center_l
        new_ar = tangent_point_circle(center_l, center_r, radius_r, normal_r, orient_r)

    else:
        # both are points (pinhole/attachment to pinhole/attachment)
        new_al = center_l
        new_ar = center_r

    seg_attachment_l[seg] = new_al
    seg_attachment_r[seg] = new_ar


@wp.kernel
def distribute_tendon_rest_lengths(
    tendon_start: wp.array(dtype=int),
    tendon_total_cable: wp.array(dtype=float),
    tendon_link_body: wp.array(dtype=int),
    tendon_link_type: wp.array(dtype=int),
    tendon_link_radius: wp.array(dtype=float),
    tendon_link_mu: wp.array(dtype=float),
    tendon_link_offset: wp.array(dtype=wp.vec3),
    tendon_link_axis: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_inv_mass: wp.array(dtype=float),
    seg_rest_length: wp.array(dtype=float),
    seg_attachment_l: wp.array(dtype=wp.vec3),
    seg_attachment_r: wp.array(dtype=wp.vec3),
    seg_link_l: wp.array(dtype=int),
    seg_compliance: wp.array(dtype=float),
):
    """Distribute cable rest length with capstan friction.

    Two regimes:

    **Proportional (frictionless or dynamic friction):** rest length
    proportional to current segment lengths.  Cable slides freely.
    Dynamic friction pulleys couple through ``solve_tendon_noslip``.

    **Ratio-preserved (kinematic friction):** rest-length ratios kept
    from previous substep, scaled to total.  Capstan projection limits
    the extension ratio to exp(mu*theta), locking cable to the fixed body.

    Launched with dim = tendon_count.
    """
    tendon_id = wp.tid()
    link_start = tendon_start[tendon_id]
    link_end = tendon_start[tendon_id + 1]
    num_links = link_end - link_start
    num_segs = num_links - 1
    if num_segs < 2:
        return

    seg_offset = int(0)
    for t in range(tendon_id):
        seg_offset = seg_offset + (tendon_start[t + 1] - tendon_start[t] - 1)

    # only kinematic friction contacts need ratio-preserved rest lengths;
    # dynamic friction pulleys use proportional + noslip coupling
    has_kinematic_friction = int(0)
    for i in range(num_links):
        if i < 1 or i >= num_links - 1:
            continue
        link_idx = link_start + i
        if tendon_link_type[link_idx] == int(TendonLinkType.ROLLING):
            if tendon_link_mu[link_idx] > 0.0:
                body_idx = tendon_link_body[link_idx]
                if body_inv_mass[body_idx] == 0.0:
                    has_kinematic_friction = has_kinematic_friction + 1

    # compute total wrap arc at rolling contacts
    total_wrap = float(0.0)
    for i in range(num_links):
        if i < 1 or i >= num_links - 1:
            continue
        link_idx = link_start + i
        if tendon_link_type[link_idx] == int(TendonLinkType.ROLLING):
            body_idx = tendon_link_body[link_idx]
            pose = body_q[body_idx]
            center = wp.transform_point(pose, tendon_link_offset[link_idx])
            normal = wp.transform_vector(pose, tendon_link_axis[link_idx])
            radius = tendon_link_radius[link_idx]

            seg_left = seg_offset + i - 1
            seg_right = seg_offset + i
            pt_dep = seg_attachment_r[seg_left]
            pt_arr = seg_attachment_l[seg_right]

            theta = wrap_angle(pt_dep, pt_arr, center, normal)
            total_wrap = total_wrap + theta * radius

    total_rest = tendon_total_cable[tendon_id] - total_wrap
    total_rest = wp.max(total_rest, 0.0)

    current_sum = float(0.0)
    total_length = float(0.0)
    for s in range(num_segs):
        seg = seg_offset + s
        current_sum = current_sum + seg_rest_length[seg]
        d = wp.length(seg_attachment_r[seg] - seg_attachment_l[seg])
        total_length = total_length + d

    if has_kinematic_friction > 0:
        # --- kinematic friction: preserve ratios + capstan projection ---
        if current_sum > 1.0e-8:
            scale = total_rest / current_sum
            for s in range(num_segs):
                seg = seg_offset + s
                seg_rest_length[seg] = seg_rest_length[seg] * scale
        elif total_length > 1.0e-8:
            for s in range(num_segs):
                seg = seg_offset + s
                d = wp.length(seg_attachment_r[seg] - seg_attachment_l[seg])
                seg_rest_length[seg] = total_rest * d / total_length

        for i in range(num_links):
            if i < 1 or i >= num_links - 1:
                continue
            link_idx = link_start + i
            if tendon_link_type[link_idx] != int(TendonLinkType.ROLLING):
                continue
            mu = tendon_link_mu[link_idx]
            if mu <= 0.0:
                continue

            body_idx = tendon_link_body[link_idx]
            pose = body_q[body_idx]
            normal_world = wp.transform_vector(pose, tendon_link_axis[link_idx])

            seg_left = seg_offset + i - 1
            seg_right = seg_offset + i

            center = wp.transform_point(pose, tendon_link_offset[link_idx])
            pt_dep = seg_attachment_r[seg_left]
            pt_arr = seg_attachment_l[seg_right]
            theta = wrap_angle(pt_dep, pt_arr, center, normal_world)
            capstan_bound = wp.exp(mu * theta)

            rest_l = seg_rest_length[seg_left]
            rest_r = seg_rest_length[seg_right]
            len_l = wp.length(seg_attachment_r[seg_left] - seg_attachment_l[seg_left])
            len_r = wp.length(seg_attachment_r[seg_right] - seg_attachment_l[seg_right])

            alpha_l = seg_compliance[seg_left]
            alpha_r = seg_compliance[seg_right]
            inv_alpha_l = 1.0
            inv_alpha_r = 1.0
            if alpha_l > 0.0:
                inv_alpha_l = 1.0 / alpha_l
            if alpha_r > 0.0:
                inv_alpha_r = 1.0 / alpha_r

            dx_l = wp.max(len_l - rest_l, 0.0) * inv_alpha_l
            dx_r = wp.max(len_r - rest_r, 0.0) * inv_alpha_r

            eps = 1.0e-8
            if dx_r > dx_l * capstan_bound + eps:
                transfer = (dx_r - capstan_bound * dx_l) * alpha_r / (inv_alpha_l * alpha_r * capstan_bound + 1.0)
                transfer = wp.clamp(transfer, 0.0, rest_l * 0.5)
                seg_rest_length[seg_left] = rest_l - transfer
                seg_rest_length[seg_right] = rest_r + transfer
            elif dx_l > dx_r * capstan_bound + eps:
                transfer = (dx_l - capstan_bound * dx_r) * alpha_l / (inv_alpha_r * alpha_l * capstan_bound + 1.0)
                transfer = wp.clamp(transfer, 0.0, rest_r * 0.5)
                seg_rest_length[seg_left] = rest_l + transfer
                seg_rest_length[seg_right] = rest_r - transfer
    else:
        # --- frictionless or dynamic friction: proportional distribution ---
        if total_length > 1.0e-8:
            for s in range(num_segs):
                seg = seg_offset + s
                d = wp.length(seg_attachment_r[seg] - seg_attachment_l[seg])
                seg_rest_length[seg] = total_rest * d / total_length


@wp.kernel
def solve_tendon_segments(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    tendon_link_body: wp.array(dtype=int),
    tendon_link_type: wp.array(dtype=int),
    tendon_link_mu: wp.array(dtype=float),
    tendon_link_radius: wp.array(dtype=float),
    tendon_link_orientation: wp.array(dtype=int),
    tendon_link_axis: wp.array(dtype=wp.vec3),
    seg_rest_length: wp.array(dtype=float),
    seg_attachment_l: wp.array(dtype=wp.vec3),
    seg_attachment_r: wp.array(dtype=wp.vec3),
    seg_compliance: wp.array(dtype=float),
    seg_damping: wp.array(dtype=float),
    seg_lambda: wp.array(dtype=float),
    seg_link_l: wp.array(dtype=int),
    relaxation: float,
    dt: float,
    # outputs
    body_deltas: wp.array(dtype=wp.spatial_vector),
):
    """Solve unilateral distance constraints for each tendon segment.

    Angular Jacobian is zeroed for all rolling contacts — the distance
    constraint moves weights without directly torquing pulleys.  Pulley
    rotation is handled by the separate ``solve_tendon_noslip`` bilateral
    constraint which only corrects the pulley angle.  Attachment links
    use the standard geometric Jacobian ``cross(r, n)``.

    Launched with dim = tendon_segment_count.
    """
    seg = wp.tid()
    link_l = seg_link_l[seg]
    link_r = link_l + 1

    body_l = tendon_link_body[link_l]
    body_r = tendon_link_body[link_r]

    type_l = tendon_link_type[link_l]
    type_r = tendon_link_type[link_r]
    mu_l = tendon_link_mu[link_l]
    mu_r = tendon_link_mu[link_r]

    pose_l = body_q[body_l]
    pose_r = body_q[body_r]
    vel_l = wp.spatial_top(body_qd[body_l])
    omega_l = wp.spatial_bottom(body_qd[body_l])
    vel_r = wp.spatial_top(body_qd[body_r])
    omega_r = wp.spatial_bottom(body_qd[body_r])

    com_l = body_com[body_l]
    com_r = body_com[body_r]
    m_inv_l = body_inv_mass[body_l]
    m_inv_r = body_inv_mass[body_r]
    I_inv_l = body_inv_inertia[body_l]
    I_inv_r = body_inv_inertia[body_r]

    x_l = seg_attachment_l[seg]
    x_r = seg_attachment_r[seg]
    rest = seg_rest_length[seg]
    compliance = seg_compliance[seg]
    damping = seg_damping[seg]

    diff = x_r - x_l
    d = wp.length(diff)

    # unilateral: only enforce when stretched beyond rest length
    err = d - rest
    if err <= 0.0:
        seg_lambda[seg] = 0.0
        return

    # constraint direction
    n = diff / wp.max(d, 1.0e-8)

    world_com_l = wp.transform_point(pose_l, com_l)
    world_com_r = wp.transform_point(pose_r, com_r)

    r_l = x_l - world_com_l
    r_r = x_r - world_com_r

    linear_l = -n
    linear_r = n
    angular_l = wp.vec3(0.0, 0.0, 0.0)
    angular_r = wp.vec3(0.0, 0.0, 0.0)

    # left link angular Jacobian — zero for rolling (pulley coupling via noslip)
    is_rolling_l = type_l == int(TendonLinkType.ROLLING)
    if not is_rolling_l:
        angular_l = -wp.cross(r_l, n)

    # right link angular Jacobian — zero for rolling
    is_rolling_r = type_r == int(TendonLinkType.ROLLING)
    if not is_rolling_r:
        angular_r = wp.cross(r_r, n)

    # constraint velocity
    derr = (
        wp.dot(linear_l, vel_l)
        + wp.dot(linear_r, vel_r)
        + wp.dot(angular_l, omega_l)
        + wp.dot(angular_r, omega_r)
    )

    # effective mass
    denom = 0.0
    denom += wp.length_sq(linear_l) * m_inv_l
    denom += wp.length_sq(linear_r) * m_inv_r

    rot_l = wp.transform_get_rotation(pose_l)
    rot_r = wp.transform_get_rotation(pose_r)
    rot_ang_l = wp.quat_rotate_inv(rot_l, angular_l)
    rot_ang_r = wp.quat_rotate_inv(rot_r, angular_r)
    denom += wp.dot(rot_ang_l, I_inv_l * rot_ang_l)
    denom += wp.dot(rot_ang_r, I_inv_r * rot_ang_r)

    alpha = compliance
    gamma = compliance * damping

    lambda_prev = seg_lambda[seg]
    d_lambda = -(err + alpha * lambda_prev + gamma * derr)
    if denom + alpha > 0.0:
        d_lambda = d_lambda / ((dt + gamma) * denom + alpha / dt)

    seg_lambda[seg] = lambda_prev + d_lambda

    lin_delta_l = linear_l * (d_lambda * relaxation)
    ang_delta_l = angular_l * (d_lambda * relaxation)
    lin_delta_r = linear_r * (d_lambda * relaxation)
    ang_delta_r = angular_r * (d_lambda * relaxation)

    wp.atomic_add(body_deltas, body_l, wp.spatial_vector(lin_delta_l, ang_delta_l))
    wp.atomic_add(body_deltas, body_r, wp.spatial_vector(lin_delta_r, ang_delta_r))


@wp.kernel
def store_tendon_theta_ref(
    tendon_start: wp.array(dtype=int),
    tendon_link_body: wp.array(dtype=int),
    tendon_link_type: wp.array(dtype=int),
    tendon_link_mu: wp.array(dtype=float),
    tendon_link_axis: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_inv_mass: wp.array(dtype=float),
    coupling_theta_ref: wp.array(dtype=float),
):
    """Store reference pulley angle BEFORE integration.

    Called before the integration step so that the velocity-driven rotation
    is included in Δθ during the iteration loop.

    Launched with dim = tendon_count.
    """
    tendon_id = wp.tid()
    link_start = tendon_start[tendon_id]
    link_end = tendon_start[tendon_id + 1]
    num_links = link_end - link_start

    coupling_theta_ref[tendon_id] = 0.0

    for i in range(num_links):
        if i < 1 or i >= num_links - 1:
            continue
        link_idx = link_start + i
        if tendon_link_type[link_idx] != int(TendonLinkType.ROLLING):
            continue
        if tendon_link_mu[link_idx] <= 0.0:
            continue
        body_idx = tendon_link_body[link_idx]
        if body_inv_mass[body_idx] == 0.0:
            continue

        axis = tendon_link_axis[link_idx]
        q = wp.transform_get_rotation(body_q[body_idx])
        q_vec = wp.vec3(q[0], q[1], q[2])
        coupling_theta_ref[tendon_id] = 2.0 * wp.atan2(wp.dot(q_vec, axis), q[3])
        return


@wp.kernel
def store_tendon_base_rest(
    tendon_start: wp.array(dtype=int),
    tendon_link_body: wp.array(dtype=int),
    tendon_link_type: wp.array(dtype=int),
    tendon_link_mu: wp.array(dtype=float),
    body_inv_mass: wp.array(dtype=float),
    seg_rest_length: wp.array(dtype=float),
    coupling_rest_l_base: wp.array(dtype=float),
    coupling_rest_r_base: wp.array(dtype=float),
):
    """Store base rest lengths after distribute_tendon_rest_lengths.

    Launched with dim = tendon_count.
    """
    tendon_id = wp.tid()
    link_start = tendon_start[tendon_id]
    link_end = tendon_start[tendon_id + 1]
    num_links = link_end - link_start

    coupling_rest_l_base[tendon_id] = 0.0
    coupling_rest_r_base[tendon_id] = 0.0

    seg_offset = int(0)
    for t in range(tendon_id):
        seg_offset = seg_offset + (tendon_start[t + 1] - tendon_start[t] - 1)

    for i in range(num_links):
        if i < 1 or i >= num_links - 1:
            continue
        link_idx = link_start + i
        if tendon_link_type[link_idx] != int(TendonLinkType.ROLLING):
            continue
        if tendon_link_mu[link_idx] <= 0.0:
            continue
        body_idx = tendon_link_body[link_idx]
        if body_inv_mass[body_idx] == 0.0:
            continue

        seg_left = seg_offset + i - 1
        seg_right = seg_offset + i
        coupling_rest_l_base[tendon_id] = seg_rest_length[seg_left]
        coupling_rest_r_base[tendon_id] = seg_rest_length[seg_right]
        return


@wp.kernel
def update_tendon_coupling_rest(
    body_q: wp.array(dtype=wp.transform),
    body_inv_mass: wp.array(dtype=float),
    tendon_start: wp.array(dtype=int),
    tendon_link_body: wp.array(dtype=int),
    tendon_link_type: wp.array(dtype=int),
    tendon_link_mu: wp.array(dtype=float),
    tendon_link_radius: wp.array(dtype=float),
    tendon_link_orientation: wp.array(dtype=int),
    tendon_link_axis: wp.array(dtype=wp.vec3),
    seg_rest_length: wp.array(dtype=float),
    coupling_theta_ref: wp.array(dtype=float),
    coupling_rest_l_base: wp.array(dtype=float),
    coupling_rest_r_base: wp.array(dtype=float),
):
    """Update rest lengths based on current pulley rotation for no-slip coupling.

    Reads the current pulley angle from body_q and sets segment rest lengths
    to enforce the no-slip condition through the distance constraints:
      rest_l = base_l - orient*R*Δθ
      rest_r = base_r + orient*R*Δθ

    When the pulley rotates, cable is transferred between sides.  This shifts
    rest lengths so the distance constraints create the correct tension
    asymmetry to drive the coupled Atwood dynamics.

    Launched with dim = tendon_count, after apply_body_deltas each iteration.
    """
    tendon_id = wp.tid()
    link_start = tendon_start[tendon_id]
    link_end = tendon_start[tendon_id + 1]
    num_links = link_end - link_start
    if num_links < 3:
        return

    seg_offset = int(0)
    for t in range(tendon_id):
        seg_offset = seg_offset + (tendon_start[t + 1] - tendon_start[t] - 1)

    for i in range(num_links):
        if i < 1 or i >= num_links - 1:
            continue
        link_idx = link_start + i
        if tendon_link_type[link_idx] != int(TendonLinkType.ROLLING):
            continue
        if tendon_link_mu[link_idx] <= 0.0:
            continue
        pulley_body = tendon_link_body[link_idx]
        if body_inv_mass[pulley_body] == 0.0:
            continue

        radius = tendon_link_radius[link_idx]
        orient = float(tendon_link_orientation[link_idx])
        axis_local = tendon_link_axis[link_idx]

        q_pulley = wp.transform_get_rotation(body_q[pulley_body])
        q_vec = wp.vec3(q_pulley[0], q_pulley[1], q_pulley[2])
        theta = 2.0 * wp.atan2(wp.dot(q_vec, axis_local), q_pulley[3])
        d_theta = theta - coupling_theta_ref[tendon_id]

        transfer = orient * radius * d_theta

        seg_left = seg_offset + i - 1
        seg_right = seg_offset + i
        seg_rest_length[seg_left] = wp.max(coupling_rest_l_base[tendon_id] - transfer, 1.0e-6)
        seg_rest_length[seg_right] = wp.max(coupling_rest_r_base[tendon_id] + transfer, 1.0e-6)
        return


@wp.func
def _is_dynamic_friction_rolling(
    link_idx: int,
    tendon_link_type: wp.array(dtype=int),
    tendon_link_mu: wp.array(dtype=float),
    tendon_link_body: wp.array(dtype=int),
    body_inv_mass: wp.array(dtype=float),
) -> int:
    if tendon_link_type[link_idx] != int(TendonLinkType.ROLLING):
        return 0
    if tendon_link_mu[link_idx] <= 0.0:
        return 0
    if body_inv_mass[tendon_link_body[link_idx]] == 0.0:
        return 0
    return 1


@wp.kernel
def store_tendon_noslip_ref(
    body_q: wp.array(dtype=wp.transform),
    body_inv_mass: wp.array(dtype=float),
    tendon_link_body: wp.array(dtype=int),
    tendon_link_type: wp.array(dtype=int),
    tendon_link_mu: wp.array(dtype=float),
    tendon_link_radius: wp.array(dtype=float),
    tendon_link_orientation: wp.array(dtype=int),
    tendon_link_offset: wp.array(dtype=wp.vec3),
    tendon_link_axis: wp.array(dtype=wp.vec3),
    seg_attachment_l: wp.array(dtype=wp.vec3),
    seg_attachment_r: wp.array(dtype=wp.vec3),
    seg_link_l: wp.array(dtype=int),
    noslip_ref: wp.array(dtype=float),
):
    """Store the no-slip coupling invariant C_ref = len + sign*orient*R*theta.

    Activates for segments where EITHER link is a dynamic friction rolling
    contact.  sign = +1 when the pulley is the right link, -1 when left.

    Launched with dim = tendon_segment_count.
    """
    seg = wp.tid()
    link_l_idx = seg_link_l[seg]
    link_r_idx = link_l_idx + 1

    is_r = _is_dynamic_friction_rolling(
        link_r_idx, tendon_link_type, tendon_link_mu, tendon_link_body, body_inv_mass
    )
    is_l = _is_dynamic_friction_rolling(
        link_l_idx, tendon_link_type, tendon_link_mu, tendon_link_body, body_inv_mass
    )

    if is_r == 0 and is_l == 0:
        noslip_ref[seg] = 0.0
        return

    sign = 1.0
    weight_idx = link_l_idx
    pulley_idx = link_r_idx
    x_tangent = seg_attachment_r[seg]
    if is_r == 0:
        sign = -1.0
        weight_idx = link_r_idx
        pulley_idx = link_l_idx
        x_tangent = seg_attachment_l[seg]

    weight_body = tendon_link_body[weight_idx]
    x_weight = wp.transform_point(body_q[weight_body], tendon_link_offset[weight_idx])
    seg_len = wp.length(x_tangent - x_weight)

    pulley_body = tendon_link_body[pulley_idx]
    radius = tendon_link_radius[pulley_idx]
    orient = float(tendon_link_orientation[pulley_idx])
    axis = tendon_link_axis[pulley_idx]

    q = wp.transform_get_rotation(body_q[pulley_body])
    q_vec = wp.vec3(q[0], q[1], q[2])
    theta = 2.0 * wp.atan2(wp.dot(q_vec, axis), q[3])

    noslip_ref[seg] = seg_len + sign * orient * radius * theta


@wp.kernel
def solve_tendon_noslip(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    tendon_link_body: wp.array(dtype=int),
    tendon_link_type: wp.array(dtype=int),
    tendon_link_mu: wp.array(dtype=float),
    tendon_link_radius: wp.array(dtype=float),
    tendon_link_orientation: wp.array(dtype=int),
    tendon_link_offset: wp.array(dtype=wp.vec3),
    tendon_link_axis: wp.array(dtype=wp.vec3),
    seg_attachment_l: wp.array(dtype=wp.vec3),
    seg_attachment_r: wp.array(dtype=wp.vec3),
    seg_link_l: wp.array(dtype=int),
    seg_compliance: wp.array(dtype=float),
    seg_damping: wp.array(dtype=float),
    noslip_ref: wp.array(dtype=float),
    noslip_lambda: wp.array(dtype=float),
    relaxation: float,
    dt: float,
    body_deltas: wp.array(dtype=wp.spatial_vector),
):
    """Bilateral no-slip constraint: len + sign*orient*R*theta = const.

    Activates for segments where EITHER link is a dynamic friction rolling
    contact.  sign = +1 when the pulley is the right link, -1 when left.
    The pulley's angular Jacobian is scaled so its effective-mass
    contribution never exceeds the weight's.

    Launched with dim = tendon_segment_count.
    """
    seg = wp.tid()
    link_l_idx = seg_link_l[seg]
    link_r_idx = link_l_idx + 1

    is_r = _is_dynamic_friction_rolling(
        link_r_idx, tendon_link_type, tendon_link_mu, tendon_link_body, body_inv_mass
    )
    is_l = _is_dynamic_friction_rolling(
        link_l_idx, tendon_link_type, tendon_link_mu, tendon_link_body, body_inv_mass
    )

    if is_r == 0 and is_l == 0:
        return

    ref = noslip_ref[seg]
    if ref == 0.0:
        return

    sign = 1.0
    weight_idx = link_l_idx
    pulley_idx = link_r_idx
    x_tangent = seg_attachment_r[seg]
    if is_r == 0:
        sign = -1.0
        weight_idx = link_r_idx
        pulley_idx = link_l_idx
        x_tangent = seg_attachment_l[seg]

    weight_body = tendon_link_body[weight_idx]
    pulley_body = tendon_link_body[pulley_idx]
    offset_local = tendon_link_offset[weight_idx]

    pose_w = body_q[weight_body]
    pose_p = body_q[pulley_body]

    x_weight = wp.transform_point(pose_w, offset_local)
    diff = x_tangent - x_weight
    seg_len = wp.length(diff)
    n = diff / wp.max(seg_len, 1.0e-8)

    radius = tendon_link_radius[pulley_idx]
    orient = float(tendon_link_orientation[pulley_idx])
    axis = tendon_link_axis[pulley_idx]

    q = wp.transform_get_rotation(pose_p)
    q_vec = wp.vec3(q[0], q[1], q[2])
    theta = 2.0 * wp.atan2(wp.dot(q_vec, axis), q[3])

    C = seg_len + sign * orient * radius * theta - ref

    com_w = body_com[weight_body]
    world_com_w = wp.transform_point(pose_w, com_w)
    r_w = x_weight - world_com_w
    J_lin_w = -n
    J_ang_w = -wp.cross(r_w, n)
    J_ang_p = sign * orient * radius * axis

    m_inv_w = body_inv_mass[weight_body]
    I_inv_w = body_inv_inertia[weight_body]
    I_inv_p = body_inv_inertia[pulley_body]

    rot_w = wp.transform_get_rotation(pose_w)
    rot_p = wp.transform_get_rotation(pose_p)
    local_J_ang_w = wp.quat_rotate_inv(rot_w, J_ang_w)
    local_J_ang_p = wp.quat_rotate_inv(rot_p, J_ang_p)

    W_w = m_inv_w * wp.dot(J_lin_w, J_lin_w) + wp.dot(local_J_ang_w, I_inv_w * local_J_ang_w)
    W_p_raw = wp.dot(local_J_ang_p, I_inv_p * local_J_ang_p)

    scale_p = 1.0
    if W_p_raw > W_w and W_p_raw > 1.0e-12:
        scale_p = wp.sqrt(W_w / W_p_raw)

    J_ang_p_eff = J_ang_p * scale_p
    local_J_ang_p_eff = local_J_ang_p * scale_p

    W = W_w + wp.dot(local_J_ang_p_eff, I_inv_p * local_J_ang_p_eff)
    if W < 1.0e-12:
        return

    vel_w = wp.spatial_top(body_qd[weight_body])
    omega_w = wp.spatial_bottom(body_qd[weight_body])
    omega_p = wp.spatial_bottom(body_qd[pulley_body])
    derr = wp.dot(J_lin_w, vel_w) + wp.dot(J_ang_w, omega_w) + wp.dot(J_ang_p_eff, omega_p)

    alpha = seg_compliance[seg]
    gamma = alpha * seg_damping[seg]

    lambda_prev = noslip_lambda[seg]
    d_lambda = -(C + alpha * lambda_prev + gamma * derr)
    if (dt + gamma) * W + alpha > 0.0:
        d_lambda = d_lambda / ((dt + gamma) * W + alpha / dt)
    noslip_lambda[seg] = lambda_prev + d_lambda

    dl = d_lambda * relaxation
    wp.atomic_add(body_deltas, weight_body, wp.spatial_vector(J_lin_w * dl, J_ang_w * dl))
    wp.atomic_add(body_deltas, pulley_body, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), J_ang_p_eff * dl))
