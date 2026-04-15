# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Warp kernels for the RAISim-style per-contact Gauss-Seidel hard contact solver."""

from __future__ import annotations

import warp as wp

from ...sim import JointType

# Maximum articulation DOFs (supports G1 with 49 DOFs).
MAX_ART_DOFS = wp.constant(64)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


@wp.func
def _dense_idx(stride: int, row: int, col: int) -> int:
    return row * stride + col


@wp.func
def _mat33_from_rows(r0: wp.vec3, r1: wp.vec3, r2: wp.vec3) -> wp.mat33:
    return wp.mat33(r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2])


@wp.func
def _build_tangent_frame(n: wp.vec3) -> wp.vec3:
    """Return a single tangent vector orthogonal to *n*.

    The second tangent is obtained via ``cross(n, t1)``.
    """
    if wp.abs(n[0]) < 0.9:
        t = wp.normalize(wp.cross(n, wp.vec3(1.0, 0.0, 0.0)))
    else:
        t = wp.normalize(wp.cross(n, wp.vec3(0.0, 1.0, 0.0)))
    return t


@wp.func
def _inv_inertia_world(body_q: wp.transform, inv_I_body: wp.mat33) -> wp.mat33:
    """Rotate inverse inertia tensor from body to world frame: R * I^{-1} * R^T."""
    rot = wp.transform_get_rotation(body_q)
    r0 = wp.quat_rotate(rot, wp.vec3(1.0, 0.0, 0.0))
    r1 = wp.quat_rotate(rot, wp.vec3(0.0, 1.0, 0.0))
    r2 = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    R = wp.mat33(r0[0], r1[0], r2[0], r0[1], r1[1], r2[1], r0[2], r1[2], r2[2])
    return R * inv_I_body * wp.transpose(R)


# ---------------------------------------------------------------------------
# Contact cache builder
# ---------------------------------------------------------------------------
# Each contact slot stores:
#   - gap (float)
#   - normal, t1, t2 (vec3 each)
#   - body_a, body_b (int) — body indices (-1 for ground)
#   - is_free_a, is_free_b (int) — 1 if the body is a single-joint free articulation
#   - art_a, art_b (int) — articulation index (-1 if ground)
#   - Gii (mat33) — Delassus diagonal block
#   - Gii_inv (mat33) — pseudo-inverse of Gii (for 3x3 local solve)
#   - bias (vec3) — Baumgarte bias
#   - lambda_n, lambda_t1, lambda_t2 (float) — contact impulse accumulators
#   - mu (float) — friction coefficient
#
# For articulated bodies the kernel also writes into dense buffers
# Wi (nv x 3 per contact) and Jr (3 x nv per contact) stored flat.


@wp.kernel
def build_contact_cache(
    # --- contact data ---
    contact_count: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_margin0: wp.array(dtype=wp.float32),
    contact_margin1: wp.array(dtype=wp.float32),
    # --- model data ---
    shape_body: wp.array(dtype=wp.int32),
    shape_material_mu: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_inv_mass: wp.array(dtype=wp.float32),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    joint_type: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    joint_ancestor: wp.array(dtype=wp.int32),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_articulation: wp.array(dtype=wp.int32),
    articulation_start: wp.array(dtype=wp.int32),
    # --- Featherstone dense data ---
    H_start: wp.array(dtype=wp.int32),
    H_rows: wp.array(dtype=wp.int32),
    dof_start_arr: wp.array(dtype=wp.int32),
    L: wp.array(dtype=wp.float32),
    # --- solver config ---
    erp: float,
    erp_velocity_clamp: float,
    dt: float,
    # --- per-contact outputs ---
    c_gap: wp.array(dtype=wp.float32),
    c_normal: wp.array(dtype=wp.vec3),
    c_t1: wp.array(dtype=wp.vec3),
    c_t2: wp.array(dtype=wp.vec3),
    c_body_a: wp.array(dtype=wp.int32),
    c_body_b: wp.array(dtype=wp.int32),
    c_is_free_a: wp.array(dtype=wp.int32),
    c_is_free_b: wp.array(dtype=wp.int32),
    c_art_a: wp.array(dtype=wp.int32),
    c_art_b: wp.array(dtype=wp.int32),
    c_Gii: wp.array(dtype=wp.mat33),
    c_bias: wp.array(dtype=wp.vec3),
    c_lambda_n: wp.array(dtype=wp.float32),
    c_lambda_t1: wp.array(dtype=wp.float32),
    c_lambda_t2: wp.array(dtype=wp.float32),
    c_mu: wp.array(dtype=wp.float32),
    c_r_a: wp.array(dtype=wp.vec3),
    c_r_b: wp.array(dtype=wp.vec3),
    # --- articulated-body dense buffers (flat) ---
    c_Jr: wp.array(dtype=wp.float32),
    c_Wi: wp.array(dtype=wp.float32),
    c_nv_a: wp.array(dtype=wp.int32),
    c_nv_b: wp.array(dtype=wp.int32),
    c_dof_start_a: wp.array(dtype=wp.int32),
    c_dof_start_b: wp.array(dtype=wp.int32),
    max_contacts: int,
):
    tid = wp.tid()
    cnt = contact_count[0]
    if tid >= cnt:
        return

    # --- unpack contact ---
    p0_raw = contact_point0[tid]
    p1_raw = contact_point1[tid]
    n = contact_normal[tid]

    s0 = contact_shape0[tid]
    s1 = contact_shape1[tid]

    m0 = contact_margin0[tid]
    m1 = contact_margin1[tid]

    # Normalize normal (should already be unit, but be safe)
    n_len = wp.length(n)
    if n_len > 1.0e-10:
        n = n / n_len

    c_normal[tid] = n

    # Body indices (-1 for ground)
    body_a = -1
    body_b = -1
    if s0 >= 0:
        body_a = shape_body[s0]
    if s1 >= 0:
        body_b = shape_body[s1]

    c_body_a[tid] = body_a
    c_body_b[tid] = body_b

    # Contact points in world frame.
    # p0_raw: for ground (body=-1) already world-frame; for dynamic bodies body-local.
    # p1_raw: same convention.
    bx_a = p0_raw
    if body_a >= 0:
        bx_a = wp.transform_point(body_q[body_a], p0_raw)
    bx_b = p1_raw
    if body_b >= 0:
        bx_b = wp.transform_point(body_q[body_b], p1_raw)

    # Actual surface points along the contact normal (n points A→B).
    # Surface of A toward B:  bx_a + m0 * n
    # Surface of B toward A:  bx_b - m1 * n
    surface_a = bx_a + m0 * n
    surface_b = bx_b - m1 * n

    # Signed gap: positive = separated, negative = penetrating.
    gap = wp.dot(n, surface_b - surface_a)
    c_gap[tid] = gap

    # Build tangent frame
    t1 = _build_tangent_frame(n)
    t2 = wp.cross(n, t1)
    c_t1[tid] = t1
    c_t2[tid] = t2

    # Friction coefficient = average of two shapes
    mu = 0.5
    if s0 >= 0 and s1 >= 0:
        mu = 0.5 * (shape_material_mu[s0] + shape_material_mu[s1])
    elif s0 >= 0:
        mu = shape_material_mu[s0]
    elif s1 >= 0:
        mu = shape_material_mu[s1]
    c_mu[tid] = mu

    # Contact point = midpoint between surface points
    cp = 0.5 * (surface_a + surface_b)

    # Lever arms from body COM (world) to contact point
    r_a = wp.vec3(0.0)
    r_b = wp.vec3(0.0)
    if body_a >= 0:
        q_a = body_q[body_a]
        com_a = wp.transform_get_translation(q_a) + wp.quat_rotate(wp.transform_get_rotation(q_a), body_com[body_a])
        r_a = cp - com_a
    if body_b >= 0:
        q_b = body_q[body_b]
        com_b = wp.transform_get_translation(q_b) + wp.quat_rotate(wp.transform_get_rotation(q_b), body_com[body_b])
        r_b = cp - com_b

    c_r_a[tid] = r_a
    c_r_b[tid] = r_b

    # Determine if bodies are free-joint (single-joint free articulation)
    # Use int() casts so Warp treats these as mutable dynamic variables
    is_free_a = int(0)
    is_free_b = int(0)
    art_a = int(-1)
    art_b = int(-1)
    joint_a = int(-1)
    joint_b = int(-1)
    nv_a = int(0)
    nv_b = int(0)
    dof_start_a_val = int(0)
    dof_start_b_val = int(0)

    # Find the joint whose child is body_a by scanning joint_child
    if body_a >= 0:
        num_joints = joint_type.shape[0]
        for j in range(num_joints):
            if joint_child[j] == body_a and joint_a < 0:
                joint_a = j
                art_a = joint_articulation[j]

        if joint_a >= 0 and art_a >= 0:
            art_start = articulation_start[art_a]
            art_end = articulation_start[art_a + 1]
            nv_a = joint_qd_start[art_end] - joint_qd_start[art_start]
            dof_start_a_val = joint_qd_start[art_start]

            # Check if this is a single-joint free articulation
            if art_end - art_start == 1:
                jtype = joint_type[art_start]
                if jtype == JointType.FREE or jtype == JointType.DISTANCE:
                    is_free_a = 1

    # Find the joint whose child is body_b
    if body_b >= 0:
        num_joints = joint_type.shape[0]
        for j in range(num_joints):
            if joint_child[j] == body_b and joint_b < 0:
                joint_b = j
                art_b = joint_articulation[j]

        if joint_b >= 0 and art_b >= 0:
            art_start = articulation_start[art_b]
            art_end = articulation_start[art_b + 1]
            nv_b = joint_qd_start[art_end] - joint_qd_start[art_start]
            dof_start_b_val = joint_qd_start[art_start]

            if art_end - art_start == 1:
                jtype = joint_type[art_start]
                if jtype == JointType.FREE or jtype == JointType.DISTANCE:
                    is_free_b = 1

    c_is_free_a[tid] = is_free_a
    c_is_free_b[tid] = is_free_b
    c_art_a[tid] = art_a
    c_art_b[tid] = art_b
    c_nv_a[tid] = nv_a
    c_nv_b[tid] = nv_b
    c_dof_start_a[tid] = dof_start_a_val
    c_dof_start_b[tid] = dof_start_b_val

    # --- Compute Delassus diagonal block Gii = J_a M_a^{-1} J_a^T + J_b M_b^{-1} J_b^T ---
    G = wp.mat33(0.0)

    # 3 contact directions
    dirs_0 = n
    dirs_1 = t1
    dirs_2 = t2

    # --- Body A contribution ---
    if body_a >= 0 and is_free_a == 1:
        # FREE body: use spatial inertia directly
        inv_m_a = body_inv_mass[body_a]
        inv_I_a = _inv_inertia_world(body_q[body_a], body_inv_inertia[body_a])
        rxn = wp.cross(r_a, dirs_0)
        rxt1 = wp.cross(r_a, dirs_1)
        rxt2 = wp.cross(r_a, dirs_2)
        # G[i][j] = dot(d_i, d_j)*inv_m + dot(cross(r,d_i), inv_I * cross(r,d_j))
        G = G + wp.mat33(
            wp.dot(dirs_0, dirs_0) * inv_m_a + wp.dot(rxn, inv_I_a * rxn),
            wp.dot(dirs_0, dirs_1) * inv_m_a + wp.dot(rxn, inv_I_a * rxt1),
            wp.dot(dirs_0, dirs_2) * inv_m_a + wp.dot(rxn, inv_I_a * rxt2),
            wp.dot(dirs_1, dirs_0) * inv_m_a + wp.dot(rxt1, inv_I_a * rxn),
            wp.dot(dirs_1, dirs_1) * inv_m_a + wp.dot(rxt1, inv_I_a * rxt1),
            wp.dot(dirs_1, dirs_2) * inv_m_a + wp.dot(rxt1, inv_I_a * rxt2),
            wp.dot(dirs_2, dirs_0) * inv_m_a + wp.dot(rxt2, inv_I_a * rxn),
            wp.dot(dirs_2, dirs_1) * inv_m_a + wp.dot(rxt2, inv_I_a * rxt1),
            wp.dot(dirs_2, dirs_2) * inv_m_a + wp.dot(rxt2, inv_I_a * rxt2),
        )

    elif body_a >= 0 and art_a >= 0 and is_free_a == 0:
        # ARTICULATED body: build Jr = contact_Jacobian @ joint_S_s for ancestor chain
        # Jr is 3 x nv_a, Wi is nv_a x 3
        # Store Jr and Wi in flat arrays indexed by contact slot
        Jr_offset_a = tid * wp.static(MAX_ART_DOFS) * 3
        Wi_offset_a = tid * wp.static(MAX_ART_DOFS) * 3

        # Zero Jr
        for d in range(nv_a):
            for row in range(3):
                c_Jr[Jr_offset_a + row * wp.static(MAX_ART_DOFS) + d] = 0.0

        # Build Jr: for each DOF that is an ancestor of body_a, compute Jr[row,dof]
        # Walk ancestor chain starting from joint_a
        j = joint_a
        while j >= 0:
            jdof_start = joint_qd_start[j]
            jdof_end = joint_qd_start[j + 1]

            art_start_j = articulation_start[art_a]
            art_dof_start = joint_qd_start[art_start_j]

            for dof in range(jdof_end - jdof_start):
                S = joint_S_s[jdof_start + dof]
                # S is a spatial vector (v, w) — Newton convention: top=linear, bottom=angular
                S_lin = wp.spatial_top(S)
                S_ang = wp.spatial_bottom(S)

                # Contact Jacobian for direction d_i at contact point:
                # v(cp) = S_lin + cross(S_ang, cp), so
                # Jr[i,dof] = dot(d_i, S_lin) + dot(cross(cp, d_i), S_ang)
                # NB: S_s is expressed at the spatial (world) origin, so the
                # lever arm must be the world-frame contact point, NOT the
                # offset from body COM.
                col = (jdof_start + dof) - art_dof_start
                rxn_a = wp.cross(cp, dirs_0)
                rxt1_a = wp.cross(cp, dirs_1)
                rxt2_a = wp.cross(cp, dirs_2)

                jr_n = wp.dot(dirs_0, S_lin) + wp.dot(rxn_a, S_ang)
                jr_t1 = wp.dot(dirs_1, S_lin) + wp.dot(rxt1_a, S_ang)
                jr_t2 = wp.dot(dirs_2, S_lin) + wp.dot(rxt2_a, S_ang)

                c_Jr[Jr_offset_a + 0 * wp.static(MAX_ART_DOFS) + col] = jr_n
                c_Jr[Jr_offset_a + 1 * wp.static(MAX_ART_DOFS) + col] = jr_t1
                c_Jr[Jr_offset_a + 2 * wp.static(MAX_ART_DOFS) + col] = jr_t2

            j = joint_ancestor[j]

        # Compute Wi = H^{-1} Jr^T using Cholesky solve
        # H^{-1} Jr^T: for each of 3 columns of Jr^T (i.e. each row of Jr),
        # solve L L^T Wi_col = Jr_row
        L_start_a = H_start[art_a]
        n_a = H_rows[art_a]

        for col3 in range(3):
            # RHS = Jr[col3, :]  (length nv_a)
            # Forward substitution: L y = b
            for i in range(n_a):
                s = c_Jr[Jr_offset_a + col3 * wp.static(MAX_ART_DOFS) + i]
                for k in range(i):
                    s -= L[L_start_a + _dense_idx(n_a, i, k)] * c_Wi[Wi_offset_a + col3 * wp.static(MAX_ART_DOFS) + k]
                diag = L[L_start_a + _dense_idx(n_a, i, i)]
                if wp.abs(diag) > 1.0e-20:
                    c_Wi[Wi_offset_a + col3 * wp.static(MAX_ART_DOFS) + i] = s / diag
                else:
                    c_Wi[Wi_offset_a + col3 * wp.static(MAX_ART_DOFS) + i] = 0.0

            # Backward substitution: L^T x = y
            for i_rev in range(n_a):
                i = n_a - 1 - i_rev
                s = c_Wi[Wi_offset_a + col3 * wp.static(MAX_ART_DOFS) + i]
                for k in range(i + 1, n_a):
                    s -= L[L_start_a + _dense_idx(n_a, k, i)] * c_Wi[Wi_offset_a + col3 * wp.static(MAX_ART_DOFS) + k]
                diag = L[L_start_a + _dense_idx(n_a, i, i)]
                if wp.abs(diag) > 1.0e-20:
                    c_Wi[Wi_offset_a + col3 * wp.static(MAX_ART_DOFS) + i] = s / diag
                else:
                    c_Wi[Wi_offset_a + col3 * wp.static(MAX_ART_DOFS) + i] = 0.0

        # Gii_a = Jr Wi = Jr H^{-1} Jr^T
        for i in range(3):
            for j2 in range(3):
                val = float(0.0)
                for k in range(n_a):
                    val += (
                        c_Jr[Jr_offset_a + i * wp.static(MAX_ART_DOFS) + k]
                        * c_Wi[Wi_offset_a + j2 * wp.static(MAX_ART_DOFS) + k]
                    )
                G[i, j2] = G[i, j2] + val

    # --- Body B contribution ---
    if body_b >= 0 and is_free_b == 1:
        inv_m_b = body_inv_mass[body_b]
        inv_I_b = _inv_inertia_world(body_q[body_b], body_inv_inertia[body_b])
        rxn_b = wp.cross(r_b, dirs_0)
        rxt1_b = wp.cross(r_b, dirs_1)
        rxt2_b = wp.cross(r_b, dirs_2)
        G = G + wp.mat33(
            wp.dot(dirs_0, dirs_0) * inv_m_b + wp.dot(rxn_b, inv_I_b * rxn_b),
            wp.dot(dirs_0, dirs_1) * inv_m_b + wp.dot(rxn_b, inv_I_b * rxt1_b),
            wp.dot(dirs_0, dirs_2) * inv_m_b + wp.dot(rxn_b, inv_I_b * rxt2_b),
            wp.dot(dirs_1, dirs_0) * inv_m_b + wp.dot(rxt1_b, inv_I_b * rxn_b),
            wp.dot(dirs_1, dirs_1) * inv_m_b + wp.dot(rxt1_b, inv_I_b * rxt1_b),
            wp.dot(dirs_1, dirs_2) * inv_m_b + wp.dot(rxt1_b, inv_I_b * rxt2_b),
            wp.dot(dirs_2, dirs_0) * inv_m_b + wp.dot(rxt2_b, inv_I_b * rxn_b),
            wp.dot(dirs_2, dirs_1) * inv_m_b + wp.dot(rxt2_b, inv_I_b * rxt1_b),
            wp.dot(dirs_2, dirs_2) * inv_m_b + wp.dot(rxt2_b, inv_I_b * rxt2_b),
        )

    elif body_b >= 0 and art_b >= 0 and is_free_b == 0:
        # Articulated body B — same logic as body A but for the second body
        # We store body B's Jr and Wi in the second half of the per-contact slot
        Jr_offset_b = (max_contacts + tid) * wp.static(MAX_ART_DOFS) * 3
        Wi_offset_b = (max_contacts + tid) * wp.static(MAX_ART_DOFS) * 3

        for d in range(nv_b):
            for row in range(3):
                c_Jr[Jr_offset_b + row * wp.static(MAX_ART_DOFS) + d] = 0.0

        j = joint_b
        while j >= 0:
            jdof_start = joint_qd_start[j]
            jdof_end = joint_qd_start[j + 1]

            art_start_j = articulation_start[art_b]
            art_dof_start = joint_qd_start[art_start_j]

            for dof in range(jdof_end - jdof_start):
                S = joint_S_s[jdof_start + dof]
                S_lin = wp.spatial_top(S)
                S_ang = wp.spatial_bottom(S)

                # S_s is expressed at the spatial (world) origin — use
                # world-frame contact point as lever arm, not body-COM offset.
                col = (jdof_start + dof) - art_dof_start
                rxn_b2 = wp.cross(cp, dirs_0)
                rxt1_b2 = wp.cross(cp, dirs_1)
                rxt2_b2 = wp.cross(cp, dirs_2)

                jr_n = wp.dot(dirs_0, S_lin) + wp.dot(rxn_b2, S_ang)
                jr_t1 = wp.dot(dirs_1, S_lin) + wp.dot(rxt1_b2, S_ang)
                jr_t2 = wp.dot(dirs_2, S_lin) + wp.dot(rxt2_b2, S_ang)

                c_Jr[Jr_offset_b + 0 * wp.static(MAX_ART_DOFS) + col] = jr_n
                c_Jr[Jr_offset_b + 1 * wp.static(MAX_ART_DOFS) + col] = jr_t1
                c_Jr[Jr_offset_b + 2 * wp.static(MAX_ART_DOFS) + col] = jr_t2

            j = joint_ancestor[j]

        L_start_b = H_start[art_b]
        n_b = H_rows[art_b]

        for col3 in range(3):
            for i in range(n_b):
                s = c_Jr[Jr_offset_b + col3 * wp.static(MAX_ART_DOFS) + i]
                for k in range(i):
                    s -= L[L_start_b + _dense_idx(n_b, i, k)] * c_Wi[Wi_offset_b + col3 * wp.static(MAX_ART_DOFS) + k]
                diag = L[L_start_b + _dense_idx(n_b, i, i)]
                if wp.abs(diag) > 1.0e-20:
                    c_Wi[Wi_offset_b + col3 * wp.static(MAX_ART_DOFS) + i] = s / diag
                else:
                    c_Wi[Wi_offset_b + col3 * wp.static(MAX_ART_DOFS) + i] = 0.0

            for i_rev in range(n_b):
                i = n_b - 1 - i_rev
                s = c_Wi[Wi_offset_b + col3 * wp.static(MAX_ART_DOFS) + i]
                for k in range(i + 1, n_b):
                    s -= L[L_start_b + _dense_idx(n_b, k, i)] * c_Wi[Wi_offset_b + col3 * wp.static(MAX_ART_DOFS) + k]
                diag = L[L_start_b + _dense_idx(n_b, i, i)]
                if wp.abs(diag) > 1.0e-20:
                    c_Wi[Wi_offset_b + col3 * wp.static(MAX_ART_DOFS) + i] = s / diag
                else:
                    c_Wi[Wi_offset_b + col3 * wp.static(MAX_ART_DOFS) + i] = 0.0

        for i in range(3):
            for j2 in range(3):
                val = float(0.0)
                for k in range(n_b):
                    val += (
                        c_Jr[Jr_offset_b + i * wp.static(MAX_ART_DOFS) + k]
                        * c_Wi[Wi_offset_b + j2 * wp.static(MAX_ART_DOFS) + k]
                    )
                G[i, j2] = G[i, j2] + val

    c_Gii[tid] = G

    # Baumgarte bias
    b_n = 0.0
    if gap < 0.0:
        b_n = -erp * gap / dt
        b_n = wp.min(b_n, erp_velocity_clamp)

    c_bias[tid] = wp.vec3(b_n, 0.0, 0.0)

    # Initialize accumulators (will be overwritten if warmstarting)
    c_lambda_n[tid] = 0.0
    c_lambda_t1[tid] = 0.0
    c_lambda_t2[tid] = 0.0


# ---------------------------------------------------------------------------
# Gauss-Seidel contact sweep (serial, dim=1)
# ---------------------------------------------------------------------------


@wp.kernel
def gs_contact_sweep(
    # --- config ---
    contact_count: wp.array(dtype=wp.int32),
    max_gs_iterations: int,
    tolerance: float,
    # --- per-contact cache ---
    c_gap: wp.array(dtype=wp.float32),
    c_normal: wp.array(dtype=wp.vec3),
    c_t1: wp.array(dtype=wp.vec3),
    c_t2: wp.array(dtype=wp.vec3),
    c_body_a: wp.array(dtype=wp.int32),
    c_body_b: wp.array(dtype=wp.int32),
    c_is_free_a: wp.array(dtype=wp.int32),
    c_is_free_b: wp.array(dtype=wp.int32),
    c_art_a: wp.array(dtype=wp.int32),
    c_art_b: wp.array(dtype=wp.int32),
    c_Gii: wp.array(dtype=wp.mat33),
    c_bias: wp.array(dtype=wp.vec3),
    c_mu: wp.array(dtype=wp.float32),
    c_r_a: wp.array(dtype=wp.vec3),
    c_r_b: wp.array(dtype=wp.vec3),
    # --- articulated-body buffers ---
    c_Jr: wp.array(dtype=wp.float32),
    c_Wi: wp.array(dtype=wp.float32),
    c_nv_a: wp.array(dtype=wp.int32),
    c_nv_b: wp.array(dtype=wp.int32),
    c_dof_start_a: wp.array(dtype=wp.int32),
    c_dof_start_b: wp.array(dtype=wp.int32),
    # --- model data ---
    body_q: wp.array(dtype=wp.transform),
    body_inv_mass: wp.array(dtype=wp.float32),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    articulation_start: wp.array(dtype=wp.int32),
    # --- mutable state ---
    joint_qd: wp.array(dtype=wp.float32),
    # --- accumulators (in/out) ---
    c_lambda_n: wp.array(dtype=wp.float32),
    c_lambda_t1: wp.array(dtype=wp.float32),
    c_lambda_t2: wp.array(dtype=wp.float32),
    max_contacts: int,
):
    """Serial GS sweep over all active contacts. Launched with dim=1."""
    cnt = contact_count[0]
    if cnt == 0:
        return

    for _gs_iter in range(max_gs_iterations):
        max_residual = float(0.0)

        for ci in range(cnt):
            gap = c_gap[ci]
            # Skip separated contacts
            if gap > 0.0:
                continue

            n = c_normal[ci]
            t1 = c_t1[ci]
            t2 = c_t2[ci]
            body_a = c_body_a[ci]
            body_b = c_body_b[ci]
            is_free_a = c_is_free_a[ci]
            is_free_b = c_is_free_b[ci]
            art_a = c_art_a[ci]
            art_b = c_art_b[ci]
            G = c_Gii[ci]
            bias = c_bias[ci]
            mu = c_mu[ci]
            r_a = c_r_a[ci]
            r_b = c_r_b[ci]

            # --- Compute contact velocity: u = J_b v_b - J_a v_a ---
            # Positive u_n means separating
            u_n = float(0.0)
            u_t1 = float(0.0)
            u_t2 = float(0.0)

            # Body A contribution (subtract because normal points A->B)
            if body_a >= 0 and is_free_a == 1:
                # Free body: read directly from joint_qd
                dof_a = c_dof_start_a[ci]
                v_a = wp.vec3(joint_qd[dof_a + 0], joint_qd[dof_a + 1], joint_qd[dof_a + 2])
                w_a = wp.vec3(joint_qd[dof_a + 3], joint_qd[dof_a + 4], joint_qd[dof_a + 5])
                vel_a = v_a + wp.cross(w_a, r_a)
                u_n -= wp.dot(vel_a, n)
                u_t1 -= wp.dot(vel_a, t1)
                u_t2 -= wp.dot(vel_a, t2)

            elif body_a >= 0 and art_a >= 0 and is_free_a == 0:
                # Articulated body: u_dir = Jr @ joint_qd
                Jr_off_a = ci * wp.static(MAX_ART_DOFS) * 3
                nv_a = c_nv_a[ci]
                dof_a = c_dof_start_a[ci]
                jr_n_val = float(0.0)
                jr_t1_val = float(0.0)
                jr_t2_val = float(0.0)
                for d in range(nv_a):
                    qd_d = joint_qd[dof_a + d]
                    jr_n_val += c_Jr[Jr_off_a + 0 * wp.static(MAX_ART_DOFS) + d] * qd_d
                    jr_t1_val += c_Jr[Jr_off_a + 1 * wp.static(MAX_ART_DOFS) + d] * qd_d
                    jr_t2_val += c_Jr[Jr_off_a + 2 * wp.static(MAX_ART_DOFS) + d] * qd_d
                u_n -= jr_n_val
                u_t1 -= jr_t1_val
                u_t2 -= jr_t2_val

            # Body B contribution (add)
            if body_b >= 0 and is_free_b == 1:
                dof_b = c_dof_start_b[ci]
                v_b = wp.vec3(joint_qd[dof_b + 0], joint_qd[dof_b + 1], joint_qd[dof_b + 2])
                w_b = wp.vec3(joint_qd[dof_b + 3], joint_qd[dof_b + 4], joint_qd[dof_b + 5])
                vel_b = v_b + wp.cross(w_b, r_b)
                u_n += wp.dot(vel_b, n)
                u_t1 += wp.dot(vel_b, t1)
                u_t2 += wp.dot(vel_b, t2)

            elif body_b >= 0 and art_b >= 0 and is_free_b == 0:
                Jr_off_b = (max_contacts + ci) * wp.static(MAX_ART_DOFS) * 3
                nv_b = c_nv_b[ci]
                dof_b = c_dof_start_b[ci]
                jr_n_val2 = float(0.0)
                jr_t1_val2 = float(0.0)
                jr_t2_val2 = float(0.0)
                for d in range(nv_b):
                    qd_d = joint_qd[dof_b + d]
                    jr_n_val2 += c_Jr[Jr_off_b + 0 * wp.static(MAX_ART_DOFS) + d] * qd_d
                    jr_t1_val2 += c_Jr[Jr_off_b + 1 * wp.static(MAX_ART_DOFS) + d] * qd_d
                    jr_t2_val2 += c_Jr[Jr_off_b + 2 * wp.static(MAX_ART_DOFS) + d] * qd_d
                u_n += jr_n_val2
                u_t1 += jr_t1_val2
                u_t2 += jr_t2_val2

            # --- Normal impulse ---
            G_nn = G[0, 0]
            if G_nn < 1.0e-20:
                continue

            old_lambda_n = c_lambda_n[ci]
            delta_lambda_n = (bias[0] - u_n) / G_nn
            new_lambda_n = wp.max(old_lambda_n + delta_lambda_n, 0.0)
            delta_lambda_n = new_lambda_n - old_lambda_n
            c_lambda_n[ci] = new_lambda_n

            # --- Tangential impulse (friction) ---
            old_lambda_t1 = c_lambda_t1[ci]
            old_lambda_t2 = c_lambda_t2[ci]

            # 2x2 tangential solve
            G_t1t1 = G[1, 1]
            G_t2t2 = G[2, 2]
            G_t1t2 = G[1, 2]

            # Account for normal impulse coupling into tangential directions
            u_t1_eff = u_t1 + G[1, 0] * delta_lambda_n
            u_t2_eff = u_t2 + G[2, 0] * delta_lambda_n

            # Solve 2x2 system [G_t1t1 G_t1t2; G_t1t2 G_t2t2] [dl_t1; dl_t2] = [-u_t1_eff; -u_t2_eff]
            det = G_t1t1 * G_t2t2 - G_t1t2 * G_t1t2
            delta_lambda_t1 = float(0.0)
            delta_lambda_t2 = float(0.0)
            if wp.abs(det) > 1.0e-20:
                delta_lambda_t1 = (-u_t1_eff * G_t2t2 + u_t2_eff * G_t1t2) / det
                delta_lambda_t2 = (u_t1_eff * G_t1t2 - u_t2_eff * G_t1t1) / det

            new_lambda_t1 = old_lambda_t1 + delta_lambda_t1
            new_lambda_t2 = old_lambda_t2 + delta_lambda_t2

            # Project onto Coulomb friction cone
            friction_limit = mu * new_lambda_n
            tang_mag = wp.sqrt(new_lambda_t1 * new_lambda_t1 + new_lambda_t2 * new_lambda_t2)
            if tang_mag > friction_limit and tang_mag > 1.0e-20:
                scale = friction_limit / tang_mag
                new_lambda_t1 = new_lambda_t1 * scale
                new_lambda_t2 = new_lambda_t2 * scale

            delta_lambda_t1 = new_lambda_t1 - old_lambda_t1
            delta_lambda_t2 = new_lambda_t2 - old_lambda_t2
            c_lambda_t1[ci] = new_lambda_t1
            c_lambda_t2[ci] = new_lambda_t2

            # --- Apply velocity corrections ---
            # Total impulse in contact frame: [delta_lambda_n, delta_lambda_t1, delta_lambda_t2]

            # Body A (negative because normal points A->B, so impulse pushes A in -n direction)
            if body_a >= 0 and is_free_a == 1:
                inv_m_a = body_inv_mass[body_a]
                inv_I_a = _inv_inertia_world(body_q[body_a], body_inv_inertia[body_a])
                impulse_a = -(n * delta_lambda_n + t1 * delta_lambda_t1 + t2 * delta_lambda_t2)
                torque_a = wp.cross(r_a, impulse_a)
                dv_a = impulse_a * inv_m_a
                dw_a = inv_I_a * torque_a
                dof_a = c_dof_start_a[ci]
                joint_qd[dof_a + 0] = joint_qd[dof_a + 0] + dv_a[0]
                joint_qd[dof_a + 1] = joint_qd[dof_a + 1] + dv_a[1]
                joint_qd[dof_a + 2] = joint_qd[dof_a + 2] + dv_a[2]
                joint_qd[dof_a + 3] = joint_qd[dof_a + 3] + dw_a[0]
                joint_qd[dof_a + 4] = joint_qd[dof_a + 4] + dw_a[1]
                joint_qd[dof_a + 5] = joint_qd[dof_a + 5] + dw_a[2]

            elif body_a >= 0 and art_a >= 0 and is_free_a == 0:
                # joint_qd -= Wi @ delta_lambda  (negative sign for body A)
                Wi_off_a = ci * wp.static(MAX_ART_DOFS) * 3
                nv_a = c_nv_a[ci]
                dof_a = c_dof_start_a[ci]
                for d in range(nv_a):
                    dqd = float(0.0)
                    dqd -= c_Wi[Wi_off_a + 0 * wp.static(MAX_ART_DOFS) + d] * delta_lambda_n
                    dqd -= c_Wi[Wi_off_a + 1 * wp.static(MAX_ART_DOFS) + d] * delta_lambda_t1
                    dqd -= c_Wi[Wi_off_a + 2 * wp.static(MAX_ART_DOFS) + d] * delta_lambda_t2
                    joint_qd[dof_a + d] = joint_qd[dof_a + d] + dqd

            # Body B (positive because impulse pushes B in +n direction)
            if body_b >= 0 and is_free_b == 1:
                inv_m_b = body_inv_mass[body_b]
                inv_I_b = _inv_inertia_world(body_q[body_b], body_inv_inertia[body_b])
                impulse_b = n * delta_lambda_n + t1 * delta_lambda_t1 + t2 * delta_lambda_t2
                torque_b = wp.cross(r_b, impulse_b)
                dv_b = impulse_b * inv_m_b
                dw_b = inv_I_b * torque_b
                dof_b = c_dof_start_b[ci]
                joint_qd[dof_b + 0] = joint_qd[dof_b + 0] + dv_b[0]
                joint_qd[dof_b + 1] = joint_qd[dof_b + 1] + dv_b[1]
                joint_qd[dof_b + 2] = joint_qd[dof_b + 2] + dv_b[2]
                joint_qd[dof_b + 3] = joint_qd[dof_b + 3] + dw_b[0]
                joint_qd[dof_b + 4] = joint_qd[dof_b + 4] + dw_b[1]
                joint_qd[dof_b + 5] = joint_qd[dof_b + 5] + dw_b[2]

            elif body_b >= 0 and art_b >= 0 and is_free_b == 0:
                Wi_off_b = (max_contacts + ci) * wp.static(MAX_ART_DOFS) * 3
                nv_b = c_nv_b[ci]
                dof_b = c_dof_start_b[ci]
                for d in range(nv_b):
                    dqd = float(0.0)
                    dqd += c_Wi[Wi_off_b + 0 * wp.static(MAX_ART_DOFS) + d] * delta_lambda_n
                    dqd += c_Wi[Wi_off_b + 1 * wp.static(MAX_ART_DOFS) + d] * delta_lambda_t1
                    dqd += c_Wi[Wi_off_b + 2 * wp.static(MAX_ART_DOFS) + d] * delta_lambda_t2
                    joint_qd[dof_b + d] = joint_qd[dof_b + d] + dqd

            # Track residual
            res = wp.abs(delta_lambda_n) + wp.abs(delta_lambda_t1) + wp.abs(delta_lambda_t2)
            max_residual = wp.max(max_residual, res)

        # Early termination
        if max_residual < tolerance:
            return


# ---------------------------------------------------------------------------
# Velocity update kernel
# ---------------------------------------------------------------------------


@wp.kernel
def update_joint_velocity(
    joint_qd_in: wp.array(dtype=wp.float32),
    joint_qdd: wp.array(dtype=wp.float32),
    angular_damping: float,
    dt: float,
    # outputs
    joint_qd_out: wp.array(dtype=wp.float32),
):
    """Compute predicted velocity: qd_out = qd_in + qdd * dt, with angular damping."""
    tid = wp.tid()
    joint_qd_out[tid] = joint_qd_in[tid] + joint_qdd[tid] * dt


@wp.kernel
def apply_angular_damping(
    joint_type: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    angular_damping: float,
    dt: float,
    # in/out
    joint_qd: wp.array(dtype=wp.float32),
):
    """Apply angular damping to angular DOFs of each joint."""
    j = wp.tid()
    dof_start = joint_qd_start[j]
    lin_count = joint_dof_dim[j, 0]
    ang_count = joint_dof_dim[j, 1]

    damp = 1.0 - angular_damping * dt
    if damp < 0.0:
        damp = 0.0

    # Angular DOFs start after linear DOFs
    for i in range(ang_count):
        idx = dof_start + lin_count + i
        joint_qd[idx] = joint_qd[idx] * damp


# ---------------------------------------------------------------------------
# Joint position integration (semi-implicit Euler)
# ---------------------------------------------------------------------------


@wp.kernel
def integrate_joint_positions(
    joint_type: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=wp.float32),
):
    """Semi-implicit Euler integration: q_new = q + qd * dt, with proper quaternion handling."""
    j = wp.tid()
    jtype = joint_type[j]
    coord_start = joint_q_start[j]
    dof_start = joint_qd_start[j]
    lin_count = joint_dof_dim[j, 0]
    ang_count = joint_dof_dim[j, 1]

    if jtype == JointType.FIXED:
        return

    if jtype == JointType.PRISMATIC or jtype == JointType.REVOLUTE:
        q = joint_q[coord_start]
        qd = joint_qd[dof_start]
        joint_q_new[coord_start] = q + qd * dt
        return

    if jtype == JointType.BALL:
        w = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
        r = wp.quat(
            joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2], joint_q[coord_start + 3]
        )
        drdt = wp.quat(w, 0.0) * r * 0.5
        r_new = wp.normalize(r + drdt * dt)
        joint_q_new[coord_start + 0] = r_new[0]
        joint_q_new[coord_start + 1] = r_new[1]
        joint_q_new[coord_start + 2] = r_new[2]
        joint_q_new[coord_start + 3] = r_new[3]
        return

    if jtype == JointType.FREE or jtype == JointType.DISTANCE:
        v = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
        w = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        p = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])
        r = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )

        dpdt = v + wp.cross(w, p)
        drdt = wp.quat(w, 0.0) * r * 0.5

        p_new = p + dpdt * dt
        r_new = wp.normalize(r + drdt * dt)

        joint_q_new[coord_start + 0] = p_new[0]
        joint_q_new[coord_start + 1] = p_new[1]
        joint_q_new[coord_start + 2] = p_new[2]
        joint_q_new[coord_start + 3] = r_new[0]
        joint_q_new[coord_start + 4] = r_new[1]
        joint_q_new[coord_start + 5] = r_new[2]
        joint_q_new[coord_start + 6] = r_new[3]
        return

    if jtype == JointType.D6:
        axis_count = lin_count + ang_count
        for i in range(axis_count):
            q = joint_q[coord_start + i]
            qd = joint_qd[dof_start + i]
            joint_q_new[coord_start + i] = q + qd * dt
        return


# ---------------------------------------------------------------------------
# Joint limit projection
# ---------------------------------------------------------------------------


@wp.kernel
def project_joint_limits(
    joint_type: wp.array(dtype=wp.int32),
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    dt: float,
    # in/out
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    """Hard-clamp joint positions to [lower, upper] and zero violating velocities."""
    j = wp.tid()
    jtype = joint_type[j]
    coord_start = joint_q_start[j]
    dof_start = joint_qd_start[j]
    lin_count = joint_dof_dim[j, 0]
    ang_count = joint_dof_dim[j, 1]

    # Only applies to 1-DOF joints (prismatic, revolute) and D6
    if jtype == JointType.PRISMATIC or jtype == JointType.REVOLUTE:
        lower = joint_limit_lower[dof_start]
        upper = joint_limit_upper[dof_start]
        if lower < upper:
            q = joint_q[coord_start]
            if q < lower:
                joint_q[coord_start] = lower
                qd = joint_qd[dof_start]
                if qd < 0.0:
                    joint_qd[dof_start] = 0.0
            elif q > upper:
                joint_q[coord_start] = upper
                qd = joint_qd[dof_start]
                if qd > 0.0:
                    joint_qd[dof_start] = 0.0
        return

    if jtype == JointType.D6:
        axis_count = lin_count + ang_count
        for i in range(axis_count):
            lower = joint_limit_lower[dof_start + i]
            upper = joint_limit_upper[dof_start + i]
            if lower < upper:
                q = joint_q[coord_start + i]
                if q < lower:
                    joint_q[coord_start + i] = lower
                    qd = joint_qd[dof_start + i]
                    if qd < 0.0:
                        joint_qd[dof_start + i] = 0.0
                elif q > upper:
                    joint_q[coord_start + i] = upper
                    qd = joint_qd[dof_start + i]
                    if qd > 0.0:
                        joint_qd[dof_start + i] = 0.0
        return
