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

"""NCP residual diagnostics for :class:`SolverRaisim`."""

from __future__ import annotations

import math

import numpy as np

from ...sim import Contacts, State
from .solver_raisim import SolverRaisim


def compute_ncp_residuals(
    solver: SolverRaisim,
    state: State,
    contacts: Contacts,
    dt: float,
) -> dict[str, float]:
    """Compute NCP (nonlinear complementarity problem) residuals after a step.

    Reads the solver's internal contact cache and computes three scalar
    residuals that characterise how well the hard-contact complementarity
    conditions are satisfied.

    Args:
        solver: The solver whose internal contact cache to inspect. Must have
            been called with :meth:`~SolverRaisim.step` at least once so that
            the contact cache is populated.
        state: The output state *after* the step (used only for reference;
            contact cache is read from the solver).
        contacts: The contacts object that was passed to the most recent step.
        dt: Time step that was used for the most recent step [s].

    Returns:
        A dict with keys:

        ``r_compl``
            Maximum complementarity residual
            ``max_i |min(lambda_n_i, u_n_i + b_n_i)|``.
        ``r_cone``
            Maximum friction-cone violation
            ``max_i max(||lambda_t_i|| - mu_i * lambda_n_i, 0)``.
        ``r_gap``
            Maximum gap penetration ``max_i max(-gap_i, 0)``.
    """
    if not getattr(solver, "_contact_cache_allocated", False):
        return {"r_compl": 0.0, "r_cone": 0.0, "r_gap": 0.0}

    cnt_arr = contacts.rigid_contact_count.numpy()
    cnt = int(cnt_arr[0]) if cnt_arr.size else 0
    if cnt == 0:
        return {"r_compl": 0.0, "r_cone": 0.0, "r_gap": 0.0}

    # Read contact cache from device
    gap = solver.c_gap.numpy()[:cnt]
    lambda_n = solver.c_lambda_n.numpy()[:cnt]
    lambda_t1 = solver.c_lambda_t1.numpy()[:cnt]
    lambda_t2 = solver.c_lambda_t2.numpy()[:cnt]
    mu = solver.c_mu.numpy()[:cnt]
    bias = solver.c_bias.numpy()[:cnt]  # shape (cnt, 3)

    # Read current joint velocities to compute contact velocity
    normal = solver.c_normal.numpy()[:cnt]
    body_a = solver.c_body_a.numpy()[:cnt]
    body_b = solver.c_body_b.numpy()[:cnt]
    is_free_a = solver.c_is_free_a.numpy()[:cnt]
    is_free_b = solver.c_is_free_b.numpy()[:cnt]
    r_a_arr = solver.c_r_a.numpy()[:cnt]
    r_b_arr = solver.c_r_b.numpy()[:cnt]
    dof_start_a_arr = solver.c_dof_start_a.numpy()[:cnt]
    dof_start_b_arr = solver.c_dof_start_b.numpy()[:cnt]

    joint_qd = state.joint_qd.numpy()

    r_compl = 0.0
    r_cone = 0.0
    r_gap = 0.0

    for i in range(cnt):
        # Gap residual
        r_gap = max(r_gap, -float(gap[i]), 0.0)

        # Contact velocity u_n (positive = separating)
        u_n = 0.0

        # Body A contribution (subtract)
        ba = int(body_a[i])
        if ba >= 0 and int(is_free_a[i]) == 1:
            dof_a = int(dof_start_a_arr[i])
            v_a = joint_qd[dof_a : dof_a + 3]
            w_a = joint_qd[dof_a + 3 : dof_a + 6]
            ra = r_a_arr[i]
            vel_a = v_a + np.cross(w_a, ra)
            u_n -= float(np.dot(vel_a, normal[i]))

        # Body B contribution (add)
        bb = int(body_b[i])
        if bb >= 0 and int(is_free_b[i]) == 1:
            dof_b = int(dof_start_b_arr[i])
            v_b = joint_qd[dof_b : dof_b + 3]
            w_b = joint_qd[dof_b + 3 : dof_b + 6]
            rb = r_b_arr[i]
            vel_b = v_b + np.cross(w_b, rb)
            u_n += float(np.dot(vel_b, normal[i]))

        b_n = float(bias[i][0])
        ln = float(lambda_n[i])

        # Complementarity: min(lambda_n, u_n + b_n) should be 0
        compl = abs(min(ln, u_n + b_n))
        r_compl = max(r_compl, compl)

        # Friction cone: ||lambda_t|| <= mu * lambda_n
        lt1 = float(lambda_t1[i])
        lt2 = float(lambda_t2[i])
        tang_mag = math.sqrt(lt1 * lt1 + lt2 * lt2)
        cone_viol = max(tang_mag - float(mu[i]) * ln, 0.0)
        r_cone = max(r_cone, cone_viol)

    return {"r_compl": r_compl, "r_cone": r_cone, "r_gap": r_gap}
