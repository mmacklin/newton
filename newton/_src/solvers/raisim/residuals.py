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

"""NCP and MDP residual diagnostics for :class:`SolverRaisim`."""

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

    t1_arr = solver.c_t1.numpy()[:cnt]
    t2_arr = solver.c_t2.numpy()[:cnt]

    r_compl = 0.0
    r_cone = 0.0
    r_gap = 0.0
    # De Saxcé / MDP residuals
    r_ds_compl = 0.0  # complementarity with Γ correction
    r_ds_dual = 0.0  # dist(c + Γ, K*_μ)
    r_mdp_dir = 0.0  # friction direction error for sliding contacts

    for i in range(cnt):
        # Gap residual
        r_gap = max(r_gap, -float(gap[i]), 0.0)

        n_i = normal[i]
        t1_i = t1_arr[i]
        t2_i = t2_arr[i]

        # Contact velocity (u_n, u_t1, u_t2): positive u_n = separating
        u_n = 0.0
        u_t1 = 0.0
        u_t2 = 0.0

        # Body A contribution (subtract)
        ba = int(body_a[i])
        if ba >= 0 and int(is_free_a[i]) == 1:
            dof_a = int(dof_start_a_arr[i])
            v_a = joint_qd[dof_a : dof_a + 3]
            w_a = joint_qd[dof_a + 3 : dof_a + 6]
            ra = r_a_arr[i]
            vel_a = v_a + np.cross(w_a, ra)
            u_n -= float(np.dot(vel_a, n_i))
            u_t1 -= float(np.dot(vel_a, t1_i))
            u_t2 -= float(np.dot(vel_a, t2_i))

        # Body B contribution (add)
        bb = int(body_b[i])
        if bb >= 0 and int(is_free_b[i]) == 1:
            dof_b = int(dof_start_b_arr[i])
            v_b = joint_qd[dof_b : dof_b + 3]
            w_b = joint_qd[dof_b + 3 : dof_b + 6]
            rb = r_b_arr[i]
            vel_b = v_b + np.cross(w_b, rb)
            u_n += float(np.dot(vel_b, n_i))
            u_t1 += float(np.dot(vel_b, t1_i))
            u_t2 += float(np.dot(vel_b, t2_i))

        b_n = float(bias[i][0])
        ln = float(lambda_n[i])
        lt1 = float(lambda_t1[i])
        lt2 = float(lambda_t2[i])
        mu_i = float(mu[i])

        # --- Standard NCP residuals ---
        # Complementarity: min(lambda_n, u_n + b_n) should be 0
        compl = abs(min(ln, u_n + b_n))
        r_compl = max(r_compl, compl)

        # Friction cone: ||lambda_t|| <= mu * lambda_n
        tang_impulse_mag = math.sqrt(lt1 * lt1 + lt2 * lt2)
        cone_viol = max(tang_impulse_mag - mu_i * ln, 0.0)
        r_cone = max(r_cone, cone_viol)

        # --- De Saxcé / MDP residuals (Le Lidec & Carpentier 2024) ---
        # Augmented velocity: ũ = c + Γ(c, μ) where Γ = [0, 0, μ‖c_T‖]
        c_T_mag = math.sqrt(u_t1 * u_t1 + u_t2 * u_t2)
        u_n_aug = u_n + mu_i * c_T_mag  # augmented normal velocity

        # ε_c: complementarity with de Saxcé correction
        # ⟨λ, c + Γ⟩ = λ_n * (u_n + μ‖c_T‖) + λ_t1 * u_t1 + λ_t2 * u_t2
        ds_inner = ln * u_n_aug + lt1 * u_t1 + lt2 * u_t2
        r_ds_compl = max(r_ds_compl, abs(ds_inner))

        # ε_d: dist(c + Γ, K*_μ) — dual cone feasibility
        # K*_μ is the dual of the friction cone: {v | v_n >= μ‖v_T‖}
        # (or equivalently: the cone with half-angle arctan(1/μ))
        # For the augmented velocity ũ = (u_t1, u_t2, u_n_aug):
        dual_viol = max(mu_i * c_T_mag - u_n_aug, 0.0)
        r_ds_dual = max(r_ds_dual, dual_viol)

        # MDP direction error: for sliding contacts, friction should
        # oppose sliding.  λ_T should be -μ λ_N (c_T / ‖c_T‖).
        if c_T_mag > 1e-8 and ln > 1e-8:
            # Expected friction direction (opposing sliding)
            expected_t1 = -mu_i * ln * (u_t1 / c_T_mag)
            expected_t2 = -mu_i * ln * (u_t2 / c_T_mag)
            # Direction error: angle between actual and expected
            dir_err = math.sqrt((lt1 - expected_t1) ** 2 + (lt2 - expected_t2) ** 2)
            # Normalize by expected magnitude for relative error
            expected_mag = mu_i * ln
            if expected_mag > 1e-8:
                dir_err /= expected_mag
            r_mdp_dir = max(r_mdp_dir, dir_err)

    return {
        "r_compl": r_compl,
        "r_cone": r_cone,
        "r_gap": r_gap,
        "r_ds_compl": r_ds_compl,
        "r_ds_dual": r_ds_dual,
        "r_mdp_dir": r_mdp_dir,
    }
