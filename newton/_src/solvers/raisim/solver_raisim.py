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

"""RAISim-style per-contact Gauss-Seidel hard contact solver.

Combines Featherstone articulated-body smooth dynamics (FK, ID, mass matrix,
Cholesky) with a per-contact Gauss-Seidel hard contact solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import BodyFlags, Contacts, Control, JointType, Model, State
from ..featherstone.kernels import (
    compute_com_transforms,
    compute_spatial_inertia,
    convert_body_force_com_to_origin,
    copy_kinematic_joint_state,
    eval_dense_cholesky_batched,
    eval_dense_gemm_batched,
    eval_dense_solve_batched,
    eval_fk_with_velocity_conversion,
    eval_rigid_fk,
    eval_rigid_id,
    eval_rigid_jacobian,
    eval_rigid_mass,
    eval_rigid_tau,
    zero_kinematic_body_forces,
    zero_kinematic_joint_qdd,
)
from ..flags import SolverNotifyFlags
from ..semi_implicit.kernels_contact import (
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
)
from ..semi_implicit.kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)
from ..solver import SolverBase
from .kernels import (
    MAX_ART_DOFS,
    apply_angular_damping,
    build_contact_cache,
    gs_contact_sweep,
    integrate_joint_positions,
    project_joint_limits,
    update_joint_velocity,
)


@dataclass
class RaisimConfig:
    """Configuration for :class:`SolverRaisim`.

    Attributes:
        max_gs_iterations: Maximum Gauss-Seidel iterations per step.
        tolerance: Complementarity residual tolerance for early termination.
        erp: Error reduction parameter (Baumgarte stabilization).
        erp_velocity_clamp: Maximum Baumgarte stabilization velocity [m/s].
        angular_damping: Angular damping factor applied each step.
        warmstart: Whether to warmstart contact impulses from previous step.
        update_mass_matrix_interval: How often to recompute the mass matrix
            (every n-th call to :meth:`SolverRaisim.step`).
        armature_min: Minimum diagonal regularization added to the mass matrix
            before Cholesky factorization [kg m^2]. Prevents numerical
            instability for articulations with near-zero effective inertia.
    """

    max_gs_iterations: int = 50
    tolerance: float = 1e-6
    erp: float = 0.2
    erp_velocity_clamp: float = 2.0
    angular_damping: float = 0.05
    warmstart: bool = True
    update_mass_matrix_interval: int = 1
    armature_min: float = 1e-1


class SolverRaisim(SolverBase):
    """RAISim-style per-contact Gauss-Seidel hard contact solver.

    This solver combines Featherstone articulated-body smooth dynamics
    (forward kinematics, inverse dynamics, mass matrix via CRBA, Cholesky
    decomposition) with a per-contact Gauss-Seidel (GS) hard contact solve
    inspired by the RAISim simulator.

    The step pipeline is:

    1. **Smooth dynamics** -- reuse Featherstone kernels to compute
       ``qdd_smooth``, then predict ``u_predict = u + h * qdd_smooth``.
    2. **Contact cache** -- for each collision contact, compute gap, contact
       Jacobians, Delassus diagonal block, and Baumgarte bias.
    3. **GS contact sweep** -- iterate over contacts solving a local 3-D
       complementarity problem (normal + Coulomb friction cone).
    4. **Integration** -- semi-implicit Euler on joint coordinates, hard
       joint-limit projection, and final forward kinematics.

    Note:
        This solver operates on generalized (reduced) coordinates and
        requires articulations with explicit joints (including
        :meth:`~newton.ModelBuilder.add_joint_free` for floating bases).

    Example:
        .. code-block:: python

            config = newton.solvers.RaisimConfig(max_gs_iterations=100)
            solver = newton.solvers.SolverRaisim(model, config=config)

            for i in range(100):
                solver.step(state_in, state_out, control, contacts, dt)
                state_in, state_out = state_out, state_in
    """

    def __init__(self, model: Model, config: RaisimConfig | None = None):
        """
        Args:
            model: The model to be simulated.
            config: Solver configuration. Uses defaults if ``None``.
        """
        super().__init__(model)

        if config is None:
            config = RaisimConfig()
        self.config = config

        self._step_count = 0
        self._mass_matrix_dirty = False

        self._update_kinematic_state()
        self._compute_articulation_indices(model)
        self._allocate_model_aux_vars(model)
        self._contact_cache_allocated = False

    # ------------------------------------------------------------------
    # Kinematic body handling (mirrored from SolverFeatherstone)
    # ------------------------------------------------------------------

    def _update_kinematic_state(self) -> None:
        """Recompute cached kinematic body/joint flags and effective armature."""
        model = self.model
        self.has_kinematic_bodies = False
        self.has_kinematic_joints = False

        # Start from model armature, enforce minimum regularization for
        # multi-joint articulations (single-joint free bodies are already
        # well-conditioned and should not be regularized).
        joint_armature = model.joint_armature.numpy().copy()
        armature_min = self.config.armature_min
        needs_copy = False
        if armature_min > 0.0 and model.joint_count:
            articulation_start = model.articulation_start.numpy()
            joint_type = model.joint_type.numpy()
            joint_qd_start = model.joint_qd_start.numpy()
            for art_idx in range(model.articulation_count):
                art_first = articulation_start[art_idx]
                art_last = articulation_start[art_idx + 1]
                joint_count_art = art_last - art_first
                # Skip single-joint free/distance articulations
                if joint_count_art == 1:
                    jtype = joint_type[art_first]
                    if jtype == int(JointType.FREE) or jtype == int(JointType.DISTANCE):
                        continue
                dof_start = int(joint_qd_start[art_first])
                dof_end = int(joint_qd_start[art_last])
                np.maximum(joint_armature[dof_start:dof_end], armature_min, out=joint_armature[dof_start:dof_end])
                needs_copy = True

        if model.body_count:
            body_flags = model.body_flags.numpy()
            kinematic_mask = (body_flags & int(BodyFlags.KINEMATIC)) != 0
            self.has_kinematic_bodies = bool(np.any(kinematic_mask))
            if model.joint_count and self.has_kinematic_bodies:
                joint_child = model.joint_child.numpy()
                joint_qd_start = model.joint_qd_start.numpy()
                for joint_idx in range(model.joint_count):
                    if not kinematic_mask[joint_child[joint_idx]]:
                        continue
                    self.has_kinematic_joints = True
                    dof_start = int(joint_qd_start[joint_idx])
                    dof_end = int(joint_qd_start[joint_idx + 1])
                    joint_armature[dof_start:dof_end] = 1.0e10
                needs_copy = True

        if needs_copy:
            self.joint_armature_effective = wp.array(joint_armature, dtype=float, device=model.device)
        else:
            self.joint_armature_effective = model.joint_armature

    @override
    def notify_model_changed(self, flags: int) -> None:
        if flags & (SolverNotifyFlags.BODY_PROPERTIES | SolverNotifyFlags.JOINT_DOF_PROPERTIES):
            self._update_kinematic_state()
            self._mass_matrix_dirty = True

    # ------------------------------------------------------------------
    # Articulation indexing (identical to SolverFeatherstone)
    # ------------------------------------------------------------------

    def _compute_articulation_indices(self, model: Model) -> None:
        """Build batched matrix offsets for Jacobian, mass matrix, and Cholesky."""
        if not model.joint_count:
            return

        self.J_size = 0
        self.M_size = 0
        self.H_size = 0

        articulation_J_start: list[int] = []
        articulation_M_start: list[int] = []
        articulation_H_start: list[int] = []

        articulation_M_rows: list[int] = []
        articulation_H_rows: list[int] = []
        articulation_J_rows: list[int] = []
        articulation_J_cols: list[int] = []

        articulation_dof_start: list[int] = []
        articulation_coord_start: list[int] = []

        articulation_start = model.articulation_start.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        for i in range(model.articulation_count):
            first_joint = articulation_start[i]
            last_joint = articulation_start[i + 1]

            first_coord = joint_q_start[first_joint]
            first_dof = joint_qd_start[first_joint]
            last_dof = joint_qd_start[last_joint]

            joint_count = last_joint - first_joint
            dof_count = last_dof - first_dof

            articulation_J_start.append(self.J_size)
            articulation_M_start.append(self.M_size)
            articulation_H_start.append(self.H_size)
            articulation_dof_start.append(first_dof)
            articulation_coord_start.append(first_coord)

            articulation_M_rows.append(joint_count * 6)
            articulation_H_rows.append(dof_count)
            articulation_J_rows.append(joint_count * 6)
            articulation_J_cols.append(dof_count)

            self.J_size += 6 * joint_count * dof_count
            self.M_size += 6 * joint_count * 6 * joint_count
            self.H_size += dof_count * dof_count

        dev = model.device
        self.articulation_J_start = wp.array(articulation_J_start, dtype=wp.int32, device=dev)
        self.articulation_M_start = wp.array(articulation_M_start, dtype=wp.int32, device=dev)
        self.articulation_H_start = wp.array(articulation_H_start, dtype=wp.int32, device=dev)

        self.articulation_M_rows = wp.array(articulation_M_rows, dtype=wp.int32, device=dev)
        self.articulation_H_rows = wp.array(articulation_H_rows, dtype=wp.int32, device=dev)
        self.articulation_J_rows = wp.array(articulation_J_rows, dtype=wp.int32, device=dev)
        self.articulation_J_cols = wp.array(articulation_J_cols, dtype=wp.int32, device=dev)

        self.articulation_dof_start = wp.array(articulation_dof_start, dtype=wp.int32, device=dev)
        self.articulation_coord_start = wp.array(articulation_coord_start, dtype=wp.int32, device=dev)

    # ------------------------------------------------------------------
    # Model auxiliary variable allocation (mirrors SolverFeatherstone)
    # ------------------------------------------------------------------

    def _allocate_model_aux_vars(self, model: Model) -> None:
        """Allocate mass, Jacobian matrices, and other auxiliary variables."""
        rg = model.requires_grad
        dev = model.device

        if model.joint_count:
            self.M = wp.zeros((self.M_size,), dtype=wp.float32, device=dev, requires_grad=rg)
            self.J = wp.zeros((self.J_size,), dtype=wp.float32, device=dev, requires_grad=rg)
            self.P = wp.empty_like(self.J, requires_grad=rg)
            self.H = wp.empty((self.H_size,), dtype=wp.float32, device=dev, requires_grad=rg)
            self.L = wp.zeros_like(self.H)

        if model.body_count:
            self.body_I_m = wp.empty((model.body_count,), dtype=wp.spatial_matrix, device=dev, requires_grad=rg)
            wp.launch(
                compute_spatial_inertia,
                model.body_count,
                inputs=[model.body_inertia, model.body_mass],
                outputs=[self.body_I_m],
                device=dev,
            )
            self.body_X_com = wp.empty((model.body_count,), dtype=wp.transform, device=dev, requires_grad=rg)
            wp.launch(
                compute_com_transforms,
                model.body_count,
                inputs=[model.body_com],
                outputs=[self.body_X_com],
                device=dev,
            )

    # ------------------------------------------------------------------
    # State-level auxiliary buffers (Featherstone temporaries)
    # ------------------------------------------------------------------

    def _allocate_state_aux_vars(self, model: Model, target: object, requires_grad: bool) -> None:
        """Allocate per-state auxiliary variables for Featherstone dynamics."""
        dev = model.device

        if model.body_count:
            target.joint_qdd = wp.zeros_like(model.joint_qd, requires_grad=requires_grad)
            target.joint_tau = wp.empty_like(model.joint_qd, requires_grad=requires_grad)
            if requires_grad:
                target.joint_solve_tmp = wp.zeros_like(model.joint_qd, requires_grad=True)
            else:
                target.joint_solve_tmp = None
            target.joint_S_s = wp.empty(
                (model.joint_dof_count,),
                dtype=wp.spatial_vector,
                device=dev,
                requires_grad=requires_grad,
            )
            target.body_q_com = wp.empty_like(model.body_q, requires_grad=requires_grad)
            target.body_I_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=dev, requires_grad=requires_grad
            )
            target.body_v_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=dev, requires_grad=requires_grad
            )
            target.body_a_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=dev, requires_grad=requires_grad
            )
            target.body_f_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=dev, requires_grad=requires_grad
            )
            target.body_ft_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=dev, requires_grad=requires_grad
            )
            target._raisim_augmented = True

    # ------------------------------------------------------------------
    # Contact cache allocation
    # ------------------------------------------------------------------

    def _allocate_contact_cache(self, max_contacts: int) -> None:
        """Allocate per-contact cache arrays for the GS solver."""
        dev = self.model.device
        self._max_contacts = max_contacts
        mc = max_contacts

        self.c_gap = wp.zeros(mc, dtype=wp.float32, device=dev)
        self.c_normal = wp.zeros(mc, dtype=wp.vec3, device=dev)
        self.c_t1 = wp.zeros(mc, dtype=wp.vec3, device=dev)
        self.c_t2 = wp.zeros(mc, dtype=wp.vec3, device=dev)
        self.c_body_a = wp.full(mc, -1, dtype=wp.int32, device=dev)
        self.c_body_b = wp.full(mc, -1, dtype=wp.int32, device=dev)
        self.c_is_free_a = wp.zeros(mc, dtype=wp.int32, device=dev)
        self.c_is_free_b = wp.zeros(mc, dtype=wp.int32, device=dev)
        self.c_art_a = wp.full(mc, -1, dtype=wp.int32, device=dev)
        self.c_art_b = wp.full(mc, -1, dtype=wp.int32, device=dev)
        self.c_Gii = wp.zeros(mc, dtype=wp.mat33, device=dev)
        self.c_bias = wp.zeros(mc, dtype=wp.vec3, device=dev)
        self.c_lambda_n = wp.zeros(mc, dtype=wp.float32, device=dev)
        self.c_lambda_t1 = wp.zeros(mc, dtype=wp.float32, device=dev)
        self.c_lambda_t2 = wp.zeros(mc, dtype=wp.float32, device=dev)
        self.c_mu = wp.zeros(mc, dtype=wp.float32, device=dev)
        self.c_r_a = wp.zeros(mc, dtype=wp.vec3, device=dev)
        self.c_r_b = wp.zeros(mc, dtype=wp.vec3, device=dev)

        # Jr and Wi: 2*mc slots (mc for body A, mc for body B), each 3 x MAX_ART_DOFS
        jr_wi_size = 2 * mc * int(MAX_ART_DOFS) * 3
        self.c_Jr = wp.zeros(jr_wi_size, dtype=wp.float32, device=dev)
        self.c_Wi = wp.zeros(jr_wi_size, dtype=wp.float32, device=dev)

        self.c_nv_a = wp.zeros(mc, dtype=wp.int32, device=dev)
        self.c_nv_b = wp.zeros(mc, dtype=wp.int32, device=dev)
        self.c_dof_start_a = wp.zeros(mc, dtype=wp.int32, device=dev)
        self.c_dof_start_b = wp.zeros(mc, dtype=wp.int32, device=dev)

        self._contact_cache_allocated = True

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance the simulation by one time step.

        Args:
            state_in: The input state.
            state_out: The output state.
            control: The control input. ``None`` uses model defaults.
            contacts: The contact information.
            dt: The time step [s].
        """
        requires_grad = state_in.requires_grad
        model = self.model
        dev = model.device
        cfg = self.config

        # Auxiliary buffer target (avoid graph-capture issues with grad)
        if requires_grad:
            state_aug = state_out
        else:
            state_aug = self

        if not getattr(state_aug, "_raisim_augmented", False):
            self._allocate_state_aux_vars(model, state_aug, requires_grad)
        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("raisim_step", False):
            # ============================================================
            # Phase 0: Evaluate particle and body forces (from semi-implicit)
            # ============================================================
            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_in.body_f
                wp.launch(
                    convert_body_force_com_to_origin,
                    dim=model.body_count,
                    inputs=[state_in.body_q, self.body_X_com],
                    outputs=[body_f],
                    device=dev,
                )

            eval_spring_forces(model, state_in, particle_f)
            eval_triangle_forces(model, state_in, control, particle_f)
            eval_bending_forces(model, state_in, particle_f)
            eval_tetrahedra_forces(model, state_in, control, particle_f)
            eval_particle_contact_forces(model, state_in, particle_f)
            eval_particle_body_contact_forces(model, state_in, contacts, particle_f, body_f, body_f_in_world_frame=True)

            # ============================================================
            # Phase 1: Smooth dynamics via Featherstone
            # ============================================================
            if model.joint_count:
                # FK
                wp.launch(
                    eval_rigid_fk,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_in.joint_q,
                        model.joint_X_p,
                        model.joint_X_c,
                        self.body_X_com,
                        model.joint_axis,
                        model.joint_dof_dim,
                    ],
                    outputs=[state_in.body_q, state_aug.body_q_com],
                    device=dev,
                )

                # Inverse dynamics (Recursive Newton-Euler)
                state_aug.body_f_s.zero_()
                wp.launch(
                    eval_rigid_id,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_qd_start,
                        state_in.joint_qd,
                        model.joint_axis,
                        model.joint_dof_dim,
                        self.body_I_m,
                        state_in.body_q,
                        state_aug.body_q_com,
                        model.joint_X_p,
                        model.body_world,
                        model.gravity,
                    ],
                    outputs=[
                        state_aug.joint_S_s,
                        state_aug.body_I_s,
                        state_aug.body_v_s,
                        state_aug.body_f_s,
                        state_aug.body_a_s,
                    ],
                    device=dev,
                )

                # Zero kinematic body forces
                if self.has_kinematic_bodies and body_f is not None:
                    wp.launch(
                        zero_kinematic_body_forces,
                        dim=model.body_count,
                        inputs=[model.body_flags],
                        outputs=[body_f],
                        device=dev,
                    )

                if model.articulation_count:
                    # Joint torques (backward pass of RNEA)
                    state_aug.body_ft_s.zero_()
                    wp.launch(
                        eval_rigid_tau,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.joint_type,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_q_start,
                            model.joint_qd_start,
                            model.joint_dof_dim,
                            control.joint_target_pos,
                            control.joint_target_vel,
                            state_in.joint_q,
                            state_in.joint_qd,
                            control.joint_f,
                            model.joint_target_ke,
                            model.joint_target_kd,
                            model.joint_limit_lower,
                            model.joint_limit_upper,
                            model.joint_limit_ke,
                            model.joint_limit_kd,
                            state_aug.joint_S_s,
                            state_aug.body_f_s,
                            body_f,
                        ],
                        outputs=[
                            state_aug.body_ft_s,
                            state_aug.joint_tau,
                        ],
                        device=dev,
                    )

                    # Mass matrix + Cholesky (optionally cached)
                    if self._mass_matrix_dirty or self._step_count % cfg.update_mass_matrix_interval == 0:
                        # Build J
                        wp.launch(
                            eval_rigid_jacobian,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                self.articulation_J_start,
                                model.joint_ancestor,
                                model.joint_qd_start,
                                state_aug.joint_S_s,
                            ],
                            outputs=[self.J],
                            device=dev,
                        )

                        # Build M
                        wp.launch(
                            eval_rigid_mass,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                self.articulation_M_start,
                                state_aug.body_I_s,
                            ],
                            outputs=[self.M],
                            device=dev,
                        )

                        # P = M*J
                        wp.launch(
                            eval_dense_gemm_batched,
                            dim=model.articulation_count,
                            inputs=[
                                self.articulation_M_rows,
                                self.articulation_J_cols,
                                self.articulation_J_rows,
                                False,
                                False,
                                self.articulation_M_start,
                                self.articulation_J_start,
                                self.articulation_J_start,
                                self.M,
                                self.J,
                            ],
                            outputs=[self.P],
                            device=dev,
                        )

                        # H = J^T * P
                        wp.launch(
                            eval_dense_gemm_batched,
                            dim=model.articulation_count,
                            inputs=[
                                self.articulation_J_cols,
                                self.articulation_J_cols,
                                self.articulation_J_rows,
                                True,
                                False,
                                self.articulation_J_start,
                                self.articulation_J_start,
                                self.articulation_H_start,
                                self.J,
                                self.P,
                            ],
                            outputs=[self.H],
                            device=dev,
                        )

                        # Cholesky decomposition
                        wp.launch(
                            eval_dense_cholesky_batched,
                            dim=model.articulation_count,
                            inputs=[
                                self.articulation_H_start,
                                self.articulation_H_rows,
                                self.articulation_dof_start,
                                self.H,
                                self.joint_armature_effective,
                            ],
                            outputs=[self.L],
                            device=dev,
                        )
                        self._mass_matrix_dirty = False

                    # Solve for qdd_smooth: H qdd = tau
                    state_aug.joint_qdd.zero_()
                    wp.launch(
                        eval_dense_solve_batched,
                        dim=model.articulation_count,
                        inputs=[
                            self.articulation_H_start,
                            self.articulation_H_rows,
                            self.articulation_dof_start,
                            self.H,
                            self.L,
                            state_aug.joint_tau,
                        ],
                        outputs=[
                            state_aug.joint_qdd,
                            state_aug.joint_solve_tmp,
                        ],
                        device=dev,
                    )

                    if self.has_kinematic_joints:
                        wp.launch(
                            zero_kinematic_joint_qdd,
                            dim=model.joint_count,
                            inputs=[model.joint_child, model.body_flags, model.joint_qd_start],
                            outputs=[state_aug.joint_qdd],
                            device=dev,
                        )

            # ============================================================
            # Phase 1b: Predict velocity: u_predict = u + h * qdd_smooth
            # ============================================================
            if model.joint_count:
                # u_predict -> state_out.joint_qd
                wp.launch(
                    update_joint_velocity,
                    dim=model.joint_dof_count,
                    inputs=[
                        state_in.joint_qd,
                        state_aug.joint_qdd,
                        cfg.angular_damping,
                        dt,
                    ],
                    outputs=[state_out.joint_qd],
                    device=dev,
                )

                # Apply angular damping
                wp.launch(
                    apply_angular_damping,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        cfg.angular_damping,
                        dt,
                    ],
                    outputs=[state_out.joint_qd],
                    device=dev,
                )

            # ============================================================
            # Phase 2 & 3: Contact cache + GS sweep
            # ============================================================
            has_contacts = contacts is not None and contacts.rigid_contact_max > 0 and model.joint_count > 0

            if has_contacts:
                mc = contacts.rigid_contact_max

                # Allocate or resize contact cache
                if not self._contact_cache_allocated or self._max_contacts < mc:
                    self._allocate_contact_cache(mc)

                # Build contact cache
                wp.launch(
                    build_contact_cache,
                    dim=mc,
                    inputs=[
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        model.shape_body,
                        model.shape_material_mu,
                        state_in.body_q,
                        model.body_com,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.joint_type,
                        model.joint_qd_start,
                        model.joint_child,
                        model.joint_ancestor,
                        state_aug.joint_S_s,
                        model.joint_articulation,
                        model.articulation_start,
                        self.articulation_H_start,
                        self.articulation_H_rows,
                        self.articulation_dof_start,
                        self.L,
                        cfg.erp,
                        cfg.erp_velocity_clamp,
                        dt,
                    ],
                    outputs=[
                        self.c_gap,
                        self.c_normal,
                        self.c_t1,
                        self.c_t2,
                        self.c_body_a,
                        self.c_body_b,
                        self.c_is_free_a,
                        self.c_is_free_b,
                        self.c_art_a,
                        self.c_art_b,
                        self.c_Gii,
                        self.c_bias,
                        self.c_lambda_n,
                        self.c_lambda_t1,
                        self.c_lambda_t2,
                        self.c_mu,
                        self.c_r_a,
                        self.c_r_b,
                        self.c_Jr,
                        self.c_Wi,
                        self.c_nv_a,
                        self.c_nv_b,
                        self.c_dof_start_a,
                        self.c_dof_start_b,
                        mc,
                    ],
                    device=dev,
                )

                # GS sweep (serial, dim=1, modifies state_out.joint_qd in-place)
                wp.launch(
                    gs_contact_sweep,
                    dim=1,
                    inputs=[
                        contacts.rigid_contact_count,
                        cfg.max_gs_iterations,
                        cfg.tolerance,
                        self.c_gap,
                        self.c_normal,
                        self.c_t1,
                        self.c_t2,
                        self.c_body_a,
                        self.c_body_b,
                        self.c_is_free_a,
                        self.c_is_free_b,
                        self.c_art_a,
                        self.c_art_b,
                        self.c_Gii,
                        self.c_bias,
                        self.c_mu,
                        self.c_r_a,
                        self.c_r_b,
                        self.c_Jr,
                        self.c_Wi,
                        self.c_nv_a,
                        self.c_nv_b,
                        self.c_dof_start_a,
                        self.c_dof_start_b,
                        state_in.body_q,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.articulation_start,
                    ],
                    outputs=[
                        state_out.joint_qd,
                        self.c_lambda_n,
                        self.c_lambda_t1,
                        self.c_lambda_t2,
                        mc,
                    ],
                    device=dev,
                )

            # ============================================================
            # Phase 4: Integration
            # ============================================================
            if model.joint_count:
                # Integrate positions: q_new = q + qd * dt
                wp.launch(
                    integrate_joint_positions,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        state_in.joint_q,
                        state_out.joint_qd,
                        dt,
                    ],
                    outputs=[state_out.joint_q],
                    device=dev,
                )

                # Project joint limits
                wp.launch(
                    project_joint_limits,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        dt,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd],
                    device=dev,
                )

                # Copy kinematic joint state through
                if self.has_kinematic_joints:
                    wp.launch(
                        copy_kinematic_joint_state,
                        dim=model.joint_count,
                        inputs=[
                            model.joint_child,
                            model.body_flags,
                            model.joint_q_start,
                            model.joint_qd_start,
                            state_in.joint_q,
                            state_in.joint_qd,
                        ],
                        outputs=[state_out.joint_q, state_out.joint_qd],
                        device=dev,
                    )

                # Final FK with velocity conversion
                eval_fk_with_velocity_conversion(model, state_out.joint_q, state_out.joint_qd, state_out)

            # Integrate particles
            self.integrate_particles(model, state_in, state_out, dt)

            self._step_count += 1
