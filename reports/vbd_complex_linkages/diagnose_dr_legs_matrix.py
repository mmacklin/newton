#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect the sparse VBD system at DR Legs' first ground contact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.vbd.rigid_sparse_articulation_kernels import solve_articulation_sparse_serial
from newton._src.solvers.vbd.rigid_vbd_kernels import accumulate_body_body_contacts_per_body
from reports.vbd_complex_linkages.bench_complex_linkages import (
    DR_LEGS_MODES,
    _joint_anchor_residuals,
    _make_dr_legs_solver,
    build_dr_legs_model,
)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av, aw = a[:3], a[3]
    bv, bw = b[:3], b[3]
    return np.concatenate((aw * bv + bw * av + np.cross(av, bv), [aw * bw - np.dot(av, bv)]))


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return v + 2.0 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)


def _perturb_pose_about_com(pose: np.ndarray, com_local: np.ndarray, delta: np.ndarray) -> np.ndarray:
    position = pose[:3]
    rotation = pose[3:]
    com = position + _quat_rotate(rotation, com_local)
    angle = np.linalg.norm(delta[3:])
    if angle > 0.0:
        dq = np.concatenate((delta[3:] / angle * np.sin(0.5 * angle), [np.cos(0.5 * angle)]))
        rotation = _quat_mul(dq, rotation)
        rotation /= np.linalg.norm(rotation)
    com += delta[:3]
    return np.concatenate((com - _quat_rotate(rotation, com_local), rotation))


def _frozen_normal_contact_energy(
    body_q: np.ndarray,
    body_q_prev: np.ndarray | None,
    shape_body: np.ndarray,
    shape0: np.ndarray,
    shape1: np.ndarray,
    point0: np.ndarray,
    point1: np.ndarray,
    normal: np.ndarray,
    margin0: np.ndarray,
    margin1: np.ndarray,
    penalty_k: np.ndarray,
    material_kd: np.ndarray | None,
    dt: float,
) -> float:
    energy = 0.0
    for i in range(len(shape0)):
        body0 = int(shape_body[shape0[i]]) if shape0[i] >= 0 else -1
        body1 = int(shape_body[shape1[i]]) if shape1[i] >= 0 else -1
        p0 = point0[i] if body0 < 0 else body_q[body0, :3] + _quat_rotate(body_q[body0, 3:], point0[i])
        p1 = point1[i] if body1 < 0 else body_q[body1, :3] + _quat_rotate(body_q[body1, 3:], point1[i])
        penetration = -(np.dot(normal[i], p1 - p0) - margin0[i] - margin1[i])
        if penetration > 0.0:
            energy += 0.5 * penalty_k[i] * penetration * penetration
            if body_q_prev is not None and material_kd is not None and material_kd[i] > 0.0:
                p0_prev = (
                    point0[i] if body0 < 0 else body_q_prev[body0, :3] + _quat_rotate(body_q_prev[body0, 3:], point0[i])
                )
                p1_prev = (
                    point1[i] if body1 < 0 else body_q_prev[body1, :3] + _quat_rotate(body_q_prev[body1, 3:], point1[i])
                )
                normal_displacement = np.dot(normal[i], p1 - p1_prev - p0 + p0_prev)
                if normal_displacement < 0.0:
                    energy += 0.5 * material_kd[i] / dt * normal_displacement * normal_displacement
    return float(energy)


def _normal_contact_gradient_error(
    model, solver, state, contacts, contact_count: int, *, include_damping: bool = False
) -> dict[str, float]:
    body_q = state.body_q.numpy().astype(np.float64)
    body_q_prev = solver.body_q_prev.numpy().astype(np.float64) if include_damping else None
    body_com = model.body_com.numpy().astype(np.float64)
    shape_body = model.shape_body.numpy()
    shape0 = contacts.rigid_contact_shape0.numpy()[:contact_count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:contact_count]
    point0 = contacts.rigid_contact_point0.numpy()[:contact_count].astype(np.float64)
    point1 = contacts.rigid_contact_point1.numpy()[:contact_count].astype(np.float64)
    normal = contacts.rigid_contact_normal.numpy()[:contact_count].astype(np.float64)
    margin0 = contacts.rigid_contact_margin0.numpy()[:contact_count].astype(np.float64)
    margin1 = contacts.rigid_contact_margin1.numpy()[:contact_count].astype(np.float64)
    penalty_k = solver.body_body_contact_penalty_k.numpy()[:contact_count].astype(np.float64)
    material_kd = (
        solver.body_body_contact_material_kd.numpy()[:contact_count].astype(np.float64) if include_damping else None
    )
    active_bodies = sorted(
        {
            int(body)
            for i in range(contact_count)
            for body in (shape_body[shape0[i]], shape_body[shape1[i]])
            if body >= 0
        }
    )

    analytic = np.concatenate(
        [
            np.concatenate((solver.body_forces.numpy()[body], solver.body_torques.numpy()[body]))
            for body in active_bodies
        ]
    ).astype(np.float64)
    numeric = np.zeros_like(analytic)
    eps = 2.0e-4
    for body_cursor, body in enumerate(active_bodies):
        for axis in range(6):
            delta = np.zeros(6)
            delta[axis] = eps
            q_plus = body_q.copy()
            q_minus = body_q.copy()
            q_plus[body] = _perturb_pose_about_com(body_q[body], body_com[body], delta)
            q_minus[body] = _perturb_pose_about_com(body_q[body], body_com[body], -delta)
            e_plus = _frozen_normal_contact_energy(
                q_plus,
                body_q_prev,
                shape_body,
                shape0,
                shape1,
                point0,
                point1,
                normal,
                margin0,
                margin1,
                penalty_k,
                material_kd,
                0.01,
            )
            e_minus = _frozen_normal_contact_energy(
                q_minus,
                body_q_prev,
                shape_body,
                shape0,
                shape1,
                point0,
                point1,
                normal,
                margin0,
                margin1,
                penalty_k,
                material_kd,
                0.01,
            )
            numeric[body_cursor * 6 + axis] = -(e_plus - e_minus) / (2.0 * eps)

    difference = analytic - numeric
    prefix = "normal_damping" if include_damping else "normal"
    return {
        f"{prefix}_contact_wrench_norm": float(np.linalg.norm(analytic)),
        f"{prefix}_contact_numeric_wrench_norm": float(np.linalg.norm(numeric)),
        f"{prefix}_contact_gradient_relative_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(numeric), 1.0e-12)
        ),
        f"{prefix}_contact_gradient_max_abs_error": float(np.max(np.abs(difference))),
    }


def _assemble_first_contact_system():
    model = build_dr_legs_model("cpu")
    state_0 = model.state()
    state_1 = model.state()
    state_1.assign(state_0)
    control = model.control()
    pipeline = __import__("newton").CollisionPipeline(model)
    contacts = model.contacts(collision_pipeline=pipeline)
    solver = _make_dr_legs_solver(model, DR_LEGS_MODES["vbd_sparse_cpu"])

    dt = 0.01
    contact_count = 0
    saw_geometric_contact = False
    for step in range(40):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        if contact_count > 0 and saw_geometric_contact:
            first_contact_step = step + 1
            break
        if contact_count > 0:
            saw_geometric_contact = True
        solver.step(state_0, state_1, control, contacts if contact_count > 0 else None, dt)
        state_0, state_1 = state_1, state_0
    else:
        raise RuntimeError("DR Legs did not reach the ground")

    solver._initialize_rigid_bodies(state_0, control, contacts, dt, True)
    layout = solver.rigid_articulation_sparse_layout
    assert layout is not None
    assert solver.rigid_articulation_sparse_values is not None
    assert solver.rigid_articulation_sparse_rhs is not None
    assert solver.rigid_articulation_sparse_delta is not None

    def accumulate_contacts() -> None:
        solver.body_torques.zero_()
        solver.body_forces.zero_()
        solver.body_hessian_aa.zero_()
        solver.body_hessian_al.zero_()
        solver.body_hessian_ll.zero_()
        wp.launch(
            kernel=accumulate_body_body_contacts_per_body,
            dim=layout.articulation_body_count * 4,
            inputs=[
                dt,
                layout.articulation_bodies,
                solver.body_q_prev,
                state_0.body_q,
                model.body_com,
                solver.body_inv_mass_effective,
                solver.friction_epsilon,
                solver.rigid_contact_tangential_stiffness_scale,
                solver.body_body_contact_penalty_k,
                solver.body_body_contact_material_ke,
                solver.body_body_contact_material_kd,
                solver.body_body_contact_material_mu,
                solver.body_body_contact_lambda,
                solver.body_body_contact_C0,
                solver.rigid_contact_alpha,
                solver.rigid_contact_hard,
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                model.shape_body,
                solver.body_body_contact_buffer_pre_alloc,
                solver.body_body_contact_counts,
                solver.body_body_contact_indices,
            ],
            outputs=[
                solver.body_forces,
                solver.body_torques,
                solver.body_hessian_ll,
                solver.body_hessian_al,
                solver.body_hessian_aa,
            ],
            device=model.device,
        )

    def accumulated_wrench_norm() -> float:
        bodies = layout.articulation_bodies.numpy()
        forces = solver.body_forces.numpy()[bodies]
        torques = solver.body_torques.numpy()[bodies]
        return float(np.linalg.norm(np.concatenate((forces, torques), axis=1)))

    material_kd = solver.body_body_contact_material_kd.numpy().copy()
    material_mu = solver.body_body_contact_material_mu.numpy().copy()
    solver.body_body_contact_material_kd.zero_()
    solver.body_body_contact_material_mu.zero_()
    original_tangential_scale = solver.rigid_contact_tangential_stiffness_scale
    solver.rigid_contact_tangential_stiffness_scale = 0.0
    accumulate_contacts()
    wp.synchronize_device(model.device)
    normal_gradient = _normal_contact_gradient_error(model, solver, state_0, contacts, contact_count)

    solver.body_body_contact_material_kd.assign(material_kd)
    accumulate_contacts()
    wp.synchronize_device(model.device)
    normal_gradient.update(
        _normal_contact_gradient_error(model, solver, state_0, contacts, contact_count, include_damping=True)
    )

    solver.body_body_contact_material_kd.zero_()
    solver.body_body_contact_material_mu.assign(material_mu)
    solver.rigid_contact_tangential_stiffness_scale = original_tangential_scale
    accumulate_contacts()
    wp.synchronize_device(model.device)
    normal_gradient["normal_friction_contact_wrench_norm"] = accumulated_wrench_norm()

    solver.body_body_contact_material_kd.assign(material_kd)
    solver.body_body_contact_material_mu.assign(material_mu)
    accumulate_contacts()
    wp.synchronize_device(model.device)
    normal_gradient["full_contact_wrench_norm"] = accumulated_wrench_norm()

    values = solver.rigid_articulation_sparse_values
    rhs = solver.rigid_articulation_sparse_rhs
    delta = solver.rigid_articulation_sparse_delta
    values.zero_()
    rhs.zero_()
    delta.zero_()
    wp.launch(
        kernel=solve_articulation_sparse_serial,
        dim=layout.articulation_count,
        inputs=[
            dt,
            layout.articulation_body_offsets,
            layout.articulation_joint_offsets,
            layout.articulation_bodies,
            layout.articulation_joints,
            layout.articulation_block_row_offsets,
            layout.articulation_block_cols,
            layout.articulation_diag_slots,
            layout.body_articulation_local,
            state_0.body_q,
            solver.body_q_prev,
            model.body_q,
            model.body_mass,
            solver.body_inv_mass_effective,
            model.body_com,
            model.body_inertia,
            solver.body_inertia_q,
            solver.body_forces,
            solver.body_torques,
            solver.body_hessian_ll,
            solver.body_hessian_al,
            solver.body_hessian_aa,
            model.joint_type,
            model.joint_enabled,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_qd_start,
            model.joint_target_q_start,
            solver.joint_constraint_start,
            solver.joint_penalty_k,
            solver.joint_penalty_kd,
            solver.joint_sigma_start,
            solver.joint_C_fric,
            model.joint_dof_dim,
            solver.joint_rest_angle,
            model.joint_target_ke,
            model.joint_target_kd,
            control.joint_target_q,
            control.joint_target_qd,
            model.joint_limit_lower,
            model.joint_limit_upper,
            model.joint_limit_ke,
            model.joint_limit_kd,
            solver.joint_lambda_lin,
            solver.joint_lambda_ang,
            solver.joint_C0_lin,
            solver.joint_C0_ang,
            solver.joint_is_hard,
            solver.rigid_joint_alpha,
            solver.rigid_articulation_relaxation,
            1,
            False,
            False,
            True,
            values,
            rhs,
            delta,
        ],
        outputs=[state_1.body_q],
        device=model.device,
    )
    wp.synchronize_device(model.device)
    sparse_delta = delta.numpy().copy()

    # Reassemble without factorization so the matrix used by the Warp solve can
    # be compared directly with a dense reference solve.
    values.zero_()
    rhs.zero_()
    delta.zero_()
    wp.launch(
        kernel=solve_articulation_sparse_serial,
        dim=layout.articulation_count,
        inputs=[
            dt,
            layout.articulation_body_offsets,
            layout.articulation_joint_offsets,
            layout.articulation_bodies,
            layout.articulation_joints,
            layout.articulation_block_row_offsets,
            layout.articulation_block_cols,
            layout.articulation_diag_slots,
            layout.body_articulation_local,
            state_0.body_q,
            solver.body_q_prev,
            model.body_q,
            model.body_mass,
            solver.body_inv_mass_effective,
            model.body_com,
            model.body_inertia,
            solver.body_inertia_q,
            solver.body_forces,
            solver.body_torques,
            solver.body_hessian_ll,
            solver.body_hessian_al,
            solver.body_hessian_aa,
            model.joint_type,
            model.joint_enabled,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_qd_start,
            model.joint_target_q_start,
            solver.joint_constraint_start,
            solver.joint_penalty_k,
            solver.joint_penalty_kd,
            solver.joint_sigma_start,
            solver.joint_C_fric,
            model.joint_dof_dim,
            solver.joint_rest_angle,
            model.joint_target_ke,
            model.joint_target_kd,
            control.joint_target_q,
            control.joint_target_qd,
            model.joint_limit_lower,
            model.joint_limit_upper,
            model.joint_limit_ke,
            model.joint_limit_kd,
            solver.joint_lambda_lin,
            solver.joint_lambda_ang,
            solver.joint_C0_lin,
            solver.joint_C0_ang,
            solver.joint_is_hard,
            solver.rigid_joint_alpha,
            solver.rigid_articulation_relaxation,
            1,
            False,
            False,
            False,
            values,
            rhs,
            delta,
        ],
        outputs=[state_1.body_q],
        device=model.device,
    )
    wp.synchronize_device(model.device)
    contact_terms = {
        "forces": solver.body_forces.numpy().copy(),
        "torques": solver.body_torques.numpy().copy(),
        "hessian_ll": solver.body_hessian_ll.numpy().copy(),
        "hessian_al": solver.body_hessian_al.numpy().copy(),
        "hessian_aa": solver.body_hessian_aa.numpy().copy(),
    }
    joint_residuals = _joint_anchor_residuals(model, state_0)
    joint_state = {
        "joint_linear_residual_norm": joint_residuals["linear_norm_m"],
        "joint_linear_residual_max": joint_residuals["linear_max_m"],
        "joint_angular_residual_norm": joint_residuals["angular_norm_rad"],
        "joint_angular_residual_max": joint_residuals["angular_max_rad"],
        "joint_lambda_linear_norm": float(np.linalg.norm(solver.joint_lambda_lin.numpy())),
        "joint_lambda_angular_norm": float(np.linalg.norm(solver.joint_lambda_ang.numpy())),
    }
    return (
        first_contact_step,
        contact_count,
        layout,
        values.numpy(),
        rhs.numpy(),
        sparse_delta,
        normal_gradient,
        contact_terms,
        joint_state,
    )


def _dense_system(layout, blocks: np.ndarray, rhs_blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    body_offsets = layout.articulation_body_offsets.numpy()
    row_offsets = layout.articulation_block_row_offsets.numpy()
    cols = layout.articulation_block_cols.numpy()
    body_start = int(body_offsets[0])
    body_end = int(body_offsets[1])
    body_count = body_end - body_start
    matrix = np.zeros((body_count * 6, body_count * 6), dtype=np.float64)
    for local_row in range(body_count):
        row = body_start + local_row
        for slot in range(int(row_offsets[row]), int(row_offsets[row + 1])):
            local_col = int(cols[slot])
            block = np.asarray(blocks[slot], dtype=np.float64)
            rs = slice(local_row * 6, (local_row + 1) * 6)
            cs = slice(local_col * 6, (local_col + 1) * 6)
            matrix[rs, cs] = block
            if local_col != local_row:
                matrix[cs, rs] = block.T
    rhs = np.asarray(rhs_blocks[body_start:body_end], dtype=np.float64).reshape(-1)
    return matrix, rhs


def _block_equilibrate(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    body_count = matrix.shape[0] // 6
    scaling = np.zeros_like(matrix)
    for body in range(body_count):
        block_slice = slice(body * 6, (body + 1) * 6)
        diagonal = matrix[block_slice, block_slice]
        regularization = max(float(np.trace(diagonal)) / 6.0, 1.0) * 1.0e-10
        chol = np.linalg.cholesky(diagonal + np.eye(6) * regularization)
        scaling[block_slice, block_slice] = np.linalg.inv(chol)
    return scaling @ matrix @ scaling.T, scaling


def main() -> None:
    (
        step,
        contacts,
        layout,
        blocks,
        rhs_blocks,
        sparse_delta_blocks,
        normal_gradient,
        contact_terms,
        joint_state,
    ) = _assemble_first_contact_system()
    matrix, rhs = _dense_system(layout, blocks, rhs_blocks)
    body_offsets = layout.articulation_body_offsets.numpy()
    body_start = int(body_offsets[0])
    body_end = int(body_offsets[1])
    sparse_solution = np.asarray(sparse_delta_blocks[body_start:body_end], dtype=np.float64).reshape(-1)
    articulation_bodies = layout.articulation_bodies.numpy()[body_start:body_end]
    contact_matrix = np.zeros_like(matrix)
    contact_rhs = np.zeros_like(rhs)
    for local_body, body in enumerate(articulation_bodies):
        body_slice = slice(local_body * 6, (local_body + 1) * 6)
        h_ll = contact_terms["hessian_ll"][body]
        h_al = contact_terms["hessian_al"][body]
        h_aa = contact_terms["hessian_aa"][body]
        contact_matrix[body_slice, body_slice] = np.block([[h_ll, h_al.T], [h_al, h_aa]])
        contact_rhs[body_slice] = np.concatenate((contact_terms["forces"][body], contact_terms["torques"][body]))
    no_contact_matrix = matrix - contact_matrix
    no_contact_rhs = rhs - contact_rhs
    symmetric_error = float(np.max(np.abs(matrix - matrix.T)))
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    positive = eigenvalues[eigenvalues > 0.0]
    equilibrated, scaling = _block_equilibrate(matrix)
    equilibrated_eigenvalues = np.linalg.eigvalsh(0.5 * (equilibrated + equilibrated.T))
    equilibrated_positive = equilibrated_eigenvalues[equilibrated_eigenvalues > 0.0]

    float32_cholesky = True
    try:
        np.linalg.cholesky(matrix.astype(np.float32))
    except np.linalg.LinAlgError:
        float32_cholesky = False
    equilibrated_float32_cholesky = True
    try:
        np.linalg.cholesky(equilibrated.astype(np.float32))
    except np.linalg.LinAlgError:
        equilibrated_float32_cholesky = False

    solution = np.linalg.solve(matrix, rhs)
    no_contact_solution = np.linalg.solve(no_contact_matrix, no_contact_rhs)
    dense_residual = matrix @ solution - rhs
    sparse_residual = matrix @ sparse_solution - rhs
    solution_difference = sparse_solution - solution
    equilibrated_rhs = scaling @ rhs
    equilibrated_solution = np.linalg.solve(equilibrated, equilibrated_rhs)
    payload = {
        "first_contact_step": step,
        "contact_count": contacts,
        "matrix_size": matrix.shape[0],
        "block_count": int(blocks.shape[0]),
        "symmetric_error": symmetric_error,
        "eigen_min": float(eigenvalues[0]),
        "eigen_max": float(eigenvalues[-1]),
        "condition_positive": float(positive[-1] / positive[0]),
        "float32_cholesky": float32_cholesky,
        "equilibrated_eigen_min": float(equilibrated_eigenvalues[0]),
        "equilibrated_eigen_max": float(equilibrated_eigenvalues[-1]),
        "equilibrated_condition_positive": float(equilibrated_positive[-1] / equilibrated_positive[0]),
        "equilibrated_float32_cholesky": equilibrated_float32_cholesky,
        "rhs_norm": float(np.linalg.norm(rhs)),
        "solution_norm": float(np.linalg.norm(solution)),
        "solution_max_abs": float(np.max(np.abs(solution))),
        "dense_relative_residual": float(np.linalg.norm(dense_residual) / np.linalg.norm(rhs)),
        "sparse_relative_residual": float(np.linalg.norm(sparse_residual) / np.linalg.norm(rhs)),
        "sparse_relative_solution_error": float(np.linalg.norm(solution_difference) / np.linalg.norm(solution)),
        "dense_rhs_dot_solution": float(np.dot(rhs, solution)),
        "contact_rhs_norm": float(np.linalg.norm(contact_rhs)),
        "no_contact_rhs_norm": float(np.linalg.norm(no_contact_rhs)),
        "no_contact_solution_norm": float(np.linalg.norm(no_contact_solution)),
        "contact_direction_relative_change": float(
            np.linalg.norm(solution - no_contact_solution) / np.linalg.norm(no_contact_solution)
        ),
        "sparse_rhs_dot_solution": float(np.dot(rhs, sparse_solution)),
        "equilibrated_solution_norm": float(np.linalg.norm(scaling.T @ equilibrated_solution)),
        **normal_gradient,
        **joint_state,
    }
    output = Path("reports/vbd_complex_linkages/dr_legs_matrix_diagnostic.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
