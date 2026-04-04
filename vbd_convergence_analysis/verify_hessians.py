#!/usr/bin/env python3
"""Verify VBD Hessians via finite difference comparison.

For each energy model (StVK, Neo-Hookean, Bending), compute the analytical
force and Hessian, then compare against finite-difference approximations.
Reports relative errors and flags any issues.
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

import warp as wp

wp.init()

from newton._src.solvers.vbd.particle_vbd_kernels import (
    evaluate_stvk_force_hessian,
    evaluate_neo_hookean_membrane_force_hessian,
    evaluate_dihedral_angle_based_bending_force_hessian,
)

# We'll test with small Warp kernels that call the energy functions
# and extract force/hessian values.

# -----------------------------------------------------------------------
# Helper: evaluate force and hessian for a single triangle / vertex
# -----------------------------------------------------------------------

@wp.kernel
def eval_stvk_kernel(
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_pose: wp.array(dtype=wp.mat22),
    area: float,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
    out_force: wp.array(dtype=wp.vec3),
    out_hessian: wp.array(dtype=wp.mat33),
):
    f, h = evaluate_stvk_force_hessian(
        0, v_order, pos, pos_anchor, tri_indices,
        tri_pose[0], area, mu, lmbd, damping, dt,
    )
    out_force[0] = f
    out_hessian[0] = h


@wp.kernel
def eval_neohookean_kernel(
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_pose: wp.array(dtype=wp.mat22),
    area: float,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
    out_force: wp.array(dtype=wp.vec3),
    out_hessian: wp.array(dtype=wp.mat33),
):
    f, h = evaluate_neo_hookean_membrane_force_hessian(
        0, v_order, pos, pos_anchor, tri_indices,
        tri_pose[0], area, mu, lmbd, damping, dt,
    )
    out_force[0] = f
    out_hessian[0] = h


@wp.kernel
def eval_bending_kernel(
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_anchor: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angle: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    stiffness: float,
    damping: float,
    dt: float,
    out_force: wp.array(dtype=wp.vec3),
    out_hessian: wp.array(dtype=wp.mat33),
):
    f, h = evaluate_dihedral_angle_based_bending_force_hessian(
        0, v_order, pos, pos_anchor, edge_indices,
        edge_rest_angle, edge_rest_length, stiffness, damping, dt,
    )
    out_force[0] = f
    out_hessian[0] = h


def finite_diff_force(eval_fn, pos_np, vertex_idx, eps=1e-5):
    """Compute force via finite difference of energy (central difference on force = -dE/dx)."""
    # We approximate the Hessian: H[i,j] = d(force_i)/d(x_j) ≈ (f(x+eps*e_j) - f(x-eps*e_j)) / (2*eps)
    base_force = eval_fn(pos_np)
    n = 3
    hessian_fd = np.zeros((n, n))
    for j in range(n):
        pos_plus = pos_np.copy()
        pos_minus = pos_np.copy()
        pos_plus[vertex_idx, j] += eps
        pos_minus[vertex_idx, j] -= eps
        f_plus = eval_fn(pos_plus)
        f_minus = eval_fn(pos_minus)
        hessian_fd[:, j] = (f_plus - f_minus) / (2 * eps)
    return base_force, hessian_fd


def make_triangle_config(seed=42, deformed=True):
    """Create a triangle configuration for testing."""
    rng = np.random.default_rng(seed)
    # Rest triangle in XY plane
    rest_verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
    ], dtype=np.float32)

    if deformed:
        # Apply random deformation
        deform = rng.normal(0, 0.3, size=(3, 3)).astype(np.float32)
        verts = rest_verts + deform
    else:
        verts = rest_verts.copy()

    # Compute DmInv (inverse of rest edge matrix)
    x01 = rest_verts[1] - rest_verts[0]
    x02 = rest_verts[2] - rest_verts[0]
    Dm = np.array([[x01[0], x02[0]], [x01[1], x02[1]]], dtype=np.float32)
    DmInv = np.linalg.inv(Dm)
    area = 0.5 * abs(np.linalg.det(Dm))

    return verts, DmInv, area


def make_bending_config(seed=42, deformed=True):
    """Create a bending edge configuration (4 vertices: 2 opposite + 2 edge)."""
    rng = np.random.default_rng(seed)
    # Two triangles sharing an edge (v2-v3), with v0 and v1 as opposite vertices
    rest_verts = np.array([
        [0.0, 1.0, 0.0],    # v0: opposite 0
        [0.0, -1.0, 0.0],   # v1: opposite 1
        [-0.5, 0.0, 0.0],   # v2: edge start
        [0.5, 0.0, 0.0],    # v3: edge end
    ], dtype=np.float32)

    if deformed:
        deform = rng.normal(0, 0.15, size=(4, 3)).astype(np.float32)
        verts = rest_verts + deform
    else:
        verts = rest_verts.copy()

    rest_length = np.linalg.norm(rest_verts[3] - rest_verts[2])
    rest_angle = 0.0  # flat rest angle

    return verts, rest_length, rest_angle


def test_stvk_hessian():
    """Verify StVK force-Hessian consistency via finite differences."""
    print("=" * 60)
    print("VERIFYING StVK HESSIAN")
    print("=" * 60)

    results = []
    for seed in range(5):
        for v_order in range(3):
            for with_damping in [False, True]:
                verts, DmInv, area = make_triangle_config(seed=seed, deformed=True)
                mu, lmbd = 1e4, 1e4
                damping_val = 1.5e-6 if with_damping else 0.0
                dt = 1.0 / 600.0
                prev_verts = verts - np.random.default_rng(seed + 100).normal(0, 0.01, size=(3, 3)).astype(np.float32)

                pos_wp = wp.array(verts, dtype=wp.vec3)
                pos_anchor_wp = wp.array(prev_verts, dtype=wp.vec3)
                tri_idx = wp.array([[0, 1, 2]], dtype=wp.int32)
                tri_pose_wp = wp.array([wp.mat22(DmInv[0, 0], DmInv[0, 1], DmInv[1, 0], DmInv[1, 1])], dtype=wp.mat22)
                out_f = wp.zeros(1, dtype=wp.vec3)
                out_h = wp.zeros(1, dtype=wp.mat33)

                wp.launch(eval_stvk_kernel, dim=1, inputs=[
                    v_order, pos_wp, pos_anchor_wp, tri_idx, tri_pose_wp,
                    area, mu, lmbd, damping_val, dt
                ], outputs=[out_f, out_h])

                force_analytical = out_f.numpy()[0]
                hessian_analytical = out_h.numpy()[0]

                # Finite difference Hessian
                vertex_idx = v_order
                eps = 1e-4

                def eval_force(pos_np):
                    p = wp.array(pos_np, dtype=wp.vec3)
                    f_out = wp.zeros(1, dtype=wp.vec3)
                    h_out = wp.zeros(1, dtype=wp.mat33)
                    wp.launch(eval_stvk_kernel, dim=1, inputs=[
                        v_order, p, pos_anchor_wp, tri_idx, tri_pose_wp,
                        area, mu, lmbd, damping_val, dt
                    ], outputs=[f_out, h_out])
                    return f_out.numpy()[0]

                _, hessian_fd = finite_diff_force(eval_force, verts, vertex_idx, eps)

                # Compare
                # The VBD convention: force = -dE/dx, hessian = d²E/dx² (stiffness matrix)
                # So dforce/dx = -hessian. FD computes dforce/dx, so compare -hessian_fd vs hessian_analytical
                hessian_fd_corrected = -hessian_fd

                err = np.linalg.norm(hessian_analytical - hessian_fd_corrected)
                ref = max(np.linalg.norm(hessian_analytical), np.linalg.norm(hessian_fd_corrected), 1e-10)
                rel_err = err / ref

                # Check symmetry
                sym_err = np.linalg.norm(hessian_analytical - hessian_analytical.T) / max(ref, 1e-10)

                # Check eigenvalues (SPD)
                eigvals = np.linalg.eigvalsh(hessian_analytical)
                min_eig = eigvals[0]

                status = "PASS" if rel_err < 0.05 else "FAIL"
                if rel_err > 0.01 or status == "FAIL":
                    damp_str = "damped" if with_damping else "undamped"
                    print(f"  seed={seed} v={v_order} {damp_str}: rel_err={rel_err:.6f} sym_err={sym_err:.6f} min_eig={min_eig:.2e} [{status}]")
                    if status == "FAIL":
                        print(f"    Analytical:\n{hessian_analytical}")
                        print(f"    FD (sign-corrected):\n{hessian_fd_corrected}")

                results.append({
                    "model": "stvk",
                    "seed": seed,
                    "v_order": v_order,
                    "damped": with_damping,
                    "rel_err": float(rel_err),
                    "sym_err": float(sym_err),
                    "min_eig": float(min_eig),
                    "pass": status == "PASS",
                })

    pass_count = sum(1 for r in results if r["pass"])
    fail_count = len(results) - pass_count
    print(f"\nStVK: {pass_count}/{len(results)} passed, {fail_count} failed")
    max_err = max(r["rel_err"] for r in results)
    print(f"  Max relative error: {max_err:.6e}")
    min_eig_worst = min(r["min_eig"] for r in results)
    print(f"  Worst min eigenvalue: {min_eig_worst:.6e}")
    non_spd = sum(1 for r in results if r["min_eig"] < -1e-6)
    if non_spd:
        print(f"  WARNING: {non_spd} cases with negative eigenvalues (non-SPD)")
    return results


def test_neohookean_hessian():
    """Verify Neo-Hookean membrane force-Hessian consistency."""
    print("\n" + "=" * 60)
    print("VERIFYING NEO-HOOKEAN MEMBRANE HESSIAN")
    print("=" * 60)

    results = []
    for seed in range(5):
        for v_order in range(3):
            for with_damping in [False, True]:
                verts, DmInv, area = make_triangle_config(seed=seed, deformed=True)
                mu, lmbd = 1e4, 1e4
                damping_val = 1.5e-6 if with_damping else 0.0
                dt = 1.0 / 600.0
                prev_verts = verts - np.random.default_rng(seed + 100).normal(0, 0.01, size=(3, 3)).astype(np.float32)

                pos_wp = wp.array(verts, dtype=wp.vec3)
                pos_anchor_wp = wp.array(prev_verts, dtype=wp.vec3)
                tri_idx = wp.array([[0, 1, 2]], dtype=wp.int32)
                tri_pose_wp = wp.array([wp.mat22(DmInv[0, 0], DmInv[0, 1], DmInv[1, 0], DmInv[1, 1])], dtype=wp.mat22)
                out_f = wp.zeros(1, dtype=wp.vec3)
                out_h = wp.zeros(1, dtype=wp.mat33)

                wp.launch(eval_neohookean_kernel, dim=1, inputs=[
                    v_order, pos_wp, pos_anchor_wp, tri_idx, tri_pose_wp,
                    area, mu, lmbd, damping_val, dt
                ], outputs=[out_f, out_h])

                force_analytical = out_f.numpy()[0]
                hessian_analytical = out_h.numpy()[0]

                vertex_idx = v_order
                eps = 1e-4

                def eval_force(pos_np):
                    p = wp.array(pos_np, dtype=wp.vec3)
                    f_out = wp.zeros(1, dtype=wp.vec3)
                    h_out = wp.zeros(1, dtype=wp.mat33)
                    wp.launch(eval_neohookean_kernel, dim=1, inputs=[
                        v_order, p, pos_anchor_wp, tri_idx, tri_pose_wp,
                        area, mu, lmbd, damping_val, dt
                    ], outputs=[f_out, h_out])
                    return f_out.numpy()[0]

                _, hessian_fd = finite_diff_force(eval_force, verts, vertex_idx, eps)
                hessian_fd_corrected = -hessian_fd

                err = np.linalg.norm(hessian_analytical - hessian_fd_corrected)
                ref = max(np.linalg.norm(hessian_analytical), np.linalg.norm(hessian_fd_corrected), 1e-10)
                rel_err = err / ref

                sym_err = np.linalg.norm(hessian_analytical - hessian_analytical.T) / max(ref, 1e-10)
                eigvals = np.linalg.eigvalsh(hessian_analytical)
                min_eig = eigvals[0]

                status = "PASS" if rel_err < 0.05 else "FAIL"
                if rel_err > 0.01 or status == "FAIL":
                    damp_str = "damped" if with_damping else "undamped"
                    print(f"  seed={seed} v={v_order} {damp_str}: rel_err={rel_err:.6f} sym_err={sym_err:.6f} min_eig={min_eig:.2e} [{status}]")
                    if status == "FAIL":
                        print(f"    Analytical:\n{hessian_analytical}")
                        print(f"    FD (sign-corrected):\n{hessian_fd_corrected}")

                results.append({
                    "model": "neohookean",
                    "seed": seed,
                    "v_order": v_order,
                    "damped": with_damping,
                    "rel_err": float(rel_err),
                    "sym_err": float(sym_err),
                    "min_eig": float(min_eig),
                    "pass": status == "PASS",
                })

    pass_count = sum(1 for r in results if r["pass"])
    fail_count = len(results) - pass_count
    print(f"\nNeo-Hookean: {pass_count}/{len(results)} passed, {fail_count} failed")
    max_err = max(r["rel_err"] for r in results)
    print(f"  Max relative error: {max_err:.6e}")
    min_eig_worst = min(r["min_eig"] for r in results)
    print(f"  Worst min eigenvalue: {min_eig_worst:.6e}")
    non_spd = sum(1 for r in results if r["min_eig"] < -1e-6)
    if non_spd:
        print(f"  WARNING: {non_spd} cases with negative eigenvalues (non-SPD)")
    return results


def test_bending_hessian():
    """Verify dihedral angle bending force-Hessian consistency."""
    print("\n" + "=" * 60)
    print("VERIFYING BENDING (DIHEDRAL ANGLE) HESSIAN")
    print("=" * 60)

    results = []
    for seed in range(5):
        for v_order in range(4):
            for with_damping in [False, True]:
                verts, rest_len, rest_angle = make_bending_config(seed=seed, deformed=True)
                stiffness = 5.0
                damping_val = 1e-2 if with_damping else 0.0
                dt = 1.0 / 600.0
                prev_verts = verts - np.random.default_rng(seed + 200).normal(0, 0.01, size=(4, 3)).astype(np.float32)

                pos_wp = wp.array(verts, dtype=wp.vec3)
                pos_anchor_wp = wp.array(prev_verts, dtype=wp.vec3)
                edge_idx = wp.array([[0, 1, 2, 3]], dtype=wp.int32)
                edge_rest_angle_wp = wp.array([rest_angle], dtype=float)
                edge_rest_len_wp = wp.array([rest_len], dtype=float)
                out_f = wp.zeros(1, dtype=wp.vec3)
                out_h = wp.zeros(1, dtype=wp.mat33)

                wp.launch(eval_bending_kernel, dim=1, inputs=[
                    v_order, pos_wp, pos_anchor_wp, edge_idx,
                    edge_rest_angle_wp, edge_rest_len_wp,
                    stiffness, damping_val, dt
                ], outputs=[out_f, out_h])

                force_analytical = out_f.numpy()[0]
                hessian_analytical = out_h.numpy()[0]

                vertex_idx = v_order
                eps = 1e-5

                def eval_force(pos_np):
                    p = wp.array(pos_np, dtype=wp.vec3)
                    f_out = wp.zeros(1, dtype=wp.vec3)
                    h_out = wp.zeros(1, dtype=wp.mat33)
                    wp.launch(eval_bending_kernel, dim=1, inputs=[
                        v_order, p, pos_anchor_wp, edge_idx,
                        edge_rest_angle_wp, edge_rest_len_wp,
                        stiffness, damping_val, dt
                    ], outputs=[f_out, h_out])
                    return f_out.numpy()[0]

                _, hessian_fd = finite_diff_force(eval_force, verts, vertex_idx, eps)
                hessian_fd_corrected = -hessian_fd

                err = np.linalg.norm(hessian_analytical - hessian_fd_corrected)
                ref = max(np.linalg.norm(hessian_analytical), np.linalg.norm(hessian_fd_corrected), 1e-10)
                rel_err = err / ref

                sym_err = np.linalg.norm(hessian_analytical - hessian_analytical.T) / max(ref, 1e-10)
                eigvals = np.linalg.eigvalsh(hessian_analytical)
                min_eig = eigvals[0]

                status = "PASS" if rel_err < 0.05 else "FAIL"
                if rel_err > 0.01 or status == "FAIL":
                    damp_str = "damped" if with_damping else "undamped"
                    print(f"  seed={seed} v={v_order} {damp_str}: rel_err={rel_err:.6f} sym_err={sym_err:.6f} min_eig={min_eig:.2e} [{status}]")
                    if status == "FAIL":
                        print(f"    Analytical:\n{hessian_analytical}")
                        print(f"    FD (sign-corrected):\n{hessian_fd_corrected}")

                results.append({
                    "model": "bending",
                    "seed": seed,
                    "v_order": v_order,
                    "damped": with_damping,
                    "rel_err": float(rel_err),
                    "sym_err": float(sym_err),
                    "min_eig": float(min_eig),
                    "pass": status == "PASS",
                })

    pass_count = sum(1 for r in results if r["pass"])
    fail_count = len(results) - pass_count
    print(f"\nBending: {pass_count}/{len(results)} passed, {fail_count} failed")
    max_err = max(r["rel_err"] for r in results)
    print(f"  Max relative error: {max_err:.6e}")
    min_eig_worst = min(r["min_eig"] for r in results)
    print(f"  Worst min eigenvalue: {min_eig_worst:.6e}")
    non_spd = sum(1 for r in results if r["min_eig"] < -1e-6)
    if non_spd:
        print(f"  NOTE: {non_spd} cases with negative eigenvalues (Gauss-Newton approx drops 2nd-order terms)")
    return results


if __name__ == "__main__":
    all_results = {}
    all_results["stvk"] = test_stvk_hessian()
    all_results["neohookean"] = test_neohookean_hessian()
    all_results["bending"] = test_bending_hessian()

    # Summary
    print("\n" + "=" * 60)
    print("OVERALL HESSIAN VERIFICATION SUMMARY")
    print("=" * 60)
    for model, results in all_results.items():
        total = len(results)
        passed = sum(1 for r in results if r["pass"])
        max_err = max(r["rel_err"] for r in results)
        non_spd = sum(1 for r in results if r["min_eig"] < -1e-6)
        max_sym = max(r["sym_err"] for r in results)
        print(f"  {model:15s}: {passed}/{total} pass, max_rel_err={max_err:.2e}, max_sym_err={max_sym:.2e}, non_SPD={non_spd}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "hessian_verification.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
