#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""NCP convergence study: compare PGS, bisection, and de Saxcé correction.

Measures complementarity residual r_compl across scenarios at varying
GS iteration counts.  Since USE_BISECTION and USE_DE_SAXCE are
compile-time constants, this script measures whichever mode is currently
compiled.  Run it once per mode to collect all data.

Usage::

    python3 -m newton._src.solvers.raisim.convergence_study
"""

from __future__ import annotations

import warp as wp

wp.config.enable_backward = False

import newton  # noqa: E402
import newton.utils  # noqa: E402

from .residuals import compute_ncp_residuals  # noqa: E402


def measure(label, model, gs_list):
    """Settle 200 steps, then measure residuals at each GS count."""
    dt = 1.0 / 360.0
    cfg0 = newton.solvers.RaisimConfig(max_gs_iterations=200)
    solver = newton.solvers.SolverRaisim(model, config=cfg0)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for _ in range(200):
        s0.clear_forces()
        model.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, dt)
        s0, s1 = s1, s0

    print(f"\n{label}:")
    print(
        f"  {'gs':>6s}  {'r_compl':>10s}  {'r_cone':>10s}  {'r_gap':>10s}"
        f"  {'r_ds_compl':>10s}  {'r_ds_dual':>10s}  {'r_mdp_dir':>10s}"
    )
    for gs in gs_list:
        cfg = newton.solvers.RaisimConfig(max_gs_iterations=gs)
        sv = newton.solvers.SolverRaisim(model, config=cfg)
        st = model.state()
        so = model.state()
        wp.copy(st.joint_q, s0.joint_q)
        wp.copy(st.joint_qd, s0.joint_qd)
        if s0.body_count:
            wp.copy(st.body_q, s0.body_q)
            wp.copy(st.body_qd, s0.body_qd)
        st.clear_forces()
        model.collide(st, contacts)
        sv.step(st, so, ctrl, contacts, dt)
        res = compute_ncp_residuals(sv, so, contacts, dt)
        print(
            f"  {gs:>6d}  {res['r_compl']:>10.6f}  {res['r_cone']:>10.6f}  {res['r_gap']:>10.6f}"
            f"  {res['r_ds_compl']:>10.6f}  {res['r_ds_dual']:>10.6f}  {res['r_mdp_dir']:>10.6f}"
        )


def build_sphere():
    b = newton.ModelBuilder()
    b.default_shape_cfg.mu = 0.5
    b.add_ground_plane()
    body = b.add_body(xform=wp.transform(p=wp.vec3(0, 0, 0.31), q=wp.quat_identity()), mass=1.0)
    b.add_shape_sphere(body=body, radius=0.3)
    return b.finalize()


def build_stack5():
    b = newton.ModelBuilder()
    b.default_shape_cfg.mu = 0.5
    b.add_ground_plane()
    r = 0.15
    for i in range(5):
        body = b.add_body(
            xform=wp.transform(p=wp.vec3(0, 0, r + 0.01 + i * (2 * r + 0.05)), q=wp.quat_identity()), mass=1.0
        )
        b.add_shape_sphere(body=body, radius=r)
    return b.finalize()


def build_tower10():
    half = 0.1
    b = newton.ModelBuilder()
    b.default_shape_cfg.mu = 0.5
    b.add_ground_plane()
    for i in range(10):
        z = half + 0.001 + i * (2.0 * half + 0.002)
        body = b.add_body(xform=wp.transform(p=wp.vec3(0, 0, z), q=wp.quat_identity()), mass=1.0)
        b.add_shape_box(body=body, hx=half, hy=half, hz=half)
    return b.finalize()


def build_g1():
    asset_path = newton.utils.download_asset("unitree_g1")
    g1 = newton.ModelBuilder()
    g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
    g1.default_shape_cfg.ke = 1.0e3
    g1.default_shape_cfg.kd = 2.0e2
    g1.default_shape_cfg.kf = 1.0e3
    g1.default_shape_cfg.mu = 0.75
    g1.add_usd(
        str(asset_path / "usd" / "g1_isaac.usd"),
        xform=wp.transform(wp.vec3(0, 0, 0.8)),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    for i in range(6, g1.joint_dof_count):
        g1.joint_target_ke[i] = 500.0
        g1.joint_target_kd[i] = 10.0
    g1.approximate_meshes("bounding_box")
    builder = newton.ModelBuilder()
    builder.replicate(g1, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 2.0e2
    builder.add_ground_plane()
    return builder.finalize()


def build_sliding_cube():
    """Cube with lateral velocity — the key MDP test case from the paper."""
    b = newton.ModelBuilder()
    b.default_shape_cfg.mu = 0.5
    b.add_ground_plane()
    body = b.add_body(xform=wp.transform(p=wp.vec3(0, 0, 0.201), q=wp.quat_identity()), mass=1.0)
    b.add_shape_box(body=body, hx=0.2, hy=0.2, hz=0.2)
    model = b.finalize()
    # Give lateral velocity
    s = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s)
    jqd = s.joint_qd.numpy()
    jqd[0] = 3.0  # lateral x velocity (sliding)
    model.joint_qd = wp.array(jqd, dtype=wp.float32, device=model.device)
    return model


def main():
    from .kernels import USE_BISECTION, USE_DE_SAXCE  # noqa: PLC0415

    bisect = int(USE_BISECTION)
    ds = int(USE_DE_SAXCE)
    mode = "PGS" if bisect == 0 else "Bisection"
    if ds == 1:
        mode += "+DS"
    print(f"Mode: {mode} (USE_BISECTION={bisect}, USE_DE_SAXCE={ds})")

    gs_list = [1, 2, 5, 10, 20, 50, 100]

    measure("Sphere on ground (1 contact)", build_sphere(), gs_list)
    measure("5-sphere stack (~10 contacts)", build_stack5(), [*gs_list, 200])
    measure("10-box tower (~40 contacts)", build_tower10(), [*gs_list, 200])
    measure("G1 robot (~20 contacts)", build_g1(), gs_list)
    measure("Sliding cube (MDP test)", build_sliding_cube(), gs_list)


if __name__ == "__main__":
    main()
