#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physics correctness and NCP convergence tests for SolverRaisim.

Section A: Analytic / physical correctness
Section B: NCP residual / convergence
Section C: Regression (no NaN, bounded penetration)

Usage::

    python3 -m newton._src.solvers.raisim.test_raisim
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

wp.config.enable_backward = False

import newton  # noqa: E402
import newton.utils  # noqa: E402

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _run_and_fk(model, num_steps, dt=1.0 / 360.0, config=None):
    """Simulate with SolverRaisim, call eval_fk, return numpy arrays.

    Returns:
        Tuple ``(body_q, body_qd, joint_q, joint_qd)`` as numpy.
    """
    if config is None:
        config = newton.solvers.RaisimConfig()
    solver = newton.solvers.SolverRaisim(model, config=config)

    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    for _ in range(num_steps):
        s0.clear_forces()
        model.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, dt)
        s0, s1 = s1, s0

    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    return (
        s0.body_q.numpy(),
        s0.body_qd.numpy(),
        s0.joint_q.numpy(),
        s0.joint_qd.numpy(),
    )


def _run_trajectory(model, num_steps, dt=1.0 / 360.0, config=None, sample_interval=10):
    """Like _run_and_fk but also returns sampled Z trajectory."""
    if config is None:
        config = newton.solvers.RaisimConfig()
    solver = newton.solvers.SolverRaisim(model, config=config)

    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    z_samples = []
    for step in range(num_steps):
        s0.clear_forces()
        model.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, dt)
        s0, s1 = s1, s0

        if step % sample_interval == 0:
            newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
            z_samples.append(s0.body_q.numpy()[:, 2].copy())

    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    return s0.body_q.numpy(), s0.body_qd.numpy(), z_samples


def _build_g1():
    """Return ModelBuilder with G1 loaded."""
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
    return g1


def _build_h1():
    """Return ModelBuilder with H1 loaded."""
    asset_path = newton.utils.download_asset("unitree_h1")
    h1 = newton.ModelBuilder()
    h1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
    h1.default_shape_cfg.ke = 2.0e3
    h1.default_shape_cfg.kd = 1.0e2
    h1.default_shape_cfg.kf = 1.0e3
    h1.default_shape_cfg.mu = 0.75
    h1.add_usd(
        str(asset_path / "usd_structured" / "h1.usda"),
        ignore_paths=["/GroundPlane"],
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    for i in range(h1.joint_dof_count):
        h1.joint_target_ke[i] = 150.0
        h1.joint_target_kd[i] = 5.0
    h1.approximate_meshes("bounding_box")
    return h1


def _check(errors, condition, msg):
    """Append *msg* to *errors* if *condition* is False."""
    if not condition:
        errors.append(msg)


def _report(test_name, errors):
    """Print PASS/FAIL and return bool."""
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False
    print("  PASSED")
    return True


# =====================================================================
# Section A: Analytic / physical correctness
# =====================================================================


def test_sphere_resting_height():
    """A1: Sphere (r=0.3) dropped from z=1 settles at z ≈ r."""
    print("=== A1: Sphere resting height ===")
    radius = 0.3
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    b = builder.add_body(
        xform=wp.transform(p=wp.vec3(0, 0, 1.0), q=wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_sphere(body=b, radius=radius)
    model = builder.finalize()

    bq, bqd, _, _ = _run_and_fk(model, num_steps=1800)  # 5s

    z = float(bq[0, 2])
    max_v = float(np.abs(bqd).max())
    z_err = abs(z - radius)
    print(f"  z={z:.4f}  expected≈{radius}  err={z_err:.4f}  max_v={max_v:.4f}")

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(bq)), "NaN in body positions")
    _check(errors, z > -0.01, f"Ground penetration: z={z:.4f}")
    _check(errors, z_err < 0.05, f"Resting height wrong: z={z:.4f} (expected≈{radius})")
    _check(errors, max_v < 0.5, f"Not settled: max_v={max_v:.4f}")
    return _report("Sphere resting height", errors)


def test_box_resting_orientation():
    """A2: Box rests upright at z ≈ half-extent."""
    print("=== A2: Box resting orientation ===")
    half = 0.2
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    b = builder.add_body(
        xform=wp.transform(p=wp.vec3(0, 0, 1.0), q=wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(body=b, hx=half, hy=half, hz=half)
    model = builder.finalize()

    bq, _bqd, _, _ = _run_and_fk(model, num_steps=1800)

    z = float(bq[0, 2])
    qw = float(bq[0, 6])
    z_err = abs(z - half)
    print(f"  z={z:.4f}  expected≈{half}  err={z_err:.4f}  |qw|={abs(qw):.4f}")

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(bq)), "NaN")
    _check(errors, z > -0.01, f"Penetration: z={z:.4f}")
    _check(errors, z_err < 0.05, f"Height wrong: z={z:.4f}")
    _check(errors, abs(qw) > 0.95, f"Rotated: |qw|={abs(qw):.4f}")
    return _report("Box resting orientation", errors)


def test_free_fall():
    """A3: Without contacts, matches analytical free-fall trajectory."""
    print("=== A3: Free-fall trajectory ===")
    builder = newton.ModelBuilder()
    # No ground plane → pure free-fall
    b = builder.add_body(
        xform=wp.transform(p=wp.vec3(0, 0, 10.0), q=wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_sphere(body=b, radius=0.1)
    model = builder.finalize()

    dt = 1.0 / 360.0
    num_steps = 180  # 0.5s
    bq, _bqd, _, _ = _run_and_fk(model, num_steps=num_steps, dt=dt)

    t = num_steps * dt
    g = -9.81
    expected_z = 10.0 + 0.5 * g * t * t
    actual_z = float(bq[0, 2])
    # Integration error bound: 0.5 * |g| * dt * t
    tol = max(0.5 * abs(g) * dt * t, 0.01)
    z_err = abs(actual_z - expected_z)

    print(f"  z={actual_z:.4f}  expected={expected_z:.4f}  err={z_err:.4f}  tol={tol:.4f}")

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(bq)), "NaN")
    _check(errors, z_err < tol, f"Free-fall error: {z_err:.4f} > {tol:.4f}")
    return _report("Free-fall trajectory", errors)


def test_static_friction():
    """A4: Object on tilted plane below friction angle stays put."""
    print("=== A4: Static friction ===")
    mu = 0.75
    # Tilt angle where object should NOT slide: arctan(mu) ≈ 36.9°
    # Use 20° — well within static friction limit
    angle_deg = 20.0
    angle_rad = np.deg2rad(angle_deg)

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = mu

    # Tilted ground (rotate around Y axis)
    q_tilt = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), float(angle_rad))
    builder.add_shape_plane(body=-1, plane=(0.0, 0.0, 1.0, 0.0), xform=wp.transform(wp.vec3(0, 0, 0), q_tilt))

    # Box sitting on the tilted plane
    height = 0.5 * np.cos(angle_rad)
    b = builder.add_body(
        xform=wp.transform(p=wp.vec3(0, 0, height + 0.2), q=q_tilt),
        mass=1.0,
    )
    builder.add_shape_box(body=b, hx=0.2, hy=0.2, hz=0.2)
    model = builder.finalize()

    bq_init, _, _, _ = _run_and_fk(model, num_steps=0)
    init_x = float(bq_init[0, 0])

    bq, bqd, _, _ = _run_and_fk(model, num_steps=1080)  # 3s

    final_x = float(bq[0, 0])
    displacement = abs(final_x - init_x)
    max_v = float(np.abs(bqd).max())

    print(f"  displacement={displacement:.4f}  max_v={max_v:.4f}")

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(bq)), "NaN")
    _check(errors, displacement < 0.1, f"Slid too much: dx={displacement:.4f}")
    return _report("Static friction", errors)


def test_g1_maintains_height():
    """A5: G1 with PD targets maintains height."""
    print("=== A5: G1 maintains height ===")
    try:
        g1 = _build_g1()
    except Exception as e:
        print(f"  SKIPPED ({e})")
        return True

    builder = newton.ModelBuilder()
    builder.replicate(g1, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 2.0e2
    builder.add_ground_plane()
    model = builder.finalize()

    bq, bqd, _, _ = _run_and_fk(model, num_steps=720)  # 2s

    root_z = float(bq[0, 2])
    root_qw = float(bq[0, 6])
    min_z = float(bq[:, 2].min())
    max_v = float(np.abs(bqd).max())
    drift = abs(root_z - 0.8)

    print(f"  root_z={root_z:.4f}  drift={drift:.4f}  |qw|={abs(root_qw):.4f}  min_z={min_z:.4f}  max_v={max_v:.3f}")

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(bq)), "NaN in body positions")
    _check(errors, drift < 0.05, f"Height drift: z={root_z:.4f}, drift={drift:.4f}")
    _check(errors, min_z > -0.02, f"Penetration: min_z={min_z:.4f}")
    _check(errors, abs(root_qw) > 0.9, f"Fell over: |qw|={abs(root_qw):.4f}")
    _check(errors, max_v < 1.0, f"Velocity too large: {max_v:.3f}")
    return _report("G1 maintains height", errors)


def test_h1_maintains_height():
    """A6: H1 with PD targets maintains height for 1 s.

    H1 with its default PD gains (ke=150, kd=5) is marginally stable
    under explicit integration.  We use stronger gains here and only
    simulate 1 second to keep the test feasible.
    """
    print("=== A6: H1 maintains height ===")
    try:
        # Build H1 with stronger PD gains for explicit integrator stability.
        asset_path = newton.utils.download_asset("unitree_h1")
        h1 = newton.ModelBuilder()
        h1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        h1.default_shape_cfg.ke = 2.0e3
        h1.default_shape_cfg.kd = 1.0e2
        h1.default_shape_cfg.kf = 1.0e3
        h1.default_shape_cfg.mu = 0.75
        h1.add_usd(
            str(asset_path / "usd_structured" / "h1.usda"),
            ignore_paths=["/GroundPlane"],
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        for i in range(h1.joint_dof_count):
            h1.joint_target_ke[i] = 500.0
            h1.joint_target_kd[i] = 10.0
        h1.approximate_meshes("bounding_box")
    except Exception as e:
        print(f"  SKIPPED ({e})")
        return True

    builder = newton.ModelBuilder()
    builder.replicate(h1, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    model = builder.finalize()

    # Get initial height
    init_s = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, init_s)
    initial_z = float(init_s.body_q.numpy()[0, 2])

    # 1 second (360 steps)
    bq, _bqd, _, _ = _run_and_fk(model, num_steps=360)

    root_z = float(bq[0, 2])
    root_qw = float(bq[0, 6])
    min_z = float(bq[:, 2].min())
    drift = abs(root_z - initial_z)

    print(
        f"  initial_z={initial_z:.4f}  root_z={root_z:.4f}  drift={drift:.4f}"
        f"  |qw|={abs(root_qw):.4f}  min_z={min_z:.4f}"
    )

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(bq)), "NaN")
    _check(errors, drift < 0.1, f"Height drift: {drift:.4f}")
    _check(errors, min_z > -0.02, f"Penetration: min_z={min_z:.4f}")
    _check(errors, abs(root_qw) > 0.9, f"Fell over: |qw|={abs(root_qw):.4f}")
    return _report("H1 maintains height", errors)


def test_g1_matches_mujoco():
    """A7: G1 root height close to MuJoCo after 1s."""
    print("=== A7: G1 vs MuJoCo ===")
    try:
        g1 = _build_g1()
    except Exception as e:
        print(f"  SKIPPED ({e})")
        return True

    # Build model for Raisim
    b_hs = newton.ModelBuilder()
    b_hs.replicate(g1, 1)
    b_hs.default_shape_cfg.ke = 1.0e3
    b_hs.default_shape_cfg.kd = 2.0e2
    b_hs.add_ground_plane()
    m_hs = b_hs.finalize()

    # Build model for MuJoCo
    g1_mj = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(g1_mj)
    g1_mj.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
    g1_mj.default_shape_cfg.ke = 1.0e3
    g1_mj.default_shape_cfg.kd = 2.0e2
    g1_mj.default_shape_cfg.kf = 1.0e3
    g1_mj.default_shape_cfg.mu = 0.75
    asset_path = newton.utils.download_asset("unitree_g1")
    g1_mj.add_usd(
        str(asset_path / "usd" / "g1_isaac.usd"),
        xform=wp.transform(wp.vec3(0, 0, 0.8)),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    for i in range(6, g1_mj.joint_dof_count):
        g1_mj.joint_target_ke[i] = 500.0
        g1_mj.joint_target_kd[i] = 10.0
    g1_mj.approximate_meshes("bounding_box")

    b_mj = newton.ModelBuilder()
    b_mj.replicate(g1_mj, 1)
    b_mj.default_shape_cfg.ke = 1.0e3
    b_mj.default_shape_cfg.kd = 2.0e2
    b_mj.add_ground_plane()
    m_mj = b_mj.finalize()

    dt = 1.0 / 360.0
    n = 360  # 1 second

    # Run Raisim
    hs_bq, _, _, _ = _run_and_fk(m_hs, num_steps=n, dt=dt)

    # Run MuJoCo
    mj_solver = newton.solvers.SolverMuJoCo(
        m_mj,
        solver="newton",
        integrator="implicitfast",
        njmax=300,
        nconmax=150,
        cone="elliptic",
        impratio=100,
        iterations=100,
        ls_iterations=50,
    )
    s0 = m_mj.state()
    s1 = m_mj.state()
    ctrl = m_mj.control()
    cts = m_mj.contacts()
    newton.eval_fk(m_mj, m_mj.joint_q, m_mj.joint_qd, s0)
    for _ in range(n):
        s0.clear_forces()
        mj_solver.step(s0, s1, ctrl, cts, dt)
        s0, s1 = s1, s0
    newton.eval_fk(m_mj, s0.joint_q, s0.joint_qd, s0)
    mj_bq = s0.body_q.numpy()

    hs_z = float(hs_bq[0, 2])
    mj_z = float(mj_bq[0, 2])
    diff = abs(hs_z - mj_z)
    print(f"  raisim z={hs_z:.4f}  mujoco z={mj_z:.4f}  diff={diff:.4f}")

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(hs_bq)), "NaN in Raisim")
    _check(errors, not np.any(np.isnan(mj_bq)), "NaN in MuJoCo")
    _check(errors, diff < 0.05, f"Root Z mismatch: diff={diff:.4f}")
    return _report("G1 vs MuJoCo", errors)


# =====================================================================
# Section B: NCP residual / convergence
# =====================================================================


def test_complementarity_residual():
    """B1: Complementarity residual is small after GS converge."""
    print("=== B1: Complementarity residual ===")
    from .residuals import compute_ncp_residuals  # noqa: PLC0415

    radius = 0.3
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    b = builder.add_body(
        xform=wp.transform(p=wp.vec3(0, 0, radius + 0.001), q=wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_sphere(body=b, radius=radius)
    model = builder.finalize()

    config = newton.solvers.RaisimConfig(max_gs_iterations=100)
    solver = newton.solvers.SolverRaisim(model, config=config)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    # Settle for a few steps
    dt = 1.0 / 360.0
    for _ in range(360):
        s0.clear_forces()
        model.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, dt)
        s0, s1 = s1, s0

    # Now do one more step and check residuals
    s0.clear_forces()
    model.collide(s0, contacts)
    solver.step(s0, s1, ctrl, contacts, dt)
    residuals = compute_ncp_residuals(solver, s1, contacts, dt)

    r_compl = residuals["r_compl"]
    r_cone = residuals["r_cone"]
    r_gap = residuals["r_gap"]
    print(f"  r_compl={r_compl:.6f}  r_cone={r_cone:.6f}  r_gap={r_gap:.6f}")

    errors: list[str] = []
    _check(errors, r_compl < 1e-4, f"Complementarity too large: {r_compl:.6f}")
    _check(errors, r_cone < 1e-5, f"Friction cone violation: {r_cone:.6f}")
    return _report("Complementarity residual", errors)


def test_friction_cone():
    """B2: Friction cone satisfied for sliding contact."""
    print("=== B2: Friction cone ===")
    from .residuals import compute_ncp_residuals  # noqa: PLC0415

    # Sphere sliding on ground with some lateral velocity
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    b = builder.add_body(
        xform=wp.transform(p=wp.vec3(0, 0, 0.31), q=wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_sphere(body=b, radius=0.3)
    model = builder.finalize()

    # Give lateral velocity
    s0 = model.state()
    s1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    jqd = s0.joint_qd.numpy()
    jqd[0] = 2.0  # lateral velocity
    s0.joint_qd = wp.array(jqd, dtype=wp.float32, device=model.device)

    config = newton.solvers.RaisimConfig(max_gs_iterations=100)
    solver = newton.solvers.SolverRaisim(model, config=config)
    ctrl = model.control()
    contacts = model.contacts()

    dt = 1.0 / 360.0
    s0.clear_forces()
    model.collide(s0, contacts)
    solver.step(s0, s1, ctrl, contacts, dt)

    residuals = compute_ncp_residuals(solver, s1, contacts, dt)
    r_cone = residuals["r_cone"]
    print(f"  r_cone={r_cone:.6f}")

    errors: list[str] = []
    _check(errors, r_cone < 1e-5, f"Friction cone violation: {r_cone:.6f}")
    return _report("Friction cone", errors)


# =====================================================================
# Section C: Regression
# =====================================================================


def test_free_body_energy():
    """C1: Dropped sphere actually falls, contacts ground, settles."""
    print("=== C1: Free body energy ===")
    radius = 0.3
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    b = builder.add_body(
        xform=wp.transform(p=wp.vec3(0, 0, 2.0), q=wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_sphere(body=b, radius=radius)
    model = builder.finalize()

    bq, _bqd, z_samples = _run_trajectory(model, num_steps=1800, sample_interval=10)

    z_traj = np.array([s[0] for s in z_samples])
    final_z = float(bq[0, 2])
    min_z = float(z_traj.min())
    has_descent = bool(np.any(np.diff(z_traj) < -1e-4))

    print(f"  start={z_traj[0]:.4f}  min={min_z:.4f}  final={final_z:.4f}  descent={has_descent}")

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(bq)), "NaN")
    _check(errors, has_descent, "Sphere never fell")
    _check(errors, min_z < 0.5, f"Never reached ground: min_z={min_z:.4f}")
    _check(errors, abs(final_z - radius) < 0.05, f"Resting height wrong: {final_z:.4f}")
    return _report("Free body energy", errors)


def test_g1_no_nan():
    """C2: G1 simulation for 2s produces no NaN."""
    print("=== C2: G1 no NaN ===")
    try:
        g1 = _build_g1()
    except Exception as e:
        print(f"  SKIPPED ({e})")
        return True

    builder = newton.ModelBuilder()
    builder.replicate(g1, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 2.0e2
    builder.add_ground_plane()
    model = builder.finalize()

    bq, _, _, _ = _run_and_fk(model, num_steps=720)

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(bq)), "NaN in G1 body positions")
    if not errors:
        min_z = float(bq[:, 2].min())
        print(f"  min_z={min_z:.4f}")
        _check(errors, min_z > -0.05, f"Deep penetration: min_z={min_z:.4f}")
    return _report("G1 no NaN", errors)


def test_penetration_bounded():
    """C3: Max penetration stays bounded for sphere stack."""
    print("=== C3: Penetration bounded ===")
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()

    r = 0.15
    for i in range(5):
        b = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0, 0, r + 0.01 + i * (2.0 * r + 0.05)),
                q=wp.quat_identity(),
            ),
            mass=1.0,
        )
        builder.add_shape_sphere(body=b, radius=r)
    model = builder.finalize()

    bq, _, _, _ = _run_and_fk(
        model,
        num_steps=1800,
        config=newton.solvers.RaisimConfig(max_gs_iterations=100),
    )

    min_z = float(bq[:, 2].min())
    print(f"  min_z={min_z:.4f}  (must be > -0.01)")

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(bq)), "NaN")
    _check(errors, min_z > -0.01, f"Penetration too deep: min_z={min_z:.4f}")
    return _report("Penetration bounded", errors)


# =====================================================================
# Section D: Quadrupeds and stacking stress tests
# =====================================================================


def _build_anymal_d():
    """Return ModelBuilder with Anymal D loaded."""
    asset_path = newton.utils.download_asset("anybotics_anymal_d")
    ab = newton.ModelBuilder()
    ab.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
    ab.default_shape_cfg.ke = 2.0e3
    ab.default_shape_cfg.kd = 1.0e2
    ab.default_shape_cfg.kf = 1.0e3
    ab.default_shape_cfg.mu = 0.75
    ab.add_usd(
        str(asset_path / "usd" / "anymal_d.usda"),
        xform=wp.transform(wp.vec3(0, 0, 0.62)),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    # Anymal D requires high PD gains under explicit integration because
    # the legs are heavy and far from the body COM.
    for i in range(ab.joint_dof_count):
        ab.joint_target_ke[i] = 2000.0
        ab.joint_target_kd[i] = 40.0
    ab.approximate_meshes("bounding_box")
    return ab


def _build_go2():
    """Return ModelBuilder with Unitree Go2 loaded."""
    asset_path = newton.utils.download_asset("unitree_go2")
    go2 = newton.ModelBuilder()
    go2.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
    go2.default_shape_cfg.ke = 2.0e3
    go2.default_shape_cfg.kd = 1.0e2
    go2.default_shape_cfg.kf = 1.0e3
    go2.default_shape_cfg.mu = 0.75
    go2.add_usd(
        str(asset_path / "usd" / "go2.usda"),
        xform=wp.transform(wp.vec3(0, 0, 0.35)),
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    for i in range(go2.joint_dof_count):
        go2.joint_target_ke[i] = 500.0
        go2.joint_target_kd[i] = 10.0
    go2.approximate_meshes("bounding_box")
    return go2


def test_anymal_d_stands():
    """D1: Anymal D quadruped maintains height for 1 s.

    Anymal D has heavy legs far from the body COM, requiring very high
    PD gains under explicit integration.  We test a 1-second window.
    """
    print("=== D1: Anymal D quadruped ===")
    try:
        ab = _build_anymal_d()
    except Exception as e:
        print(f"  SKIPPED ({e})")
        return True

    builder = newton.ModelBuilder()
    builder.replicate(ab, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    model = builder.finalize()

    init_s = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, init_s)
    initial_z = float(init_s.body_q.numpy()[0, 2])

    bq, bqd, _, _ = _run_and_fk(model, num_steps=360)  # 1s

    root_z = float(bq[0, 2])
    root_qw = float(bq[0, 6])
    min_z = float(bq[:, 2].min())
    max_v = float(np.abs(bqd).max())
    drift = abs(root_z - initial_z)

    print(
        f"  initial_z={initial_z:.4f}  root_z={root_z:.4f}  drift={drift:.4f}"
        f"  |qw|={abs(root_qw):.4f}  min_z={min_z:.4f}  max_v={max_v:.3f}"
    )

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(bq)), "NaN in body positions")
    _check(errors, drift < 0.15, f"Height drift: z={root_z:.4f}, drift={drift:.4f}")
    _check(errors, min_z > -0.05, f"Penetration: min_z={min_z:.4f}")
    _check(errors, abs(root_qw) > 0.85, f"Fell over: |qw|={abs(root_qw):.4f}")
    return _report("Anymal D quadruped", errors)


def test_go2_stands():
    """D2: Unitree Go2 quadruped maintains height for 1 s."""
    print("=== D2: Unitree Go2 quadruped ===")
    try:
        go2 = _build_go2()
    except Exception as e:
        print(f"  SKIPPED ({e})")
        return True

    builder = newton.ModelBuilder()
    builder.replicate(go2, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    model = builder.finalize()

    init_s = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, init_s)
    initial_z = float(init_s.body_q.numpy()[0, 2])

    bq, _bqd, _, _ = _run_and_fk(model, num_steps=360)  # 1s

    root_z = float(bq[0, 2])
    root_qw = float(bq[0, 6])
    min_z = float(bq[:, 2].min())
    drift = abs(root_z - initial_z)

    print(
        f"  initial_z={initial_z:.4f}  root_z={root_z:.4f}  drift={drift:.4f}"
        f"  |qw|={abs(root_qw):.4f}  min_z={min_z:.4f}"
    )

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(bq)), "NaN in body positions")
    _check(errors, drift < 0.1, f"Height drift: z={root_z:.4f}, drift={drift:.4f}")
    _check(errors, min_z > -0.02, f"Penetration: min_z={min_z:.4f}")
    _check(errors, abs(root_qw) > 0.85, f"Fell over: |qw|={abs(root_qw):.4f}")
    return _report("Go2 quadruped", errors)


def test_box_tower():
    """D3: Tower of 20 boxes — all settle without deep penetration."""
    print("=== D3: Box tower (20 boxes) ===")
    half = 0.1
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()

    for i in range(20):
        z = half + 0.001 + i * (2.0 * half + 0.002)
        b = builder.add_body(
            xform=wp.transform(p=wp.vec3(0, 0, z), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(body=b, hx=half, hy=half, hz=half)
    model = builder.finalize()

    config = newton.solvers.RaisimConfig(max_gs_iterations=100)
    bq, bqd, _, _ = _run_and_fk(model, num_steps=1800, config=config)  # 5s

    min_z = float(bq[:, 2].min())
    max_z = float(bq[:, 2].max())
    max_v = float(np.abs(bqd).max())
    has_nan = bool(np.any(np.isnan(bq)))

    print(f"  boxes=20  min_z={min_z:.4f}  max_z={max_z:.4f}  max_v={max_v:.3f}")

    errors: list[str] = []
    _check(errors, not has_nan, "NaN in body positions")
    _check(errors, min_z > -0.02, f"Penetration: min_z={min_z:.4f}")
    _check(errors, max_v < 5.0, f"Not settled: max_v={max_v:.3f}")
    # Bottom box should be near z=half, top near z=20*2*half
    _check(errors, min_z < half + 0.1, f"Bottom box too high: min_z={min_z:.4f}")
    return _report("Box tower (20 boxes)", errors)


def test_box_pyramid():
    """D4: Pyramid of 15 boxes (5-row triangle) — stable for 3 s."""
    print("=== D4: Box pyramid (15 boxes) ===")
    half = 0.1
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.6
    builder.add_ground_plane()

    rows = 5
    box_count = 0
    for row in range(rows):
        n = rows - row
        for col in range(n):
            x = (col - (n - 1) / 2.0) * (2.0 * half + 0.005)
            z = half + 0.001 + row * (2.0 * half + 0.002)
            b = builder.add_body(
                xform=wp.transform(p=wp.vec3(x, 0, z), q=wp.quat_identity()),
                mass=1.0,
            )
            builder.add_shape_box(body=b, hx=half, hy=half, hz=half)
            box_count += 1
    model = builder.finalize()

    config = newton.solvers.RaisimConfig(max_gs_iterations=100)
    bq, bqd, _, _ = _run_and_fk(model, num_steps=1080, config=config)  # 3s

    min_z = float(bq[:, 2].min())
    max_v = float(np.abs(bqd).max())
    has_nan = bool(np.any(np.isnan(bq)))

    print(f"  boxes={box_count}  min_z={min_z:.4f}  max_v={max_v:.3f}")

    errors: list[str] = []
    _check(errors, not has_nan, "NaN in body positions")
    _check(errors, min_z > -0.02, f"Penetration: min_z={min_z:.4f}")
    _check(errors, max_v < 5.0, f"Not settled: max_v={max_v:.3f}")
    return _report("Box pyramid (15 boxes)", errors)


def test_long_stability():
    """D5: G1 robot stable for 10 s (3600 steps) — no drift or NaN."""
    print("=== D5: G1 long stability (10 s) ===")
    try:
        g1 = _build_g1()
    except Exception as e:
        print(f"  SKIPPED ({e})")
        return True

    builder = newton.ModelBuilder()
    builder.replicate(g1, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 2.0e2
    builder.add_ground_plane()
    model = builder.finalize()

    bq, _bqd, _, _ = _run_and_fk(model, num_steps=3600)  # 10s

    root_z = float(bq[0, 2])
    root_qw = float(bq[0, 6])
    min_z = float(bq[:, 2].min())
    drift = abs(root_z - 0.8)

    print(f"  root_z={root_z:.4f}  drift={drift:.4f}  |qw|={abs(root_qw):.4f}  min_z={min_z:.4f}")

    errors: list[str] = []
    _check(errors, not np.any(np.isnan(bq)), "NaN in body positions")
    _check(errors, drift < 0.1, f"Height drift after 10s: {drift:.4f}")
    _check(errors, abs(root_qw) > 0.85, f"Fell over: |qw|={abs(root_qw):.4f}")
    _check(errors, min_z > -0.05, f"Penetration: min_z={min_z:.4f}")
    return _report("G1 long stability (10 s)", errors)


# =====================================================================
# Main
# =====================================================================


def main():
    results = []

    # Section A: Physical correctness
    results.append(("A1 Sphere resting height", test_sphere_resting_height()))
    results.append(("A2 Box resting orientation", test_box_resting_orientation()))
    results.append(("A3 Free-fall trajectory", test_free_fall()))
    results.append(("A4 Static friction", test_static_friction()))
    results.append(("A5 G1 maintains height", test_g1_maintains_height()))
    results.append(("A6 H1 maintains height", test_h1_maintains_height()))
    results.append(("A7 G1 vs MuJoCo", test_g1_matches_mujoco()))

    # Section B: NCP residuals
    results.append(("B1 Complementarity residual", test_complementarity_residual()))
    results.append(("B2 Friction cone", test_friction_cone()))

    # Section C: Regression
    results.append(("C1 Free body energy", test_free_body_energy()))
    results.append(("C2 G1 no NaN", test_g1_no_nan()))
    results.append(("C3 Penetration bounded", test_penetration_bounded()))

    # Section D: Quadrupeds and stacking
    results.append(("D1 Anymal D quadruped", test_anymal_d_stands()))
    results.append(("D2 Go2 quadruped", test_go2_stands()))
    results.append(("D3 Box tower (20)", test_box_tower()))
    results.append(("D4 Box pyramid (15)", test_box_pyramid()))
    results.append(("D5 G1 long stability", test_long_stability()))

    print()
    n_pass = sum(1 for _, p in results if p)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print(f"\n{n_pass}/{len(results)} tests passed.")
    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
