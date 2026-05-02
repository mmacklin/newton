# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


def _identity_inertia():
    return wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _quat_from_angle_z(theta: float):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), theta)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    qv = np.array([x, y, z], dtype=float)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def _transform_point_np(xform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return xform[:3] + _quat_rotate_np(xform[3:7], point)


def _deformed_endpoint_world(model, state, joint: int, side: str) -> np.ndarray:
    body = int(model.joint_parent.numpy()[joint]) if side == "parent" else int(model.joint_child.numpy()[joint])
    xforms = model.joint_X_p.numpy() if side == "parent" else model.joint_X_c.numpy()
    local = np.array(xforms[joint, :3], dtype=float)
    endpoint_ids = (
        model.joint_parent_elastic_endpoint.numpy() if side == "parent" else model.joint_child_elastic_endpoint.numpy()
    )
    endpoint = int(endpoint_ids[joint])
    if endpoint >= 0:
        elastic = int(model.body_elastic_index.numpy()[body])
        owner_joint = int(model.elastic_joint.numpy()[elastic])
        q_start = int(model.joint_q_start.numpy()[owner_joint])
        mode_count = int(model.elastic_mode_count.numpy()[elastic])
        phi = model.elastic_endpoint_phi.numpy().reshape((-1, 3))
        max_modes = int(model.elastic_max_mode_count)
        q = state.joint_q.numpy()
        for i in range(mode_count):
            local += phi[endpoint * max_modes + i] * q[q_start + 7 + i]
    if body < 0:
        return local
    return _transform_point_np(state.body_q.numpy()[body], local)


def _solve_fourbar(theta2: float, a: float, b: float, c: float, d: float) -> tuple[float, float]:
    """Return open-branch coupler and rocker angles for a planar four-bar."""
    k1 = d / a
    k2 = d / c
    k3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c)
    A = k1 - math.cos(theta2)
    B = -math.sin(theta2)
    C = k2 * math.cos(theta2) - k3
    denom = math.sqrt(A * A + B * B)
    theta4 = math.atan2(B, A) + math.acos(np.clip(C / denom, -1.0, 1.0))
    cx = d + c * math.cos(theta4) - a * math.cos(theta2)
    cy = c * math.sin(theta4) - a * math.sin(theta2)
    theta3 = math.atan2(cy, cx)
    return theta3, theta4


def test_elastic_link_layout(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=2,
        mode_mass=[2.0, 3.0],
        mode_stiffness=[5.0, 7.0],
        mode_damping=[0.2, 0.3],
        mode_q=[0.1, -0.2],
        mode_qd=[0.4, -0.5],
        label="elastic",
    )
    builder.color()
    model = builder.finalize(device=device)

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])

    test.assertEqual(body, 0)
    test.assertEqual(model.elastic_body_count, 1)
    test.assertEqual(model.elastic_max_mode_count, 2)
    test.assertEqual(int(model.body_elastic_index.numpy()[body]), 0)
    test.assertEqual(int(model.joint_type.numpy()[owner_joint]), int(newton.JointType.ELASTIC))
    test.assertEqual(int(model.joint_q_start.numpy()[owner_joint + 1] - model.joint_q_start.numpy()[owner_joint]), 9)
    test.assertEqual(int(model.joint_qd_start.numpy()[owner_joint + 1] - model.joint_qd_start.numpy()[owner_joint]), 8)
    np.testing.assert_allclose(model.joint_q.numpy()[q_start + 7 : q_start + 9], [0.1, -0.2], atol=1.0e-7)
    np.testing.assert_allclose(model.joint_qd.numpy()[qd_start + 6 : qd_start + 8], [0.4, -0.5], atol=1.0e-7)
    np.testing.assert_allclose(model.elastic_mode_mass.numpy(), [2.0, 3.0], atol=1.0e-7)


def test_elastic_fk_sync(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1)
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])

    q = state.joint_q.numpy()
    q[q_start : q_start + 7] = [1.0, 2.0, 3.0, 0.0, 0.0, math.sin(0.25), math.cos(0.25)]
    q[q_start + 7] = 0.35
    state.joint_q.assign(q)

    qd = state.joint_qd.numpy()
    qd[qd_start : qd_start + 7] = [0.4, 0.5, 0.6, 0.1, 0.2, 0.3, -0.7]
    state.joint_qd.assign(qd)

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    np.testing.assert_allclose(state.body_q.numpy()[body], q[q_start : q_start + 7], atol=1.0e-6)
    np.testing.assert_allclose(state.body_qd.numpy()[body], qd[qd_start : qd_start + 6], atol=1.0e-6)
    np.testing.assert_allclose(state.joint_q.numpy()[q_start + 7], 0.35, atol=1.0e-7)


def test_elastic_endpoint_shape_sampling(test, device):
    def shape_fn(x):
        return np.array([[2.0 * x[0], -x[1], 0.5]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=0.0)
    parent = builder.add_body(mass=1.0, inertia=_identity_inertia())
    child = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1, mode_shape_fn=shape_fn)
    joint = builder.add_joint_revolute(
        parent=parent,
        child=child,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5, -0.25, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(int(model.joint_parent_elastic_endpoint.numpy()[joint]), -1)
    endpoint = int(model.joint_child_elastic_endpoint.numpy()[joint])
    test.assertGreaterEqual(endpoint, 0)
    phi = model.elastic_endpoint_phi.numpy().reshape((-1, 3))
    np.testing.assert_allclose(phi[endpoint], [1.0, 0.25, 0.5], atol=1.0e-7)


def test_elastic_render_shape_sampling(test, device):
    def shape_fn(x):
        xi = x[0] / 0.5
        return np.array([[0.0, 1.0 - xi * xi, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_shape_fn=shape_fn,
    )
    builder.add_shape_box(body, hx=0.5, hy=0.05, hz=0.03)
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(model.elastic_render_point_total_count, 33)
    test.assertEqual(int(model.elastic_render_point_start.numpy()[0]), 0)
    test.assertEqual(int(model.elastic_render_point_count.numpy()[0]), 33)

    local = model.elastic_render_point_local.numpy()
    np.testing.assert_allclose(local[0, 0], -0.5, atol=1.0e-7)
    np.testing.assert_allclose(local[-1, 0], 0.5, atol=1.0e-7)
    test.assertGreater(local[0, 2], 0.03)

    phi = model.elastic_render_point_phi.numpy().reshape((-1, 3))
    np.testing.assert_allclose(phi[16], [0.0, 1.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(phi[0], [0.0, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(phi[-1], [0.0, 0.0, 0.0], atol=1.0e-6)


def test_elastic_shape_mesh_sampling(test, device):
    length = 1.0
    hy = 0.05
    hz = 0.03

    def shape_fn(x):
        s = float(x[0] + 0.5 * length)
        xi = s / length
        phi = math.sin(math.pi * xi)
        slope = (math.pi / length) * math.cos(math.pi * xi)
        return np.array([[-float(x[1]) * slope, phi, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1, mode_shape_fn=shape_fn)
    builder.add_shape_box(body, hx=0.5 * length, hy=hy, hz=hz)
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(model.elastic_shape_count, 1)
    test.assertGreater(model.elastic_shape_vertex_total_count, 8)
    test.assertGreater(model.elastic_shape_index_total_count, 36)
    test.assertEqual(int(model.elastic_shape_shape.numpy()[0]), 0)

    local = model.elastic_shape_vertex_local.numpy()
    phi = model.elastic_shape_vertex_phi.numpy().reshape((-1, 3))
    mid = int(np.argmin(np.abs(local[:, 0]) + np.abs(local[:, 1] - hy) + np.abs(local[:, 2] - hz)))
    np.testing.assert_allclose(phi[mid], [0.0, 1.0, 0.0], atol=1.0e-6)

    left = np.where(np.isclose(local[:, 0], -0.5 * length))[0]
    right = np.where(np.isclose(local[:, 0], 0.5 * length))[0]
    np.testing.assert_allclose(phi[left, 1], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(phi[right, 1], 0.0, atol=1.0e-6)


def test_torsion_render_shape_sampling(test, device):
    length = 1.0
    hy = 0.06
    hz = 0.025

    def torsion_shape_fn(x):
        s = float(x[0] + 0.5 * length)
        twist = s / length
        return np.array([[0.0, -float(x[2]) * twist, float(x[1]) * twist]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1, mode_shape_fn=torsion_shape_fn)
    builder.add_shape_box(body, hx=0.5 * length, hy=hy, hz=hz)
    builder.color()
    model = builder.finalize(device=device)

    np.testing.assert_allclose(torsion_shape_fn(np.array([0.0, 0.0, 0.0], dtype=np.float32))[0], 0.0, atol=1.0e-7)

    local = model.elastic_shape_vertex_local.numpy()
    phi = model.elastic_shape_vertex_phi.numpy().reshape((-1, 3))
    tip = int(np.argmin(np.abs(local[:, 0] - 0.5 * length) + np.abs(local[:, 1] - hy) + np.abs(local[:, 2] - hz)))
    np.testing.assert_allclose(phi[tip], [0.0, -hz, hy], atol=1.0e-6)


def test_cantilever_tip_load_solution(test, device):
    length = 0.8
    ei = 0.32
    tip_load = 0.2
    stiffness = 3.0 * ei / (length**3)

    def cantilever_shape_fn(x):
        s = float(x[0] + 0.5 * length)
        s = min(max(s, 0.0), length)
        phi = (s * s * (3.0 * length - s)) / (2.0 * length**3)
        slope = (3.0 * s * (2.0 * length - s)) / (2.0 * length**3)
        return np.array([[-float(x[2]) * slope, 0.0, phi]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[0.0],
        mode_stiffness=[stiffness],
        mode_damping=[0.0],
        mode_shape_fn=cantilever_shape_fn,
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])
    jf = control.joint_f.numpy()
    jf[qd_start + 6] = tip_load
    control.joint_f.assign(jf)

    solver = newton.solvers.SolverVBD(model, iterations=0)
    solver.step(state_0, state_1, control, None, 0.01)

    expected_tip_deflection = tip_load * length**3 / (3.0 * ei)
    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], expected_tip_deflection, atol=1.0e-7)
    tip_phi = cantilever_shape_fn(np.array([0.5 * length, 0.0, 0.0], dtype=np.float32))[0]
    np.testing.assert_allclose(tip_phi, [0.0, 0.0, 1.0], atol=1.0e-7)


def test_elastic_modal_implicit_solution_vbd(test, device):
    mode_mass = 2.0
    stiffness = 18.0
    damping = 0.5
    q0 = 0.1
    v0 = -0.2
    force = 1.3
    dt = 0.01

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[mode_mass],
        mode_stiffness=[stiffness],
        mode_damping=[damping],
        mode_q=[q0],
        mode_qd=[v0],
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])
    jf = control.joint_f.numpy()
    jf[qd_start + 6] = force
    control.joint_f.assign(jf)

    solver = newton.solvers.SolverVBD(model, iterations=0)
    solver.step(state_0, state_1, control, None, dt)

    v_expected = (mode_mass * v0 + dt * (force - stiffness * q0)) / (mode_mass + dt * damping + dt * dt * stiffness)
    q_expected = q0 + dt * v_expected
    np.testing.assert_allclose(state_0.joint_q.numpy()[q_start + 7], q0, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(state_0.joint_qd.numpy()[qd_start + 6], v0, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], q_expected, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(state_1.joint_qd.numpy()[qd_start + 6], v_expected, rtol=1.0e-6, atol=1.0e-7)


def test_vbd_revolute_uses_elastic_endpoint(test, device):
    eta = 0.2
    rest_anchor = -0.5

    def shape_fn(x):
        return np.array([[x[0] / abs(rest_anchor), 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(abs(rest_anchor) + eta, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_q=[eta],
        mode_shape_fn=shape_fn,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(rest_anchor, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    before = _deformed_endpoint_world(model, state_0, joint, "child")
    np.testing.assert_allclose(before, [0.0, 0.0, 0.0], atol=1.0e-6)

    solver = newton.solvers.SolverVBD(model, iterations=8, rigid_joint_linear_ke=1.0e5, rigid_joint_angular_ke=1.0e5)
    solver.step(state_0, state_1, control, None, 0.01)

    after = _deformed_endpoint_world(model, state_1, joint, "child")
    np.testing.assert_allclose(after, [0.0, 0.0, 0.0], atol=1.0e-4)
    np.testing.assert_allclose(state_1.body_q.numpy()[body, :3], [abs(rest_anchor) + eta, 0.0, 0.0], atol=1.0e-4)


def test_vbd_revolute_constraint_solves_elastic_mode(test, device):
    rest_anchor = -0.5
    target_anchor = 0.1

    def shape_fn(x):
        return np.array([[x[0] / abs(rest_anchor), 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(abs(rest_anchor), 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_mass=[1.0e-6],
        mode_q=[0.0],
        mode_shape_fn=shape_fn,
        is_kinematic=True,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(target_anchor, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(rest_anchor, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])

    solver = newton.solvers.SolverVBD(
        model, iterations=4, rigid_joint_linear_k_start=1.0e6, rigid_joint_linear_ke=1.0e6
    )
    solver.step(state_0, state_1, control, None, 0.01)

    np.testing.assert_allclose(state_1.joint_q.numpy()[q_start + 7], -target_anchor, atol=1.0e-4)
    after = _deformed_endpoint_world(model, state_1, joint, "child")
    np.testing.assert_allclose(after, [target_anchor, 0.0, 0.0], atol=1.0e-4)


def test_fourbar_elastic_coupler_geometry(test, device):
    a, b_rest, c, d = 0.2, 0.5, 0.4, 0.5
    eta = 0.06
    b_eff = b_rest + eta
    theta2 = 0.35
    theta3, theta4 = _solve_fourbar(theta2, a, b_eff, c, d)

    A = np.array([0.0, 0.0, 0.0])
    D = np.array([d, 0.0, 0.0])
    e2 = np.array([math.cos(theta2), math.sin(theta2), 0.0])
    e3 = np.array([math.cos(theta3), math.sin(theta3), 0.0])
    e4 = np.array([math.cos(theta4), math.sin(theta4), 0.0])
    B = A + a * e2
    C = B + b_eff * e3
    np.testing.assert_allclose(C, D + c * e4, atol=1.0e-6)

    def axial_shape_fn(x):
        return np.array([[x[0] / b_rest, 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=0.0)
    crank = builder.add_body(
        xform=wp.transform(wp.vec3(*(A + 0.5 * a * e2)), _quat_from_angle_z(theta2)),
        mass=1.0,
        inertia=_identity_inertia(),
    )
    coupler = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(*(B + 0.5 * b_eff * e3)), _quat_from_angle_z(theta3)),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_q=[eta],
        mode_shape_fn=axial_shape_fn,
    )
    rocker = builder.add_body(
        xform=wp.transform(wp.vec3(*(D + 0.5 * c * e4)), _quat_from_angle_z(theta4)),
        mass=1.0,
        inertia=_identity_inertia(),
    )

    j_ab = builder.add_joint_revolute(
        parent=crank,
        child=coupler,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(a / 2.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-b_rest / 2.0, 0.0, 0.0), wp.quat_identity()),
    )
    j_bc = builder.add_joint_revolute(
        parent=coupler,
        child=rocker,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(b_rest / 2.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(c / 2.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    p_ab_parent = _deformed_endpoint_world(model, state, j_ab, "parent")
    p_ab_child = _deformed_endpoint_world(model, state, j_ab, "child")
    p_bc_parent = _deformed_endpoint_world(model, state, j_bc, "parent")
    p_bc_child = _deformed_endpoint_world(model, state, j_bc, "child")

    np.testing.assert_allclose(p_ab_parent, B, atol=2.0e-6)
    np.testing.assert_allclose(p_ab_child, B, atol=2.0e-6)
    np.testing.assert_allclose(p_bc_parent, C, atol=2.0e-6)
    np.testing.assert_allclose(p_bc_child, C, atol=2.0e-6)
    np.testing.assert_allclose(np.linalg.norm(p_bc_parent - p_ab_child), b_eff, atol=2.0e-6)


class TestReducedElasticBody(unittest.TestCase):
    pass


for device in devices:
    add_function_test(TestReducedElasticBody, "test_elastic_link_layout", test_elastic_link_layout, devices=[device])
    add_function_test(TestReducedElasticBody, "test_elastic_fk_sync", test_elastic_fk_sync, devices=[device])
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_endpoint_shape_sampling",
        test_elastic_endpoint_shape_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_render_shape_sampling",
        test_elastic_render_shape_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_shape_mesh_sampling",
        test_elastic_shape_mesh_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_torsion_render_shape_sampling",
        test_torsion_render_shape_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_cantilever_tip_load_solution",
        test_cantilever_tip_load_solution,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_modal_implicit_solution_vbd",
        test_elastic_modal_implicit_solution_vbd,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_revolute_uses_elastic_endpoint",
        test_vbd_revolute_uses_elastic_endpoint,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_revolute_constraint_solves_elastic_mode",
        test_vbd_revolute_constraint_solves_elastic_mode,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_fourbar_elastic_coupler_geometry",
        test_fourbar_elastic_coupler_geometry,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
