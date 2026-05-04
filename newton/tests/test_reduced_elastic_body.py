# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.basic._reduced_elastic import (
    beam_render_sample_points,
    box_surface_mesh,
    finite_torsion_displacement,
    joint_endpoint_world,
    mesh_volume,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


class _ElasticMeshColorProbe(newton.viewer.ViewerNull):
    def __init__(self):
        super().__init__(num_frames=1)
        self.mesh_colors = {}

    def log_mesh(
        self,
        name,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden=False,
        backface_culling=True,
        colors=None,
    ):
        if name.startswith("/model/elastic_shapes/"):
            self.mesh_colors[name] = None if colors is None else colors.numpy()


def _identity_inertia():
    return wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _quat_from_angle_z(theta: float):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), theta)


def _quat_angle_error(q_a: np.ndarray, q_b: np.ndarray) -> float:
    dot = abs(float(np.dot(q_a, q_b)))
    dot = min(max(dot, -1.0), 1.0)
    return 2.0 * math.acos(dot)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    qv = np.array([x, y, z], dtype=float)
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)


def _transform_point_np(xform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return xform[:3] + _quat_rotate_np(xform[3:7], point)


def test_modal_basis_add_sample(test, device):
    basis = newton.ModalBasis(
        sample_points=[[0.0, 0.0, 0.0]],
        sample_phi=[[[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]],
        mode_mass=[2.0, 3.0],
        mode_stiffness=[4.0, 5.0],
        mode_damping=[0.1, 0.2],
    )

    test.assertEqual(basis.mode_count, 2)
    test.assertEqual(basis.sample_count, 1)
    test.assertEqual(basis.add_sample([0.0, 0.0, 0.0]), 0)
    sample = basis.add_sample([1.0, 0.0, 0.0], phi=[[0.5, 0.0, 0.0], [0.0, 0.0, 0.25]])
    test.assertEqual(sample, 1)
    np.testing.assert_allclose(basis.sample_value(sample), [[0.5, 0.0, 0.0], [0.0, 0.0, 0.25]], atol=1.0e-7)


def test_modal_generator_beam_samples(test, device):
    length = 1.0
    basis = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.1,
        half_width_z=0.05,
        mode_specs=[
            {"type": newton.ModalGeneratorBeam.Mode.AXIAL},
            {
                "type": newton.ModalGeneratorBeam.Mode.BENDING_Y,
                "boundary": newton.ModalGeneratorBeam.Boundary.PINNED_PINNED,
            },
            {
                "type": newton.ModalGeneratorBeam.Mode.TORSION,
                "boundary": newton.ModalGeneratorBeam.Boundary.LINEAR,
            },
        ],
        sample_count=3,
    ).build()

    sample = basis.add_sample([0.0, 0.1, 0.05])
    phi = basis.sample_value(sample)
    np.testing.assert_allclose(phi[0], [0.0, 0.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(phi[1], [0.0, 1.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(phi[2], [0.0, -0.025, 0.05], atol=1.0e-7)
    test.assertGreater(float(basis.mode_stiffness[0]), 0.0)


def test_modal_generator_pod_rank_one(test, device):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    displacements = np.array([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=np.float32)

    basis = newton.ModalGeneratorPOD(
        sample_points=points,
        displacements=displacements,
        mode_count=1,
        total_mass=2.0,
        stiffness_scale=3.0,
    ).build()

    test.assertEqual(basis.mode_count, 1)
    np.testing.assert_allclose(np.abs(basis.sample_value(1)[0]), [0.0, 1.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(basis.mode_mass, [1.0], atol=1.0e-7)
    np.testing.assert_allclose(basis.mode_stiffness, [3.0], atol=1.0e-7)


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


def test_elastic_endpoint_modal_basis_sampling(test, device):
    basis = newton.ModalBasis(
        sample_points=[[0.5, -0.25, 0.0]],
        sample_phi=[[[1.0, 0.25, 0.5]]],
        mode_mass=[1.0],
    )

    builder = newton.ModelBuilder(gravity=0.0)
    parent = builder.add_body(mass=1.0, inertia=_identity_inertia())
    child = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis)
    joint = builder.add_joint_revolute(
        parent=parent,
        child=child,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5, -0.25, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)

    endpoint = int(model.joint_child_elastic_endpoint.numpy()[joint])
    test.assertGreaterEqual(endpoint, 0)
    test.assertEqual(int(model.elastic_endpoint_sample.numpy()[endpoint]), 0)
    test.assertEqual(model.modal_basis_count, 1)
    np.testing.assert_allclose(model.elastic_basis.numpy(), [0], atol=0)
    phi = model.elastic_endpoint_phi.numpy().reshape((-1, 3))
    np.testing.assert_allclose(phi[endpoint], [1.0, 0.25, 0.5], atol=1.0e-7)


def test_modal_basis_shared_by_elastic_bodies(test, device):
    basis = newton.ModalGeneratorBeam(
        length=1.0,
        half_width_y=0.05,
        half_width_z=0.03,
        mode_specs=[{"type": newton.ModalGeneratorBeam.Mode.AXIAL}],
        sample_count=3,
    ).build()

    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        modal_basis=basis,
    )
    builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.0, 0.4, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
        modal_basis=basis,
    )
    builder.color()
    model = builder.finalize(device=device)

    test.assertEqual(model.elastic_body_count, 2)
    test.assertEqual(model.modal_basis_count, 1)
    np.testing.assert_allclose(model.elastic_basis.numpy(), [0, 0], atol=0)


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


def test_elastic_shape_box_winding(test, device):
    hx = 0.5
    hy = 0.05
    hz = 0.03

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), mode_count=1)
    builder.add_shape_box(body, hx=hx, hy=hy, hz=hz)
    builder.color()
    model = builder.finalize(device=device)

    vertices = model.elastic_shape_vertex_local.numpy()
    indices = model.elastic_shape_indices.numpy()
    test.assertGreater(len(indices), 36)

    for tri in indices.reshape((-1, 3)):
        a, b, c = vertices[tri]
        normal = np.cross(b - a, c - a)
        center = (a + b + c) / 3.0
        axis = int(np.argmax(np.abs(center / np.array([hx, hy, hz], dtype=np.float32))))
        expected_sign = 1.0 if center[axis] > 0.0 else -1.0
        test.assertGreater(float(normal[axis] * expected_sign), 0.0)


def test_elastic_shape_box_exact_modal_samples(test, device):
    length = 0.8
    hy = 0.04
    hz = 0.03
    surface_points, _ = box_surface_mesh(length, hy, hz)
    for axis in range(3):
        axis_values = np.unique(np.round(surface_points[:, axis], decimals=8))
        test.assertLessEqual(float(np.max(np.diff(axis_values))), 0.010001)
    test.assertGreater(np.unique(surface_points[:, 1]).size, 2)
    test.assertGreater(np.unique(surface_points[:, 2]).size, 2)

    basis = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=hy,
        half_width_z=hz,
        mode_specs=[
            {
                "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_TIP,
            }
        ],
    ).build(
        sample_points=beam_render_sample_points(
            length,
            hy,
            hz,
            extra_points=((-0.5 * length, 0.0, 0.0), (0.5 * length, 0.0, 0.0)),
        )
    )

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(mass=1.0, inertia=_identity_inertia(), modal_basis=basis)
    builder.add_shape_box(body, hx=0.5 * length, hy=hy, hz=hz)
    builder.color()
    model = builder.finalize(device=device)

    samples = model.elastic_shape_vertex_sample.numpy()
    np.testing.assert_array_equal(samples, np.arange(surface_points.shape[0], dtype=np.int32))
    np.testing.assert_allclose(model.elastic_shape_vertex_local.numpy(), surface_points, atol=1.0e-7)

    phi = model.elastic_shape_vertex_phi.numpy().reshape((-1, model.elastic_max_mode_count, 3))[:, 0]
    expected_phi = np.asarray([basis.sample_value(i)[0] for i in range(surface_points.shape[0])], dtype=np.float32)
    np.testing.assert_allclose(phi, expected_phi, atol=1.0e-7)


def test_elastic_strain_visualization_colors(test, device):
    length = 1.0

    def axial_gradient_shape(x):
        s = float(x[0] + 0.5 * length)
        return np.array([[s / length, 0.0, 0.0]], dtype=np.float32)

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_count=1,
        mode_shape_fn=axial_gradient_shape,
    )
    builder.add_shape_box(body, hx=0.5 * length, hy=0.04, hz=0.03)
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])

    viewer = _ElasticMeshColorProbe()
    viewer.set_model(model)
    viewer.show_elastic_strain = True

    q = state.joint_q.numpy()
    q[q_start + 7] = 0.04
    state.joint_q.assign(q)
    viewer.log_state(state)
    colors_small = viewer.mesh_colors["/model/elastic_shapes/shape_0"].copy()

    q[q_start + 7] = 0.08
    state.joint_q.assign(q)
    viewer.log_state(state)

    test.assertIn("/model/elastic_shapes/shape_0", viewer.mesh_colors)
    colors = viewer.mesh_colors["/model/elastic_shapes/shape_0"]
    test.assertIsNotNone(colors)
    test.assertEqual(colors.shape[1], 3)
    test.assertGreater(float(np.max(np.abs(colors - colors_small))), 0.2)
    test.assertGreater(float(np.ptp(colors[:, 0])), 0.1)
    test.assertGreater(float(np.ptp(colors[:, 2])), 0.1)
    test.assertTrue(bool(np.all(colors >= -1.0e-6)))
    test.assertTrue(bool(np.all(colors <= 1.0 + 1.0e-6)))


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


def test_finite_torsion_exemplar_preserves_volume(test, device):
    length = 1.0
    hy = 0.085
    hz = 0.05
    tip_twist = math.pi / 2.0
    vertices, indices = box_surface_mesh(length, hy, hz)
    rest_volume = mesh_volume(vertices, indices)

    finite_vertices = vertices + finite_torsion_displacement(vertices, length, tip_twist)
    finite_ratio = mesh_volume(finite_vertices, indices) / rest_volume
    np.testing.assert_allclose(finite_ratio, 1.0, atol=2.0e-3)

    s = np.clip(vertices[:, 0] + 0.5 * length, 0.0, length)
    theta = tip_twist * s / length
    linear_vertices = vertices.copy()
    linear_vertices[:, 1] -= vertices[:, 2] * theta
    linear_vertices[:, 2] += vertices[:, 1] * theta
    linear_ratio = mesh_volume(linear_vertices, indices) / rest_volume
    test.assertGreater(linear_ratio, 1.5)


def test_finite_torsion_pod_modes_project_exemplar(test, device):
    length = 1.0
    hy = 0.085
    hz = 0.05
    tip_twist = math.pi / 2.0
    mode_count = 8
    vertices, indices = box_surface_mesh(length, hy, hz)
    sample_points = beam_render_sample_points(
        length,
        hy,
        hz,
        extra_points=((-0.5 * length, 0.0, 0.0), (0.5 * length, 0.0, 0.0)),
    )
    snapshot_amplitudes = np.linspace(-1.0, 1.0, 17, dtype=np.float32)
    snapshot_amplitudes = snapshot_amplitudes[np.abs(snapshot_amplitudes) > 1.0e-6]
    snapshots = np.asarray(
        [
            finite_torsion_displacement(sample_points, length, tip_twist * float(amplitude))
            for amplitude in snapshot_amplitudes
        ],
        dtype=np.float32,
    )

    basis = newton.ModalGeneratorPOD(
        sample_points=sample_points,
        displacements=snapshots,
        mode_count=mode_count,
        total_mass=1.0,
        stiffness_scale=1.0,
    ).build()
    phi = basis.sample_phi.reshape((sample_points.shape[0], mode_count, 3))
    projection_matrix = np.transpose(phi, (0, 2, 1)).reshape((-1, mode_count))
    target = finite_torsion_displacement(sample_points, length, tip_twist)
    q, *_ = np.linalg.lstsq(projection_matrix, target.reshape(-1), rcond=None)
    projected = np.einsum("smc,m->sc", phi, q)

    test.assertEqual(basis.mode_count, mode_count)
    test.assertLess(float(np.max(np.linalg.norm(projected - target, axis=1))), 1.0e-5)
    ratio = mesh_volume(vertices + projected[: vertices.shape[0]], indices) / mesh_volume(vertices, indices)
    np.testing.assert_allclose(ratio, 1.0, atol=2.0e-3)


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


def test_cantilever_modal_vibration_solution(test, device):
    length = 0.9
    basis = newton.ModalGeneratorBeam(
        length=length,
        half_width_y=0.045,
        half_width_z=0.035,
        mode_specs=[
            {
                "type": newton.ModalGeneratorBeam.Mode.BENDING_Z,
                "boundary": newton.ModalGeneratorBeam.Boundary.CANTILEVER_TIP,
            }
        ],
        density=250.0,
        young_modulus=3.2e7,
        damping_ratio=0.001,
    ).build()

    mode_mass = float(basis.mode_mass[0])
    stiffness = float(basis.mode_stiffness[0])
    damping = float(basis.mode_damping[0])
    q_expected = 0.09
    qd_expected = 0.0
    dt = 1.0 / 480.0
    min_q = q_expected

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_q=[q_expected],
        mode_qd=[qd_expected],
        modal_basis=basis,
    )
    builder.add_shape_box(body, hx=0.5 * length, hy=0.045, hz=0.035)
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    owner_joint = int(model.elastic_joint.numpy()[0])
    q_start = int(model.joint_q_start.numpy()[owner_joint])
    qd_start = int(model.joint_qd_start.numpy()[owner_joint])

    solver = newton.solvers.SolverVBD(model, iterations=0)
    for _ in range(180):
        solver.step(state_0, state_1, control, None, dt)
        denom = mode_mass + dt * damping + dt * dt * stiffness
        qd_expected = (mode_mass * qd_expected - dt * stiffness * q_expected) / denom
        q_expected = q_expected + dt * qd_expected
        min_q = min(min_q, q_expected)
        state_0, state_1 = state_1, state_0

    np.testing.assert_allclose(state_0.joint_q.numpy()[q_start + 7], q_expected, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(state_0.joint_qd.numpy()[qd_start + 6], qd_expected, rtol=1.0e-6, atol=1.0e-5)
    test.assertLess(min_q, -0.01)


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


def test_vbd_prismatic_rotates_elastic_child(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    inertia = wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02)
    parent = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
    )
    child = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(0.4, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
        mode_count=1,
    )
    parent_joint = builder.add_joint_revolute(
        parent=-1,
        child=parent,
        axis=(0.0, 0.0, 1.0),
    )
    builder.add_joint_prismatic(
        parent=parent,
        child=child,
        axis=(1.0, 0.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.4, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        target_pos=0.0,
    )
    builder.color()
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    parent_qd_start = int(model.joint_qd_start.numpy()[parent_joint])
    joint_f = control.joint_f.numpy()
    joint_f[parent_qd_start] = 1.0
    control.joint_f.assign(joint_f)

    solver = newton.solvers.SolverVBD(
        model,
        iterations=20,
        rigid_joint_linear_k_start=1.0e5,
        rigid_joint_angular_k_start=1.0e5,
        rigid_joint_linear_ke=1.0e6,
        rigid_joint_angular_ke=1.0e6,
    )
    for _ in range(60):
        solver.step(state_0, state_1, control, None, 1.0 / 180.0)
        state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    parent_angle = 2.0 * math.atan2(float(body_q[parent, 5]), float(body_q[parent, 6]))
    child_angle = 2.0 * math.atan2(float(body_q[child, 5]), float(body_q[child, 6]))
    test.assertGreater(abs(parent_angle), 0.1)
    test.assertLess(_quat_angle_error(body_q[parent, 3:7], body_q[child, 3:7]), 1.0e-4)
    np.testing.assert_allclose(child_angle, parent_angle, atol=1.0e-4)


def test_crank_slider_elastic_analytic_geometry(test, device):
    crank_length = 0.24
    rod_rest_length = 0.64
    axial_q = 0.05
    theta = 0.72
    rod_length = rod_rest_length + axial_q
    slider_x = crank_length * math.cos(theta) + math.sqrt(
        rod_length * rod_length - (crank_length * math.sin(theta)) ** 2
    )

    crank_pin = np.array([crank_length * math.cos(theta), crank_length * math.sin(theta), 0.0])
    slider_pin = np.array([slider_x, 0.0, 0.0])
    rod_theta = math.atan2(float(slider_pin[1] - crank_pin[1]), float(slider_pin[0] - crank_pin[0]))

    basis = newton.ModalGeneratorBeam(
        length=rod_rest_length,
        half_width_y=0.03,
        half_width_z=0.02,
        mode_specs=[{"type": newton.ModalGeneratorBeam.Mode.AXIAL}],
    ).build(sample_points=[[-0.5 * rod_rest_length, 0.0, 0.0], [0.5 * rod_rest_length, 0.0, 0.0]])

    builder = newton.ModelBuilder(gravity=0.0)
    crank = builder.add_body(
        xform=wp.transform(
            wp.vec3(*(0.5 * crank_pin)),
            _quat_from_angle_z(theta),
        ),
        mass=1.0,
        inertia=_identity_inertia(),
    )
    rod = builder.add_body_elastic(
        xform=wp.transform(wp.vec3(*(0.5 * (crank_pin + slider_pin))), _quat_from_angle_z(rod_theta)),
        mass=1.0,
        inertia=_identity_inertia(),
        mode_q=[axial_q],
        modal_basis=basis,
    )
    slider = builder.add_body(
        xform=wp.transform(wp.vec3(*slider_pin), wp.quat_identity()),
        mass=1.0,
        inertia=_identity_inertia(),
    )
    j_crank_rod = builder.add_joint_revolute(
        parent=crank,
        child=rod,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.5 * crank_length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5 * rod_rest_length, 0.0, 0.0), wp.quat_identity()),
    )
    j_rod_slider = builder.add_joint_revolute(
        parent=rod,
        child=slider,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.5 * rod_rest_length, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    p_crank = joint_endpoint_world(model, state, j_crank_rod, "parent")
    p_rod_left = joint_endpoint_world(model, state, j_crank_rod, "child")
    p_rod_right = joint_endpoint_world(model, state, j_rod_slider, "parent")
    p_slider = joint_endpoint_world(model, state, j_rod_slider, "child")

    np.testing.assert_allclose(p_crank, crank_pin, atol=1.0e-6)
    np.testing.assert_allclose(p_rod_left, crank_pin, atol=1.0e-6)
    np.testing.assert_allclose(p_rod_right, slider_pin, atol=1.0e-6)
    np.testing.assert_allclose(p_slider, slider_pin, atol=1.0e-6)
    np.testing.assert_allclose(np.linalg.norm(p_rod_right - p_rod_left), rod_length, atol=1.0e-6)


def test_elastic_multiple_attachment_samples(test, device):
    basis = newton.ModalBasis(
        sample_points=[[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        sample_phi=[
            [[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.25, 0.0]],
            [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        mode_mass=[1.0, 1.0],
    )

    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body_elastic(
        mass=1.0,
        inertia=_identity_inertia(),
        mode_q=[0.2, -0.4],
        modal_basis=basis,
    )
    j_left = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(-0.6, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
    )
    j_mid = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.1, 0.0), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    j_right = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.6, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    model = builder.finalize(device=device)
    state = model.state()

    endpoint_samples = model.elastic_endpoint_sample.numpy()
    child_endpoints = model.joint_child_elastic_endpoint.numpy()
    sample_ids = [int(endpoint_samples[int(child_endpoints[j])]) for j in (j_left, j_mid, j_right)]
    test.assertEqual(sample_ids, [0, 1, 2])

    np.testing.assert_allclose(joint_endpoint_world(model, state, j_left, "child"), [-0.6, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(joint_endpoint_world(model, state, j_mid, "child"), [0.0, 0.1, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(joint_endpoint_world(model, state, j_right, "child"), [0.6, 0.0, 0.0], atol=1.0e-6)


class TestReducedElasticBody(unittest.TestCase):
    pass


for device in devices:
    add_function_test(
        TestReducedElasticBody, "test_modal_basis_add_sample", test_modal_basis_add_sample, devices=[device]
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_beam_samples",
        test_modal_generator_beam_samples,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_generator_pod_rank_one",
        test_modal_generator_pod_rank_one,
        devices=[device],
    )
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
        "test_elastic_endpoint_modal_basis_sampling",
        test_elastic_endpoint_modal_basis_sampling,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_modal_basis_shared_by_elastic_bodies",
        test_modal_basis_shared_by_elastic_bodies,
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
        "test_elastic_shape_box_winding",
        test_elastic_shape_box_winding,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_shape_box_exact_modal_samples",
        test_elastic_shape_box_exact_modal_samples,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_strain_visualization_colors",
        test_elastic_strain_visualization_colors,
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
        "test_finite_torsion_exemplar_preserves_volume",
        test_finite_torsion_exemplar_preserves_volume,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_finite_torsion_pod_modes_project_exemplar",
        test_finite_torsion_pod_modes_project_exemplar,
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
        "test_cantilever_modal_vibration_solution",
        test_cantilever_modal_vibration_solution,
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
    add_function_test(
        TestReducedElasticBody,
        "test_vbd_prismatic_rotates_elastic_child",
        test_vbd_prismatic_rotates_elastic_child,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_crank_slider_elastic_analytic_geometry",
        test_crank_slider_elastic_analytic_geometry,
        devices=[device],
    )
    add_function_test(
        TestReducedElasticBody,
        "test_elastic_multiple_attachment_samples",
        test_elastic_multiple_attachment_samples,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
