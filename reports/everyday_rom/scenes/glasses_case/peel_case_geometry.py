# SPDX-License-Identifier: Apache-2.0
"""Procedural geometry and auditable trajectory export for the peel/case demos."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp
from contact_capacity import GLOBAL_CAPACITY, ContactTrace

import newton
import newton.solvers


def grid_faces(rows, cols):
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a = i * cols + j
            faces.extend(((a, a + cols, a + cols + 1), (a, a + cols + 1, a + 1)))
    return np.asarray(faces, dtype=np.int32)


def thick_patch(mid, thickness):
    """Close a rectangular surface grid with normal-offset skins and a rim."""
    rows, cols, _ = mid.shape
    normal = np.cross(np.gradient(mid, axis=0), np.gradient(mid, axis=1))
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    points = np.concatenate(
        ((mid + thickness / 2 * normal).reshape(-1, 3), (mid - thickness / 2 * normal).reshape(-1, 3))
    )
    n = rows * cols
    top = grid_faces(rows, cols)
    faces = list(top) + list(top[:, ::-1] + n)
    boundary = list(range(0, n, cols))
    boundary += list(range(n - cols + 1, n))
    boundary += list(range(n - cols - 1, cols - 2, -cols))
    boundary += list(range(cols - 2, 0, -1))
    for a, b in zip(boundary, boundary[1:] + boundary[:1], strict=True):
        faces.extend(((a, a + n, b + n), (a, b + n, b)))
    return points.astype(np.float32), np.asarray(faces, dtype=np.int32)


def join_meshes(meshes):
    points, faces, count = [], [], 0
    for p, f in meshes:
        points.append(p)
        faces.append(f + count)
        count += len(p)
    return np.concatenate(points).astype(np.float32), np.concatenate(faces).astype(np.int32)


def tube(path, radius, sides=8, closed=False):
    path = np.asarray(path, dtype=float)
    tangent = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0) if closed else np.gradient(path, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    helper = np.tile([0.0, 0.0, 1.0], (len(path), 1))
    helper[np.abs(tangent[:, 2]) > 0.9] = [0.0, 1.0, 0.0]
    u = np.cross(tangent, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(tangent, u)
    theta = np.arange(sides) * 2 * np.pi / sides
    rad = np.broadcast_to(np.asarray(radius), (len(path),))
    points = path[:, None] + rad[:, None, None] * (
        np.cos(theta)[None, :, None] * u[:, None] + np.sin(theta)[None, :, None] * v[:, None]
    )
    faces = []
    for i in range(len(path) if closed else len(path) - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            c, d = ((i + 1) % len(path)) * sides + j, ((i + 1) % len(path)) * sides + (j + 1) % sides
            faces.extend(((a, b, d), (a, d, c)))
    points = points.reshape(-1, 3)
    if not closed:
        n = len(points)
        points = np.concatenate((points, path[[0, -1]]))
        for j in range(sides):
            faces.extend(
                (
                    (n, (j + 1) % sides, j),
                    (n + 1, (len(path) - 1) * sides + j, (len(path) - 1) * sides + (j + 1) % sides),
                )
            )
    return points.astype(np.float32), np.asarray(faces, dtype=np.int32)


def ellipsoid(radii, center=(0, 0, 0), rings=14, sides=32):
    theta = np.arange(sides) * 2 * np.pi / sides
    latitude = np.linspace(0, np.pi, rings + 2)[1:-1]
    points = np.array(
        [[np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)] for a in latitude for b in theta]
        + [[0, 0, 1], [0, 0, -1]]
    )
    faces = []
    for i in range(rings - 1):
        for j in range(sides):
            a, b = i * sides + j, i * sides + (j + 1) % sides
            faces.extend(((a, a + sides, b + sides), (a, b + sides, b)))
    for j in range(sides):
        faces.extend(
            (
                (rings * sides, j, (j + 1) % sides),
                (rings * sides + 1, (rings - 1) * sides + (j + 1) % sides, (rings - 1) * sides + j),
            )
        )
    return (points * radii + center).astype(np.float32), np.asarray(faces, dtype=np.int32)


def box(half, center=(0, 0, 0)):
    p = np.asarray(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        dtype=float,
    )
    f = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return (p * half + center).astype(np.float32), f


def transform(points, pose):
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)
    return np.asarray(points) @ rotation.T + pose[:3]


def surface_strain_metrics(rest, recovered, faces):
    """Measure triangle edges, areas and principal stretches in a local frame.

    Normal reversal is a bounded-deformation check, not a general large-fold
    inversion predicate. Both skins and all boundary triangles are included.
    """
    initial = np.asarray(rest, dtype=np.float64)[faces]
    current = np.asarray(recovered, dtype=np.float64)[faces]
    initial_edges = initial[:, [1, 2, 0]] - initial
    current_edges = current[:, [1, 2, 0]] - current
    edge_ratio = np.linalg.norm(current_edges, axis=-1) / np.linalg.norm(initial_edges, axis=-1)
    initial_normal = np.cross(initial_edges[:, 0], -initial_edges[:, 2])
    current_normal = np.cross(current_edges[:, 0], -current_edges[:, 2])
    area_ratio = np.linalg.norm(current_normal, axis=-1) / np.linalg.norm(initial_normal, axis=-1)
    # Express the rest triangle in its orthonormal tangent frame; singular
    # values of the resulting 3x2 deformation gradient are principal stretches.
    edge_length = np.linalg.norm(initial_edges[:, 0], axis=-1)
    projection = np.einsum("tc,tc->t", -initial_edges[:, 2], initial_edges[:, 0]) / edge_length
    height = np.linalg.norm(initial_normal, axis=-1) / edge_length
    tangent0 = current_edges[:, 0] / edge_length[:, None]
    tangent1 = (-current_edges[:, 2] - tangent0 * projection[:, None]) / height[:, None]
    stretches = np.linalg.svd(np.stack((tangent0, tangent1), axis=-1), compute_uv=False)
    return {
        "max_abs_edge_strain": float(np.abs(edge_ratio - 1).max()),
        "max_abs_area_strain": float(np.abs(area_ratio - 1).max()),
        "max_abs_principal_strain": float(np.abs(stretches - 1).max()),
        "max_principal_compression": max(0.0, 1.0 - float(stretches.min())),
        "max_principal_extension": max(0.0, float(stretches.max()) - 1.0),
        "reversed_surface_triangles": int(np.count_nonzero(np.einsum("tc,tc->t", initial_normal, current_normal) <= 0)),
    }


@wp.kernel
def closest_skin_points(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    faces: wp.array[int],
    closest: wp.array[wp.vec3],
):
    i = wp.tid()
    query = wp.mesh_query_point(mesh, points[i], 2.0)
    faces[i] = -1
    if query.result:
        faces[i] = query.face
        closest[i] = wp.mesh_eval_position(mesh, query.face, query.u, query.v)


def skin_embedded_modes(points, skin_points, skin_faces, skin_phi):
    """Embed offset vertices with fixed barycentric modes and director rotation.

    Tangent displacement gradients determine the infinitesimal skin rotation.
    Normal offsets rotate with that frame, without an invented normal strain.
    This computes a fixed linear basis, never a per-frame geometry correction.
    """
    mesh = wp.Mesh(
        points=wp.array(skin_points, dtype=wp.vec3, device="cpu"),
        indices=wp.array(skin_faces.ravel(), dtype=int, device="cpu"),
    )
    queries = wp.array(points, dtype=wp.vec3, device="cpu")
    face_ids = wp.empty(len(points), dtype=int, device="cpu")
    closest = wp.empty(len(points), dtype=wp.vec3, device="cpu")
    wp.launch(closest_skin_points, len(points), inputs=[mesh.id, queries, face_ids, closest], device="cpu")
    indices = face_ids.numpy()
    assert np.all(indices >= 0)
    triangles = np.asarray(skin_points, dtype=np.float64)[skin_faces[indices]]
    values = np.asarray(skin_phi, dtype=np.float64)[skin_faces[indices]]
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    tangent_inverse = np.linalg.pinv(np.stack((e1, e2), axis=-1))
    uv = np.einsum("qic,qc->qi", tangent_inverse, closest.numpy() - triangles[:, 0])
    weights = np.clip(np.column_stack((1 - uv.sum(axis=1), uv)), 0, 1)
    weights /= weights.sum(axis=1, keepdims=True)
    offset = points - np.einsum("qt,qtc->qc", weights, triangles)
    delta = np.stack((values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]), axis=-1)
    gradient = np.einsum("qmci,qij->qmcj", delta, tangent_inverse)
    normal = np.cross(e1, e2)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    director_rotation = -np.einsum("qmij,qi->qmj", gradient, normal)
    gradient += director_rotation[..., :, None] * normal[:, None, None, :]
    rotation = 0.5 * np.stack(
        (
            gradient[:, :, 2, 1] - gradient[:, :, 1, 2],
            gradient[:, :, 0, 2] - gradient[:, :, 2, 0],
            gradient[:, :, 1, 0] - gradient[:, :, 0, 1],
        ),
        axis=-1,
    )
    embedded = np.einsum("qt,qtmc->qmc", weights, values) + np.cross(rotation, offset[:, None, :])
    return embedded.astype(np.float32)


def pose(position=(0, 0, 0), axis=(1, 0, 0), angle=0.0):
    return wp.transform(wp.vec3(*position), wp.quat_from_axis_angle(wp.vec3(*axis), float(angle)))


def make_basis(points, phi, mass, stiffness, damping_ratio=0.3, label="Ritz shell"):
    """Whiten the full lumped mass and diagonalize the transformed Ritz energy."""
    phi = np.asarray(phi, dtype=np.float64)
    weights = np.full(len(points), mass / len(points), dtype=np.float32)
    raw_mass = np.einsum("n,nic,njc->ij", weights, phi, phi)
    raw_stiffness = np.asarray(stiffness, dtype=np.float64)
    if raw_stiffness.ndim == 1:
        raw_stiffness = np.diag(raw_stiffness)
    eigenvalues, vectors = np.linalg.eigh(raw_mass)
    if eigenvalues.min() <= eigenvalues.max() * 1e-10:
        raise ValueError("Linearly dependent trial modes")
    whitening = vectors / np.sqrt(eigenvalues)[None, :]
    eigen_stiffness, elastic_vectors = np.linalg.eigh(whitening.T @ raw_stiffness @ whitening)
    coordinate_transform = whitening @ elastic_vectors
    normalized = np.einsum("nic,ij->njc", phi, coordinate_transform)
    basis = newton.ModalBasis(
        sample_points=points,
        sample_phi=normalized,
        sample_mass=weights,
        mode_stiffness=eigen_stiffness,
        mode_damping=2 * damping_ratio * np.sqrt(eigen_stiffness),
        label=label,
    )
    basis.raw_sample_phi = phi
    basis.raw_mass_matrix = raw_mass
    basis.raw_stiffness_matrix = raw_stiffness
    basis.coordinate_transform = coordinate_transform
    gram = np.einsum("n,nic,njc->ij", weights, basis.sample_phi, basis.sample_phi, dtype=np.float64)
    np.testing.assert_allclose(gram, np.eye(basis.mode_count), atol=2e-6)
    np.testing.assert_allclose(
        coordinate_transform.T @ raw_stiffness @ coordinate_transform,
        np.diag(basis.mode_stiffness),
        rtol=2e-6,
        atol=1e-6,
    )
    return basis


def solid_angle_inside(points, surface_points, faces):
    """Generalized winding-number containment for a closed oriented triangle mesh."""
    triangles = np.asarray(surface_points, dtype=np.float64)[faces]
    inside = []
    for offset in range(0, len(points), 48):
        delta = triangles[None] - np.asarray(points[offset : offset + 48], dtype=np.float64)[:, None, None]
        a, b, c = delta[:, :, 0], delta[:, :, 1], delta[:, :, 2]
        la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
        numerator = np.einsum("ntc,ntc->nt", a, np.cross(b, c))
        denominator = (
            la * lb * lc
            + np.einsum("ntc,ntc->nt", a, b) * lc
            + np.einsum("ntc,ntc->nt", b, c) * la
            + np.einsum("ntc,ntc->nt", c, a) * lb
        )
        winding = np.sum(2 * np.arctan2(numerator, denominator), axis=1) / (4 * np.pi)
        inside.extend(np.abs(winding) > 0.5)
    return np.asarray(inside, dtype=bool)


def initial_case_clearance(hinge_z=0.068):
    """Global vertical gap proves disjoint surfaces before any simulation motion."""
    shell, shell_faces = case_shell()[0]
    flap, _ = case_flap()[0]
    shell = shell + np.array([0, 0, 0.047])
    flap = flap + np.array([0, 0.030, hinge_z])
    vertical_gap = float(flap[:, 2].min() - shell[:, 2].max())
    inside = solid_angle_inside(flap, shell, shell_faces)
    return {
        "initial_flap_vertices_inside_shell": int(inside.sum()),
        "initial_vertical_clearance_m": vertical_gap,
        "initial_clearance_passed": not inside.any() and vertical_gap > 0.001,
    }


@dataclass
class Config:
    substeps: int = 4
    iterations: int = 16
    contact_enabled: bool = True
    drive_enabled: bool = True
    friction: float = 0.45
    contact_ke: float = 8000.0
    contact_kd: float = 0.03
    contact_gap: float = 0.0003
    joint_angular_ke: float = 10.0
    elastic_relaxation: float = 0.7


def shape_config(cfg, collision=True):
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=cfg.contact_ke,
        kd=cfg.contact_kd,
        mu=cfg.friction,
        margin=0.0,
        gap=cfg.contact_gap,
        has_shape_collision=collision and cfg.contact_enabled,
        has_particle_collision=False,
    )


@wp.func
def ramp(t: float, start: float, duration: float):
    u = wp.clamp((t - start) / duration, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@wp.func
def write_kinematic(
    body: int,
    current: wp.transform,
    nxt: wp.transform,
    dt: float,
    p0: wp.array[wp.transform],
    v0: wp.array[wp.spatial_vector],
    p1: wp.array[wp.transform],
    v1: wp.array[wp.spatial_vector],
):
    velocity = (wp.transform_get_translation(nxt) - wp.transform_get_translation(current)) / dt
    dq = wp.transform_get_rotation(nxt) * wp.quat_inverse(wp.transform_get_rotation(current))
    axis, angle = wp.quat_to_axis_angle(dq)
    vel = wp.spatial_vector(velocity, axis * (angle / dt))
    p0[body] = current
    p1[body] = current
    v0[body] = vel
    v1[body] = vel


@wp.kernel
def tick(clock: wp.array[float], dt: float):
    clock[0] += dt


@wp.kernel
def sync_frames(
    bodies: wp.array[int],
    starts: wp.array[int],
    speeds: wp.array[int],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    q0: wp.array[float],
    q1: wp.array[float],
    qd0: wp.array[float],
    qd1: wp.array[float],
):
    i = wp.tid()
    b, s, v = bodies[i], starts[i], speeds[i]
    x, velocity = poses[b], velocities[b]
    # Only the seven rigid-frame coordinates are boundary conditions, never q_modal.
    for k in range(7):
        q0[s + k] = x[k]
        q1[s + k] = x[k]
    for k in range(6):
        qd0[v + k] = velocity[k]
        qd1[v + k] = velocity[k]


class Simulation:
    """Shared graph stepping and exact sampled-surface export, without rendering."""

    def make_solver(self, model, **options):
        return newton.solvers.SolverVBD(model, **options)

    def begin(self, viewer, cfg):
        self.viewer, self.cfg = viewer, cfg
        self.builder = newton.ModelBuilder(gravity=(0, 0, -9.81), up_axis="Z")
        self.builder.num_rigid_contacts_per_world = 16384
        self.parts, self.bases = [], {}
        self.records, self.saved_poses, self.saved_q, self.saved_qd, self.saved_v = [], [], [], [], []
        self.timings, self.sim_time = [], 0.0
        self.dt = 1 / (60 * cfg.substeps)

    def elastic(self, points, basis, mass, position, name, kinematic=True):
        weights = basis.sample_mass.astype(np.float64)
        samples = basis.sample_points.astype(np.float64)
        np.testing.assert_allclose(weights.sum(), mass, rtol=1e-6)
        com = np.einsum("n,nc->c", weights, samples) / mass
        relative = samples - com
        inertia = np.eye(3) * np.einsum("n,nc,nc->", weights, relative, relative)
        inertia -= np.einsum("n,ni,nj->ij", weights, relative, relative)
        body = self.builder.add_body_elastic(
            xform=pose(position),
            mass=mass,
            com=wp.vec3(*com),
            inertia=wp.mat33(*inertia.ravel()),
            modal_basis=basis,
            is_kinematic=kinematic,
            lock_inertia=True,
            label=name,
        )
        self.bases[body] = basis
        return body

    def mesh(
        self,
        name,
        points,
        faces,
        body=-1,
        color=(0.5, 0.5, 0.5),
        collision=False,
        roughness=0.65,
        metallic=0.0,
        opacity=1.0,
    ):
        shape = self.builder.add_shape_mesh(
            body,
            mesh=newton.Mesh(points, faces.ravel()),
            cfg=shape_config(self.cfg, collision),
            color=color,
            opacity=opacity,
            label=name,
        )
        part = {
            "name": name,
            "rest": np.asarray(points, dtype=np.float32),
            "faces": np.asarray(faces, dtype=np.int32),
            "body": body,
            "shape": shape,
            "color": list(color),
            "roughness": roughness,
            "metallic": metallic,
            "opacity": opacity,
            "world": [],
        }
        self.parts.append(part)
        return len(self.parts) - 1

    def finish(self):
        self.builder.color()
        self.model = self.builder.finalize()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.pipeline = newton.CollisionPipeline(self.model, reduce_contacts=False, rigid_contact_max=GLOBAL_CAPACITY)
        self.control, self.contacts = self.model.control(), self.pipeline.contacts()
        self.solver = self.make_solver(
            self.model,
            iterations=self.cfg.iterations,
            rigid_body_contact_buffer_size=4096,
            rigid_contact_k_start=self.cfg.contact_ke,
            rigid_compliant_alm=False,
            rigid_contact_hard=False,
            rigid_joint_linear_ke=40000.0,
            rigid_joint_linear_k_start=40000.0,
            rigid_joint_angular_ke=self.cfg.joint_angular_ke,
            rigid_joint_angular_k_start=self.cfg.joint_angular_ke,
            rigid_joint_linear_kd=0.03,
            rigid_joint_angular_kd=0.0002,
            rigid_joint_adaptive_stiffness=False,
            elastic_body_relaxation=self.cfg.elastic_relaxation,
            elastic_solver_metrics=True,
            friction_epsilon=1e-4,
        )
        self.contact_trace = ContactTrace(self.model, self.contacts, self.solver)
        self.executed_frames = 0
        self.clock = wp.zeros(1, dtype=float)
        kinematic_joints = [
            j
            for j, (b, kind) in enumerate(zip(self.builder.joint_child, self.builder.joint_type, strict=True))
            if b >= 0
            and self.builder.body_flags[b] & int(newton.BodyFlags.KINEMATIC)
            and kind in (newton.JointType.FREE, newton.JointType.ELASTIC)
        ]
        self.kin_bodies = wp.array([self.builder.joint_child[j] for j in kinematic_joints], dtype=int)
        self.kin_starts = wp.array([self.builder.joint_q_start[j] for j in kinematic_joints], dtype=int)
        self.kin_speeds = wp.array([self.builder.joint_qd_start[j] for j in kinematic_joints], dtype=int)
        shape_ids = self.model.elastic_shape_shape.numpy()
        starts = self.model.elastic_shape_vertex_start.numpy()
        counts = self.model.elastic_shape_vertex_count.numpy()
        all_phi = self.model.elastic_shape_vertex_phi.numpy().reshape(-1, self.model.elastic_max_mode_count, 3)
        locals_ = self.model.elastic_shape_vertex_local.numpy()
        self.contact_phi, self.contact_rest = all_phi, locals_
        self.shape_bodies = self.model.shape_body.numpy()
        self.q_starts = {}
        for body in self.bases:
            elastic = int(self.model.body_elastic_index.numpy()[body])
            owner = int(self.model.elastic_joint.numpy()[elastic])
            self.q_starts[body] = int(self.model.joint_q_start.numpy()[owner]) + 7
        for p in self.parts:
            if p["body"] in self.bases:
                i = int(np.flatnonzero(shape_ids == p["shape"])[0])
                sl = slice(int(starts[i]), int(starts[i] + counts[i]))
                np.testing.assert_allclose(locals_[sl], p["rest"], atol=1e-7)
                p["phi"] = all_phi[sl, : self.bases[p["body"]].mode_count].copy()
                p["q_start"] = self.q_starts[p["body"]]
            else:
                p["phi"], p["q_start"] = np.zeros((len(p["rest"]), 0, 3), np.float32), -1
        self.graph = None
        start = time.perf_counter()
        if self.model.device.is_cuda:
            if self.cfg.substeps % 2:
                raise ValueError("Graph stepping requires an even substep count")
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        self.capture_s = time.perf_counter() - start
        if self.viewer is not None:
            self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.cfg.substeps):
            self.drive()
            wp.launch(
                sync_frames,
                dim=len(self.kin_bodies),
                inputs=[
                    self.kin_bodies,
                    self.kin_starts,
                    self.kin_speeds,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.joint_q,
                    self.state_1.joint_q,
                    self.state_0.joint_qd,
                    self.state_1.joint_qd,
                ],
            )
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.contact_trace.collision(self.contacts, self.clock)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.contact_trace.solved(self.solver)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(tick, dim=1, inputs=[self.clock, self.dt])

    def step(self):
        start = time.perf_counter()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        wp.synchronize()
        self.timings.append(time.perf_counter() - start)
        self.executed_frames += 1
        self.contact_trace.check(self.executed_frames * self.cfg.substeps, self.dt)
        self.sim_time += 1 / 60
        poses, q = self.state_0.body_q.numpy(), self.state_0.joint_q.numpy()
        assert all(
            np.isfinite(a).all() for a in (poses, q, self.state_0.joint_qd.numpy(), self.state_0.body_qd.numpy())
        ), "Nonfinite simulation state"
        self.saved_poses.append(poses.copy())
        self.saved_q.append(q.copy())
        self.saved_qd.append(self.state_0.joint_qd.numpy())
        self.saved_v.append(self.state_0.body_qd.numpy())
        maximum, by_body = 0.0, {}
        for p in self.parts:
            amplitudes = q[p["q_start"] : p["q_start"] + p["phi"].shape[1]]
            displacement = np.einsum("nmc,m->nc", p["phi"], amplitudes)
            local = p["rest"] + displacement
            world = transform(local, poses[p["body"]]) if p["body"] >= 0 else local
            p["world"].append(world.astype(np.float32))
            d = float(np.linalg.norm(displacement, axis=1).max())
            maximum = max(maximum, d)
            by_body[p["body"]] = max(d, by_body.get(p["body"], 0.0))
        record = {
            "time": self.sim_time,
            "displacement_m": maximum,
            "contacts": int(self.contacts.rigid_contact_count.numpy()[0]),
            "body_displacement_m": {str(k): v for k, v in by_body.items() if k in self.bases},
        }
        record.update(self.measure())
        record["contact_penetration_m"] = self.contact_penetration()
        record["penetration_m"] = max(record["penetration_m"], record["contact_penetration_m"])
        self.records.append(record)

    def contact_penetration(self):
        count = min(int(self.contacts.rigid_contact_count.numpy()[0]), self.contacts.rigid_contact_max)
        if not count:
            return 0.0
        points = []
        for side in (0, 1):
            shapes = getattr(self.contacts, f"rigid_contact_shape{side}").numpy()[:count]
            bodies = self.shape_bodies[shapes]
            samples = getattr(self.contacts, f"rigid_contact_elastic_sample{side}").numpy()[:count]
            local = getattr(self.contacts, f"rigid_contact_point{side}").numpy()[:count].copy()
            for body in np.unique(bodies):
                mask = bodies == body
                elastic = mask & (samples >= 0)
                if np.any(elastic):
                    modes = self.bases[int(body)].mode_count
                    q0 = self.q_starts[int(body)]
                    phi = self.contact_phi[samples[elastic], :modes]
                    local[elastic] = self.contact_rest[samples[elastic]] + np.einsum(
                        "nmc,m->nc", phi, self.saved_q[-1][q0 : q0 + modes]
                    )
                if body >= 0:
                    local[mask] = transform(local[mask], self.saved_poses[-1][body])
            points.append(local)
        normal = self.contacts.rigid_contact_normal.numpy()[:count]
        separation = np.einsum("nc,nc->n", normal, points[1] - points[0])
        for side in (0, 1):
            separation -= getattr(self.contacts, f"rigid_contact_margin{side}").numpy()[:count]
        return max(0.0, -float(separation.min()))

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

    def summary(self):
        basis_audit = {}
        for body, basis in self.bases.items():
            mass = np.einsum("n,nic,njc->ij", basis.sample_mass, basis.sample_phi, basis.sample_phi, dtype=np.float64)
            stiffness = basis.coordinate_transform.T @ basis.raw_stiffness_matrix @ basis.coordinate_transform
            diagonal = np.diag(np.diag(mass))
            basis_audit[str(body)] = {
                "normalized_mass_max_offdiagonal": float(np.abs(mass - diagonal).max()),
                "normalized_mass_identity_error": float(np.abs(mass - np.eye(len(mass))).max()),
                "stiffness_diagonalization_relative_error": float(
                    np.linalg.norm(stiffness - np.diag(basis.mode_stiffness)) / max(np.linalg.norm(stiffness), 1e-30)
                ),
            }
        return {
            "test_passed": False,
            "negative_control_passed": False,
            "config": asdict(self.cfg),
            "max_displacement_m": max(r["displacement_m"] for r in self.records),
            "max_penetration_m": max(r["penetration_m"] for r in self.records),
            "final_displacement_m": self.records[-1]["displacement_m"],
            "max_contacts": max(r["contacts"] for r in self.records),
            "frames": len(self.records),
            "duration_s": self.sim_time,
            "graph_enabled": self.graph is not None,
            "device": str(self.model.device),
            "graph_capture_s": self.capture_s,
            "graph_sim_frame_ms": float(np.mean(self.timings[10:] or self.timings) * 1000),
            "timing_scope": "Synchronized warm CUDA graph replay, excludes readback/export/render; metrics kernels enabled",
            "finite_state": True,
            "basis_audit": basis_audit,
            "records": self.records,
        }

    def export(self, output, metadata, metrics):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        metrics["contact_capacity"] = self.contact_trace.save(output, self.expected_frames * self.cfg.substeps, self.dt)
        arrays = {"times": np.asarray([r["time"] for r in self.records], dtype=np.float64)}
        audit = {}
        meshes = []
        recovery_error = 0.0
        for i, p in enumerate(self.parts):
            world = np.asarray(p["world"], dtype=np.float32)
            arrays[f"vertices_{i}"] = world if p["body"] >= 0 else world[0]
            arrays[f"faces_{i}"] = p["faces"]
            meshes.append(
                {k: p[k] for k in ("name", "color", "roughness", "metallic", "opacity")}
                | {"vertices": f"vertices_{i}", "faces": f"faces_{i}"}
            )
            audit[f"rest_{i}"] = p["rest"]
            audit[f"phi_{i}"] = p["phi"]
            audit[f"body_{i}"] = np.asarray(p["body"], dtype=np.int32)
            audit[f"q_start_{i}"] = np.asarray(p["q_start"], dtype=np.int32)
            for frame in (0, len(world) // 2, len(world) - 1):
                q = self.saved_q[frame]
                local = p["rest"] + np.einsum("nmc,m->nc", p["phi"], q[p["q_start"] : p["q_start"] + p["phi"].shape[1]])
                recovered = transform(local, self.saved_poses[frame][p["body"]]) if p["body"] >= 0 else local
                recovery_error = max(recovery_error, float(np.abs(recovered - world[frame]).max()))
        for body, basis in self.bases.items():
            for name in (
                "sample_points",
                "sample_phi",
                "sample_psi",
                "sample_mass",
                "mode_mass",
                "mode_stiffness",
                "mode_damping",
                "raw_sample_phi",
                "raw_mass_matrix",
                "raw_stiffness_matrix",
                "coordinate_transform",
            ):
                audit[f"body_{body}_{name}"] = getattr(basis, name)
            for name in ("training_snapshots", "training_energy_fraction", "training_reconstruction_max_abs_error_m"):
                if hasattr(basis, name):
                    audit[f"body_{body}_{name}"] = getattr(basis, name)
        for name in ("body_mass", "body_com", "body_inertia"):
            audit[f"model_{name}"] = getattr(self.model, name).numpy()
        np.savez_compressed(output / "scene.npz", **arrays)
        np.savez_compressed(output / "basis.npz", **audit)
        state_arrays = {
            "times": arrays["times"],
            "body_q": self.saved_poses,
            "joint_q": self.saved_q,
            "joint_qd": self.saved_qd,
            "body_qd": self.saved_v,
        }
        if hasattr(self, "saved_attachment_enabled"):
            state_arrays.update(
                attachment_enabled=self.saved_attachment_enabled,
                attachment_joint_ids=self.holding_joints.numpy(),
                prescribed_attachment_release_times=self.release_times.numpy(),
            )
        np.savez_compressed(output / "state.npz", **state_arrays)
        metrics["surface_recovery_max_abs_error_m"] = recovery_error
        metrics["surface_recovery_passed"] = recovery_error < 1e-7
        assert metrics["surface_recovery_passed"]
        metadata.update(
            fps=60,
            meshes=meshes,
            mode_count=sum(b.mode_count for b in self.bases.values()),
            parameters=asdict(self.cfg),
            poster_frame=int(np.argmax([r["displacement_m"] for r in self.records])),
        )
        sources = output / "source"
        sources.mkdir(exist_ok=True)
        metadata["source_sha256"] = {}
        for name in metadata["source_files"]:
            source = Path(__file__).parent / name
            shutil.copyfile(source, sources / name)
            metadata["source_sha256"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
        (output / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def case_shell():
    x = np.linspace(-0.083, 0.083, 35)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 25)
    r = np.sqrt(1 - (x / 0.086) ** 8)
    xx, tt = np.meshgrid(x, theta, indexing="ij")
    mid = np.stack((xx, 0.034 * r[:, None] * np.sin(tt), 0.010 - 0.025 * r[:, None] * np.cos(tt)), axis=-1)
    return thick_patch(mid, 0.0017), mid


def case_flap():
    x = np.linspace(-1, 1, 35)
    s = np.linspace(0, 1, 29)
    xx, ss = np.meshgrid(x, s, indexing="ij")
    width = 0.080 - 0.011 * ss**3
    mid = np.stack(
        (xx * width, -0.068 * ss + 0.013 * ss**4 * xx**2, -0.008 * ss + 0.005 * np.sin(np.pi * ss) * (1 - xx**2)),
        axis=-1,
    )
    return thick_patch(mid, 0.0017), mid


def pouch_shell():
    """Tall oval pouch wall, with a rear/top opening and closed separate end panels."""
    x = np.linspace(-0.073, 0.073, 49)
    theta = np.linspace(0.68, 2 * np.pi - 0.68, 57)
    xx, tt = np.meshgrid(x, theta, indexing="ij")
    grain = 0.000055 * np.sin(3100 * xx + 13 * tt) * np.sin(73 * tt - 1100 * xx)
    mid = np.stack((xx, (0.020 + grain) * np.sin(tt), (0.035 + grain) * np.cos(tt)), axis=-1)
    return thick_patch(mid, 0.0016), mid


def pouch_flap():
    """Broad wrap-over panel ending in a rounded central triangular tongue."""
    xx, ss = np.meshgrid(np.linspace(-1, 1, 49), np.linspace(0, 1, 43), indexing="ij")
    end_angle = -0.55 - 1.40 * np.maximum(0, 1 - xx**2) ** 1.35
    theta = 0.68 + (end_angle - 0.68) * ss
    x = 0.072 * xx
    grain = 0.000055 * np.sin(3100 * x + 13 * theta) * np.sin(73 * theta - 1100 * x)
    mid = np.stack((x, (0.0235 + grain) * np.sin(theta), (0.039 + grain) * np.cos(theta)), axis=-1)
    return thick_patch(mid, 0.0013), mid


def pouch_endcaps():
    return [ellipsoid((0.007, 0.020, 0.035), (sign * 0.074, 0, 0), rings=24, sides=48) for sign in (-1, 1)]


def pouch_clearance():
    shell, faces = pouch_shell()[0]
    flap, _ = pouch_flap()[0]
    # All flap points must lie outside an inflated ellipse containing the entire shell and ends.
    shell_radius = np.sqrt((shell[:, 1] / 0.020) ** 2 + (shell[:, 2] / 0.035) ** 2)
    flap_radius = np.sqrt((flap[:, 1] / 0.020) ** 2 + (flap[:, 2] / 0.035) ** 2)
    radial_gap = float((flap_radius.min() - shell_radius.max()) * 0.020)
    contained = int(solid_angle_inside(flap, shell, faces).sum())
    end_contained = sum(int(solid_angle_inside(flap, p, f).sum()) for p, f in pouch_endcaps())
    return {
        "initial_flap_vertices_inside_shell": contained,
        "initial_flap_vertices_inside_endcaps": end_contained,
        "initial_clearance_lower_bound_m": radial_gap,
        "initial_clearance_passed": contained == 0 and end_contained == 0 and radial_gap > 0.0005,
    }


def glasses_mesh():
    meshes = []
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for cx in (-0.033, 0.033):
        x = cx + 0.027 * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** 0.65
        y = 0.020 * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** 0.75
        meshes.append(tube(np.column_stack((x, y, np.zeros_like(x))), 0.0020, closed=True))
    meshes.append(tube([[-0.008, 0.006, 0], [0, 0.009, 0.001], [0.008, 0.006, 0]], 0.0018))
    for sign in (-1, 1):
        meshes.append(
            tube(
                [
                    [sign * 0.060, 0.006, 0],
                    [sign * 0.065, 0.019, -0.004],
                    [sign * 0.045, 0.024, -0.005],
                    [sign * 0.015, 0.022, -0.005],
                ],
                0.0015,
            )
        )
    return join_meshes(meshes)


def banana_center(s):
    s = np.asarray(s)
    return np.stack((0.030 * np.sin(np.pi * s), np.zeros_like(s), 0.180 + 0.210 * s), axis=-1)


def banana_radius(s):
    return 0.018 * np.sqrt(np.sin(np.pi * (0.065 + 0.87 * np.asarray(s))))


def banana_surface(s, theta, extra=0.0):
    s, theta = np.broadcast_arrays(s, theta)
    r = banana_radius(s) + extra
    ribs = 1 + 0.035 * np.cos(5 * theta)
    return banana_center(s) + np.stack((r * ribs * np.cos(theta), r * ribs * np.sin(theta), np.zeros_like(s)), axis=-1)


def banana_flesh():
    ss = np.linspace(0, 1, 65)
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    p = banana_surface(ss[:, None], theta[None, :], -0.0010).reshape(-1, 3)
    faces = []
    for i in range(len(ss) - 1):
        for j in range(len(theta)):
            a = i * len(theta) + j
            b = i * len(theta) + (j + 1) % len(theta)
            faces.extend(((a, b, b + len(theta)), (a, b + len(theta), a + len(theta))))
    n = len(p)
    p = np.concatenate((p, banana_center([0, 1])))
    for j in range(len(theta)):
        faces.extend(
            (
                (n, (j + 1) % len(theta), j),
                (n + 1, (len(ss) - 1) * len(theta) + j, (len(ss) - 1) * len(theta) + (j + 1) % len(theta)),
            )
        )
    return p.astype(np.float32), np.asarray(faces, dtype=np.int32)


def banana_strip_snapshot(strip, front=1.0, bend=np.pi, fold_width=0.18, rows=65, cols=17):
    """Smooth training geometry only; never assigned to a simulated peel state."""
    s = np.linspace(0, 1, rows)
    angle = 2 * np.pi * strip / 3 - np.pi / 2
    theta = np.linspace(angle - np.pi / 3 + 0.025, angle + np.pi / 3 - 0.025, cols)
    original = banana_surface(s[:, None], theta[None], 0.00035)
    center = banana_surface(s, angle, 0.00035)
    radial = np.array([np.cos(angle), np.sin(angle), 0.0])
    tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
    u = np.clip((s - front) / fold_width, 0, 1)
    flatten = np.clip(u / 0.45, 0, 1)
    flatten = flatten**2 * (3 - 2 * flatten)
    turn = np.clip((u - 0.35) / 0.65, 0, 1)
    beta = bend * turn**2 * (3 - 2 * turn)

    def rotate(vectors, angles):
        return (
            vectors * np.cos(angles)[..., None]
            + np.cross(tangent, vectors) * np.sin(angles)[..., None]
            + tangent * np.sum(vectors * tangent, axis=-1, keepdims=True) * (1 - np.cos(angles)[..., None])
        )

    increment = rotate(np.diff(center, axis=0), (beta[1:] + beta[:-1]) / 2)
    curved_center = np.concatenate((center[:1], center[:1] + np.cumsum(increment, axis=0)))
    # Flatten transverse edge directions before folding, retaining every edge length.
    transverse = np.diff(original, axis=1)
    lengths = np.linalg.norm(transverse, axis=-1)
    directions = np.arctan2(transverse @ radial, transverse @ tangent)
    directions *= 1 - flatten[:, None]
    transverse = lengths[..., None] * (np.cos(directions)[..., None] * tangent + np.sin(directions)[..., None] * radial)
    cross_section = np.concatenate((np.zeros((rows, 1, 3)), np.cumsum(transverse, axis=1)), axis=1)
    cross_section -= cross_section[:, cols // 2 : cols // 2 + 1]
    mid = curved_center[:, None] + rotate(cross_section, beta[:, None])
    points, faces = thick_patch(mid, 0.0016)
    return points, faces, mid, curved_center[[0, -1]]


def banana_whole_strip(strip, modes=24):
    """Whole-strip POD with full projected quadratic membrane/bending energy.

    This is a small-strain linear shell ROM. Finite-rotation training snapshots
    improve its span but do not enforce an inextensible finite-rotation manifold.
    """
    rows, cols = 65, 17
    rest, faces, mid, anchors = banana_strip_snapshot(strip)
    origin = anchors[0]
    attachment_rows = np.repeat([8, 16, 24, 32, 40, 48, 56], 3)
    attachment_cols = np.tile([2, 8, 14], 7)
    sample_rest = np.concatenate((rest, mid[attachment_rows, attachment_cols], anchors)) - origin
    snapshots = []
    for width in (0.12, 0.20):
        for bend in (1.8, 2.5, np.pi):
            for front in np.linspace(0.97, 0.04, 25):
                deformed, _, moved_mid, moved_anchors = banana_strip_snapshot(strip, front, bend, width)
                snapshots.append(
                    np.concatenate((deformed, moved_mid[attachment_rows, attachment_cols], moved_anchors))
                    - origin
                    - sample_rest
                )
    snapshots = np.asarray(snapshots, dtype=np.float64)
    matrix = snapshots.reshape(len(snapshots), -1).T
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    raw_phi = u[:, :modes].reshape(len(sample_rest), 3, modes).transpose(0, 2, 1)
    count = rows * cols
    mid_phi = ((raw_phi[:count] + raw_phi[count : 2 * count]) / 2).reshape(rows, cols, modes, 3)
    ds = np.gradient(mid, axis=0)
    dt = np.gradient(mid, axis=1)
    ls, lt = np.linalg.norm(ds, axis=-1), np.linalg.norm(dt, axis=-1)
    ts, tt = ds / ls[..., None], dt / lt[..., None]
    normal = np.cross(ts, tt)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    area = np.linalg.norm(np.cross(ds, dt), axis=-1)
    us = np.gradient(mid_phi, axis=0) / ls[..., None, None]
    ut = np.gradient(mid_phi, axis=1) / lt[..., None, None]
    es = np.einsum("rcmj,rcj->rcm", us, ts)
    et = np.einsum("rcmj,rcj->rcm", ut, tt)
    shear = np.einsum("rcmj,rcj->rcm", us, tt) + np.einsum("rcmj,rcj->rcm", ut, ts)
    curvature_s = np.einsum("rcmj,rcj->rcm", np.gradient(us, axis=0) / ls[..., None, None], normal)
    curvature_t = np.einsum("rcmj,rcj->rcm", np.gradient(ut, axis=1) / lt[..., None, None], normal)

    def gram(a, b):
        return np.einsum("rc,rcm,rcn->mn", area, a, b)

    young, thickness, poisson = 0.5e6, 0.0016, 0.35
    membrane = young * thickness / (1 - poisson**2)
    bending = young * thickness**3 / (12 * (1 - poisson**2))
    stiffness = membrane * (
        gram(es, es) + gram(et, et) + poisson * (gram(es, et) + gram(et, es)) + 0.5 * (1 - poisson) * gram(shear, shear)
    )
    stiffness += bending * (gram(curvature_s, curvature_s) + gram(curvature_t, curvature_t))
    stiffness += np.eye(modes) * 1e-8
    mass = float(900 * thickness * area.sum())
    basis = make_basis(sample_rest, raw_phi, mass, stiffness, 0.45, label=f"whole_peel_{strip}_POD_linear_shell")
    projection = u[:, :modes] @ (u[:, :modes].T @ matrix)
    basis.training_snapshots = snapshots.astype(np.float32)
    basis.training_energy_fraction = float(np.sum(singular[:modes] ** 2) / np.sum(singular**2))
    basis.training_reconstruction_max_abs_error_m = float(np.abs(projection - matrix).max())
    return (rest - origin).astype(np.float32), faces, basis, origin, anchors - origin, mass, mid - origin


def strip_geometry_metrics(rest, current, rows=65, cols=17):
    """Objective local edge/area changes and through-thickness orientation checks."""
    n = rows * cols
    initial = ((rest[:n] + rest[n : 2 * n]) / 2).reshape(rows, cols, 3)
    deformed = ((current[:n] + current[n : 2 * n]) / 2).reshape(rows, cols, 3)
    strains = []
    for axis in (0, 1):
        a, b = np.diff(initial, axis=axis), np.diff(deformed, axis=axis)
        strains.extend((np.linalg.norm(b, axis=-1) / np.linalg.norm(a, axis=-1) - 1).ravel())
    a = np.cross(initial[1:, :-1] - initial[:-1, :-1], initial[:-1, 1:] - initial[:-1, :-1])
    b = np.cross(deformed[1:, :-1] - deformed[:-1, :-1], deformed[:-1, 1:] - deformed[:-1, :-1])
    director0 = (rest[:n] - rest[n : 2 * n]).reshape(rows, cols, 3)[:-1, :-1]
    director1 = (current[:n] - current[n : 2 * n]).reshape(rows, cols, 3)[:-1, :-1]
    signed = np.einsum("rcj,rcj->rc", b, director1) / np.einsum("rcj,rcj->rc", a, director0)
    area_ratio = np.linalg.norm(b, axis=-1) / np.linalg.norm(a, axis=-1)
    return {
        "max_abs_edge_strain": float(np.abs(strains).max()),
        "max_abs_area_strain": float(np.abs(area_ratio - 1).max()),
        "minimum_signed_volume_ratio": float(signed.min()),
        "inverted_surface_cells": int(np.count_nonzero(signed <= 0)),
    }
