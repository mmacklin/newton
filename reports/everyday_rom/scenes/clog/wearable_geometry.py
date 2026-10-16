# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Procedural wearable meshes and auditable, contact-driven Newton ROM runner.

Only NumPy, Warp, the standard library, and Newton's public API are used.
Material parameters are illustrative, not a fit to manufactured products.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import contact_capacity
import numpy as np
import warp as wp

import newton
import newton.solvers


@dataclass
class Config:
    substeps: int = 4
    iterations: int = 24
    contact_enabled: bool = True
    contact_ke: float = 8.0e4
    contact_kd: float = 0.004
    contact_gap: float = 0.0004
    friction: float = 0.50
    damping_ratio: float = 0.08


def surface_mass(vertices, faces):
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    mass = np.zeros(len(vertices))
    np.add.at(mass, faces.flatten(), np.repeat(area / 3, 3))
    return mass


def triangle_reference(vertices, faces):
    """Return inverse rest-edge coordinates and mean-ratio quality for every face."""
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    length = np.linalg.norm(a, axis=1)
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    if np.any(length <= 0) or np.any(cross <= 0):
        raise ValueError("Degenerate material triangle")
    reference = np.zeros((len(faces), 2, 2))
    reference[:, 0, 0] = length
    reference[:, 0, 1] = np.einsum("fc,fc->f", a, b) / length
    reference[:, 1, 1] = cross / length
    edge_squared = np.sum(a * a + b * b + (a - b) ** 2, axis=1)
    quality = 2 * np.sqrt(3) * cross / edge_squared
    return np.linalg.inv(reference), quality


def triangle_principal_stretches(vertices, faces, inverse_reference):
    """Singular values of each triangle's 3-by-2 material deformation gradient."""
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    edges = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1)
    gradient = edges @ inverse_reference
    return np.linalg.svd(gradient, compute_uv=False)


def thicken(vertices, faces, thickness):
    """Close a manifold midsurface, including its hole walls [m]."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], normal)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-14)
    n = len(vertices)
    result = faces.tolist() + (faces[:, ::-1] + n).tolist()
    edges = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.setdefault(tuple(sorted((int(u), int(v)))), []).append((int(u), int(v)))
    for directions in edges.values():
        if len(directions) > 2:
            raise ValueError("Nonmanifold midsurface")
        if len(directions) == 1:
            a, b = directions[0]
            result.extend(((a, a + n, b), (b, a + n, b + n)))
    return np.concatenate((vertices + normals * thickness / 2, vertices - normals * thickness / 2)), np.asarray(
        result, dtype=np.int32
    )


def grid_mesh(points, periodic=False):
    points = np.asarray(points, dtype=float)
    rows, columns = points.shape[:2]
    faces = []
    for i in range(rows - 1):
        for j in range(columns if periodic else columns - 1):
            k = (j + 1) % columns
            a, b, c, d = i * columns + j, i * columns + k, (i + 1) * columns + k, (i + 1) * columns + j
            faces.extend(((a, b, c), (a, c, d)))
    return points.reshape(-1, 3), np.asarray(faces, dtype=np.int32)


def tube_mesh(path, radius, sides=8, closed=False):
    path = np.asarray(path)
    tangent = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    if not closed:
        tangent[0], tangent[-1] = path[1] - path[0], path[-1] - path[-2]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    reference = np.tile((0.0, 0.0, 1.0), (len(path), 1))
    reference[np.abs(tangent[:, 2]) > 0.9] = (1.0, 0.0, 0.0)
    n = np.cross(tangent, reference)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    b = np.cross(tangent, n)
    angles = np.arange(sides) * 2 * np.pi / sides
    rings = path[:, None] + radius * (
        np.cos(angles)[None, :, None] * n[:, None] + np.sin(angles)[None, :, None] * b[:, None]
    )
    vertices, faces = grid_mesh(rings, periodic=True)
    if closed:
        last = (len(path) - 1) * sides
        closure = []
        for j in range(sides):
            k = (j + 1) % sides
            closure.extend(((last + j, last + k, k), (last + j, k, j)))
        faces = np.concatenate((faces, closure))
    else:
        count = len(vertices)
        vertices = np.concatenate((vertices, path[[0, -1]]))
        caps = []
        for j in range(sides):
            k = (j + 1) % sides
            caps.extend(((count, k, j), (count + 1, (len(path) - 1) * sides + j, (len(path) - 1) * sides + k)))
        faces = np.concatenate((faces, caps))
    return vertices, faces


def ellipsoid_mesh(radii, center=(0, 0, 0), segments=48, rings=24):
    vertices = [(0.0, 0.0, 1.0)]
    for i in range(1, rings):
        a = np.pi * i / rings
        for j in range(segments):
            b = 2 * np.pi * j / segments
            vertices.append((np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)))
    bottom = len(vertices)
    vertices.append((0.0, 0.0, -1.0))
    faces = []
    for j in range(segments):
        k = (j + 1) % segments
        faces.extend(((0, 1 + j, 1 + k), (bottom, bottom - segments + k, bottom - segments + j)))
    for i in range(rings - 2):
        for j in range(segments):
            k = (j + 1) % segments
            a, b = 1 + i * segments + j, 1 + i * segments + k
            faces.extend(((a, a + segments, b + segments), (a, b + segments, b)))
    return np.asarray(vertices) * radii + center, np.asarray(faces, dtype=np.int32)


def capsule_mesh(radius, half_height, segments=48, rings=12):
    """Closed Y-axis capsule with one vertex per pole and explicit equator rings."""
    polar = np.linspace(0, np.pi / 2, rings + 1)[1:]
    radial = radius * np.r_[np.sin(polar), np.sin(polar[::-1])]
    height = np.r_[half_height + radius * np.cos(polar), -half_height - radius * np.cos(polar[::-1])]
    angles = np.arange(segments) * 2 * np.pi / segments
    points = np.stack(
        (
            radial[:, None] * np.cos(angles),
            np.broadcast_to(height[:, None], (len(height), segments)),
            -radial[:, None] * np.sin(angles),
        ),
        axis=-1,
    ).reshape(-1, 3)
    vertices = np.concatenate(([[0, half_height + radius, 0]], points, [[0, -half_height - radius, 0]]))
    bottom = len(vertices) - 1
    faces = []
    for j in range(segments):
        k = (j + 1) % segments
        faces.extend(((0, 1 + j, 1 + k), (bottom, bottom - segments + k, bottom - segments + j)))
    for i in range(len(height) - 1):
        for j in range(segments):
            k = (j + 1) % segments
            a, b = 1 + i * segments + j, 1 + i * segments + k
            faces.extend(((a, a + segments, b + segments), (a, b + segments, b)))
    return vertices, np.asarray(faces, dtype=np.int32)


class Assembly:
    """Colored components sampled into one body and one modal basis."""

    def __init__(self):
        self.vertices, self.faces, self.parts = [], [], []

    def add(self, name, mesh, color, mass, region, roughness=0.65):
        vertices, faces = mesh
        start = sum(len(v) for v in self.vertices)
        self.vertices.append(np.asarray(vertices, dtype=float))
        self.faces.append(np.asarray(faces, dtype=np.int32) + start)
        self.parts.append(
            {
                "name": name,
                "start": start,
                "count": len(vertices),
                "color": color,
                "mass": mass,
                "region": region,
                "roughness": roughness,
            }
        )

    def finish(self):
        vertices, faces = np.concatenate(self.vertices).astype(np.float32), np.concatenate(self.faces)
        mass = np.zeros(len(vertices))
        region = np.zeros(len(vertices), dtype=np.int32)
        for part, local_v, global_f in zip(self.parts, self.vertices, self.faces, strict=True):
            ids = slice(part["start"], part["start"] + part["count"])
            weights = surface_mass(local_v, global_f - part["start"])
            mass[ids] = weights * part["mass"] / weights.sum()
            region[ids] = part["region"]
        return vertices, faces, mass, region


def mass_properties(vertices, sample_mass):
    mass = float(sample_mass.sum())
    center = np.sum(vertices * sample_mass[:, None], axis=0) / mass
    relative = vertices - center
    inertia = np.eye(3) * np.sum(sample_mass * np.sum(relative**2, axis=1))
    inertia -= np.einsum("n,ni,nj->ij", sample_mass, relative, relative)
    return mass, center, inertia


def remove_rigid_modes(vertices, raw, sample_mass):
    """Remove the six mass-weighted rigid fields before floating-frame reduction."""
    _, center, _ = mass_properties(vertices, sample_mass)
    relative = vertices - center
    rigid = np.zeros((len(vertices), 3, 6))
    rigid[:, :, :3] = np.eye(3)
    for axis in range(3):
        rigid[:, :, axis + 3] = np.cross(np.eye(3)[axis], relative)
    rigid = rigid.reshape(-1, 6)
    weights = np.repeat(sample_mass, 3)
    matrix = raw.transpose(0, 2, 1).reshape(-1, raw.shape[1])
    gram = rigid.T @ (weights[:, None] * rigid)
    matrix -= rigid @ np.linalg.solve(gram, rigid.T @ (weights[:, None] * matrix))
    return matrix.reshape(len(vertices), 3, -1).transpose(0, 2, 1)


def orthogonal_basis(vertices, raw, sample_mass, frequency_hz, damping_ratio, floating=False):
    """Mass-weighted QR retains the analytic subspace without modal mass coupling."""
    if floating:
        raw = remove_rigid_modes(vertices, raw.copy(), sample_mass)
    n, m, _ = raw.shape
    matrix = raw.transpose(0, 2, 1).reshape(3 * n, m)
    weights = np.repeat(sample_mass, 3)
    q, r = np.linalg.qr(np.sqrt(weights)[:, None] * matrix, mode="reduced")
    if np.min(np.abs(np.diag(r))) < 1e-9:
        raise ValueError("Dependent analytic modes")
    matrix = q / np.sqrt(weights)[:, None]
    scales = np.linalg.norm(matrix.reshape(n, 3, m), axis=1).max(axis=0)
    matrix /= scales
    phi = matrix.reshape(n, 3, m).transpose(0, 2, 1).astype(np.float32)
    gram = np.einsum("n,nic,njc->ij", sample_mass, phi.astype(float), phi.astype(float))
    mode_mass = np.diag(gram)
    stiffness = mode_mass * (2 * np.pi * np.asarray(frequency_hz)) ** 2
    damping = 2 * damping_ratio * np.sqrt(mode_mass * stiffness)
    relative = gram / np.sqrt(mode_mass[:, None] * mode_mass[None, :])
    basis = newton.ModalBasis(
        sample_points=vertices,
        sample_phi=phi,
        sample_mass=sample_mass,
        mode_mass=mode_mass,
        mode_stiffness=stiffness,
        mode_damping=damping,
        label="area_lumped_mass_QR_analytic_wearable",
    )
    audit = {
        "method": "Analytic displacement fields; local area-lumped mass-weighted QR; no FEM or trained material model",
        "mass_kg": float(sample_mass.sum()),
        "mode_count": m,
        "max_mass_orthogonality_error": float(np.abs(relative - np.eye(m)).max()),
        "frequencies_hz": list(frequency_hz),
        "mode_mass_kg": mode_mass.tolist(),
        "mode_stiffness_N_per_m": stiffness.tolist(),
        "mode_damping_Ns_per_m": damping.tolist(),
        "stiffness_assumption": "Diagonal K_i = M_i (2 pi f_i)^2 with assumed frequencies; not material calibrated",
        "rigid_fields_removed": floating,
        "max_modal_translation_coupling_kg": float(np.abs(np.einsum("n,nmc->mc", sample_mass, phi)).max()),
    }
    return basis, audit


@wp.func
def ramp(t: float, start: float, duration: float):
    s = wp.clamp((t - start) / duration, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@wp.func
def press_position(t: float, start: wp.vec3, stroke: float, schedule: wp.vec4):
    amount = ramp(t, schedule[0], schedule[1]) - ramp(t, schedule[2], schedule[3])
    return start - wp.vec3(0.0, 0.0, stroke * amount)


@wp.kernel
def drive_press(
    clock: wp.array[float],
    dt: float,
    body: int,
    start: wp.vec3,
    stroke: float,
    schedule: wp.vec4,
    pose0: wp.array[wp.transform],
    velocity0: wp.array[wp.spatial_vector],
    pose1: wp.array[wp.transform],
    velocity1: wp.array[wp.spatial_vector],
):
    position = press_position(clock[0], start, stroke, schedule)
    nxt = press_position(clock[0] + dt, start, stroke, schedule)
    pose = wp.transform(position, wp.quat_identity())
    velocity = wp.spatial_vector((nxt - position) / dt, wp.vec3(0.0))
    pose0[body] = pose
    pose1[body] = pose
    velocity0[body] = velocity
    velocity1[body] = velocity


@wp.kernel
def tick(clock: wp.array[float], dt: float):
    clock[0] = clock[0] + dt


def transform_points(vertices, pose):
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)
    return vertices @ rotation.T + pose[:3]


class WearableExample:
    """Shared Example implementation; only the press rigid pose is prescribed."""

    def __init__(self, viewer, cfg, spec):
        self.viewer, self.cfg, self.spec = viewer, cfg, spec
        self.source_snapshot = {
            name: Path(__file__).with_name(name).read_text()
            for name in ("clog.py", "cap.py", "wearable_geometry.py", "test_wearables.py")
        }
        self.assembly = spec["assembly"]
        self.source_snapshot["contact_capacity.py"] = Path(contact_capacity.__file__).read_text()
        self.dynamic = spec.get("dynamic", True)
        self.schedule = spec.get("schedule", [0.85, 0.90, 2.35, 0.055])
        self.vertices, self.faces, self.sample_mass, self.region = self.assembly.finish()
        self.raw = spec["raw_modes"] if "raw_modes" in spec else spec["modes"](self.vertices, self.region)
        self.basis, self.audit = orthogonal_basis(
            self.vertices, self.raw, self.sample_mass, spec["frequencies"], cfg.damping_ratio, floating=self.dynamic
        )
        self.mode_count = self.basis.mode_count
        if "pod_audit" in spec:
            self.audit["pod"] = spec["pod_audit"]
            self.audit["method"] = (
                "Mass-weighted POD of geometric shell poses; rigid-field projection and local mass orthogonalization"
            )
        if "stiffness_audit" in spec:
            self.audit["projected_shell"] = spec["stiffness_audit"]
            self.audit["stiffness_assumption"] = spec["stiffness_audit"]["method"]
        self.sim_time, self.dt = 0.0, 1 / (60 * cfg.substeps)
        self.records, self.saved_body, self.saved_q, self.saved_qd, self.saved_body_qd, self.world = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.timings = []
        self.edges = np.unique(
            np.sort(np.concatenate((self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]])), axis=1),
            axis=0,
        )
        self.rest_edges = np.linalg.norm(self.vertices[self.edges[:, 1]] - self.vertices[self.edges[:, 0]], axis=1)
        assert np.all(self.rest_edges > 0), "Every material edge must be nondegenerate"
        self.inverse_reference, self.rest_quality = triangle_reference(self.vertices, self.faces)
        rest_triangles = self.vertices[self.faces]
        self.rest_normals = np.cross(
            rest_triangles[:, 1] - rest_triangles[:, 0], rest_triangles[:, 2] - rest_triangles[:, 0]
        )
        self.rest_areas = np.linalg.norm(self.rest_normals, axis=1)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis="Z")
        builder.num_rigid_contacts_per_world = 32768
        mass, self.center, inertia = mass_properties(self.vertices, self.sample_mass)
        self.audit["center_of_mass_m"] = self.center.tolist()
        self.audit["inertia_kg_m2"] = inertia.tolist()
        self.body = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0, 0, spec.get("initial_height", 0.055)), wp.quat_identity()),
            mass=mass,
            com=wp.vec3(*self.center),
            inertia=wp.mat33(inertia),
            modal_basis=self.basis,
            is_kinematic=not self.dynamic,
            lock_inertia=True,
            label=f"whole_{spec['slug']}_ROM",
        )

        def shape_config(collision=True):
            return newton.ModelBuilder.ShapeConfig(
                density=0,
                ke=cfg.contact_ke,
                kd=cfg.contact_kd,
                mu=cfg.friction,
                margin=0,
                gap=cfg.contact_gap,
                has_shape_collision=collision,
                has_particle_collision=False,
            )

        self.elastic_shape = builder.add_shape_mesh(
            self.body,
            mesh=newton.Mesh(self.vertices, self.faces.flatten()),
            cfg=shape_config(),
            color=self.assembly.parts[0]["color"],
        )
        self.presses = []
        pads = spec.get("pads", [{"xy": spec.get("press_xy", [0, 0]), "radius": spec.get("press_radius", 0.028)}])
        self.stroke = spec.get("press_clearance", 0.035) + spec["indentation"]
        for index, pad in enumerate(pads):
            radius, xy = pad["radius"], np.asarray(pad["xy"])
            d2 = np.sum((self.vertices[:, :2] - xy) ** 2, axis=1)
            inside = d2 < radius**2
            touching_z = np.max(self.vertices[inside, 2] + np.sqrt(radius**2 - d2[inside]))
            start = np.r_[xy, touching_z + spec.get("rest_height", 0.038) + spec.get("press_clearance", 0.035)]
            body = builder.add_body(
                xform=wp.transform(wp.vec3(*start), wp.quat_identity()),
                is_kinematic=True,
                label=f"prescribed_press_{index}",
            )
            shape = builder.add_shape_sphere(
                body, radius=radius, cfg=shape_config(cfg.contact_enabled), color=(0.67, 0.72, 0.76)
            )
            mesh = ellipsoid_mesh((radius, radius, radius))
            rod = tube_mesh([(0, 0, radius * 0.8), (0, 0, radius + 0.065)], 0.007, sides=24)
            builder.add_shape_mesh(
                body, mesh=newton.Mesh(rod[0], rod[1].flatten()), cfg=shape_config(False), color=(0.18, 0.21, 0.23)
            )
            self.presses.append(
                {"body": body, "shape": shape, "start": start, "radius": radius, "mesh": mesh, "rod": rod}
            )
        self.press = self.presses[0]["body"]
        self.floor_shape = builder.add_ground_plane(cfg=shape_config(), label="physical_ground")
        self.support_shape = -1
        self.static_meshes = list(spec.get("static", []))
        self.support = spec.get("support")
        if self.support:
            support = self.support
            rotation = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), -np.pi / 2)
            self.support_shape = builder.add_shape_capsule(
                -1,
                xform=wp.transform(wp.vec3(*support["center"]), rotation),
                radius=support["radius"],
                half_height=support["half_height"],
                cfg=shape_config(),
                color=(0.21, 0.25, 0.29),
                label="visible_ground_flex_support",
            )
            vertices, faces = capsule_mesh(support["radius"], support["half_height"])
            self.static_meshes.append(
                (
                    "rounded_flex_support",
                    (vertices + support["center"], faces),
                    [0.21, 0.25, 0.29],
                )
            )
        for name, mesh, color in spec.get("static", []):
            builder.add_shape_mesh(
                -1, mesh=newton.Mesh(mesh[0], mesh[1].flatten()), cfg=shape_config(False), color=color, label=name
            )
        builder.color()
        self.model = builder.finalize()
        self.phi = (
            self.model.elastic_shape_vertex_phi.numpy().reshape(-1, self.mode_count, 3)[: len(self.vertices)].copy()
        )
        np.testing.assert_allclose(
            self.model.elastic_shape_vertex_local.numpy()[: len(self.vertices)], self.vertices, atol=1e-7
        )
        # Builder attachment samples can extend the basis; the shape still has
        # exactly the original surface vertices, in their original order.
        np.testing.assert_allclose(self.phi, self.basis.sample_phi[: len(self.vertices)], atol=2e-6)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.pipeline = newton.CollisionPipeline(
            self.model, reduce_contacts=False, rigid_contact_max=contact_capacity.GLOBAL_CAPACITY
        )
        self.control, self.contacts = self.model.control(), self.pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=cfg.iterations,
            rigid_body_contact_buffer_size=16384,
            rigid_contact_k_start=cfg.contact_ke,
            rigid_compliant_alm=False,
            rigid_contact_hard=False,
            friction_epsilon=1e-4,
            elastic_body_relaxation=0.6,
            elastic_solver_metrics=True,
        )
        elastic = int(self.model.body_elastic_index.numpy()[self.body])
        owner = int(self.model.elastic_joint.numpy()[elastic])
        self.q_start = int(self.model.joint_q_start.numpy()[owner]) + 7
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.contact_recorder = contact_capacity.ContactRecorder(self.contacts, self.solver, self.clock)
        self.graph = None
        capture_start = time.perf_counter()
        if self.model.device.is_cuda:
            if cfg.substeps % 2:
                raise ValueError("Use an even substep count for state-pair graph replay")
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            wp.synchronize()
        self.capture_seconds = time.perf_counter() - capture_start
        if viewer is not None:
            viewer.set_model(self.model)
        self.observe()

    def simulate(self):
        for _ in range(self.cfg.substeps):
            for pad in self.presses:
                wp.launch(
                    drive_press,
                    dim=1,
                    inputs=[
                        self.clock,
                        self.dt,
                        pad["body"],
                        wp.vec3(*pad["start"]),
                        self.stroke,
                        wp.vec4(*self.schedule),
                        self.state_0.body_q,
                        self.state_0.body_qd,
                        self.state_1.body_q,
                        self.state_1.body_qd,
                    ],
                    device=self.model.device,
                )
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.contact_recorder.collide()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.contact_recorder.solved()
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(tick, dim=1, inputs=[self.clock, self.dt], device=self.model.device)

    def step(self):
        start = time.perf_counter()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        wp.synchronize()
        self.timings.append(time.perf_counter() - start)
        self.sim_time = len(self.timings) / 60
        self.observe()

    def observe(self):
        poses = self.state_0.body_q.numpy()
        q, qd = self.state_0.joint_q.numpy(), self.state_0.joint_qd.numpy()
        amplitudes = q[self.q_start : self.q_start + self.mode_count]
        displacement = np.einsum("nmc,m->nc", self.phi, amplitudes)
        vertices = self.vertices + displacement
        world = transform_points(vertices, poses[self.body])
        triangles = world[self.faces]
        samples = np.concatenate(
            (
                world,
                triangles.mean(axis=1),
                (triangles[:, 0] + triangles[:, 1]) / 2,
                (triangles[:, 1] + triangles[:, 2]) / 2,
                (triangles[:, 2] + triangles[:, 0]) / 2,
            )
        )
        press_penetration = max(
            max(0.0, -float((np.linalg.norm(samples - poses[pad["body"], :3], axis=1) - pad["radius"]).min()))
            for pad in self.presses
        )
        ground_penetration = max(0.0, -float(samples[:, 2].min()))
        support_penetration = 0.0
        if self.support:
            local = samples - np.asarray(self.support["center"])
            local[:, 1] = np.maximum(np.abs(local[:, 1]) - self.support["half_height"], 0.0)
            support_penetration = max(0.0, -float((np.linalg.norm(local, axis=1) - self.support["radius"]).min()))
        edge_lengths = np.linalg.norm(vertices[self.edges[:, 1]] - vertices[self.edges[:, 0]], axis=1)
        strain = edge_lengths / self.rest_edges - 1
        stretches = triangle_principal_stretches(vertices, self.faces, self.inverse_reference)
        local_triangles = vertices[self.faces]
        normals = np.cross(local_triangles[:, 1] - local_triangles[:, 0], local_triangles[:, 2] - local_triangles[:, 0])
        areas = np.linalg.norm(normals, axis=1)
        cosine = np.einsum("nc,nc->n", normals, self.rest_normals) / np.maximum(areas * self.rest_areas, 1e-30)
        normal_rotation = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
        solve = {name: float(value[0]) for name, value in self.solver.elastic_mode_solve_metrics().items()}
        contacts = int(self.contacts.rigid_contact_count.numpy()[0])
        assert contacts < self.contacts.rigid_contact_max, "Global contact overflow/saturation"
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:contacts]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:contacts]
        other = np.where(shape0 == self.elastic_shape, shape1, np.where(shape1 == self.elastic_shape, shape0, -2))
        press_contacts = int(np.isin(other, [pad["shape"] for pad in self.presses]).sum())
        support_contacts = int(np.isin(other, [self.floor_shape, self.support_shape]).sum())
        com = np.sum(world * self.sample_mass[:, None], axis=0) / self.sample_mass.sum()
        com_velocity = (com - np.asarray(self.records[-1]["center_of_mass_m"])) * 60 if self.records else np.zeros(3)
        sole = np.zeros(len(self.vertices), dtype=bool)
        for part in self.assembly.parts:
            if part["name"] == "rounded_foam_sole":
                sole[part["start"] : part["start"] + part["count"]] = True
        flex = float(np.ptp(displacement[sole, 2])) if np.any(sole) else 0.0
        record = {
            "time": self.sim_time,
            "max_deformation_m": float(np.linalg.norm(displacement, axis=1).max()),
            "max_crown_deformation_m": float(np.linalg.norm(displacement[self.region == 2], axis=1).max(initial=0)),
            "penetration_m": max(
                ground_penetration, support_penetration, press_penetration if self.cfg.contact_enabled else 0.0
            ),
            "press_penetration_m": press_penetration,
            "ground_penetration_m": ground_penetration,
            "support_penetration_m": support_penetration,
            "max_edge_length_change": float(np.abs(strain).max()),
            "max_principal_stretch": float(stretches.max()),
            "min_principal_stretch": float(stretches.min()),
            "max_principal_strain": float(np.abs(stretches - 1).max()),
            "max_face_normal_rotation_deg": float(normal_rotation.max()),
            "min_triangle_area_ratio": float((areas / self.rest_areas).min()),
            "contacts": contacts,
            "press_contacts": press_contacts,
            "ground_support_contacts": support_contacts,
            "ground_contacts": int((other == self.floor_shape).sum()),
            "support_contacts": int((other == self.support_shape).sum()) if self.support else 0,
            "center_of_mass_m": com.tolist(),
            "com_velocity_m_s": com_velocity.tolist(),
            "sole_flex_range_m": flex,
            "body_pose": poses[self.body].tolist(),
            "q": amplitudes.tolist(),
            "max_fixed_displacement_m": float(
                np.linalg.norm(displacement[np.all(self.raw == 0, axis=(1, 2))], axis=1).max(initial=0)
            ),
            "min_surface_z_m": float(world[:, 2].min()),
            "modal_solver": solve,
        }
        assert np.isfinite(world).all() and np.isfinite(q).all() and np.isfinite(qd).all(), record
        self.records.append(record)
        self.saved_body.append(poses.copy())
        self.saved_q.append(q.copy())
        self.saved_qd.append(qd.copy())
        self.saved_body_qd.append(self.state_0.body_qd.numpy())
        self.world.append(world.astype(np.float32))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def summary(self):
        hold = [
            r
            for r in self.records
            if self.schedule[0] + self.schedule[1] + 0.15 <= r["time"] <= self.schedule[2] - 0.05
        ]
        release = [r for r in self.records if r["time"] >= 5.0]
        recoil = [r for r in self.records if self.schedule[2] + self.schedule[3] <= r["time"] <= self.schedule[2] + 1.0]
        displacement = [r["max_deformation_m"] for r in hold]
        relative = [
            r["modal_solver"]["solve_residual_norm"] / max(r["modal_solver"]["initial_residual_norm"], 1e-12)
            for r in self.records[1:]
        ]
        return {
            "slug": self.spec["slug"],
            "config": asdict(self.cfg),
            "contact_capacity": self.contact_recorder.summary(
                len(self.timings) * self.cfg.substeps, self.dt, self.schedule[0]
            ),
            "basis": self.audit,
            "source_sha256": {
                name: hashlib.sha256(source.encode()).hexdigest() for name, source in self.source_snapshot.items()
            },
            "device": str(self.model.device),
            "graph_enabled": self.graph is not None,
            "elastic_body_count": int(self.model.elastic_body_count),
            "floating_frame_dynamic": self.dynamic,
            "body_mass_kg": float(self.model.body_mass.numpy()[self.body]),
            "body_inverse_mass_kg_inv": float(self.model.body_inv_mass.numpy()[self.body]),
            "max_rigid_frame_displacement_m": float(
                np.linalg.norm(
                    np.array([r["body_pose"][:3] for r in self.records]) - self.records[0]["body_pose"][:3], axis=1
                ).max()
            ),
            "max_rigid_frame_rotation_deg": float(
                np.rad2deg(
                    2
                    * np.arccos(
                        np.clip(
                            np.abs(
                                np.array([r["body_pose"][3:] for r in self.records])
                                @ np.array(self.records[0]["body_pose"][3:])
                            ),
                            0,
                            1,
                        )
                    )
                ).max()
            ),
            "max_sole_flex_range_m": max(r["sole_flex_range_m"] for r in self.records),
            "frames": len(self.records),
            "duration_s": self.sim_time,
            "max_deformation_m": max(r["max_deformation_m"] for r in self.records),
            "max_crown_deformation_m": max(r["max_crown_deformation_m"] for r in self.records),
            "max_displacement_m": max(r["max_deformation_m"] for r in self.records),
            "max_penetration_m": max(r["penetration_m"] for r in self.records),
            "max_ground_penetration_m": max(r["ground_penetration_m"] for r in self.records),
            "max_support_penetration_m": max(r["support_penetration_m"] for r in self.records),
            "max_edge_length_change": max(r["max_edge_length_change"] for r in self.records),
            "max_principal_stretch": max(r["max_principal_stretch"] for r in self.records),
            "min_principal_stretch": min(r["min_principal_stretch"] for r in self.records),
            "max_principal_strain": max(r["max_principal_strain"] for r in self.records),
            "minimum_rest_triangle_quality": float(self.rest_quality.min()),
            "surface_strain_coverage": "Every triangle and edge of every elastic material component; no area or quality exclusions",
            "max_face_normal_rotation_deg": max(r["max_face_normal_rotation_deg"] for r in self.records),
            "min_triangle_area_ratio": min(r["min_triangle_area_ratio"] for r in self.records),
            "max_fixed_displacement_m": max(r["max_fixed_displacement_m"] for r in self.records),
            "min_surface_z_m": min(r["min_surface_z_m"] for r in self.records),
            "hold_contact_dropouts": sum(r["press_contacts"] == 0 for r in hold),
            "hold_mean_deformation_m": float(np.mean(displacement)) if hold else None,
            "hold_max_deformation_m": max(displacement, default=0),
            "initial_drop_max_deformation_m": max(
                r["max_deformation_m"] for r in self.records if r["time"] < self.schedule[0]
            ),
            "hold_peak_to_peak_m": float(np.ptp(displacement)) if hold else None,
            "release_max_deformation_m": max((r["max_deformation_m"] for r in release), default=None),
            "final_deformation_m": self.records[-1]["max_deformation_m"],
            "final_displacement_m": self.records[-1]["max_deformation_m"],
            "maximum_contact_count": max(r["contacts"] for r in self.records),
            "maximum_press_contacts": max(r["press_contacts"] for r in self.records),
            "first_press_contact_s": next((r["time"] for r in self.records if r["press_contacts"]), None),
            "maximum_ground_support_contacts": max(r["ground_support_contacts"] for r in self.records),
            "final_ground_contacts": self.records[-1]["ground_contacts"],
            "final_support_contacts": self.records[-1]["support_contacts"],
            "post_release_airborne_frames": sum(
                r["ground_support_contacts"] == 0 and r["press_contacts"] == 0 for r in recoil
            ),
            "post_release_max_upward_speed_m_s": max((r["com_velocity_m_s"][2] for r in recoil), default=0.0),
            "post_release_com_vertical_range_m": float(np.ptp([r["center_of_mass_m"][2] for r in recoil]))
            if recoil
            else 0.0,
            "settled_com_peak_to_peak_m": float(np.ptp([r["center_of_mass_m"] for r in release], axis=0).max())
            if release
            else None,
            "release_duration_s": self.schedule[3],
            "max_relative_linear_solve_residual": max(relative, default=0),
            "mean_warm_graph_frame_ms": float(np.mean(self.timings[10:]) * 1000) if len(self.timings) > 10 else None,
            "capture_compile_seconds": self.capture_seconds,
            "contact_formulation": "Finite-stiffness penalty contact, legacy AVBD hard contacts disabled; no compliant ALM",
            "timing_scope": "Synchronized graph replay only, after 10 frames; excludes observation, readback, export, and rendering",
            "prescribed": "Rigid pressing-pad poses only. Elastic floating frame and every modal amplitude are solved by VBD; no pose reset, clamp, joint drive, or hidden modal animation.",
            "gravity": "9.81 m/s^2; free-body fall and physical ground/support contact remain active in the no-pad-contact control",
            "press_radii_m": [pad["radius"] for pad in self.presses],
            "schedule_s": self.schedule,
            "scientific_status": self.spec.get("status", "prototype"),
            "validation_scope": self.spec.get("validation_scope", "Bounded elastic loading and recovery"),
            "indentation_m": self.spec["indentation"],
            "bounds": self.spec["bounds"],
            "limitations": self.spec["limitations"],
        }

    def test_final(self):
        self.contact_recorder.validate(len(self.timings) * self.cfg.substeps, self.dt)
        s, bounds = self.summary(), self.spec["bounds"]
        assert self.sim_time >= 5.5, "A full loading/unloading trajectory is required"
        assert s["elastic_body_count"] == 1
        assert s["floating_frame_dynamic"] and s["body_inverse_mass_kg_inv"] > 0, s
        assert s["max_rigid_frame_displacement_m"] > 0.008, s
        assert s["max_deformation_m"] < bounds["displacement_m"], s
        assert s["max_edge_length_change"] < bounds["edge_strain"], s
        assert s["max_principal_strain"] < bounds.get("principal_strain", bounds["edge_strain"]), s
        assert s["minimum_rest_triangle_quality"] > 0.01, s
        assert s["max_face_normal_rotation_deg"] < 45, s
        assert s["min_triangle_area_ratio"] > 0.5, s
        assert s["release_max_deformation_m"] < bounds["release_m"], s
        assert s["max_penetration_m"] < bounds.get("penetration_m", 0.002), s
        assert s["max_relative_linear_solve_residual"] < 1e-4, s
        assert s["maximum_ground_support_contacts"] > 0, s
        assert s["settled_com_peak_to_peak_m"] < bounds.get("settled_motion_m", 0.004), s
        if self.cfg.contact_enabled:
            assert s["first_press_contact_s"] is not None and s["first_press_contact_s"] > self.schedule[0], s
            assert s["max_deformation_m"] > bounds["minimum_deformation_m"], s
            if "minimum_sole_flex_m" in bounds:
                assert s["max_sole_flex_range_m"] > bounds["minimum_sole_flex_m"], s
            if "minimum_crown_deformation_m" in bounds:
                assert s["max_crown_deformation_m"] > bounds["minimum_crown_deformation_m"], s
            assert s["hold_contact_dropouts"] == 0, s
            assert s["hold_peak_to_peak_m"] < bounds.get("hold_chatter_m", 0.002), s
            assert s["post_release_max_upward_speed_m_s"] > bounds.get("minimum_recoil_speed_m_s", 0.0), s
        else:
            assert s["maximum_press_contacts"] == 0, s
            assert s["hold_max_deformation_m"] < bounds["control_m"], s

    def export(self, output, status="passed"):
        output.mkdir(parents=True, exist_ok=True)
        collision_mesh = self.model.shape_source[self.elastic_shape]
        np.savez_compressed(
            output / "model_geometry.npz",
            collision_vertices=collision_mesh.vertices,
            collision_faces=collision_mesh.indices.reshape(-1, 3),
            elastic_vertex_local=self.model.elastic_shape_vertex_local.numpy(),
            elastic_vertex_phi=self.model.elastic_shape_vertex_phi.numpy(),
            body_mass=self.model.body_mass.numpy(),
            body_com=self.model.body_com.numpy(),
            body_inertia=self.model.body_inertia.numpy(),
            shape_type=self.model.shape_type.numpy(),
            shape_transform=self.model.shape_transform.numpy(),
            shape_scale=self.model.shape_scale.numpy(),
            shape_body=self.model.shape_body.numpy(),
            support_shape=np.int32(self.support_shape),
        )
        self.contact_recorder.export(output)
        summary = self.summary()
        summary["test_status"] = status
        summary["test_passed"] = status == "passed"
        summary["negative_control_passed"] = False
        (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output / "frame_metrics.json").write_text(json.dumps(self.records, indent=2) + "\n")
        np.savez_compressed(
            output / "state.npz",
            times=np.array([r["time"] for r in self.records]),
            body_q=self.saved_body,
            body_qd=self.saved_body_qd,
            joint_q=self.saved_q,
            joint_qd=self.saved_qd,
            modal_q_start=np.int32(self.q_start),
            elastic_body=np.int32(self.body),
            press_body=np.int32(self.press),
            press_bodies=np.asarray([pad["body"] for pad in self.presses], dtype=np.int32),
        )
        np.savez_compressed(
            output / "basis.npz",
            rest_vertices=self.vertices,
            faces=self.faces,
            phi=self.phi,
            raw_analytic_modes=self.raw,
            sample_mass=self.sample_mass,
            region=self.region,
            mode_mass=self.basis.mode_mass,
            mode_stiffness=self.basis.mode_stiffness,
            mode_damping=self.basis.mode_damping,
            geometric_snapshots=self.spec.get("snapshots", np.empty((0, len(self.vertices), 3))),
            geometric_pod_modes=self.spec.get("pod_modes", np.empty((len(self.vertices), 0, 3))),
            **{f"projected_{key}": value for key, value in self.spec.get("stiffness_matrices", {}).items()},
        )
        arrays = {"times": np.array([r["time"] for r in self.records])}
        meshes = []

        def mesh_entry(name, vertices, faces, color, roughness=0.65, metallic=0.0, **extra):
            index = len(meshes)
            arrays[f"vertices_{index}"], arrays[f"faces_{index}"] = (
                np.asarray(vertices, dtype=np.float32),
                np.asarray(faces, dtype=np.int32),
            )
            meshes.append(
                dict(
                    name=name,
                    vertices=f"vertices_{index}",
                    faces=f"faces_{index}",
                    color=color,
                    roughness=roughness,
                    metallic=metallic,
                    **extra,
                )
            )

        world = np.asarray(self.world)
        for part, faces in zip(self.assembly.parts, self.assembly.faces, strict=True):
            start, count = part["start"], part["count"]
            mesh_entry(
                part["name"],
                world[:, start : start + count],
                faces - start,
                part["color"],
                part["roughness"],
                basis_vertex_start=start,
                basis_vertex_count=count,
            )
        for index, pad in enumerate(self.presses):
            for name, mesh, color, roughness, metallic in (
                (f"rounded_press_{index}", pad["mesh"], [0.67, 0.72, 0.76], 0.24, 0.65),
                (f"press_shaft_{index}", pad["rod"], [0.18, 0.21, 0.23], 0.3, 0.7),
            ):
                animated = [transform_points(mesh[0], poses[pad["body"]]) for poses in self.saved_body]
                mesh_entry(name, animated, mesh[1], color, roughness, metallic)
        for name, mesh, color in self.static_meshes:
            mesh_entry(name, mesh[0], mesh[1], color)
        np.savez_compressed(output / "scene.npz", **arrays)
        root = Path(newton.__file__).resolve().parents[1]
        helpers = Path(__file__).resolve().parent
        slug = self.spec["slug"]
        environment = (
            f'env PYTHONPATH="{root}" OPENBLAS_NUM_THREADS=1 '
            "PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/tmp/newton-everyday-wearables-warp "
            f'uv run --project "{root}" python '
        )
        job = f'flock /tmp/newton-everyday-rom-gpu.lock timeout 5m {environment}"{helpers}/{slug}.py"'
        commands = [
            f'{job} --output "{output}"',
            f'{job} --no-contact --output "{output}/negative-control"',
            f'{environment}"{helpers}/test_wearables.py" --validate-pair "{output}"',
        ]
        scene = {
            "slug": slug,
            "title": self.spec["title"],
            "status": self.spec.get("status", "prototype; negative control pending") if status == "passed" else status,
            "fps": 60,
            "meshes": meshes,
            "camera": self.spec["camera"],
            "floor_z": 0.0,
            "poster_frame": int(np.argmax([r["max_deformation_m"] for r in self.records])),
            "description": self.spec["description"],
            "model": self.spec["model"],
            "limitations": self.spec["limitations"],
            "driven": summary["prescribed"],
            "mode_count": self.mode_count,
            "parameters": asdict(self.cfg),
            "source_files": ["clog.py", "cap.py", "wearable_geometry.py", "test_wearables.py"],
            "reproduce": "\n".join(commands),
            "audit": {"state_file": "state.npz", "basis_file": "basis.npz", "frame_metrics": "frame_metrics.json"},
        }
        (output / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
        for name, source in self.source_snapshot.items():
            (output / name).write_text(source)
        return summary


def run_cli(example_class, slug):
    parser = argparse.ArgumentParser(description=example_class.__doc__)
    parser.add_argument("--output", type=Path, default=Path("output") / slug)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--iterations", type=int, default=Config.iterations)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--no-contact", action="store_true")
    parser.add_argument("--with-control", action="store_true")
    args = parser.parse_args()
    wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", "/tmp/newton-everyday-wearables-warp")
    wp.init()
    wp.set_device(args.device)

    def run(enabled, output):
        example = example_class(
            None, Config(contact_enabled=enabled, iterations=args.iterations, substeps=args.substeps)
        )
        for i in range(args.frames):
            example.step()
            if (i + 1) % 60 == 0:
                r = example.records[-1]
                print(
                    f"{slug} contact={enabled} frame={i + 1} displacement_mm={1000 * r['max_deformation_m']:.3f} penetration_mm={1000 * r['penetration_m']:.3f} contacts={r['contacts']}",
                    flush=True,
                )
        status = "passed"
        try:
            example.test_final()
        except AssertionError:
            status = "failed"
            raise
        finally:
            summary = example.export(output, status)
            print(json.dumps(summary, indent=2), flush=True)
        return summary

    run(not args.no_contact, args.output)
    if args.with_control and not args.no_contact:
        run(False, args.output / "negative-control")
        from test_wearables import validate_pair  # noqa: PLC0415 - avoid the test module's example import cycle

        validate_pair(args.output)
