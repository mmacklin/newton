# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Accepted 32-mode hand mechanics and unchanged all-substep physical gates.

The baseline bar schedule is retained for timing and source regression tests.
The five-finger entry point is individual.py; only the derived ROM is loaded.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from hand_geometry import (
    IntersectionAudit,
    SubstepAudit,
    box,
    box_sdf,
    count_intersections,
    transform_points,
)
from load_hand_rom import load_hand_rom

import newton
import newton.solvers

ROOT = Path("output/everyday-rom/hand").resolve()
ACCEPTED = Path(__file__).with_name("assets")
RUNTIME = Path(newton.__file__).resolve().parents[1]
PIN = "8228bf6f690fff4dbb359d5de6447b09cb77c1e7"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def private_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path == ROOT.resolve():
        raise ValueError("Outputs must be a new private hand-cache subdirectory")
    return path


def smooth_cpu(t, start, duration):
    s = np.clip((np.asarray(t) - start) / duration, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def pulse_cpu(t):
    return sum(smooth_cpu(t, start, 1.1) * (1 - smooth_cpu(t, start + 1.5, 0.2)) for start in (1.0, 4.5))


def finite_strain(rest, current, faces):
    """Principal engineering strain from singular values of triangle F (3x2).

    The reference chart is orthonormal, not a raw edge-coordinate metric.
    Rigid rotations leave this strain invariant. No edge-only approximation.
    """
    a, b, c = np.moveaxis(np.asarray(rest, dtype=float)[faces], 1, 0)
    u, v = b - a, c - a
    length = np.linalg.norm(u, axis=-1)
    if np.any(length <= 0):
        raise ValueError("Degenerate reference face")
    x = np.einsum("fc,fc->f", u, v) / length
    y = np.linalg.norm(np.cross(u, v), axis=-1) / length
    if np.any(length <= 0) or np.any(y <= 0):
        raise ValueError("Degenerate reference face")
    tri = np.asarray(current, dtype=float)[..., faces, :]
    e, f = tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :]
    f0 = e / length[:, None]
    f1 = (f - f0 * x[:, None]) / y[:, None]
    aa, bb, ab = (np.sum(z, axis=-1) for z in (f0 * f0, f1 * f1, f0 * f1))
    disc = np.sqrt(np.maximum(0, (aa - bb) ** 2 + 4 * ab**2))
    stretches = np.sqrt(np.maximum(0, np.stack((aa + bb - disc, aa + bb + disc), axis=-1) / 2))
    return stretches - 1


def recover(rest, phi, q):
    return np.asarray(rest) + np.einsum("nmc,...m->...nc", phi, q, optimize=True)


def certify_export(rest, phi, faces, bq, jq, q_start, body, scene):
    if not np.array_equal(scene["faces_0"], faces):
        raise AssertionError("Export topology changed")
    worst = 0.0
    for i in range(len(bq)):
        local = recover(rest, phi, jq[i, q_start : q_start + phi.shape[1]])
        expected = transform_points(local, bq[i, body])
        worst = max(worst, float(np.max(np.abs(expected - scene["vertices_0"][i]))))
    if worst > 2e-7:
        raise AssertionError(f"Export does not recover solved geometry: {worst}")
    return worst


@wp.func
def smooth(t: float, start: float, duration: float):
    s = wp.clamp((t - start) / duration, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@wp.func
def pulse(t: float):
    return smooth(t, 1.0, 1.1) * (1.0 - smooth(t, 2.5, 0.2)) + smooth(t, 4.5, 1.1) * (1.0 - smooth(t, 6.0, 0.2))


@wp.kernel
def drive(
    clock: wp.array[int],
    dt: float,
    travel: float,
    bar_height: float,
    support: int,
    pusher: int,
    b0: wp.array[wp.transform],
    b1: wp.array[wp.transform],
    v0: wp.array[wp.spatial_vector],
    v1: wp.array[wp.spatial_vector],
):
    body = support
    position = wp.vec3(0.0, 0.0, 0.12)
    velocity = wp.vec3(0.0)
    if wp.tid() == 1:
        body = pusher
        t = float(clock[0]) * dt
        position = wp.vec3(0.0, 0.025 - travel * pulse(t), 0.12 + bar_height)
        velocity = wp.vec3(0.0, -travel * (pulse(t + dt) - pulse(t)) / dt, 0.0)
    b0[body] = wp.transform(position, wp.quat_identity())
    b1[body] = b0[body]
    v0[body] = wp.spatial_vector(velocity, wp.vec3(0.0))
    v1[body] = v0[body]


@wp.kernel
def advance_clock(cursor: wp.array[int]):
    cursor[0] += 1


@wp.kernel
def record_contacts(
    cursor: wp.array[int],
    raw: wp.array[int],
    s0: wp.array[int],
    s1: wp.array[int],
    p0: wp.array[wp.vec3],
    p1: wp.array[wp.vec3],
    hand: int,
    pusher: int,
    saved: wp.array2d[int],
    locations: wp.array2d[float],
):
    slot = cursor[0]
    saved[slot, 0] = raw[0]
    count = int(0)
    left, middle, right = int(0), int(0), int(0)
    min_z, max_z = float(1.0e6), float(-1.0e6)
    min_x, max_x = float(1.0e6), float(-1.0e6)
    for i in range(wp.min(raw[0], s0.shape[0])):
        if (s0[i] == hand and s1[i] == pusher) or (s1[i] == hand and s0[i] == pusher):
            count += 1
            p = p0[i]
            if s1[i] == hand:
                p = p1[i]
            min_z, max_z = wp.min(min_z, p[2]), wp.max(max_z, p[2])
            min_x, max_x = wp.min(min_x, p[0]), wp.max(max_x, p[0])
            if p[0] < -0.015:
                left += 1
            elif p[0] > 0.015:
                right += 1
            else:
                middle += 1
    saved[slot, 1] = count
    saved[slot, 2] = left
    saved[slot, 3] = middle
    saved[slot, 4] = right
    if count > 0:
        locations[slot, 0] = min_z
        locations[slot, 1] = max_z
        locations[slot, 2] = min_x
        locations[slot, 3] = max_x


@wp.kernel
def record_state(
    cursor: wp.array[int],
    pairs: wp.array[int],
    intersections: wp.array[int],
    bq: wp.array[wp.transform],
    jq: wp.array[float],
    saved_bq: wp.array2d[wp.transform],
    saved_jq: wp.array2d[float],
):
    slot = cursor[0]
    intersections[slot] = pairs[0]
    for i in range(bq.shape[0]):
        saved_bq[slot, i] = bq[i]
    for i in range(jq.shape[0]):
        saved_jq[slot, i] = jq[i]
    cursor[0] += 1


def pose(position):
    return wp.transform(wp.vec3(*position), wp.quat_identity())


def shape_config(collision=True):
    return newton.ModelBuilder.ShapeConfig(
        density=0,
        ke=3e4,
        kd=15,
        mu=0.45,
        gap=0.001,
        margin=0,
        has_shape_collision=collision,
        has_particle_collision=False,
    )


class HandContact:
    def __init__(
        self,
        travel=0.036,
        enabled=True,
        duration=8.5,
        device="cuda:0",
        modes=32,
        bar_height=0.123,
        *,
        audited=True,
        rom_path=None,
        substeps=4,
        iterations=20,
    ):
        self.audited = audited
        self.travel, self.enabled, self.duration = travel, enabled, duration
        self.bar_height = bar_height
        if substeps not in (2, 4):
            raise ValueError("This graph harness supports two or four substeps")
        self.dt, self.substeps, self.modes = 1 / (60 * substeps), substeps, modes
        self.iterations = iterations
        self.capacity = round(duration / self.dt)
        self.frames = round(duration * 60)
        if modes != 32 and rom_path is None:
            raise ValueError("The accepted derived ROM has exactly 32 modes")
        rom_path = Path(rom_path) if rom_path else ACCEPTED / "rom-48.npz"
        basis, fields = load_hand_rom(rom_path)
        if basis.sample_phi.shape[1] != modes:
            raise ValueError("Basis and requested mode count differ")
        self.rest, self.faces = fields["rest"], fields["faces"]
        self.used, self.anchor = fields["used"], fields["anchor"]
        self.phi = basis.sample_phi[self.used]
        mass, com, inertia = fields["body_mass"], fields["body_com"], fields["body_inertia"]
        self.basis_audit = {
            "method": f"Loaded derived {modes}-mode fields",
            "sha256": digest(rom_path),
        }
        self.basis = basis
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81), up_axis="Z")
        self.floor = builder.add_ground_plane(cfg=shape_config())
        self.hand = builder.add_body_elastic(
            xform=pose((0, 0, 0.12)),
            mass=float(mass),
            com=wp.vec3(*com),
            inertia=wp.mat33(inertia.flatten()),
            modal_basis=basis,
            lock_inertia=True,
            label="dynamic_hand",
        )
        self.meshes = []
        self.hand_shape = self.add_mesh(
            builder, "native_hand", self.hand, self.rest, self.faces, (0.68, 0.69, 0.72), True
        )
        self.support = builder.add_body(xform=pose((0, 0, 0.12)), is_kinematic=True, label="wrist_support")
        self.support_joint = builder.add_joint_fixed(
            self.support,
            self.hand,
            parent_xform=pose(self.anchor),
            child_xform=pose(self.anchor),
            label="permanent_support",
        )
        for name, half, center in (
            ("cradle", (0.036, 0.028, 0.004), (0.004, -0.012, -0.026)),
            ("left_jaw", (0.004, 0.025, 0.013), (-0.034, -0.009, -0.010)),
            ("right_jaw", (0.004, 0.025, 0.013), (0.046, -0.009, -0.010)),
            ("support_back", (0.045, 0.008, 0.012), (0.004, 0.029, -0.015)),
        ):
            v, f = box(half, center)
            shape = self.add_mesh(builder, name, self.support, v, f, (0.36, 0.39, 0.43))
            builder.add_shape_collision_filter_pair(self.hand_shape, shape)
        self.half = np.asarray(getattr(self, "probe_half", [0.054, 0.008, 0.023]))
        self.pusher = builder.add_body(xform=pose((0, 0.025, 0.12 + bar_height)), is_kinematic=True, label="finger_bar")
        self.pusher_shape = builder.add_shape_box(
            self.pusher,
            hx=self.half[0],
            hy=self.half[1],
            hz=self.half[2],
            cfg=shape_config(enabled),
            color=(0.8, 0.24, 0.12),
            label="finger_bar",
        )
        v, f = box(self.half)
        self.meshes.append(
            {
                "name": "finger_bar",
                "body": self.pusher,
                "rest": v,
                "faces": f,
                "color": (0.8, 0.24, 0.12),
                "elastic": False,
            }
        )
        builder.color()
        self.model = builder.finalize(device=device)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(
            self.model,
            rigid_contact_max=32768,
            contact_matching="latest",
            reduce_contacts=False,
        )
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=iterations,
            rigid_compliant_alm=True,
            rigid_contact_history=True,
            rigid_joint_linear_ke=1e5,
            rigid_joint_angular_ke=100,
            rigid_joint_linear_kd=1.0,
            rigid_joint_angular_kd=0.01,
            rigid_body_contact_buffer_size=16384,
            elastic_body_relaxation=0.8,
            elastic_solver_metrics=audited,
        )
        owner = int(self.model.elastic_joint.numpy()[0])
        self.q_start = int(self.model.joint_q_start.numpy()[owner]) + 7
        np.testing.assert_array_equal(self.model.elastic_shape_vertex_phi.numpy().reshape(-1, self.modes, 3), self.phi)
        np.testing.assert_array_equal(self.model.elastic_shape_vertex_sample.numpy(), self.used)
        self.endpoint = int(self.model.joint_child_elastic_endpoint.numpy()[self.support_joint])
        self.anchor_phi = self.model.elastic_endpoint_phi.numpy().reshape(-1, self.modes, 3)[self.endpoint]
        self.cursor = wp.zeros(1, dtype=int, device=device)
        self.prepare_drive(device)
        if audited:
            self.audit = SubstepAudit(self.rest, self.faces, self.phi, self.capacity, device=device)
            self.checker = IntersectionAudit(self.rest, self.faces, device=device)
            if self.checker.count(self.rest) != 0:
                raise AssertionError("Reference hand intersects")
            self.raw_counts = wp.zeros((self.capacity, 5), dtype=int, device=device)
            self.contact_locations = wp.zeros((self.capacity, 4), dtype=float, device=device)
            self.crossings = wp.zeros(self.capacity, dtype=int, device=device)
            self.sub_body = wp.empty((self.capacity, self.model.body_count), dtype=wp.transform, device=device)
            self.sub_joint = wp.empty((self.capacity, self.model.joint_coord_count), dtype=float, device=device)
        self.bodies, self.joints, self.velocities, self.joint_velocities = [], [], [], []
        if audited:
            self.record()
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph

    def add_mesh(self, builder, name, body, vertices, faces, color, elastic=False):
        shape = builder.add_shape_mesh(
            body,
            mesh=newton.Mesh(vertices, faces.flatten()),
            cfg=shape_config(),
            color=color,
            label=name,
        )
        self.meshes.append(
            {"name": name, "body": body, "rest": vertices, "faces": faces, "color": color, "elastic": elastic}
        )
        return shape

    def prepare_drive(self, device):
        pass

    def drive_step(self):
        wp.launch(
            drive,
            2,
            inputs=[
                self.cursor,
                self.dt,
                self.travel,
                self.bar_height,
                self.support,
                self.pusher,
                self.s0.body_q,
                self.s1.body_q,
                self.s0.body_qd,
                self.s1.body_qd,
            ],
            device=self.model.device,
        )

    def audit_contact_regions(self):
        pass

    def simulate(self):
        device = self.model.device
        for _ in range(self.substeps):
            self.drive_step()
            self.s0.clear_forces()
            self.pipeline.collide(self.s0, self.contacts)
            if not self.audited:
                self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
                self.s0, self.s1 = self.s1, self.s0
                wp.launch(advance_clock, 1, inputs=[self.cursor], device=device)
                continue
            self.audit_contact_regions()
            wp.launch(
                record_contacts,
                1,
                inputs=[
                    self.cursor,
                    self.contacts.rigid_contact_count,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self.contacts.rigid_contact_point0,
                    self.contacts.rigid_contact_point1,
                    self.hand_shape,
                    self.pusher_shape,
                    self.raw_counts,
                    self.contact_locations,
                ],
                device=device,
            )
            self.solver.step(self.s0, self.s1, self.control, self.contacts, self.dt)
            self.s0, self.s1 = self.s1, self.s0
            self.audit.record(self.s0.joint_q, self.s0.body_q, self.q_start, self.hand)
            wp.copy(self.checker.points, self.audit.local)
            self.checker.mesh.refit()
            self.checker.pair_count.zero_()
            wp.launch(
                count_intersections,
                len(self.faces),
                inputs=[
                    self.checker.mesh.id,
                    self.checker.points,
                    self.checker.faces,
                    self.checker.counts,
                    self.checker.pair_count,
                    self.checker.pair_keys,
                ],
                device=device,
            )
            wp.launch(
                record_state,
                1,
                inputs=[
                    self.cursor,
                    self.checker.pair_count,
                    self.crossings,
                    self.s0.body_q,
                    self.s0.joint_q,
                    self.sub_body,
                    self.sub_joint,
                ],
                device=device,
            )

    def record(self):
        for target, state in (
            (self.bodies, self.s0.body_q),
            (self.joints, self.s0.joint_q),
            (self.velocities, self.s0.body_qd),
            (self.joint_velocities, self.s0.joint_qd),
        ):
            target.append(state.numpy())
        if not all(
            np.isfinite(x[-1]).all() for x in (self.bodies, self.joints, self.velocities, self.joint_velocities)
        ):
            raise AssertionError("Nonfinite solved state")

    def run(self):
        start = time.perf_counter()
        for i in range(self.frames):
            if self.graph is None:
                self.simulate()
            else:
                wp.capture_launch(self.graph)
            self.record()
            if i % 60 == 0:
                print(json.dumps({"frame": i, "frames": self.frames, "contact": self.enabled}), flush=True)
        wp.synchronize_device(self.model.device)
        return time.perf_counter() - start

    def export(self, output):
        output.mkdir(parents=True, exist_ok=False)
        bq, jq = np.asarray(self.bodies), np.asarray(self.joints)
        sub_bq, sub_jq = self.sub_body.numpy(), self.sub_joint.numpy()
        geom, q, poses = self.audit.arrays()
        assert int(self.cursor.numpy()[0]) == self.capacity == len(q)
        np.testing.assert_array_equal(sub_bq[self.substeps - 1 :: self.substeps], bq[1:])
        np.testing.assert_array_equal(sub_jq[self.substeps - 1 :: self.substeps], jq[1:])
        np.testing.assert_array_equal(q, sub_jq[:, self.q_start : self.q_start + self.modes])
        np.testing.assert_array_equal(poses, sub_bq[:, self.hand])
        np.testing.assert_array_equal(sub_jq[:, self.q_start - 7 : self.q_start], sub_bq[:, self.hand])
        np.testing.assert_allclose(
            sub_bq[:, self.support, :3], np.broadcast_to([0, 0, 0.12], (self.capacity, 3)), atol=1e-8
        )
        times = np.arange(len(bq)) / 60
        arrays = {"times": times}
        meta = {
            "slug": "private-nimble-hand-contact",
            "poster_frame": 138,
            "fps": 60,
            "meshes": [],
            "floor_z": 0.0,
            "camera": {"position": [0.255, -0.37, 0.345], "target": [0, -0.022, 0.184], "fov": 30},
            "private": True,
            "model": "Uncalibrated native NIMBLE pose-POD shell; not anatomical mechanics",
        }
        for i, item in enumerate(self.meshes):
            arrays[f"faces_{i}"] = item["faces"]
            vertices = []
            for j in range(len(bq)):
                local = (
                    recover(item["rest"], self.phi, jq[j, self.q_start : self.q_start + self.modes])
                    if item["elastic"]
                    else item["rest"]
                )
                vertices.append(transform_points(local, bq[j, item["body"]]))
            arrays[f"vertices_{i}"] = np.asarray(vertices)
            meta["meshes"].append(
                {
                    "name": item["name"],
                    "vertices": f"vertices_{i}",
                    "faces": f"faces_{i}",
                    "color": item["color"],
                    "roughness": 0.45,
                    "metallic": 0 if i == 0 else 0.15,
                }
            )
        error = certify_export(self.rest, self.phi, self.faces, bq, jq, self.q_start, self.hand, arrays)
        np.savez_compressed(output / "scene.npz", **arrays)
        (output / "scene.json").write_text(json.dumps(meta, indent=2) + "\n")
        (output / "basis_audit.json").write_text(json.dumps(self.basis_audit, indent=2) + "\n")
        original = {
            "times": times,
            "body_q": bq,
            "joint_q": jq,
            "body_qd": self.velocities,
            "joint_qd": self.joint_velocities,
            "q_start": self.q_start,
            "hand_body": self.hand,
            "pusher_body": self.pusher,
            "support_body": self.support,
            "hand_shape": self.hand_shape,
            "pusher_shape": self.pusher_shape,
            "rest": self.rest,
            "faces": self.faces,
            "vertex_phi": self.phi,
            "used": self.used,
            "anchor": self.anchor,
            "anchor_phi": self.anchor_phi,
            "sample_points": self.basis.sample_points,
            "sample_phi": self.basis.sample_phi,
            "sample_psi": self.basis.sample_psi,
            "sample_mass": self.basis.sample_mass,
            "mode_mass": self.basis.mode_mass,
            "mode_stiffness": self.basis.mode_stiffness,
            "mode_damping": self.basis.mode_damping,
        }
        np.savez_compressed(output / "original_states.npz", **original)
        raw, crossings = self.raw_counts.numpy(), self.crossings.numpy()
        np.savez_compressed(
            output / "substeps.npz",
            times=(np.arange(self.capacity) + 1) * self.dt,
            body_q=sub_bq,
            joint_q=sub_jq,
            q=q,
            geometry=geom,
            contacts=raw,
            contact_locations=self.contact_locations.numpy(),
            contact_location_columns=["rest_local_min_z", "rest_local_max_z", "rest_local_min_x", "rest_local_max_x"],
            intersections=crossings,
            geometry_columns=self.audit.columns,
            contact_columns=["raw_generated", "hand_pusher", "local_x_left", "local_x_middle", "local_x_right"],
        )
        # Independent float64 all-face audit of every saved integrator state.
        strain_rows, worst_faces = [], []
        min_clearance, min_floor, max_gap = 0.0, np.inf, 0.0
        for start in range(0, len(q), 32):
            local = recover(self.rest, self.phi.astype(float), q[start : start + 32].astype(float))
            strain = np.max(np.abs(finite_strain(self.rest, local, self.faces)), axis=-1)
            strain_rows.extend(strain.max(axis=1))
            worst_faces.extend(strain.argmax(axis=1))
            for k, vertices in enumerate(local):
                slot = start + k
                world = transform_points(vertices, sub_bq[slot, self.hand])
                tri = world[self.faces]
                samples = np.concatenate(
                    (
                        world,
                        tri.mean(axis=1),
                        (tri[:, 0] + tri[:, 1]) / 2,
                        (tri[:, 1] + tri[:, 2]) / 2,
                        (tri[:, 0] + tri[:, 2]) / 2,
                    )
                )
                min_clearance = min(min_clearance, float(box_sdf(samples, sub_bq[slot, self.pusher], self.half).min()))
                min_floor = min(min_floor, float(world[:, 2].min()))
                anchor = self.anchor + np.einsum("mc,m->c", self.anchor_phi, q[slot])
                gap = np.linalg.norm(
                    transform_points(anchor[None], sub_bq[slot, self.hand])
                    - transform_points(self.anchor[None], sub_bq[slot, self.support])
                )
                max_gap = max(max_gap, float(gap))
        strains = np.asarray(strain_rows)
        np.savez_compressed(
            output / "finite_strain.npz",
            maximum=strains,
            face=worst_faces,
            definition="max(abs(singular_values(F_3x2)-1)); orthonormal reference triangle chart",
        )
        worst = int(strains.argmax())
        raw_peak = int(raw[:, 0].max())
        checks = {
            "finite": bool(
                np.isfinite(strains).all()
                and np.isfinite(geom).all()
                and np.isfinite(sub_bq).all()
                and np.isfinite(sub_jq).all()
            ),
            "principal_strain": bool(strains.max() <= 0.15),
            "edge_strain": bool(geom[:, 1].max() <= 0.25),
            "area": bool(geom[:, 2].min() >= 0.7 and geom[:, 3].max() <= 1.4),
            "normals": bool(geom[:, 4].min() > 0),
            "self_intersections": bool(crossings.max() == 0),
            "global_capacity": raw_peak < self.contacts.rigid_contact_max,
            "body_capacity": 2 * raw_peak < 16384,
            "pusher_penetration": bool(not self.enabled or min_clearance >= -0.002),
            "floor_clearance": min_floor > 0.02,
            "wrist_support": max_gap < 0.001,
            "export": error <= 2e-7,
        }
        result = {
            "checks": checks,
            "geometry_passed": all(checks.values()),
            "substeps": len(q),
            "frames": len(bq),
            "contact_enabled": self.enabled,
            "travel_m": self.travel,
            "bar_height_m": self.bar_height,
            "max_principal_strain": float(strains.max()),
            "worst_strain_slot": worst,
            "worst_strain_face": int(worst_faces[worst]),
            "worst_strain_time_s": (worst + 1) * self.dt,
            "max_edge_strain": float(geom[:, 1].max()),
            "min_area_ratio": float(geom[:, 2].min()),
            "max_area_ratio": float(geom[:, 3].max()),
            "min_normal_alignment": float(geom[:, 4].min()),
            "max_intersections": int(crossings.max()),
            "raw_contact_peak": raw_peak,
            "hand_pusher_peak": int(raw[:, 1].max()),
            "contact_spatial_bin_peaks": raw[:, 2:].max(axis=0).tolist(),
            "global_capacity": int(self.contacts.rigid_contact_max),
            "body_capacity": 16384,
            "conservative_body_entries_upper_bound": 2 * raw_peak,
            "max_sampled_pusher_penetration_m": -min_clearance,
            "min_floor_clearance_m": min_floor,
            "max_wrist_gap_m": max_gap,
            "exact_export_max_error_m": error,
            "all_substep_checks": "All faces; all post-solve states; raw contacts before solve. Adjacent triangles sharing a vertex excluded from intersections. Pusher penetration uses vertices, centroids and edge midpoints, not a continuous triangle-box proof.",
        }
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


def paired_metrics(main, control):
    a, b = np.load(main / "original_states.npz"), np.load(control / "original_states.npz")
    sa, sb = np.load(main / "substeps.npz"), np.load(control / "substeps.npz")
    rest, phi = a["rest"], a["vertex_phi"].astype(float)
    np.testing.assert_array_equal(rest, b["rest"])
    np.testing.assert_array_equal(a["vertex_phi"], b["vertex_phi"])
    np.testing.assert_array_equal(sa["body_q"][:, int(a["pusher_body"])], sb["body_q"][:, int(b["pusher_body"])])
    ts = sa["times"]
    qa, qb = sa["q"], sb["q"]
    baseline = qa[(ts >= 0.7) & (ts <= 0.95)].mean(axis=0)
    patches = [
        (rest[:, 2] > 0.09) & (rest[:, 0] >= lo) & (rest[:, 0] < hi)
        for lo, hi in ((-0.06, -0.015), (-0.015, 0.008), (0.008, 0.03), (0.03, 0.06))
    ]
    response, causal, distal = [], [], []
    for lo in range(0, len(qa), 32):
        dis = recover(np.zeros_like(rest), phi, qa[lo : lo + 32] - baseline)
        delta = recover(np.zeros_like(rest), phi, qa[lo : lo + 32] - qb[lo : lo + 32])
        response.extend(np.linalg.norm(dis, axis=-1).max(axis=1))
        causal.extend(np.linalg.norm(delta, axis=-1).max(axis=1))
        distal.extend(np.stack([np.linalg.norm(delta[:, patch], axis=-1).max(axis=1) for patch in patches], axis=1))
    response, causal, distal = map(np.asarray, (response, causal, distal))
    load = ((ts >= 1) & (ts <= 2.7)) | ((ts >= 4.5) & (ts <= 6.2))
    recovered = ts >= 8.0
    peak = float(response[load].max())
    residual = float(response[recovered].max())
    causal_peak = float(causal[load].max())
    patch_peak = distal[load].max(axis=0)
    contact = sa["contacts"][:, 1]
    hold = ((ts >= 2.15) & (ts <= 2.5)) | ((ts >= 5.65) & (ts <= 6.0))
    checks = {
        "response_3mm": peak >= 0.003,
        "causal_response_3mm": causal_peak >= 0.003,
        "distal_three_patches_1mm": int((patch_peak >= 0.001).sum()) >= 3,
        "recovery_90pct": residual <= 0.1 * peak,
        "recovery_1mm": residual <= 0.001,
        "negative_no_pusher_contacts": int(sb["contacts"][:, 1].max()) == 0,
        "sustained_hold_contact": bool((contact[hold] > 0).all()),
    }
    np.savez_compressed(
        main / "paired_response.npz",
        times=ts,
        baseline_response=response,
        contact_causal_response=causal,
        distal_patch_response=distal,
    )
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "response_m": peak,
        "causal_response_m": causal_peak,
        "recovery_residual_m": residual,
        "recovery_fraction": 1 - residual / max(peak, 1e-30),
        "distal_spatial_patch_peaks_m": patch_peak.tolist(),
        "distal_patch_definition": "Rest z>90mm, x bands [-60,-15,8,30,60]mm; spatial regions, not anatomical labels",
        "hold_contact_min": int(contact[hold].min()),
    }
