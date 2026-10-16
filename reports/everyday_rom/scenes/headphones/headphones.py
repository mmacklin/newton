# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Spread and release dynamic over-ear headphones with a jointed VBD cable.

Only the contact paddles and a visible carrying gripper are prescribed. The
finite-mass headband, earcups, cable, and jack are advanced by Newton SolverVBD.
A real fixed support joint releases the whole assembly to the floor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp
from capacity_audit import CapacityAudit
from headphone_geometry import (
    CROWN_Z,
    CUP_X,
    CUP_Z,
    RADIUS,
    box,
    build_headband,
    capsule,
    cushion,
    ellipsoid,
    transform_points,
)
from test_headphones import GeometryTests, audit_export, audit_rejects_corruption

import newton
import newton.solvers
import newton.utils


@wp.func
def smooth(t: float, start: float, duration: float):
    s = wp.clamp((t - start) / duration, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@wp.func
def paddle_x(t: float):
    return 0.040 + 0.067 * smooth(t, 1.2, 1.4) * (1.0 - smooth(t, 3.1, 0.5))


@wp.kernel
def drive_paddles(
    clock: wp.array[float],
    dt: float,
    bodies: wp.array[int],
    z: float,
    poses0: wp.array[wp.transform],
    poses1: wp.array[wp.transform],
    velocities0: wp.array[wp.spatial_vector],
    velocities1: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    sign = float(2 * i - 1)
    t = clock[0]
    pose = wp.transform(wp.vec3(sign * paddle_x(t), 0.0, z), wp.quat_identity())
    velocity = wp.spatial_vector(sign * (paddle_x(t + dt) - paddle_x(t)) / dt, 0.0, 0.0, 0.0, 0.0, 0.0)
    body = bodies[i]
    poses0[body] = pose
    poses1[body] = pose
    velocities0[body] = velocity
    velocities1[body] = velocity


@wp.kernel
def tick(clock: wp.array[float], dt: float):
    clock[0] += dt


@wp.func
def gripper_position(t: float):
    return wp.vec3(0.0, -0.18 * smooth(t, 10.0, 1.0), 0.4 + 0.08 * smooth(t, 11.2, 0.5))


@wp.kernel
def drive_gripper(
    clock: wp.array[float],
    dt: float,
    body: int,
    joint: int,
    enabled: wp.array[wp.bool],
    poses0: wp.array[wp.transform],
    poses1: wp.array[wp.transform],
    velocities0: wp.array[wp.spatial_vector],
    velocities1: wp.array[wp.spatial_vector],
):
    t = clock[0]
    p = gripper_position(t)
    v = (gripper_position(t + dt) - p) / dt
    xform = wp.transform(p, wp.quat_identity())
    velocity = wp.spatial_vector(v, wp.vec3(0.0))
    poses0[body] = xform
    poses1[body] = xform
    velocities0[body] = velocity
    velocities1[body] = velocity
    if t >= 11.2:
        enabled[joint] = False


def pose(position):
    return wp.transform(wp.vec3(*position), wp.quat_identity())


def shape_config(*, collision=True, density=0.0):
    return newton.ModelBuilder.ShapeConfig(
        density=density,
        ke=3.0e4,
        kd=15.0,
        mu=0.45,
        gap=0.0005,
        margin=0.0,
        has_shape_collision=collision,
        has_particle_collision=False,
    )


def box_sdf(points, body_pose, half_extents):
    """Measure signed distance to the actual actuator box [m]."""
    q = np.asarray(body_pose[3:], dtype=float)
    inv = np.r_[np.zeros(3), -q[:3], q[3]]
    local = transform_points(np.asarray(points) - body_pose[:3], inv)
    d = np.abs(local) - half_extents
    return np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(np.max(d, axis=1), 0)


class Example:
    @dataclass
    class Config:
        substeps: int = 4
        iterations: int = 20
        contact_enabled: bool = True
        cable: bool = True
        device: str = "cuda:0"
        duration: float = 15.0

    limitations = (
        "Generic procedural headphones; the 240 N/m side stiffness is illustrative, not material-calibrated.",
        "Two linear curved-beam modes with infinitesimal cross-section rotations; no large-fold, buckling, or fracture model.",
        "A prescribed gripper carries the finite-mass arch through a real fixed joint, then disables that joint at 11.2 s.",
        "Earcup housings and cushions are rigid. Cushion compression and acoustic behavior are not modeled.",
        "The headband, housing, cushions, yokes, pins, cable, jack, and stand have collision geometry. The mount is filtered against its attached band.",
        "Joint-connected neighbors are collision-filtered. The cable has no cable-cable self-contact.",
        "Cable is a short 12-segment VBD rod with assumed joint stiffness and a dynamic 12 g jack, not a calibrated cable.",
        "Mass and inertia use explicit arc-length/cross-section sample quadrature; the ROM is not a volumetric FEM reduction.",
        "Modal damping is specified relative to the unloaded band; the heavy jointed earcups reduce the effective damping ratio.",
    )

    def __init__(self, viewer=None, args=None, *, config=None):
        self.cfg = c = config or self.Config()
        if c.substeps <= 0 or c.substeps % 2:
            raise ValueError("Use a positive even number of substeps for graph replay")
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / c.substeps
        self.sim_time = 0.0
        self.visuals, self.history, self.body_states, self.joint_states = [], [], [], []
        self.body_velocities, self.joint_velocities, self.graph_times = [], [], []
        self.support_states = []
        self.vertices, self.faces, self.basis, self.basis_audit = build_headband()
        b = newton.ModelBuilder(gravity=(0, 0, -9.81), up_axis="Z")
        b.num_rigid_contacts_per_world = 4096
        ground_cfg = shape_config()
        # Capture approaching impacts before a 4.17 ms step crosses the plane.
        # Gap changes contact generation, not the plane's zero-height surface.
        ground_cfg.gap = 0.010
        self.ground_shape = b.add_ground_plane(cfg=ground_cfg)
        self.band = b.add_body_elastic(
            xform=pose((0, 0, CROWN_Z)),
            mass=self.basis_audit["mass_kg"],
            com=wp.vec3(*self.basis_audit["com_m"]),
            inertia=wp.mat33(np.asarray(self.basis_audit["inertia_kg_m2"]).flatten()),
            modal_basis=self.basis,
            is_kinematic=False,
            lock_inertia=True,
            label="dynamic_headband",
        )
        self.band_shape = self.add_mesh(
            b,
            "elastic_headband",
            self.band,
            self.vertices,
            self.faces,
            (0.035, 0.25, 0.24),
            collision=True,
            elastic=True,
        )
        self.support = b.add_body(xform=pose((0, 0, CROWN_Z)), is_kinematic=True, label="visible_carrying_gripper")
        self.support_joint = b.add_joint_fixed(
            self.support,
            self.band,
            parent_xform=pose((0, 0, 0)),
            child_xform=pose((0, 0, 0)),
            label="releasable_crown_support",
        )
        v, f = box((0.014, 0.055, 0.008), center=(0, 0.051, 0.0105))
        self.add_mesh(b, "crown_gripper", self.support, v, f, (0.49, 0.53, 0.56), collision=True, metallic=0.6)
        self.cups, self.cup_joints, self.cup_shapes = [], [], []
        for side, label in ((-1, "left"), (1, "right")):
            mass = 0.115
            radii = np.array([0.018, 0.034, 0.047])
            diagonal = mass / 5 * (np.sum(radii**2) - radii**2)
            cup = b.add_body(
                xform=pose((side * CUP_X, 0, CUP_Z)),
                mass=mass,
                inertia=wp.mat33(np.diag(diagonal).flatten()),
                lock_inertia=True,
                label=f"dynamic_{label}_earcup",
            )
            self.cups.append(cup)
            v, f = ellipsoid(radii)
            shape = b.add_shape_ellipsoid(
                cup, rx=radii[0], ry=radii[1], rz=radii[2], cfg=shape_config(), color=(0.035, 0.26, 0.25)
            )
            self.visuals.append(
                {
                    "name": f"{label}_housing",
                    "vertices": v,
                    "faces": f,
                    "body": cup,
                    "color": (0.035, 0.26, 0.25),
                    "shape": shape,
                    "collision": True,
                }
            )
            self.cup_shapes.append(shape)
            v, f = cushion(side)
            self.cup_shapes.append(
                self.add_mesh(b, f"{label}_oval_cushion", cup, v, f, (0.045, 0.052, 0.059), collision=True)
            )
            v, f = ellipsoid((0.0007, 0.019, 0.029), center=(-side * 0.020, 0, 0))
            self.add_mesh(b, f"{label}_speaker_fabric", cup, v, f, (0.11, 0.13, 0.14))
            v, f = ellipsoid((0.0012, 0.021, 0.031), center=(side * 0.017, 0, 0))
            self.add_mesh(b, f"{label}_outer_inlay", cup, v, f, (0.12, 0.48, 0.44), metallic=0.4)
            v, f = box((0.004, 0.009, 0.019), center=(side * 0.008, 0, 0.037))
            self.add_mesh(b, f"{label}_yoke", cup, v, f, (0.53, 0.58, 0.60), collision=True, metallic=0.8)
            v, f = capsule(0.006, 0.015)
            v = v[:, [0, 2, 1]] + (0, 0, 0.051)
            f = f[:, [0, 2, 1]]
            self.add_mesh(b, f"{label}_pivot_pin", cup, v, f, (0.65, 0.70, 0.72), collision=True, metallic=0.8)
            joint = b.add_joint_revolute(
                self.band,
                cup,
                parent_xform=pose((side * RADIUS, 0, -RADIUS)),
                child_xform=pose((0, 0, 0.051)),
                axis=(0, 0, 1),
                target_ke=0.04,
                target_kd=0.004,
                limit_lower=-0.22,
                limit_upper=0.22,
                limit_ke=1.0,
                limit_kd=0.01,
                label=f"{label}_earcup_swivel",
            )
            self.cup_joints.append(joint)

        self.paddles, self.paddle_shapes = [], []
        self.paddle_half = np.array([0.008, 0.038, 0.032])
        for side, label in ((-1, "left"), (1, "right")):
            actuator = b.add_body(
                xform=pose((side * 0.040, 0, CUP_Z)), is_kinematic=True, label=f"prescribed_{label}_spreading_paddle"
            )
            self.paddles.append(actuator)
            shape = b.add_shape_box(
                actuator,
                hx=self.paddle_half[0],
                hy=self.paddle_half[1],
                hz=self.paddle_half[2],
                cfg=shape_config(collision=c.contact_enabled),
                color=(0.84, 0.22, 0.12),
            )
            self.paddle_shapes.append(shape)
            v, f = box(self.paddle_half)
            self.visuals.append(
                {
                    "name": f"{label}_contact_paddle",
                    "vertices": v,
                    "faces": f,
                    "body": actuator,
                    "color": (0.84, 0.22, 0.12),
                    "shape": shape,
                    "collision": c.contact_enabled,
                }
            )
            v, f = box((0.004, 0.004, 0.075), center=(0, 0.030, -0.107))
            self.add_mesh(b, f"{label}_paddle_arm", actuator, v, f, (0.64, 0.66, 0.68), metallic=0.7)

        # The gripper carries the object forward before release to clear this fixture.
        for label, half, center in (
            ("stand_base", (0.16, 0.075, 0.008), (0, 0.085, 0.008)),
            ("stand_post", (0.009, 0.009, 0.202), (0, 0.106, 0.218)),
            ("actuator_rail", (0.15, 0.015, 0.008), (0, 0.03, 0.052)),
        ):
            v, f = box(half, center=center)
            self.add_mesh(b, label, -1, v, f, (0.49, 0.53, 0.56), collision=True, metallic=0.6)

        self.cable_bodies, self.cable_joints, self.jack = [], [], None
        if c.cable:
            self.build_cable(b)
        b.color()
        self.model = b.finalize(device=c.device)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(
            self.model, contact_matching="latest", reduce_contacts=False, rigid_contact_max=16384
        )
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=c.iterations,
            rigid_compliant_alm=True,
            rigid_contact_history=True,
            rigid_joint_linear_ke=1.0e5,
            rigid_joint_angular_ke=100.0,
            rigid_joint_linear_kd=1.0,
            rigid_joint_angular_kd=0.01,
            rigid_body_contact_buffer_size=2048,
            elastic_body_relaxation=0.8,
            elastic_solver_metrics=True,
        )
        elastic = int(self.model.body_elastic_index.numpy()[self.band])
        owner = int(self.model.elastic_joint.numpy()[elastic])
        self.q_start = int(self.model.joint_q_start.numpy()[owner]) + 7
        self.vertex_phi = self.model.elastic_shape_vertex_phi.numpy().reshape(-1, 2, 3)
        np.testing.assert_allclose(self.vertex_phi, self.basis.sample_phi[: len(self.vertices)], atol=1e-6)
        np.testing.assert_allclose(self.model.elastic_shape_vertex_local.numpy(), self.vertices, atol=0)
        self.endpoint_phi = self.model.elastic_endpoint_phi.numpy().reshape(-1, 2, 3)
        self.parent_endpoints = self.model.joint_parent_elastic_endpoint.numpy()
        self.joint_parents = self.model.joint_parent.numpy()
        self.joint_children = self.model.joint_child.numpy()
        self.joint_Xp = self.model.joint_X_p.numpy()
        self.joint_Xc = self.model.joint_X_c.numpy()
        self.dynamic_bodies = np.array(
            [self.band, *self.cups, *self.cable_bodies, *([] if self.jack is None else [self.jack])], dtype=int
        )
        self.paddle_ids = wp.array(self.paddles, dtype=int, device=self.model.device)
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.capacity_audit = CapacityAudit(self.model, self.contacts, self.solver, self.clock, self.sim_dt)
        self.graph = None
        self.graph_launch_count = 0
        if viewer is not None:
            viewer.set_model(self.model)
            viewer.show_elastic_strain = False
        capture_start = time.perf_counter()
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph
        self.capture_seconds = time.perf_counter() - capture_start
        self.record()
        self.initial_clearances = self.clearances(self.body_states[0], self.joint_states[0])

    def release(self):
        """Release the physical support early without prescribing object state."""
        enabled = self.model.joint_enabled.numpy()
        enabled[self.support_joint] = False
        self.model.joint_enabled.assign(enabled)

    def add_mesh(self, builder, name, body, vertices, faces, color, *, collision=False, elastic=False, metallic=0.0):
        shape = builder.add_shape_mesh(
            body,
            mesh=newton.Mesh(vertices, faces.flatten()),
            cfg=shape_config(collision=collision),
            color=color,
            label=name,
        )
        self.visuals.append(
            {
                "name": name,
                "vertices": vertices,
                "faces": faces,
                "body": body,
                "color": color,
                "shape": shape,
                "collision": collision,
                "elastic": elastic,
                "metallic": metallic,
            }
        )
        return shape

    def build_cable(self, b):
        anchor_local = np.array([0, -0.016, -0.047])
        anchor = np.array([-CUP_X, 0, CUP_Z]) + anchor_local
        points = anchor + np.linspace(0, 1, 13)[:, None] * np.array([0, 0, -0.140])
        v, f = capsule(0.0027, 0.006)
        v += (0, -0.016, -0.044)
        self.add_mesh(b, "left_cable_strain_relief", self.cups[0], v, f, (0.045, 0.052, 0.059), collision=True)
        quaternions = newton.utils.rod_parallel_transport_quaternions([wp.vec3(*p) for p in points])
        self.cable_bodies, self.cable_joints = b.add_rod(
            positions=points.tolist(),
            quaternions=quaternions,
            radius=0.0017,
            cfg=shape_config(density=1200),
            stretch_stiffness=2.0e4,
            stretch_damping=0.5,
            bend_stiffness=0.0003,
            bend_damping=0.003,
            label="headphone_cable",
            body_frame_origin="com",
            color=(0.028, 0.033, 0.038),
        )
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        for i, (body, length) in enumerate(zip(self.cable_bodies, lengths, strict=True)):
            v, f = capsule(0.0017, length / 2)
            self.visuals.append(
                {
                    "name": f"cable_segment_{i:02}",
                    "vertices": v,
                    "faces": f,
                    "body": body,
                    "color": (0.028, 0.033, 0.038),
                    "collision": True,
                    "shape": b.body_shapes[body][0],
                }
            )
        self.cable_joints.append(
            b.add_joint_fixed(
                self.cups[0],
                self.cable_bodies[0],
                parent_xform=wp.transform(wp.vec3(*anchor_local), quaternions[0]),
                child_xform=pose((0, 0, -lengths[0] / 2)),
                label="cable_to_left_earcup",
            )
        )
        direction = (points[-1] - points[-2]) / lengths[-1]
        jack_pose = wp.transform(wp.vec3(*(points[-1] + 0.012 * direction)), quaternions[-1])
        self.jack = b.add_body(
            xform=jack_pose,
            mass=0.012,
            inertia=wp.mat33(7e-7, 0, 0, 0, 7e-7, 0, 0, 0, 1e-7),
            lock_inertia=True,
            label="dynamic_audio_jack",
        )
        shape = b.add_shape_capsule(
            self.jack, radius=0.004, half_height=0.008, cfg=shape_config(), color=(0.12, 0.14, 0.15)
        )
        v, f = capsule(0.004, 0.008)
        self.visuals.append(
            {
                "name": "jack_grip",
                "vertices": v,
                "faces": f,
                "body": self.jack,
                "color": (0.12, 0.14, 0.15),
                "shape": shape,
                "collision": True,
            }
        )
        v, f = capsule(0.00175, 0.005)
        v[:, 2] += 0.016
        self.add_mesh(b, "jack_metal_plug", self.jack, v, f, (0.76, 0.72, 0.49), collision=True, metallic=0.9)
        self.cable_joints.append(
            b.add_joint_fixed(
                self.cable_bodies[-1],
                self.jack,
                parent_xform=pose((0, 0, lengths[-1] / 2)),
                child_xform=pose((0, 0, -0.012)),
                label="cable_to_dynamic_jack",
            )
        )
        for i, body in enumerate(self.cable_bodies):
            for other in self.cable_bodies[i + 1 :]:
                for s in b.body_shapes[body]:
                    for t in b.body_shapes[other]:
                        b.add_shape_collision_filter_pair(s, t)

    def simulate(self):
        for _ in range(self.cfg.substeps):
            wp.launch(
                drive_gripper,
                dim=1,
                inputs=[
                    self.clock,
                    self.sim_dt,
                    self.support,
                    self.support_joint,
                    self.model.joint_enabled,
                    self.state_0.body_q,
                    self.state_1.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_qd,
                ],
                device=self.model.device,
            )
            wp.launch(
                drive_paddles,
                dim=2,
                inputs=[
                    self.clock,
                    self.sim_dt,
                    self.paddle_ids,
                    CUP_Z,
                    self.state_0.body_q,
                    self.state_1.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_qd,
                ],
                device=self.model.device,
            )
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.capacity_audit.after_collide()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.capacity_audit.after_solve()
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(tick, dim=1, inputs=[self.clock, self.sim_dt], device=self.model.device)

    def step(self):
        begin = time.perf_counter()
        if self.graph is not None:
            wp.capture_launch(self.graph)
            self.graph_launch_count += 1
            wp.synchronize_device(self.model.device)
        else:
            self.simulate()
        self.graph_times.append(time.perf_counter() - begin)
        self.sim_time += self.frame_dt
        self.record()

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

    def world_mesh(self, mesh, body_q, joint_q):
        local = mesh["vertices"]
        if mesh.get("elastic"):
            local = local + np.einsum("nmc,m->nc", self.vertex_phi, joint_q[self.q_start : self.q_start + 2])
        return local.copy() if mesh["body"] < 0 else transform_points(local, body_q[mesh["body"]])

    def joint_gap(self, joint, body_q, joint_q):
        parent, child = self.joint_parents[joint], self.joint_children[joint]
        p = self.joint_Xp[joint, :3].copy()
        endpoint = self.parent_endpoints[joint]
        if endpoint >= 0:
            p += np.einsum("mc,m->c", self.endpoint_phi[endpoint], joint_q[self.q_start : self.q_start + 2])
        p = transform_points(p[None], body_q[parent])[0] if parent >= 0 else p
        c = transform_points(self.joint_Xc[joint, :3][None], body_q[child])[0]
        return float(np.linalg.norm(p - c))

    def clearances(self, body_q, joint_q):
        minimum_paddle = float("inf")
        floor = float("inf")
        minimum_cup_gap = float("inf")
        cup_bounds = []
        for cup, paddle in zip(self.cups, self.paddles, strict=True):
            samples = []
            for mesh in self.visuals:
                if mesh["body"] == cup and mesh.get("collision"):
                    world = self.world_mesh(mesh, body_q, joint_q)
                    tri = world[mesh["faces"]]
                    samples.extend(
                        (
                            world,
                            tri.mean(axis=1),
                            (tri[:, 0] + tri[:, 1]) / 2,
                            (tri[:, 1] + tri[:, 2]) / 2,
                            (tri[:, 0] + tri[:, 2]) / 2,
                        )
                    )
            samples = np.concatenate(samples)
            cup_bounds.append((samples[:, 0].min(), samples[:, 0].max()))
            minimum_paddle = min(minimum_paddle, float(box_sdf(samples, body_q[paddle], self.paddle_half).min()))
        minimum_cup_gap = float(cup_bounds[1][0] - cup_bounds[0][1])
        for mesh in self.visuals:
            if mesh["body"] >= 0 and mesh["body"] not in self.paddles:
                floor = min(floor, float(self.world_mesh(mesh, body_q, joint_q)[:, 2].min()))
        return {
            "paddle_surface_clearance_m": minimum_paddle,
            "floor_clearance_m": floor,
            "earcup_gap_m": minimum_cup_gap,
        }

    def record(self):
        body_q, joint_q = self.state_0.body_q.numpy(), self.state_0.joint_q.numpy()
        body_qd, joint_qd = self.state_0.body_qd.numpy(), self.state_0.joint_qd.numpy()
        finite = all(np.isfinite(a).all() for a in (body_q, joint_q, body_qd, joint_qd))
        if not finite:
            raise AssertionError(f"Nonfinite state at {self.sim_time:g}s")
        q = joint_q[self.q_start : self.q_start + 2]
        disp = np.einsum("nmc,m->nc", self.vertex_phi, q)
        local = self.vertices + disp
        rest_rings = self.vertices[:-2].reshape(81, 16, 3).mean(axis=1)
        rings = local[:-2].reshape(81, 16, 3).mean(axis=1)
        strain = (
            np.linalg.norm(np.diff(rings, axis=0), axis=1) / np.linalg.norm(np.diff(rest_rings, axis=0), axis=1) - 1
        )
        clearances = self.clearances(body_q, joint_q)
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        s0 = self.contacts.rigid_contact_shape0.numpy()[:count]
        s1 = self.contacts.rigid_contact_shape1.numpy()[:count]
        cup_contact = (np.isin(s0, self.paddle_shapes) & np.isin(s1, self.cup_shapes)) | (
            np.isin(s1, self.paddle_shapes) & np.isin(s0, self.cup_shapes)
        )
        support_enabled = bool(self.model.joint_enabled.numpy()[self.support_joint])
        self.support_states.append(support_enabled)
        self.history.append(
            {
                "time": self.sim_time,
                "finite": bool(finite),
                "q": q.tolist(),
                "deformation_m": float(np.linalg.norm(disp, axis=1).max()),
                "cup_spread_m": float(body_q[self.cups[1], 0] - body_q[self.cups[0], 0]),
                "arch_centerline_strain": float(np.abs(strain).max()),
                "pivot_gap_m": max(self.joint_gap(j, body_q, joint_q) for j in self.cup_joints),
                "cable_joint_gap_m": max((self.joint_gap(j, body_q, joint_q) for j in self.cable_joints), default=0.0),
                "contacts": count,
                "paddle_contacts": int(cup_contact.sum()),
                "floor_contacts": int(((s0 == self.ground_shape) | (s1 == self.ground_shape)).sum()),
                "support_enabled": support_enabled,
                "support_gap_m": self.joint_gap(self.support_joint, body_q, joint_q),
                "band_position_m": body_q[self.band, :3].tolist(),
                "max_assembly_linear_speed_m_s": float(np.linalg.norm(body_qd[self.dynamic_bodies, :3], axis=1).max()),
                "modal_solve": {k: float(v[0]) for k, v in self.solver.elastic_mode_solve_metrics().items()},
                **clearances,
            }
        )
        self.body_states.append(body_q)
        self.joint_states.append(joint_q)
        self.body_velocities.append(body_qd)
        self.joint_velocities.append(joint_qd)

    def summary(self):
        h = self.history
        baseline = [r for r in h if 0.8 <= r["time"] <= 1.15]
        settled_start = 9.5
        final = [r for r in h if settled_start <= r["time"] <= 10.0]
        drop = [r for r in h if not r["support_enabled"]]
        landed = [r for r in drop if r["time"] >= self.cfg.duration - 0.5]
        baseline_q = np.mean([r["q"] for r in baseline], axis=0) if baseline else np.zeros(2)
        increments = [
            float(np.linalg.norm(np.einsum("nmc,m->nc", self.vertex_phi, np.array(r["q"]) - baseline_q), axis=1).max())
            for r in h
        ]
        peak = max((value for value, row in zip(increments, h, strict=True) if 1.2 <= row["time"] <= 10.0), default=0.0)
        recovery = max(
            (increments[i] for i, r in enumerate(h) if settled_start <= r["time"] <= 10.0), default=float("inf")
        )
        return {
            "finite": all(r["finite"] for r in h),
            "cuda_graph": self.graph is not None,
            "graph_launch_count": self.graph_launch_count,
            "frames": len(h),
            "device": str(self.model.device),
            "newton_source": str(Path(newton.__file__).resolve()),
            "duration_s": self.sim_time,
            "settled_window_s": [settled_start, 10.0],
            "max_deformation_m": max(r["deformation_m"] for r in h),
            "max_displacement_m": max(r["deformation_m"] for r in h),
            "final_displacement_m": h[-1]["deformation_m"],
            "max_penetration_m": max(
                0.0, -min(min(r["paddle_surface_clearance_m"], r["floor_clearance_m"]) for r in h)
            ),
            "metric_definitions": {
                "displacement": "Maximum modal vertex displacement from the undeformed mesh in the body frame; excludes rigid carry/drop motion.",
                "penetration": "Maximum measured collision-surface penetration into the paddles or floor across all saved frames.",
                "recovery": "Modal displacement relative to the gravity-loaded pre-actuation equilibrium, measured before carry/drop.",
            },
            "max_deformation_from_loaded_equilibrium_m": peak,
            "recovery_error_m": recovery,
            "recovery_fraction": 1 - recovery / max(peak, 1e-12),
            "cup_spread_increase_m": max((r["cup_spread_m"] for r in h if 1.2 <= r["time"] <= 10.0), default=0)
            - np.mean([r["cup_spread_m"] for r in baseline])
            if baseline
            else 0,
            "max_pivot_gap_m": max(r["pivot_gap_m"] for r in h),
            "max_cable_joint_gap_m": max(r["cable_joint_gap_m"] for r in h),
            "max_paddle_penetration_m": max(0.0, -min(r["paddle_surface_clearance_m"] for r in h)),
            "max_floor_penetration_m": max(0.0, -min(r["floor_clearance_m"] for r in h)),
            "max_arch_centerline_strain": max(r["arch_centerline_strain"] for r in h),
            "peak_paddle_contacts": max(r["paddle_contacts"] for r in h),
            "contact_frames": sum(r["paddle_contacts"] > 0 for r in h),
            "settled_cup_spread_jitter_m": float(np.ptp([r["cup_spread_m"] for r in final])) if final else float("inf"),
            "settled_paddle_contacts": sum(r["paddle_contacts"] for r in final),
            "max_relative_modal_solve_residual": max(
                r["modal_solve"]["solve_residual_norm"] / max(r["modal_solve"]["initial_residual_norm"], 1e-12)
                for r in h
            ),
            "capture_s": self.capture_seconds,
            "warm_graph_step_ms": 1000 * float(np.median(self.graph_times[10:]))
            if len(self.graph_times) > 10
            else None,
            "initial_clearances": self.initial_clearances,
            "basis": self.basis_audit,
            "drop": {
                "release_observed": bool(drop),
                "release_time_s": drop[0]["time"] if drop else None,
                "support_enabled_final": h[-1]["support_enabled"],
                "max_supported_gap_m": max(r["support_gap_m"] for r in h if r["support_enabled"]),
                "gripper_carried_distance_m": float(-min(r["band_position_m"][1] for r in h if r["support_enabled"])),
                "band_drop_m": max((CROWN_Z - r["band_position_m"][2] for r in drop), default=0.0),
                "floor_contact_frames": sum(r["floor_contacts"] > 0 for r in drop),
                "final_floor_contact_frames": sum(r["floor_contacts"] > 0 for r in landed),
                "peak_assembly_speed_m_s": max((r["max_assembly_linear_speed_m_s"] for r in drop), default=0.0),
                "settled_max_assembly_speed_m_s": max(
                    (r["max_assembly_linear_speed_m_s"] for r in landed), default=0.0
                ),
                "band_mass_kg": float(self.model.body_mass.numpy()[self.band]),
                "band_is_dynamic": not bool(
                    int(self.model.body_flags.numpy()[self.band]) & int(newton.BodyFlags.KINEMATIC)
                ),
            },
            "physics_parameters": {
                "gravity_m_s2": [0, 0, -9.81],
                "contact_stiffness_N_m": float(self.model.shape_material_ke.numpy()[self.paddle_shapes[0]]),
                "contact_damping_Ns_m": float(self.model.shape_material_kd.numpy()[self.paddle_shapes[0]]),
                "friction": float(self.model.shape_material_mu.numpy()[self.paddle_shapes[0]]),
                "floor_contact_generation_gap_m": float(self.model.shape_gap.numpy()[self.ground_shape]),
                "joint_linear_stiffness_N_m": self.solver.rigid_joint_linear_ke,
                "joint_angular_stiffness_Nm_rad": self.solver.rigid_joint_angular_ke,
                "modal_damping": self.basis.mode_damping.tolist(),
                "earcup_mass_kg": self.model.body_mass.numpy()[self.cups].tolist(),
                "jack_mass_kg": float(self.model.body_mass.numpy()[self.jack]) if self.jack is not None else 0.0,
                "rod_bend_stiffness_Nm_rad": 0.0003,
                "rod_bend_damping_Nms_rad": 0.003,
                "rod_stretch_stiffness_N_m": 2e4,
                "rod_stretch_damping_Ns_m": 0.5,
            },
        }

    def test_final(self, *, require_actuation=None):
        """Reject instability, missing contact actuation, loose joints, or poor recovery."""
        self.capacity_audit.check((len(self.history) - 1) * self.cfg.substeps)
        m = self.summary()
        checks = {
            "finite": m["finite"],
            "cuda_graph": m["cuda_graph"] and m["graph_launch_count"] == m["frames"] - 1,
            "complete_sequence": m["duration_s"] >= 14.9,
            "valid_initial_clearance": min(m["initial_clearances"].values()) > 0.002,
            "orthogonal_basis": m["basis"]["mass_orthogonality_error"] < 2e-6,
            "pivot_attachment": m["max_pivot_gap_m"] < 0.002,
            "cable_attachment": m["max_cable_joint_gap_m"] < 0.003,
            "floor_stability": m["max_floor_penetration_m"] < 0.003,
            "bounded_strain": m["max_arch_centerline_strain"] < 0.12,
            "settled": m["settled_cup_spread_jitter_m"] < 0.001,
            "dynamic_headband": m["drop"]["band_is_dynamic"] and m["drop"]["band_mass_kg"] > 0,
            "real_support": m["drop"]["max_supported_gap_m"] < 0.002,
            "carried": m["drop"]["gripper_carried_distance_m"] > 0.15,
            "support_released": m["drop"]["release_observed"] and not m["drop"]["support_enabled_final"],
            "dropped": m["drop"]["band_drop_m"] > 0.12 and m["drop"]["peak_assembly_speed_m_s"] > 0.8,
            "ground_contact": m["drop"]["floor_contact_frames"] > 30 and m["drop"]["final_floor_contact_frames"] > 20,
            "landed_stably": m["drop"]["settled_max_assembly_speed_m_s"] < 0.15,
        }
        if self.cfg.contact_enabled if require_actuation is None else require_actuation:
            checks.update(
                contact_driven=m["peak_paddle_contacts"] >= 2 and m["contact_frames"] >= 30,
                deformed=m["max_deformation_from_loaded_equilibrium_m"] > 0.008,
                spread=m["cup_spread_increase_m"] > 0.015,
                recovered=m["recovery_error_m"] < 0.002 and m["recovery_fraction"] > 0.85,
                contact_stability=m["max_paddle_penetration_m"] < 0.002,
                fully_released=m["settled_paddle_contacts"] == 0,
            )
        else:
            checks["interaction_disabled"] = m["peak_paddle_contacts"] == 0
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise AssertionError(f"Failed {failed}: {json.dumps(m)}")
        return {k: bool(v) for k, v in checks.items()}

    def export(self, directory, metrics):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        states = {
            "times": np.array([r["time"] for r in self.history]),
            "body_q": np.asarray(self.body_states),
            "joint_q": np.asarray(self.joint_states),
            "body_qd": np.asarray(self.body_velocities),
            "joint_qd": np.asarray(self.joint_velocities),
            "q_start": np.array(self.q_start),
            "band_body": np.array(self.band),
            "support_body": np.array(self.support),
            "support_joint": np.array(self.support_joint),
            "support_enabled": np.asarray(self.support_states, dtype=bool),
            "sample_points": self.basis.sample_points,
            "sample_phi": self.basis.sample_phi,
            "sample_psi": self.basis.sample_psi,
            "sample_mass": self.basis.sample_mass,
            "mode_mass": self.basis.mode_mass,
            "mode_stiffness": self.basis.mode_stiffness,
            "mode_damping": self.basis.mode_damping,
            "runtime_vertex_phi": self.vertex_phi,
            "runtime_vertex_local": self.model.elastic_shape_vertex_local.numpy(),
            "runtime_vertex_sample": self.model.elastic_shape_vertex_sample.numpy(),
            "body_mass": self.model.body_mass.numpy(),
            "body_flags": self.model.body_flags.numpy(),
            "body_com": self.model.body_com.numpy(),
            "body_inertia": self.model.body_inertia.numpy(),
            "cup_bodies": np.array(self.cups),
            "paddle_bodies": np.array(self.paddles),
            "cable_bodies": np.array(self.cable_bodies, dtype=np.int32),
            "jack_body": np.array(-1 if self.jack is None else self.jack),
            "joint_parent": self.joint_parents,
            "joint_child": self.joint_children,
            "joint_X_p": self.joint_Xp,
            "joint_X_c": self.joint_Xc,
            "joint_type": self.model.joint_type.numpy(),
            "cup_joints": np.array(self.cup_joints),
            "cable_joints": np.array(self.cable_joints, dtype=np.int32),
            "endpoint_phi": self.endpoint_phi,
            "parent_endpoints": self.parent_endpoints,
        }
        for name in (
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_mu",
            "shape_gap",
            "shape_margin",
            "joint_target_ke",
            "joint_target_kd",
            "joint_q_start",
            "joint_qd_start",
        ):
            states[name] = getattr(self.model, name).numpy()
        scene = {"times": states["times"]}
        descriptors = []
        for i, mesh in enumerate(self.visuals):
            body = mesh["body"]
            if body < 0:
                world = mesh["vertices"]
            else:
                world = np.asarray(
                    [self.world_mesh(mesh, bq, jq) for bq, jq in zip(self.body_states, self.joint_states, strict=True)]
                )
            scene[f"vertices_{i}"] = world
            scene[f"faces_{i}"] = mesh["faces"]
            states[f"rest_vertices_{i}"] = mesh["vertices"]
            states[f"faces_{i}"] = mesh["faces"]
            states[f"body_{i}"] = np.array(body)
            descriptors.append(
                {
                    "name": mesh["name"],
                    "vertices": f"vertices_{i}",
                    "faces": f"faces_{i}",
                    "color": list(mesh["color"]),
                    "roughness": 0.38,
                    "metallic": mesh.get("metallic", 0.0),
                    "body": body,
                    "elastic": mesh.get("elastic", False),
                    "collision": mesh.get("collision", False),
                }
            )
        np.savez_compressed(directory / "original_states.npz", **states)
        np.savez_compressed(directory / "scene.npz", **scene)
        metadata = {
            "slug": "headphones",
            "title": "Over-ear headphones: flex, carry, release, and drop",
            "status": "validated prototype" if metrics.get("all_tests_passed", False) else "WIP",
            "fps": self.fps,
            "meshes": descriptors,
            "floor_z": 0.0,
            "camera": {"position": [0.65, -1.03, 0.58], "target": [-0.015, -0.06, 0.21], "fov": 34},
            "description": "Paddles spread the jointed earcups. After spring-back, a real support joint carries the finite-mass headphones forward and releases the assembly onto the floor.",
            "model": "Dynamic two-mode elastic arch, two dynamic swivel-jointed earcups, and a releasable fixed support joint to a visible kinematic gripper."
            + (
                " Includes 12 VBD rod segments and a fixed-joint-attached dynamic audio jack." if self.cfg.cable else ""
            ),
            "limitations": [text for text in self.limitations if self.cfg.cable or "cable" not in text.lower()],
            "mode_count": 2,
            "parameters": asdict(self.cfg),
            "physics_parameters": metrics["physics_parameters"],
            "driven": "Only the visible paddles and carrying gripper are prescribed. At 11.2 s the real support joint is disabled; no headphone rigid or modal state is prescribed.",
            "base_commit": subprocess.run(
                ["git", "-C", str(Path(newton.__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip(),
            "gravity_m_s2": [0, 0, -9.81],
            "friction": 0.45,
            "source_files": ["headphones.py", "headphone_geometry.py", "test_headphones.py"],
            "reproduce": reproduction_command(directory),
            "state_proof": "original_states.npz: original body/joint positions and velocities, sample basis, runtime vertex basis, per-mesh rest vertices and body IDs",
            "phases_s": {
                "gravity_settle": [0, 1.2],
                "spread": [1.2, 2.6],
                "hold": [2.6, 3.1],
                "release": [3.1, 3.6],
                "recover": [3.6, 10.0],
                "carry": [10.0, 11.0],
                "drop": [11.2, self.cfg.duration],
            },
        }
        (directory / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        (directory / "history.json").write_text(json.dumps(self.history, indent=2) + "\n")
        for name in metadata["source_files"]:
            source = Path(__file__).with_name(name)
            shutil.copy2(source, directory / name)


def reproduction_command(directory):
    script = Path(__file__).resolve()
    root = Path(newton.__file__).resolve().parents[1]
    return (
        "flock /tmp/newton-everyday-rom-gpu.lock timeout 5m env "
        f'PYTHONPATH="{root}" OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 '
        "WARP_CACHE_PATH=/tmp/newton-everyday-headphones-warp "
        f'uv run --project "{root}" python "{script}" --output "{directory}"'
    )


def run(config):
    example = Example(config=config)
    for frame in range(round(config.duration * example.fps)):
        example.step()
        if (frame + 1) % 60 == 0:
            print(json.dumps(example.history[-1]), flush=True)
    return example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/headphones"))
    parser.add_argument("--no-cable", action="store_true")
    parser.add_argument(
        "--quick", action="store_true", help="Diagnostic run without negative control or validated status"
    )
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()
    if args.duration < 15.0:
        parser.error("The complete spread/carry/drop validation needs --duration >= 15 seconds")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("Geometry/basis unit checks failed")
    positive = run(Example.Config(cable=not args.no_cable, duration=args.duration))
    metrics = positive.summary()
    metrics["unit_tests"] = {"passed": result.wasSuccessful(), "run": result.testsRun}
    metrics["parameters"] = asdict(positive.cfg)
    metrics["tests"] = {}
    try:
        metrics["tests"] = positive.test_final()
    except AssertionError as error:
        metrics["test_final_error"] = str(error)
    if not args.quick:
        negative = run(Example.Config(contact_enabled=False, cable=not args.no_cable, duration=args.duration))
        metrics["negative_control"] = negative.summary()
        metrics["negative_control"]["tests"] = negative.test_final()
        rejected = False
        try:
            negative.test_final(require_actuation=True)
        except AssertionError:
            rejected = True
        metrics["negative_control"]["claimed_response_test_rejected"] = rejected
        prefix = np.array([r["time"] for r in positive.history]) <= 1.15
        prefix_error = float(
            np.max(np.abs(np.asarray(positive.body_states)[prefix] - np.asarray(negative.body_states)[prefix]))
        )
        metrics["negative_control"]["predrive_body_state_error"] = prefix_error
        ratio = metrics["negative_control"]["max_deformation_from_loaded_equilibrium_m"] / max(
            metrics["max_deformation_from_loaded_equilibrium_m"], 1e-12
        )
        metrics["negative_control"]["deformation_ratio"] = ratio
        metrics["negative_control"]["passed"] = bool(ratio < 0.25 and rejected and prefix_error < 2e-6)
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output / "negative_control.npz",
            body_q=negative.body_states,
            joint_q=negative.joint_states,
            body_qd=negative.body_velocities,
            joint_qd=negative.joint_velocities,
            times=[r["time"] for r in negative.history],
        )
        (args.output / "negative_history.json").write_text(json.dumps(negative.history, indent=2) + "\n")
    metrics["test_passed"] = False
    metrics["negative_control_passed"] = bool(metrics.get("negative_control", {}).get("passed", False))
    metrics["all_tests_passed"] = False
    positive.export(args.output, metrics)
    metrics["source_proof"] = audit_export(args.output)
    metrics["source_proof"]["corrupted_vertex_rejected"] = audit_rejects_corruption(args.output)
    metrics["test_passed"] = bool(
        metrics["tests"]
        and all(metrics["tests"].values())
        and metrics["unit_tests"]["passed"]
        and metrics["source_proof"]["passed"]
        and metrics["source_proof"]["corrupted_vertex_rejected"]
    )
    metrics["all_tests_passed"] = metrics["test_passed"] and metrics["negative_control_passed"]
    metrics["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("headphones.py", "headphone_geometry.py", "test_headphones.py")
    }
    metrics["artifact_sha256"] = {
        name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        for name in ("scene.npz", "original_states.npz")
    }
    positive.export(args.output, metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    if not args.quick and not metrics["all_tests_passed"]:
        raise AssertionError("Headphones validation failed; see metrics.json")


if __name__ == "__main__":
    main()
