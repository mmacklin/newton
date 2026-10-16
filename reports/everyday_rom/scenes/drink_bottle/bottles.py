# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact-driven squeezing of two open, empty, single-ROM bottle shells.

Run under the shared GPU flock; see scene.json for the complete command.
No modal coordinates, modal forces, or display amplification are prescribed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import warp as wp
from contact_capacity import GLOBAL_CAPACITY, ContactTrace

import newton
import newton.solvers

try:
    from .bottle_geometry import (
        SPECS,
        box_mesh,
        build_bottle_rom,
        build_shell,
        capsule_mesh,
        cavity_volume,
        signed_volume,
        triangle_reference,
        triangle_stretches,
    )
except ImportError:
    from bottle_geometry import (
        SPECS,
        box_mesh,
        build_bottle_rom,
        build_shell,
        capsule_mesh,
        cavity_volume,
        signed_volume,
        triangle_reference,
        triangle_stretches,
    )


RENDER_REVISION = "ring-aligned-collars-no-labels-v3"


@dataclass
class Config:
    slug: str = "drink-bottle"
    substeps: int = 4
    iterations: int = 16
    contact_enabled: bool = True
    pad_radius: float = 0.012
    pad_half_height: float = 0.020
    indentation: float = 0.005
    contact_ke: float = 3e3
    contact_kd: float = 0.002
    contact_gap: float = 0.00035
    ground_detection_gap: float = 0.005
    friction: float = 0.40
    damping_ratio: float = 0.20
    drop_height: float = 0.060
    relaxation: float = 0.25
    compliant_alm: bool = True
    frames: int = 360


@wp.func
def ramp(t: float, start: float, duration: float):
    s = wp.clamp((t - start) / duration, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@wp.func
def pad_distance(t: float, home: float, closed: float):
    amount = ramp(t, 0.6, 1.2) * (1.0 - ramp(t, 3.0, 0.10))
    return home + (closed - home) * amount


@wp.kernel
def drive_pads(
    clock: wp.array[float],
    dt: float,
    left: int,
    right: int,
    home: float,
    closed: float,
    height: float,
    pose0: wp.array[wp.transform],
    vel0: wp.array[wp.spatial_vector],
    pose1: wp.array[wp.transform],
    vel1: wp.array[wp.spatial_vector],
):
    side = wp.tid()
    body, sign = left, float(-1.0)
    if side == 1:
        body, sign = right, 1.0
    t = clock[0]
    x = pad_distance(t, home, closed)
    nxt = pad_distance(t + dt, home, closed)
    pose = wp.transform(wp.vec3(sign * x, 0.0, height), wp.quat_identity())
    velocity = wp.spatial_vector(wp.vec3(sign * (nxt - x) / dt, 0.0, 0.0), wp.vec3(0.0))
    pose0[body] = pose
    pose1[body] = pose
    vel0[body] = velocity
    vel1[body] = velocity


@wp.kernel
def drive_shelves(
    clock: wp.array[float],
    dt: float,
    left: int,
    right: int,
    height: float,
    pose0: wp.array[wp.transform],
    vel0: wp.array[wp.spatial_vector],
    pose1: wp.array[wp.transform],
    vel1: wp.array[wp.spatial_vector],
):
    body, sign = left, float(-1.0)
    if wp.tid() == 1:
        body, sign = right, 1.0
    t = clock[0]
    x = 0.027 + 0.095 * ramp(t, 3.1, 0.20)
    nxt = 0.027 + 0.095 * ramp(t + dt, 3.1, 0.20)
    pose = wp.transform(wp.vec3(sign * x, 0.0, height - 0.004), wp.quat_identity())
    velocity = wp.spatial_vector(wp.vec3(sign * (nxt - x) / dt, 0.0, 0.0), wp.vec3(0.0))
    pose0[body] = pose
    pose1[body] = pose
    vel0[body] = velocity
    vel1[body] = velocity


@wp.kernel
def advance_clock(clock: wp.array[float], dt: float):
    clock[0] = clock[0] + dt


def transform_points(points, pose):
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)
    return np.asarray(points) @ rotation.T + pose[:3]


def capsule_sdf(points, pose, radius, half_height):
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)
    p = (np.asarray(points) - pose[:3]) @ rotation
    p[:, 2] -= np.clip(p[:, 2], -half_height, half_height)
    return np.linalg.norm(p, axis=1) - radius


def box_sdf(points, pose, half_extents):
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)
    p = np.abs((np.asarray(points) - pose[:3]) @ rotation) - half_extents
    return np.linalg.norm(np.maximum(p, 0), axis=1) + np.minimum(p.max(axis=1), 0)


def shape_config(cfg: Config, collision=True):
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=cfg.contact_ke,
        kd=cfg.contact_kd,
        mu=cfg.friction,
        margin=0.0,
        gap=cfg.contact_gap,
        has_shape_collision=collision,
        has_particle_collision=False,
    )


class Example:
    def __init__(self, viewer=None, cfg: Config | None = None):
        self.cfg = cfg or Config()
        c = self.cfg
        if c.substeps not in (2, 4):
            raise ValueError("use 2 or 4 substeps for graph-safe state swapping")
        self.spec = SPECS[c.slug]
        self.viewer = viewer
        self.sim_time = 0.0
        self.dt = 1 / (60 * c.substeps)
        self.records, self.step_times = [], []
        self.saved_poses, self.saved_q, self.saved_qd, self.saved_velocities = [], [], [], []
        self.shell, self.basis, self.basis_audit = build_bottle_rom(self.spec, c.damping_ratio)
        self.vertices, self.faces = self.shell["vertices"], self.shell["faces"]
        wall_x = float(self.vertices[(np.abs(self.vertices[:, 2] - self.spec.squeeze_z) < 0.004), 0].max())
        self.home = wall_x + c.pad_radius + 0.013
        self.closed = wall_x + c.pad_radius - c.indentation
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis="Z")
        b.num_rigid_contacts_per_world = 8192
        mass = self.basis_audit["wall_mass_kg"]
        inertia = np.asarray(self.basis_audit["inertia_about_com_kg_m2"])
        self.bottle = b.add_body_elastic(
            xform=wp.transform(wp.vec3(0, 0, c.drop_height + c.contact_gap), wp.quat_identity()),
            mass=mass,
            com=wp.vec3(*self.basis_audit["center_of_mass_local_m"]),
            inertia=wp.mat33(*inertia.ravel()),
            modal_basis=self.basis,
            lock_inertia=True,
            label="finite_mass_free_floating_bottle_ROM",
        )
        self.bottle_shape = b.add_shape_mesh(
            self.bottle,
            mesh=newton.Mesh(self.vertices, self.faces.ravel()),
            cfg=shape_config(c),
            color=self.spec.color,
            label="hollow_wall",
        )
        self.pads, self.pad_shapes = [], []
        for side, name in ((-1, "left"), (1, "right")):
            body = b.add_body(
                xform=wp.transform(
                    wp.vec3(side * self.home, 0, self.spec.squeeze_z + c.drop_height), wp.quat_identity()
                ),
                is_kinematic=True,
                label=f"prescribed_{name}_finger_pad",
            )
            shape = b.add_shape_capsule(
                body, radius=c.pad_radius, half_height=c.pad_half_height, cfg=shape_config(c), color=(0.26, 0.30, 0.33)
            )
            self.pads.append(body)
            self.pad_shapes.append(shape)
            if not c.contact_enabled:
                b.add_shape_collision_filter_pair(self.bottle_shape, shape)
        self.shelves, self.shelf_shapes = [], []
        self.shelf_half_extents = (0.027, 0.045, 0.004)
        for side in (-1, 1):
            body = b.add_body(
                xform=wp.transform(wp.vec3(side * 0.027, 0, c.drop_height - 0.004), wp.quat_identity()),
                is_kinematic=True,
                label=f"retracting_support_{side}",
            )
            shape = b.add_shape_box(body, hx=0.027, hy=0.045, hz=0.004, cfg=shape_config(c))
            self.shelves.append(body)
            self.shelf_shapes.append(shape)
        ground_cfg = shape_config(c)
        # Detect the approaching base before it crosses the plane in a substep.
        # This is a detection envelope, not an offset of the physical ground.
        ground_cfg.gap = c.ground_detection_gap
        self.ground_shape = b.add_ground_plane(cfg=ground_cfg)
        # Actuators only collide with the free object, never each other.
        actuators = self.pad_shapes + self.shelf_shapes + [self.ground_shape]
        for i, first in enumerate(actuators):
            for second in actuators[i + 1 :]:
                b.add_shape_collision_filter_pair(first, second)
        b.color()
        self.model = b.finalize()
        self.phi = self.model.elastic_shape_vertex_phi.numpy().reshape(-1, self.basis.mode_count, 3).copy()
        np.testing.assert_allclose(self.model.elastic_shape_vertex_local.numpy(), self.vertices, atol=1e-7)
        np.testing.assert_allclose(self.phi, self.basis.sample_phi[: len(self.vertices)], atol=1e-6)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.pipeline = newton.CollisionPipeline(self.model, reduce_contacts=False, rigid_contact_max=GLOBAL_CAPACITY)
        self.control, self.contacts = self.model.control(), self.pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=c.iterations,
            rigid_body_contact_buffer_size=4096,
            rigid_contact_k_start=c.contact_ke,
            rigid_compliant_alm=c.compliant_alm,
            rigid_contact_hard=c.compliant_alm,
            friction_epsilon=1e-4,
            elastic_body_relaxation=c.relaxation,
            elastic_solver_metrics=True,
        )
        self.contact_trace = ContactTrace(self.model, self.contacts, self.solver)
        self.executed_frames = 0
        elastic_id = int(self.model.body_elastic_index.numpy()[self.bottle])
        owner = int(self.model.elastic_joint.numpy()[elastic_id])
        self.q_start = int(self.model.joint_q_start.numpy()[owner]) + 7
        self.qd_start = int(self.model.joint_qd_start.numpy()[owner]) + 6
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.graph = None
        start = time.perf_counter()
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        self.capture_seconds = time.perf_counter() - start
        self.fixed = (self.vertices[:, 2] <= self.spec.fixed_z) | (self.vertices[:, 2] >= self.spec.neck_z)
        self.probes = []
        for side in (-1, 1):
            eligible = np.arange(self.shell["outer_count"])
            cost = (self.vertices[eligible, 2] - self.spec.squeeze_z) ** 2 + self.vertices[eligible, 1] ** 2
            cost[self.vertices[eligible, 0] * side < 0] = np.inf
            self.probes.append(int(eligible[np.argmin(cost)]))
        edges = np.sort(np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]), axis=1)
        self.edges = np.unique(edges, axis=0)
        self.edge_lengths = np.linalg.norm(self.vertices[self.edges[:, 1]] - self.vertices[self.edges[:, 0]], axis=1)
        self.surface_reference = triangle_reference(self.vertices, self.faces)
        self.initial_body_q = self.state_0.body_q.numpy().copy()
        self.initial_joint_q = self.state_0.joint_q.numpy().copy()
        if viewer is not None:
            viewer.set_model(self.model)
            viewer.show_elastic_strain = False
            viewer.set_camera(pos=wp.vec3(0.27, -0.45, 0.27), pitch=-17.0, yaw=119)
        self.update_metrics()

    def simulate(self):
        c = self.cfg
        for _ in range(c.substeps):
            wp.launch(
                drive_pads,
                dim=2,
                inputs=[
                    self.clock,
                    self.dt,
                    *self.pads,
                    self.home,
                    self.closed,
                    self.spec.squeeze_z + c.drop_height,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_q,
                    self.state_1.body_qd,
                ],
                device=self.model.device,
            )
            wp.launch(
                drive_shelves,
                dim=2,
                inputs=[
                    self.clock,
                    self.dt,
                    *self.shelves,
                    c.drop_height,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_q,
                    self.state_1.body_qd,
                ],
                device=self.model.device,
            )
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.contact_trace.collision(self.contacts, self.clock)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.contact_trace.solved(self.solver)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_clock, dim=1, inputs=[self.clock, self.dt], device=self.model.device)

    def step(self):
        start = time.perf_counter()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        wp.synchronize_device(self.model.device)
        self.step_times.append(time.perf_counter() - start)
        self.executed_frames += 1
        self.contact_trace.check(self.executed_frames * self.cfg.substeps, self.dt)
        self.sim_time += 1 / 60
        self.update_metrics()

    def update_metrics(self):
        poses, q = self.state_0.body_q.numpy(), self.state_0.joint_q.numpy()
        qd, velocities = self.state_0.joint_qd.numpy(), self.state_0.body_qd.numpy()
        if not all(np.isfinite(a).all() for a in (poses, q, qd, velocities)):
            raise AssertionError("nonfinite Newton state")
        amplitudes = q[self.q_start : self.q_start + self.basis.mode_count]
        disp = np.einsum("vmc,m->vc", self.phi, amplitudes)
        local = self.vertices + disp
        world = transform_points(local, poses[self.bottle])
        tri = world[self.faces]
        samples = np.concatenate(
            [
                world,
                tri.mean(axis=1),
                (tri[:, 0] + tri[:, 1]) / 2,
                (tri[:, 1] + tri[:, 2]) / 2,
                (tri[:, 2] + tri[:, 0]) / 2,
            ]
        )
        penetrations = [
            max(0.0, -float(capsule_sdf(samples, poses[pad], self.cfg.pad_radius, self.cfg.pad_half_height).min()))
            for pad in self.pads
        ]
        shelf_penetrations = [
            max(0.0, -float(box_sdf(samples, poses[shelf], self.shelf_half_extents).min())) for shelf in self.shelves
        ]
        ground_penetration = max(0.0, -float(world[:, 2].min()))
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:count]
        contacts = [int(np.sum((shape0 == shape) | (shape1 == shape))) for shape in self.pad_shapes]
        support_contacts = [int(np.sum((shape0 == shape) | (shape1 == shape))) for shape in self.shelf_shapes]
        ground_contacts = int(np.sum((shape0 == self.ground_shape) | (shape1 == self.ground_shape)))
        edge_lengths = np.linalg.norm(local[self.edges[:, 1]] - local[self.edges[:, 0]], axis=1)
        stretches = triangle_stretches(local, self.faces, self.surface_reference)
        strain = np.max(np.abs(stretches - 1), axis=1)
        areas = self.surface_reference[1]
        solver_metrics = {k: float(v[0]) for k, v in self.solver.elastic_mode_solve_metrics().items()}
        solver_metrics["relative_solve_residual"] = solver_metrics["solve_residual_norm"] / max(
            solver_metrics["initial_residual_norm"], 1e-12
        )
        self.records.append(
            {
                "time": self.sim_time,
                "contacts_per_pad": contacts,
                "indentation_m": [float(disp[self.probes[0], 0]), -float(disp[self.probes[1], 0])],
                "max_deformation_m": float(np.linalg.norm(disp, axis=1).max()),
                "max_penetration_m": max(
                    shelf_penetrations + [ground_penetration] + (penetrations if self.cfg.contact_enabled else [])
                ),
                "penetration_per_pad_m": penetrations,
                "ground_penetration_m": ground_penetration,
                "shelf_penetration_m": shelf_penetrations,
                "contacts_per_shelf": support_contacts,
                "ground_contacts": ground_contacts,
                "airborne": sum(contacts + support_contacts) + ground_contacts == 0,
                "frame_translation_m": poses[self.bottle, :3].tolist(),
                "frame_linear_speed_m_s": float(np.linalg.norm(velocities[self.bottle, :3])),
                "frame_angular_speed_rad_s": float(np.linalg.norm(velocities[self.bottle, 3:])),
                "frame_rotation_rad": float(2 * np.arccos(np.clip(abs(poses[self.bottle, 6]), 0, 1))),
                "locally_rigid_region_deformation_m": float(np.linalg.norm(disp[self.fixed], axis=1).max()),
                "modal_velocity_norm_m_s": float(
                    np.linalg.norm(qd[self.qd_start : self.qd_start + self.basis.mode_count])
                ),
                "max_edge_length_change": float(np.max(np.abs(edge_lengths / self.edge_lengths - 1))),
                "min_principal_stretch": float(stretches.min()),
                "max_principal_stretch": float(stretches.max()),
                "max_principal_stretch_deviation": float(strain.max()),
                "surface_area_fraction_above_10pct": float(areas[strain > 0.10].sum() / areas.sum()),
                "surface_area_fraction_above_5pct": float(areas[strain > 0.05].sum() / areas.sum()),
                "cavity_volume_ml": cavity_volume(local, self.shell) * 1e6,
                "material_skin_volume_ml": signed_volume(local, self.faces) * 1e6,
                "modal_coordinates_m": amplitudes.tolist(),
                "solver": solver_metrics,
            }
        )
        self.saved_poses.append(poses.copy())
        self.saved_q.append(q.copy())
        self.saved_qd.append(qd.copy())
        self.saved_velocities.append(velocities.copy())

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

    def summary(self):
        hold = [r for r in self.records if 2.0 <= r["time"] <= 2.4]
        release = [r for r in self.records if r["time"] >= 5.0]
        hold_indent = np.asarray([r["indentation_m"] for r in hold])
        translations = np.asarray([r["frame_translation_m"] for r in self.records])
        pre_release = [r for r in self.records if 2.8 <= r["time"] <= 3.0]
        peak = max(r["max_deformation_m"] for r in self.records)
        recovered = [r["time"] - 3.1 for r in self.records if r["time"] >= 3.1 and r["max_deformation_m"] < 0.1 * peak]
        return {
            "slug": self.cfg.slug,
            "config": asdict(self.cfg),
            "basis": self.basis_audit,
            "device": str(self.model.device),
            "cuda_graph": self.graph is not None,
            "elastic_body_count": int(self.model.elastic_body_count),
            "floating_frame_dynamic": bool(
                self.model.body_inv_mass.numpy()[self.bottle] > 0
                and not (int(self.model.body_flags.numpy()[self.bottle]) & int(newton.BodyFlags.KINEMATIC))
            ),
            "max_frame_translation_m": float(np.linalg.norm(translations - translations[0], axis=1).max()),
            "final_frame_translation_m": float(np.linalg.norm(translations[-1] - translations[0])),
            "frame_vertical_range_m": float(np.ptp(translations[:, 2])),
            "max_frame_rotation_rad": max(r["frame_rotation_rad"] for r in self.records),
            "max_frame_linear_speed_m_s": max(r["frame_linear_speed_m_s"] for r in self.records),
            "max_frame_angular_speed_rad_s": max(r["frame_angular_speed_rad_s"] for r in self.records),
            "airborne_drop_frames": sum(r["airborne"] for r in self.records if r["time"] > 3.1),
            "ground_contact_frames": sum(r["ground_contacts"] > 0 for r in self.records),
            "pre_pad_release_mean_height_m": float(np.mean([r["frame_translation_m"][2] for r in pre_release]))
            if pre_release
            else None,
            "frames": len(self.records),
            "duration_s": self.sim_time,
            "max_deformation_m": max(r["max_deformation_m"] for r in self.records),
            "max_displacement_m": max(r["max_deformation_m"] for r in self.records),
            "final_displacement_m": self.records[-1]["max_deformation_m"],
            "recovery_to_ten_percent_after_release_s": min(recovered) if recovered else None,
            "max_penetration_m": max(r["max_penetration_m"] for r in self.records),
            "hold_mean_indentation_per_side_m": hold_indent.mean(axis=0).tolist() if hold else [],
            "hold_indentation_peak_to_peak_m": float(np.ptp(hold_indent, axis=0).max()) if hold else None,
            "hold_contact_dropouts": sum(min(r["contacts_per_pad"]) == 0 for r in hold),
            "hold_samples": len(hold),
            "release_samples": len(release),
            "max_contacts_per_pad": np.max([r["contacts_per_pad"] for r in self.records], axis=0).tolist(),
            "release_max_deformation_m": max(r["max_deformation_m"] for r in release) if release else None,
            "max_locally_rigid_region_deformation_m": max(
                r["locally_rigid_region_deformation_m"] for r in self.records
            ),
            "settled_deformation_peak_to_peak_m": float(np.ptp([r["max_deformation_m"] for r in release]))
            if release
            else None,
            "settled_max_modal_velocity_m_s": max(r["modal_velocity_norm_m_s"] for r in release) if release else None,
            "settled_max_frame_speed_m_s": max(r["frame_linear_speed_m_s"] for r in release) if release else None,
            "max_edge_length_change": max(r["max_edge_length_change"] for r in self.records),
            "min_principal_stretch": min(r["min_principal_stretch"] for r in self.records),
            "max_principal_stretch": max(r["max_principal_stretch"] for r in self.records),
            "max_principal_stretch_deviation": max(r["max_principal_stretch_deviation"] for r in self.records),
            "max_surface_area_fraction_above_10pct": max(r["surface_area_fraction_above_10pct"] for r in self.records),
            "max_surface_area_fraction_above_5pct": max(r["surface_area_fraction_above_5pct"] for r in self.records),
            "strain_measure": "Singular values of each triangle's 3x2 rest-to-current tangent map, all triangles without exclusions; area fractions weighted by reference area",
            "min_cavity_volume_ml": min(r["cavity_volume_ml"] for r in self.records),
            "final_cavity_volume_ml": self.records[-1]["cavity_volume_ml"],
            "max_relative_solve_residual": max(r["solver"]["relative_solve_residual"] for r in self.records),
            "max_final_iteration_modal_update_m": max(r["solver"]["update_norm"] for r in self.records),
            "graph_capture_s": self.capture_seconds,
            "warm_simulation_frame_ms": float(np.mean(self.step_times[10:]) * 1000)
            if len(self.step_times) > 10
            else None,
            "timing_scope": "Synchronized graph replay; excludes setup/capture, metrics/readbacks/export; first 10 frames omitted",
            "finite_state_all_frames": True,
        }

    def test_final(self):
        s = self.summary()
        assert s["elastic_body_count"] == 1, s
        assert s["floating_frame_dynamic"], s
        assert s["max_locally_rigid_region_deformation_m"] < 1e-7, s
        assert s["hold_samples"] > 20 and s["release_samples"] > 20, "run at least 5.5 seconds"
        assert 0.045 < s["frame_vertical_range_m"] < 0.09, s
        assert 0.045 < s["final_frame_translation_m"] < 0.10, s
        assert s["max_frame_translation_m"] < 0.10, s
        assert s["max_frame_rotation_rad"] < 0.35, s
        assert s["max_frame_linear_speed_m_s"] < 1.5, s
        assert s["max_frame_angular_speed_rad_s"] < 8.0, s
        assert s["airborne_drop_frames"] >= 3, s
        assert s["ground_contact_frames"] > 60, s
        assert s["max_penetration_m"] < 0.001, s
        assert s["max_edge_length_change"] < 0.12, s
        assert s["max_principal_stretch_deviation"] < 0.12, s
        assert s["max_surface_area_fraction_above_10pct"] < 0.01, s
        assert s["release_max_deformation_m"] < 0.0003, s
        assert s["settled_deformation_peak_to_peak_m"] < 0.00015, s
        assert s["settled_max_modal_velocity_m_s"] < 0.01, s
        assert s["settled_max_frame_speed_m_s"] < 0.025, s
        if not self.cfg.contact_enabled:
            assert max(s["max_contacts_per_pad"]) == 0, s
            return
        assert min(s["hold_mean_indentation_per_side_m"]) > 0.0035, s
        assert 0.0035 < s["max_deformation_m"] < 0.010, s
        assert s["hold_contact_dropouts"] == 0, s
        assert s["hold_indentation_peak_to_peak_m"] < 0.0006, s
        assert s["recovery_to_ten_percent_after_release_s"] is not None, s
        assert s["recovery_to_ten_percent_after_release_s"] < 0.25, s
        assert s["max_relative_solve_residual"] < 0.01, s


def compare_control(example, control):
    """Compare identical gravity/support trajectories with only pad contact removed."""
    assert replace(example.cfg, contact_enabled=False) == control.cfg
    np.testing.assert_array_equal(example.initial_body_q, control.initial_body_q)
    np.testing.assert_array_equal(example.initial_joint_q, control.initial_joint_q)
    actuators = example.pads + example.shelves
    np.testing.assert_array_equal(
        np.asarray(example.saved_poses)[:, actuators], np.asarray(control.saved_poses)[:, actuators]
    )
    positive, negative = example.summary(), control.summary()
    hold = [r for r in control.records if 2.0 <= r["time"] <= 2.4]
    baseline_peak = max(r["max_deformation_m"] for r in hold)
    contact_indent = np.asarray(positive["hold_mean_indentation_per_side_m"])
    baseline_indent = np.asarray(negative["hold_mean_indentation_per_side_m"])
    excess = contact_indent - baseline_indent
    ratio = baseline_peak / float(contact_indent.min())
    assert ratio < 0.10, (ratio, negative)
    assert excess.min() > 0.0035, excess
    negative.update(
        test_status="passed",
        comparison="Same finite mass, inertia, gravity, initial state, ground and shelf withdrawal; only bottle/pad collision pairs disabled. Baseline deformation is measured, not assumed zero.",
        hold_baseline_peak_deformation_m=baseline_peak,
        hold_deformation_ratio_to_contact=ratio,
        contact_excess_indentation_per_side_m=excess.tolist(),
    )
    return negative


def limitations():
    return [
        "Open/vented EMPTY containers: no liquid, gas pressure, suction, or fluid simulation.",
        "One linear four-mode ROM per bottle; tangentially compensated ovalization with smooth axial fade, not arbitrary shell buckling.",
        "Finite-mass floating frame is solved under gravity/contact. Base and neck are locally rigid within this four-mode approximation, not fixed in world space.",
        "Analytic mass-orthogonalized modes and assumed effective stiffness/damping; no material calibration or incompressibility claim.",
        "Linear ring/axial compensation is not exact finite-strain isometry. Every triangle is gated at 12% maximum principal-stretch deviation, with less than 1% reference area above 10%.",
        "Finite radial wall thickness, same displacement on inner/outer skins; not a geometrically exact thick-shell model.",
        "Vertex contact against rigid capsules, retracting shelves and ground; no self-contact, plasticity, creases, or fracture. Edge change is a strain proxy, not constitutive stress.",
        "Cavity volume is a polygon-section geometric estimate with a virtual vent closure; it is not a fluid or pressure measurement.",
        "Colored shell regions are procedural generic decoration, not proprietary assets or a product replica.",
    ]


def shell_render_parts(shell, world, spec):
    """Partition original shell triangles without overlays or geometry offsets."""
    for material, name, color in [
        (0, "hollow-bottle-wall", spec.color),
        (1, "collar", spec.collar_color),
        (2, "open-nozzle", spec.tip_color),
    ]:
        faces = shell["faces"][shell["materials"] == material]
        if not len(faces):
            continue
        ids, inverse = np.unique(faces.ravel(), return_inverse=True)
        yield name, world[:, ids], inverse.reshape(-1, 3).astype(np.int32), color, ids


def verify_surface_recovery(output: Path):
    """Audit every saved shell frame and reject missing or overlapping faces."""
    with (
        np.load(output / "scene.npz") as scene_file,
        np.load(output / "trajectory.npz") as trajectory_file,
        np.load(output / "basis.npz") as basis_file,
        np.load(output / "surface_map.npz") as mapping_file,
    ):
        scene, trajectory, basis, mapping = map(dict, (scene_file, trajectory_file, basis_file, mapping_file))
    recovered_faces = np.concatenate(
        [mapping[f"mesh_{i}_source_vertex"][scene[f"faces_{i}"]] for i in range(len(mapping))]
    )
    assert len(recovered_faces) == len(basis["faces"])
    np.testing.assert_array_equal(np.unique(recovered_faces, axis=0), np.unique(basis["faces"], axis=0))
    error = 0.0
    for frame in range(len(trajectory["times"])):
        recovered = basis["rest_vertices"] + np.einsum("vmc,m->vc", basis["phi"], trajectory["modal_q"][frame])
        recovered = transform_points(recovered, trajectory["body_q"][frame, int(trajectory["bottle_body"])])
        for i in range(len(mapping)):
            difference = scene[f"vertices_{i}"][frame] - recovered[mapping[f"mesh_{i}_source_vertex"]]
            error = max(error, float(np.max(np.abs(difference))))
    assert error < 3e-8, error
    return error


def render_notes():
    return (
        "Keep both rigid finger pads visible. No front labels; collar boundaries follow complete mesh rings. "
        "All shell triangles are disjoint material subsets, with no overlays or offsets. No cap closes either vent."
    )


def refresh_render(output: Path):
    """Repartition saved surfaces on CPU, preserving simulation files verbatim."""
    metadata = json.loads((output / "scene.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    frozen = ("trajectory.npz", "basis.npz", "frame_metrics.json")
    original_hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in frozen}
    with (
        np.load(output / "trajectory.npz") as trajectory_file,
        np.load(output / "basis.npz") as basis_file,
        np.load(output / "scene.npz") as scene_file,
        np.load(output / "surface_map.npz") as mapping_file,
    ):
        trajectory, basis, old_scene, old_mapping = map(dict, (trajectory_file, basis_file, scene_file, mapping_file))
    spec = SPECS[metadata["slug"]]
    shell = build_shell(spec)
    np.testing.assert_array_equal(shell["vertices"], basis["rest_vertices"])
    np.testing.assert_array_equal(shell["faces"], basis["faces"])
    world = np.asarray(
        [
            transform_points(
                basis["rest_vertices"] + np.einsum("vmc,m->vc", basis["phi"], q), poses[int(trajectory["bottle_body"])]
            )
            for q, poses in zip(trajectory["modal_q"], trajectory["body_q"], strict=True)
        ],
        dtype=np.float32,
    )
    arrays, meshes, mapping = {"times": trajectory["times"]}, [], {}
    for name, vertices, faces, color, ids in shell_render_parts(shell, world, spec):
        i = len(meshes)
        arrays[f"vertices_{i}"], arrays[f"faces_{i}"] = vertices, faces
        mapping[f"mesh_{i}_source_vertex"] = ids
        meshes.append(
            {
                "name": name,
                "vertices": f"vertices_{i}",
                "faces": f"faces_{i}",
                "color": list(color),
                "roughness": 0.38,
                "metallic": 0.0,
            }
        )
    for old_index, item in enumerate(metadata["meshes"]):
        if f"mesh_{old_index}_source_vertex" in old_mapping:
            continue
        i = len(meshes)
        arrays[f"vertices_{i}"] = old_scene[item["vertices"]]
        arrays[f"faces_{i}"] = old_scene[item["faces"]]
        meshes.append({**item, "vertices": f"vertices_{i}", "faces": f"faces_{i}"})
    metadata["meshes"] = meshes
    metadata["render_notes"] = render_notes()
    metadata["render_revision"] = RENDER_REVISION
    np.savez_compressed(output / "scene.npz", **arrays)
    np.savez_compressed(output / "surface_map.npz", **mapping)
    recovery_error = verify_surface_recovery(output)
    current_hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in frozen}
    assert current_hashes == original_hashes, "A render-only refresh modified simulation artifacts"
    metrics.setdefault("simulation_source_sha256", metrics["source_sha256"])
    metrics["source_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in metadata["source_files"]
    }
    metrics["surface_recovery_max_error_m"] = recovery_error
    metrics["surface_recovery_passed"] = True
    metrics["render_only_refresh"] = {
        "physics_files_unchanged": True,
        "physics_sha256": current_hashes,
        "exact_surface_face_partition": True,
        "revision": metadata["render_revision"],
    }
    metrics["render_artifact_sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ("scene.npz", "surface_map.npz")
    }
    (output / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"{output}: render-only refresh; recovery error={recovery_error}; physics hashes unchanged", flush=True)


def export(example: Example, output: Path, negative_control=None, test_status="passed"):
    output.mkdir(parents=True, exist_ok=True)
    capacity = example.contact_trace.save(output, example.cfg.frames * example.cfg.substeps, example.dt)
    c, spec = example.cfg, example.spec
    body_q = np.asarray(example.saved_poses)
    joint_q = np.asarray(example.saved_q)
    modal_q = joint_q[:, example.q_start : example.q_start + example.basis.mode_count]
    local = example.vertices[None, :, :] + np.einsum("vmc,fm->fvc", example.phi, modal_q)
    world = np.asarray(
        [transform_points(v, poses[example.bottle]) for v, poses in zip(local, body_q, strict=True)], dtype=np.float32
    )
    times = np.asarray([r["time"] for r in example.records])
    arrays, meshes, surface_map = {"times": times}, [], {}

    def add_mesh(name, vertices, faces, color, roughness=0.38, metallic=0.0, source=None):
        i = len(meshes)
        arrays[f"vertices_{i}"] = np.asarray(vertices, dtype=np.float32)
        arrays[f"faces_{i}"] = np.asarray(faces, dtype=np.int32)
        meshes.append(
            {
                "name": name,
                "vertices": f"vertices_{i}",
                "faces": f"faces_{i}",
                "color": list(color),
                "roughness": roughness,
                "metallic": metallic,
            }
        )
        if source is not None:
            surface_map[f"mesh_{i}_source_vertex"] = source

    for name, vertices, faces, color, ids in shell_render_parts(example.shell, world, spec):
        add_mesh(name, vertices, faces, color, source=ids)
    pv, pf = capsule_mesh(c.pad_radius, c.pad_half_height)
    for i, pad in enumerate(example.pads):
        points = np.asarray([transform_points(pv, poses[pad]) for poses in body_q])
        add_mesh(f"{'left' if i == 0 else 'right'}-rigid-finger-pad", points, pf, (0.24, 0.28, 0.31), 0.42)
        stem, sf = box_mesh((0.020, 0.007, 0.007))
        stem[:, 0] += (-1 if i == 0 else 1) * 0.025
        points = np.asarray([transform_points(stem, poses[pad]) for poses in body_q])
        add_mesh(f"pad-{i}-rigid-drive-stem", points, sf, (0.62, 0.66, 0.68), 0.24, 0.65)
    platform, pf = box_mesh(example.shelf_half_extents)
    for i, shelf in enumerate(example.shelves):
        points = np.asarray([transform_points(platform, poses[shelf]) for poses in body_q])
        add_mesh(f"retracting-support-shelf-{i}", points, pf, (0.17, 0.20, 0.22), 0.4, 0.15)
    source = Path(__file__).resolve()
    root = Path(newton.__file__).resolve().parents[1]
    command = (
        f"flock /tmp/newton-everyday-rom-gpu.lock timeout 5m env PYTHONPATH={shlex.quote(str(root))} "
        "OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/tmp/newton-everyday-bottles-warp "
        f"uv run --project {shlex.quote(str(root))} "
        f"python {shlex.quote(str(source))} --slug {c.slug} --device cuda:0 "
        f"--frames {c.frames} --iterations {c.iterations} --substeps {c.substeps} "
        f"--output {shlex.quote(str(output))}"
    )
    for key in (
        "pad_radius",
        "pad_half_height",
        "indentation",
        "contact_ke",
        "contact_kd",
        "contact_gap",
        "ground_detection_gap",
        "friction",
        "damping_ratio",
        "relaxation",
        "drop_height",
    ):
        command += f" --{key.replace('_', '-')} {getattr(c, key)}"
    if not c.contact_enabled:
        command += " --no-contact"
    if c.compliant_alm:
        command += " --compliant-alm"
    else:
        command += " --no-compliant-alm"
    if test_status == "not_run":
        command += " --no-test"
    elif negative_control:
        command += " --validate"
    metadata = {
        "slug": c.slug,
        "title": spec.title,
        "status": "validated prototype" if test_status == "passed" and negative_control else "WIP",
        "fps": 60,
        "meshes": meshes,
        "poster_frame": int(np.argmax([r["max_deformation_m"] for r in example.records])),
        "camera": {
            "position": [0.33, -0.57, 0.32],
            "target": [0, 0, 0.138 if c.slug == "drink-bottle" else 0.125],
            "fov": 39,
        },
        "floor_z": 0.0,
        "description": "Opposing rigid finger pads ovalize both walls and open rapidly. Physical shelves then retract, and the free bottle rebounds and drops onto the ground.",
        "model": "One finite-mass, free-floating four-mode hollow bottle with compensated ovalization and locally rigid base/neck regions.",
        "limitations": limitations(),
        "driven": "Two rigid pad poses and two retracting shelf poses only. Bottle floating frame and every modal coordinate solved by Newton VBD with gravity/contact.",
        "mode_count": example.basis.mode_count,
        "parameters": asdict(c),
        "source_files": ["bottles.py", "bottle_geometry.py", "test_bottles.py", "contact_capacity.py"],
        "reproduce": command,
        "surface_recovery": "rest_vertices + einsum('vmc,fm->fvc', phi, modal_q), then body_q rotation/translation; no amplification",
        "simulation_state": "trajectory.npz",
        "original_basis": "basis.npz",
        "render_vertex_map": "surface_map.npz",
        "timeline_s": {
            "approach": [0.6, 1.8],
            "hold": [1.8, 3.0],
            "support_withdrawal": [3.1, 3.3],
            "fast_unload": [3.0, 3.1],
            "rebound_and_drop": [3.1, 3.8],
            "settled": [5.0, 6.0],
        },
        "render_notes": render_notes(),
        "render_revision": RENDER_REVISION,
    }
    summary = example.summary()
    summary["contact_capacity"] = capacity
    summary.update(
        {
            "test_status": test_status,
            "test_passed": test_status == "passed",
            "negative_control_passed": negative_control is not None and negative_control["test_status"] == "passed",
            "limitations": limitations(),
            "negative_control": negative_control,
            "source_sha256": {
                name: hashlib.sha256((source.parent / name).read_bytes()).hexdigest()
                for name in metadata["source_files"]
            },
        }
    )
    np.savez_compressed(output / "scene.npz", **arrays)
    np.savez_compressed(
        output / "trajectory.npz",
        times=times,
        body_q=body_q,
        joint_q=joint_q,
        joint_qd=np.asarray(example.saved_qd),
        body_qd=np.asarray(example.saved_velocities),
        modal_q=modal_q,
        q_start=example.q_start,
        qd_start=example.qd_start,
        bottle_body=example.bottle,
        pad_bodies=example.pads,
        shelf_bodies=example.shelves,
        body_mass=example.model.body_mass.numpy(),
        body_com=example.model.body_com.numpy(),
        body_inertia=example.model.body_inertia.numpy(),
        body_flags=example.model.body_flags.numpy(),
        initial_body_q=example.initial_body_q,
        initial_joint_q=example.initial_joint_q,
    )
    np.savez_compressed(
        output / "basis.npz",
        rest_vertices=example.vertices,
        faces=example.faces,
        phi=example.phi,
        sample_mass=example.basis.sample_mass[: len(example.vertices)],
        mode_mass=example.basis.mode_mass,
        mode_stiffness=example.basis.mode_stiffness,
        mode_damping=example.basis.mode_damping,
        locally_rigid_region_mask=example.fixed,
        outer_count=example.shell["outer_count"],
        ring_count=example.shell["ring_count"],
        segments=example.shell["segments"],
        probe_indices=example.probes,
    )
    np.savez_compressed(output / "surface_map.npz", **surface_map)
    recovery_error = verify_surface_recovery(output)
    summary["surface_recovery_max_error_m"] = recovery_error
    summary["surface_recovery_passed"] = recovery_error < 3e-8
    assert summary["surface_recovery_passed"], recovery_error
    summary["render_artifact_sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ("scene.npz", "surface_map.npz")
    }
    (output / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "frame_metrics.json").write_text(json.dumps(example.records, indent=2) + "\n")


def run(cfg):
    example = Example(cfg=cfg)
    for frame in range(cfg.frames):
        example.step()
        if (frame + 1) % 60 == 0:
            r = example.records[-1]
            print(
                f"{cfg.slug} contact={cfg.contact_enabled} frame={frame + 1} "
                f"indent_mm={np.round(np.array(r['indentation_m']) * 1000, 3)} "
                f"contacts={r['contacts_per_pad']} penetration_mm={r['max_penetration_m'] * 1000:.3f}",
                flush=True,
            )
    return example


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slug", choices=SPECS, default="drink-bottle")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path)
    p.add_argument("--frames", type=int, default=360)
    p.add_argument("--iterations", type=int, default=16)
    p.add_argument("--substeps", type=int, choices=(2, 4), default=4)
    numeric_options = (
        "pad_radius",
        "pad_half_height",
        "indentation",
        "contact_ke",
        "contact_kd",
        "contact_gap",
        "ground_detection_gap",
        "friction",
        "damping_ratio",
        "relaxation",
        "drop_height",
    )
    for key in numeric_options:
        p.add_argument("--" + key.replace("_", "-"), type=float, default=getattr(Config, key))
    p.add_argument(
        "--validate", action="store_true", help="Also save the matched gravity/support/drop control without pad contact"
    )
    p.add_argument(
        "--no-contact", action="store_true", help="Disable only bottle/pad pairs; keep support and ground contacts"
    )
    p.add_argument("--compliant-alm", action=argparse.BooleanOptionalAction, default=Config.compliant_alm)
    p.add_argument("--no-test", action="store_true", help="Diagnostic partial trajectory; status remains WIP")
    p.add_argument("--refresh-render", action="store_true", help="CPU-only material refresh from saved state and basis")
    args = p.parse_args()
    if args.refresh_render:
        wp.init()
        wp.set_device("cpu")
        output = args.output or Path("output") / args.slug
        refresh_render(output)
        if (output / "negative-control" / "scene.json").exists():
            refresh_render(output / "negative-control")
        return
    wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", "/tmp/newton-everyday-bottles-warp")
    wp.init()
    wp.set_device(args.device)
    cfg = Config(
        slug=args.slug,
        frames=args.frames,
        iterations=args.iterations,
        substeps=args.substeps,
        contact_enabled=not args.no_contact,
        compliant_alm=args.compliant_alm,
        **{key: getattr(args, key) for key in numeric_options},
    )
    output = args.output or Path("output") / cfg.slug
    example = run(cfg)
    negative = None
    status = "not_run"
    try:
        if not args.no_test:
            example.test_final()
            status = "passed"
        if args.validate:
            if not cfg.contact_enabled or args.no_test:
                raise ValueError("--validate requires the positive-contact test")
            control = run(replace(cfg, contact_enabled=False))
            control_status = "failed"
            try:
                control.test_final()
                negative = compare_control(example, control)
                control_status = "passed"
            finally:
                export(control, output / "negative-control", test_status=control_status)
    except (AssertionError, ValueError):
        status = "failed"
        raise
    finally:
        export(example, output, negative, status)
        print(json.dumps(example.summary(), indent=2), flush=True)


if __name__ == "__main__":
    main()
