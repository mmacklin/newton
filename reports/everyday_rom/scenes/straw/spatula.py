# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Finite-mass silicone spatula pressed into a bowl, released, and dropped."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp
from utensil_geometry import (
    Config as BaseConfig,
)
from utensil_geometry import (
    UtensilExample,
    capsule_mesh,
    connect_rings,
    diagonalize_basis,
    lathe,
    ramp,
    run_cli,
    shape_config,
    tick,
    transform_vertices,
)

import newton


@dataclass
class Config(BaseConfig):
    length: float = 0.110
    young_modulus: float = 24.0e6
    poisson_ratio: float = 0.45
    damping_ratio: float = 0.12
    contact_ke: float = 4.0e4
    handle_mass: float = 0.065
    release_time: float = 3.2


def blade_section(s):
    width = np.interp(s, [0, 0.12, 0.30, 0.65, 0.84, 0.94, 1], [0.008, 0.012, 0.026, 0.033, 0.033, 0.031, 0.025])
    half_thickness = 0.0030 - 0.0017 * s
    return width, half_thickness


def build_blade(cfg):
    n, k = 37, 48
    s = np.linspace(0, 1, n)
    width, half_thickness = blade_section(s)
    theta = np.arange(k) * 2 * np.pi / k
    sn = np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** 0.55
    cs = np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** 0.55
    rings = [
        np.column_stack((np.full(k, t * cfg.length), w * sn, h * cs))
        for t, w, h in zip(s, width, half_thickness, strict=True)
    ]
    vertices, faces = connect_rings(rings, cap=True)
    x = vertices[:, 0] / cfg.length
    y, z = vertices[:, 1], vertices[:, 2]
    raw = np.zeros((len(vertices), 5, 3))
    for j, power in enumerate((2, 3)):
        raw[:, j, 2] = x**power
        raw[:, j, 0] = -z * power * x ** (power - 1) / cfg.length
    for j, power in enumerate((2, 3)):
        theta_field = x**power / 0.033
        raw[:, 2 + j, 2] = y * theta_field
        raw[:, 2 + j, 1] = -z * theta_field
    raw[:, 4, 1] = x**2
    raw[:, 4, 0] = -y * 2 * x / cfg.length
    dx = np.full(n, cfg.length / (n - 1))
    dx[[0, -1]] /= 2
    # Rounded-rectangle area is measured from its actual polygon, not a box.
    section_area = np.array(
        [0.5 * abs(np.dot(r[:, 1], np.roll(r[:, 2], 1)) - np.dot(r[:, 2], np.roll(r[:, 1], 1))) for r in rings]
    )
    sample_mass = np.r_[np.repeat(cfg.density * section_area * dx / k, k), 0.0, 0.0]
    nodes, weights = np.polynomial.legendre.leggauss(96)
    t, weights = (nodes + 1) / 2, weights / 2 * cfg.length
    w, h = blade_section(t)
    inertia_y = (2 * w) * (2 * h) ** 3 / 12
    inertia_z = (2 * h) * (2 * w) ** 3 / 12
    torsion = (2 * w) * (2 * h) ** 3 / 3 * (1 - 0.63 * h / w)
    shear = cfg.young_modulus / (2 * (1 + cfg.poisson_ratio))
    stiffness = np.zeros((5, 5))
    for i, p in enumerate((2, 3)):
        for j, q in enumerate((2, 3)):
            curvature_product = p * (p - 1) * q * (q - 1) * t ** (p + q - 4) / cfg.length**4
            stiffness[i, j] = np.sum(weights * cfg.young_modulus * inertia_y * curvature_product)
            stiffness[2 + i, 2 + j] = np.sum(
                weights * shear * torsion * p * q * t ** (p + q - 2) / (cfg.length * 0.033) ** 2
            )
    stiffness[4, 4] = np.sum(weights * cfg.young_modulus * inertia_z * 4 / cfg.length**4)
    basis, audit = diagonalize_basis(vertices, raw, sample_mass, stiffness, cfg, "tapered_spatula_bending_and_torsion")
    audit.update(
        young_modulus_Pa=cfg.young_modulus,
        blade_length_m=cfg.length,
        energy_model="Variable rectangular-section Euler-Bernoulli bending and Saint-Venant torsion; rounded section approximated in stiffness",
    )
    return vertices, faces, basis, audit


def handle_mesh():
    theta = np.arange(48) * 2 * np.pi / 48
    x = np.array([-0.150, -0.147, -0.140, -0.125, -0.080, -0.035, -0.014, 0.000])
    w = np.array([0.006, 0.010, 0.011, 0.011, 0.010, 0.009, 0.0065, 0.0065])
    h = np.array([0.003, 0.0042, 0.0048, 0.0048, 0.0045, 0.0042, 0.003, 0.003])
    rings = [
        np.column_stack((np.full(48, a), b * np.sin(theta), c * np.cos(theta))) for a, b, c in zip(x, w, h, strict=True)
    ]
    return connect_rings(rings, cap=True)


def bowl_geometry(segments=128):
    """Closed revolved ceramic wall with a rounded lip and a broad concave interior."""
    angle = np.linspace(0, np.pi / 2, 22)
    inner = np.column_stack((0.025 + 0.065 * np.sin(angle), 0.018 + 0.080 * (1 - np.cos(angle))))
    lip_angle = np.linspace(0, np.pi, 9)
    lip = np.column_stack((0.092 - 0.002 * np.cos(lip_angle), 0.098 + 0.002 * np.sin(lip_angle)))
    outer = np.column_stack((0.025 + 0.069 * np.sin(angle), 0.012 + 0.086 * (1 - np.cos(angle))))
    profile = np.vstack(([0.0001, 0.018], inner, lip[1:], outer[-2::-1], [0.0001, 0.012]))
    vertices, faces = lathe(profile, segments)
    vertices[:, :2] += [0.155, 0.025]
    return vertices, faces, profile


def bowl_distance(points, profile):
    """Signed distance to the revolved polygon; mesh faceting error is bounded separately."""
    p = np.column_stack((np.linalg.norm(points[:, :2] - [0.155, 0.025], axis=1), points[:, 2]))
    distance2 = np.full(len(p), np.inf)
    inside = np.zeros(len(p), dtype=bool)
    for a, b in zip(profile, np.roll(profile, -1, axis=0), strict=True):
        edge = b - a
        t = np.clip((p - a) @ edge / (edge @ edge), 0, 1)
        distance2 = np.minimum(distance2, np.sum((p - a - t[:, None] * edge) ** 2, axis=1))
        if edge[1] != 0:
            crossing = (a[1] > p[:, 1]) != (b[1] > p[:, 1])
            inside ^= crossing & (p[:, 0] < a[0] + (p[:, 1] - a[1]) * edge[0] / edge[1])
    return np.sqrt(distance2) * np.where(inside, -1, 1)


@wp.func
def grip_position(t: float, side: float, release: float):
    load = ramp(t, 0.6, 1.6)
    sweep = ramp(t, 2.3, 0.6)
    retreat = ramp(t, release, 0.4)
    return wp.vec3(0.0, 0.008 * sweep + side * 0.020 * retreat, 0.145 - 0.059 * load + 0.085 * retreat)


@wp.kernel
def drive_grip(
    clock: wp.array[float],
    dt: float,
    body: int,
    side: float,
    release: float,
    pose0: wp.array[wp.transform],
    pose1: wp.array[wp.transform],
    velocity0: wp.array[wp.spatial_vector],
    velocity1: wp.array[wp.spatial_vector],
):
    p = grip_position(clock[0], side, release)
    velocity = (grip_position(clock[0] + dt, side, release) - p) / dt
    pose0[body] = wp.transform(p, wp.quat_identity())
    pose1[body] = pose0[body]
    velocity0[body] = wp.spatial_vector(velocity, wp.vec3(0.0))
    velocity1[body] = velocity0[body]


class Example(UtensilExample):
    """A finite-mass handle and elastic blade press into a bowl, then the grip opens."""

    slug, title, kind = "spatula", "Flexible silicone spatula", 1
    elastic_color = (0.73, 0.085, 0.14)
    camera: ClassVar[dict] = {"position": [0.37, -0.57, 0.40], "target": [0.025, 0.0, 0.08], "fov": 42}
    description = "A raspberry silicone spatula bends and twists against a ceramic bowl. The grip opens, releasing the finite-mass utensil to rebound and fall under gravity."
    model_description = "Dynamic rigid handle fixed-jointed to a dynamic five-mode elastic blade; releasable grip joint; actual concave bowl mesh and floor collisions."
    driven = "Only the visible grip jaws are kinematic. The handle, blade frame, and all modal coordinates are solved by Newton VBD. The grip joint is disabled at 3.2 seconds."
    limitations: ClassVar[list[str]] = [
        "Generic procedural spatula with assumed 24 MPa linear modulus; not calibrated silicone.",
        "Linear bending/torsion ROM; no local wrinkling, nonlinear rubber, self-contact, failure, or food. Tests bound edge strain below 12 percent and neutral-axis length drift below 6 percent.",
        "Rounded blade stiffness uses rectangular-section formulas; surface-ring lumping approximates rotary inertia.",
        "Visible grip restraint is a switchable fixed joint, not frictional finger contact; blade-to-handle joint remains attached after release.",
        "Bowl is static; its rendered ceramic shell is the collision mesh. The tiny axial construction hole is 0.2 mm wide.",
        "Penetration is sampled at vertices, edge midpoints and face centers against the revolved bowl profile with a conservative faceting allowance, and the floor.",
        "After rebound the utensil can rest across bowl and floor with residual bending under load; this is not a stress-free final pose.",
    ]
    solver_options: ClassVar[dict] = {
        "rigid_joint_linear_ke": 2.0e4,
        "rigid_joint_angular_ke": 100.0,
        "rigid_joint_linear_kd": 0.02,
        "rigid_joint_angular_kd": 0.001,
    }

    def __init__(self, viewer=None, cfg=None):
        self.cfg, self.viewer = cfg or Config(), viewer
        c = self.cfg
        self.vertices, self.faces, self.basis, self.audit = build_blade(c)
        self.clamped = self.vertices[:, 0] < 1e-7
        self.tip = self.vertices[:, 0] > c.length - 0.001
        self.tip_left = self.tip & (self.vertices[:, 1] > 0.015)
        self.tip_right = self.tip & (self.vertices[:, 1] < -0.015)
        self.visuals = []
        self.released = False
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81), up_axis="Z")
        builder.num_rigid_contacts_per_world = 8192
        mass = self.audit["mass_kg"]
        sample_mass = np.asarray(self.basis.sample_mass)
        com = np.sum(self.vertices * sample_mass[:, None], axis=0) / mass
        r = self.vertices - com
        inertia = np.eye(3) * np.sum(sample_mass * np.sum(r * r, axis=1)) - (r * sample_mass[:, None]).T @ r
        initial = wp.transform(wp.vec3(0, 0, 0.145), wp.quat_identity())
        self.elastic_body = builder.add_body_elastic(
            xform=initial,
            mass=mass,
            com=wp.vec3(*com),
            inertia=wp.mat33(inertia),
            modal_basis=self.basis,
            lock_inertia=True,
            label="dynamic_elastic_blade",
        )
        self.blade_shape = builder.add_shape_mesh(
            self.elastic_body,
            mesh=newton.Mesh(self.vertices, self.faces.flatten()),
            cfg=shape_config(c),
            color=self.elastic_color,
        )
        m = c.handle_mass
        self.handle_body = builder.add_body(
            xform=initial,
            mass=m,
            com=wp.vec3(-0.075, 0, 0),
            inertia=wp.mat33(
                m * (0.02**2 + 0.009**2) / 12,
                0,
                0,
                0,
                m * (0.15**2 + 0.009**2) / 12,
                0,
                0,
                0,
                m * (0.15**2 + 0.02**2) / 12,
            ),
            lock_inertia=True,
            label="dynamic_rigid_handle",
        )
        self.root_joint = builder.add_joint_fixed(parent=self.handle_body, child=self.elastic_body, label="blade_root")
        self.handle_vertices, self.handle_faces = handle_mesh()
        builder.add_shape_mesh(
            self.handle_body,
            mesh=newton.Mesh(self.handle_vertices, self.handle_faces.flatten()),
            cfg=shape_config(c),
            label="handle_collision",
        )
        self.add_visual(
            builder, "rigid_handle", self.handle_vertices, self.handle_faces, (0.16, 0.19, 0.21), self.handle_body
        )
        for x in np.linspace(-0.120, -0.055, 9):
            v, f = capsule_mesh([x, -0.0080, 0.0038], [x, 0.0080, 0.0038], 0.00055, 24, 4)
            self.add_visual(builder, f"handle_grip_{x:.3f}", v, f, (0.29, 0.32, 0.34), self.handle_body)
        profile = [
            (0.0042, -0.0025),
            (0.008, -0.0025),
            (0.0087, -0.0015),
            (0.0087, 0.0015),
            (0.008, 0.0025),
            (0.0042, 0.0025),
        ]
        v, f = lathe(profile, 64)
        v[:, 0] -= 0.157
        self.add_visual(builder, "open_hanging_eye", v, f, (0.16, 0.19, 0.21), self.handle_body)
        builder.add_shape_mesh(
            self.handle_body, mesh=newton.Mesh(v, f.flatten()), cfg=shape_config(c), label="hanging_eye_collision"
        )
        handle_samples = []
        for part in self.visuals:
            if part["body"] == self.handle_body:
                points, triangles = part["vertices"], part["vertices"][part["faces"]]
                handle_samples.extend((points, triangles.mean(axis=1)))
        self.handle_surface_samples = np.concatenate(handle_samples)
        self.grip_bodies = []
        for side in (-1, 1):
            body = builder.add_body(xform=initial, is_kinematic=True, label=f"driven_grip_jaw_{side}")
            self.grip_bodies.append(body)
            v, f = capsule_mesh([-0.102, side * 0.014, -0.001], [-0.102, side * 0.014, 0.033], 0.005, 32, 8)
            self.add_visual(builder, f"grip_jaw_{side}", v, f, (0.11, 0.39, 0.43), body, metallic=0.3)
            v, f = capsule_mesh([-0.111, side * 0.014, 0], [-0.093, side * 0.014, 0], 0.004, 32, 6)
            self.add_visual(builder, f"grip_pad_{side}", v, f, (0.08, 0.11, 0.12), body)
        self.grip_joint = builder.add_joint_fixed(
            parent=self.grip_bodies[0],
            child=self.handle_body,
            parent_xform=wp.transform(wp.vec3(-0.102, 0, 0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.102, 0, 0), wp.quat_identity()),
            label="releasable_grip",
        )
        v, f, self.bowl_profile = bowl_geometry()
        self.bowl_vertices, self.bowl_faces = v.copy(), f.copy()
        self.bowl_faceting_bound = float(self.bowl_profile[:, 0].max() * (1 - np.cos(np.pi / 128)))
        self.bowl_shape = builder.add_shape_mesh(
            -1,
            mesh=newton.Mesh(v, f.flatten()),
            cfg=shape_config(c, c.contact_enabled),
            label="ceramic_bowl_collision",
        )
        self.add_visual(builder, "ceramic_bowl", v, f, (0.73, 0.80, 0.77), roughness=0.20)
        self.floor_shape = builder.add_ground_plane(cfg=shape_config(c))
        self.finish_setup(builder)

    def simulate(self):
        for _ in range(self.cfg.substeps):
            for side, body in zip((-1.0, 1.0), self.grip_bodies, strict=True):
                wp.launch(
                    drive_grip,
                    dim=1,
                    inputs=[
                        self.clock,
                        self.dt,
                        body,
                        side,
                        self.cfg.release_time,
                        self.state_0.body_q,
                        self.state_1.body_q,
                        self.state_0.body_qd,
                        self.state_1.body_qd,
                    ],
                    device=self.model.device,
                )
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(tick, dim=1, inputs=[self.clock, self.dt], device=self.model.device)

    def step(self):
        if not self.released and self.sim_time >= self.cfg.release_time - 1e-8:
            enabled = self.model.joint_enabled.numpy()
            enabled[self.grip_joint] = False
            self.model.joint_enabled.assign(enabled)
            self.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
            self.released = True
        super().step()

    def surface_distance(self, samples, poses):
        handle = transform_vertices(self.handle_surface_samples, poses[self.handle_body])
        samples = np.concatenate((samples, handle))
        self.current_floor_penetration = max(0.0, -float(samples[:, 2].min()))
        self.current_bowl_penetration = max(
            0.0, -float(bowl_distance(samples, self.bowl_profile).min()) + self.bowl_faceting_bound
        )
        return np.array(
            [-max(self.current_floor_penetration, self.current_bowl_penetration if self.cfg.contact_enabled else 0)]
        )

    def update_metrics(self):
        super().update_metrics()
        record = self.records[-1]
        poses, velocities = self.saved_body_q[-1], self.saved_body_qd[-1]
        handle, blade = poses[self.handle_body], poses[self.elastic_body]
        dot = np.clip(abs(np.dot(handle[3:], blade[3:])), 0, 1)
        count = record["contacts"]
        shapes0 = self.contacts.rigid_contact_shape0.numpy()[:count]
        shapes1 = self.contacts.rigid_contact_shape1.numpy()[:count]
        grip_point = np.array([[-0.102, 0, 0]])
        q = np.asarray(record["q"])
        local = self.vertices + np.einsum("nmc,m->nc", self.phi, q)
        centers = local[: 37 * 48].reshape(37, 48, 3).mean(axis=1)
        center_lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        record.update(
            centerline_length_gain=float(center_lengths.sum() / self.cfg.length - 1),
            max_centerline_segment_strain=float(np.abs(center_lengths / (self.cfg.length / 36) - 1).max()),
            root_joint_error_m=float(np.linalg.norm(handle[:3] - blade[:3])),
            root_joint_angle_error_rad=float(2 * np.arccos(dot)),
            grip_joint_error_m=float(
                np.linalg.norm(
                    transform_vertices(grip_point, handle) - transform_vertices(grip_point, poses[self.grip_bodies[0]])
                )
            ),
            grip_enabled=not self.released,
            bowl_contacts=int(np.sum((shapes0 == self.bowl_shape) | (shapes1 == self.bowl_shape))),
            floor_contacts=int(np.sum((shapes0 == self.floor_shape) | (shapes1 == self.floor_shape))),
            bowl_penetration_m=self.current_bowl_penetration,
            floor_penetration_m=self.current_floor_penetration,
            handle_speed_m_s=float(np.linalg.norm(velocities[self.handle_body, :3])),
            handle_angular_speed_rad_s=float(np.linalg.norm(velocities[self.handle_body, 3:])),
            handle_origin_z_m=float(handle[2]),
            handle_com_z_m=float(transform_vertices(np.array([[-0.075, 0, 0]]), handle)[0, 2]),
        )

    def summary(self):
        result = super().summary()
        loading = [r for r in self.records if r["grip_enabled"]]
        free = [r for r in self.records if not r["grip_enabled"]]
        early = [r for r in free if r["time_s"] < self.cfg.release_time + 0.35]
        final = [r for r in free if r["time_s"] > 5.5]
        unloaded = [r for r in free if r["contacts"] == 0]
        unloaded_min = min((r["max_deformation_m"] for r in unloaded), default=None)
        result.update(
            dynamic_release=True,
            driven_body_indices=self.grip_bodies,
            handle_body_index=self.handle_body,
            blade_body_index=self.elastic_body,
            handle_mass_kg=self.cfg.handle_mass,
            blade_mass_kg=self.audit["mass_kg"],
            release_time_s=self.cfg.release_time,
            gravity_m_s2=[0, 0, -9.81],
            loading_max_deformation_m=max(r["max_deformation_m"] for r in loading),
            max_bowl_contacts=max(r["bowl_contacts"] for r in self.records),
            max_floor_contacts=max(r["floor_contacts"] for r in self.records),
            max_root_joint_error_m=max(r["root_joint_error_m"] for r in self.records),
            max_root_joint_angle_error_rad=max(r["root_joint_angle_error_rad"] for r in self.records),
            max_grip_joint_error_m=max(r["grip_joint_error_m"] for r in loading),
            max_free_handle_speed_m_s=max((r["handle_speed_m_s"] for r in free), default=0),
            max_free_handle_angular_speed_rad_s=max((r["handle_angular_speed_rad_s"] for r in free), default=0),
            free_handle_z_range_m=float(np.ptp([r["handle_origin_z_m"] for r in free])) if free else 0,
            free_handle_com_z_range_m=float(np.ptp([r["handle_com_z_m"] for r in free])) if free else 0,
            rebound_tip_peak_to_peak_m=float(np.ptp([r["tip_displacement_m"] for r in early], axis=0).max())
            if early
            else 0,
            recovery_residual_m=max((r["max_deformation_m"] for r in final), default=None),
            recovery_definition="Maximum deformation over the final second, relative to local rest shape; bowl/floor still load the landed utensil. See unloaded_min_deformation_m for contact-free recovery.",
            contact_free_frames=len(unloaded),
            unloaded_min_deformation_m=unloaded_min,
            unloaded_recovery_fraction=1 - unloaded_min / max(r["max_deformation_m"] for r in loading)
            if unloaded
            else None,
            penetration_samples="Dense blade skin and handle vertices against bowl profile (plus faceting bound) and exact floor plane",
            bowl_faceting_error_bound_m=self.bowl_faceting_bound,
            max_centerline_length_gain=max(r["centerline_length_gain"] for r in self.records),
            max_centerline_segment_strain=max(r["max_centerline_segment_strain"] for r in self.records),
        )
        return result

    def test_final(self):
        s = self.summary()
        assert s["graph_enabled"] and s["frames"] >= 360 and self.cfg.substeps == 4, s
        assert self.released and not bool(self.model.joint_enabled.numpy()[self.grip_joint]), s
        masses = self.model.body_inv_mass.numpy()
        assert masses[self.handle_body] > 0 and masses[self.elastic_body] > 0, "Spatula must be dynamic"
        flags = self.model.body_flags.numpy()
        assert not any(flags[b] & int(newton.BodyFlags.KINEMATIC) for b in (self.handle_body, self.elastic_body))
        assert s["max_clamp_displacement_m"] < 1e-7 and s["max_root_joint_error_m"] < 0.001, s
        assert s["max_root_joint_angle_error_rad"] < 0.03 and s["max_grip_joint_error_m"] < 0.001, s
        assert s["max_edge_strain"] < 0.12 and s["max_deformation_m"] < 0.045, s
        assert s["max_centerline_length_gain"] < 0.06 and s["max_centerline_segment_strain"] < 0.10, s
        assert s["max_penetration_m"] < 0.001 and s["max_modal_solve_residual_ratio"] < 0.03, s
        assert s["free_handle_com_z_range_m"] > 0.035 and s["max_free_handle_speed_m_s"] > 0.15, s
        assert s["max_free_handle_angular_speed_rad_s"] > 0.5 and s["max_floor_contacts"] > 0, s
        assert s["recovery_residual_m"] < 0.008, s
        if self.cfg.contact_enabled:
            assert 0.030 < s["loading_max_deformation_m"] < 0.050, s
            assert s["max_bowl_contacts"] > 0 and s["max_twist_displacement_m"] > 0.002, s
            assert s["rebound_tip_peak_to_peak_m"] > 0.010, s
            assert s["contact_free_frames"] >= 2 and s["unloaded_min_deformation_m"] < 0.005, s
            assert s["unloaded_recovery_fraction"] > 0.85, s
        else:
            assert s["max_bowl_contacts"] == 0 and s["loading_max_deformation_m"] < 0.008, s

    def export(self, output, result):
        super().export(output, result)
        np.savez_compressed(
            output / "mechanism.npz",
            body_mass=self.model.body_mass.numpy(),
            body_com=self.model.body_com.numpy(),
            body_inertia=self.model.body_inertia.numpy(),
            body_flags=self.model.body_flags.numpy(),
            root_joint=self.root_joint,
            grip_joint=self.grip_joint,
            grip_joint_enabled=np.array([r["grip_enabled"] for r in self.records]),
            joint_parent=self.model.joint_parent.numpy(),
            joint_child=self.model.joint_child.numpy(),
            joint_parent_xform=self.model.joint_X_p.numpy(),
            joint_child_xform=self.model.joint_X_c.numpy(),
            bowl_profile=self.bowl_profile,
            bowl_vertices=self.bowl_vertices,
            bowl_faces=self.bowl_faces,
            bowl_collision_enabled=self.cfg.contact_enabled,
            handle_rest_vertices=self.handle_vertices,
            handle_faces=self.handle_faces,
            driven_bodies=self.grip_bodies,
        )
        path = output / "scene.json"
        scene = json.loads(path.read_text())
        scene["audit"]["mechanism"] = "mechanism.npz"
        scene["poster_frame"] = int(
            np.argmax([r["max_deformation_m"] if r["grip_enabled"] else 0 for r in self.records])
        )
        if not self.cfg.contact_enabled:
            scene["description"] = (
                "Only bowl collisions disabled; floor, gravity and grip release remain enabled. " + self.description
            )
        path.write_text(json.dumps(scene, indent=2) + "\n")
        source = output / "source"
        source.mkdir(exist_ok=True)
        for name in ("spatula.py", "utensil_geometry.py", "test_utensils.py"):
            shutil.copy2(Path(__file__).with_name(name), source / name)


if __name__ == "__main__":
    run_cli(Example, Config, "spatula")
