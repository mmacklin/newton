# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Hollow silicone straw with a curved 30-degree tip and contact-driven bending."""

import json
import shutil
from dataclasses import dataclass
from itertools import pairwise
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
    capsule_sdf,
    diagonalize_basis,
    drive_frame,
    lathe,
    orient_mesh,
    ramp,
    run_cli,
    shape_config,
    target_position,
    tick,
    transform_vertices,
)

import newton


@dataclass
class Config(BaseConfig):
    length: float = 0.215
    clamp_length: float = 0.035
    outer_radius: float = 0.0045
    inner_radius: float = 0.0030
    young_modulus: float = 6.0e6
    poisson_ratio: float = 0.49
    tip_angle_degrees: float = 30.0
    tip_length: float = 0.032
    neck_length: float = 0.018
    release_time: float = 3.8
    iterations: int = 24
    contact_ke: float = 20000.0
    elastic_relaxation: float = 0.25
    diagnostic_floor_enabled: bool = True
    floating_basis: bool = True
    floor_contact_gap: float = 0.0015


def build_straight_straw(cfg):
    """Preserve the previously validated straight geometry and four-mode basis."""
    n, k = 49, 40
    z = np.r_[np.linspace(0, cfg.clamp_length, 6)[:-1], np.linspace(cfg.clamp_length, cfg.length, n - 5)]
    angle = np.arange(k) * 2 * np.pi / k
    vertices = []
    for radius in (cfg.outer_radius, cfg.inner_radius):
        for height in z:
            # A small rolled lip leaves a real opening, never a capped cylinder.
            bevel = 0.00015 * (height == z[-1] or height == z[0])
            r = radius - bevel if radius == cfg.outer_radius else radius + bevel
            vertices.extend(zip(r * np.cos(angle), r * np.sin(angle), np.full(k, height), strict=True))
    faces = []
    for side in range(2):
        for j in range(n - 1):
            for i in range(k):
                a, b = side * n * k + j * k + i, side * n * k + j * k + (i + 1) % k
                c, d = b + k, a + k
                faces.extend(((a, b, c), (a, c, d)) if side == 0 else ((a, c, b), (a, d, c)))
    for j in (0, n - 1):
        for i in range(k):
            a, b = j * k + i, j * k + (i + 1) % k
            c, d = b + n * k, a + n * k
            faces.extend(((a, c, b), (a, d, c)) if j == 0 else ((a, b, c), (a, c, d)))
    vertices, faces = orient_mesh(np.asarray(vertices), np.asarray(faces, dtype=np.int32))
    length = cfg.length - cfg.clamp_length
    s = np.maximum((vertices[:, 2] - cfg.clamp_length) / length, 0)
    raw = np.zeros((len(vertices), 4, 3))
    for axis in range(2):
        for j, power in enumerate((2, 3)):
            mode = 2 * axis + j
            raw[:, mode, axis] = s**power
            raw[:, mode, 2] = -vertices[:, axis] * power * s ** (power - 1) / length
    dz = np.r_[np.diff(z)[0] / 2, (z[2:] - z[:-2]) / 2, np.diff(z)[-1] / 2]
    ring_mass = cfg.density * np.pi * (cfg.outer_radius**2 - cfg.inner_radius**2) * dz
    sample_mass = np.tile(np.repeat(ring_mass / (2 * k), k), 2)
    moment = np.pi / 4 * (cfg.outer_radius**4 - cfg.inner_radius**4)
    stiffness = np.zeros((4, 4))
    for axis in range(2):
        for i, p in enumerate((2, 3)):
            for j, q in enumerate((2, 3)):
                stiffness[axis * 2 + i, axis * 2 + j] = (
                    cfg.young_modulus * moment * p * (p - 1) * q * (q - 1) / (length**3 * (p + q - 3))
                )
    basis, audit = diagonalize_basis(vertices, raw, sample_mass, stiffness, cfg, "hollow_straw_four_bending_modes")
    audit.update(
        young_modulus_Pa=cfg.young_modulus,
        outer_diameter_m=2 * cfg.outer_radius,
        inner_diameter_m=2 * cfg.inner_radius,
        free_length_m=length,
    )
    return vertices, faces, basis, audit


def centerline(cfg, arclength):
    """Return rest centers, tangent, normal, and curvature along the tube [m]."""
    s = np.atleast_1d(np.asarray(arclength, dtype=float))
    start = cfg.length - cfg.tip_length - cfg.neck_length
    angle = np.deg2rad(cfg.tip_angle_degrees)
    if not (cfg.clamp_length < start < cfg.length and 0 <= angle <= np.pi / 3):
        raise ValueError("Need a free neck and a tip angle between zero and 60 degrees")
    if angle == 0:
        return (
            np.column_stack((0 * s, 0 * s, s)),
            np.tile([0, 0, 1], (len(s), 1)),
            np.tile([1, 0, 0], (len(s), 1)),
            np.zeros_like(s),
        )
    radius = cfg.neck_length / angle
    if radius <= 2 * cfg.outer_radius:
        raise ValueError("Neck radius must exceed two outer tube radii")
    neck_s = np.clip(s - start, 0, cfg.neck_length)
    theta = neck_s / radius
    beyond = np.maximum(s - start - cfg.neck_length, 0)
    centers = np.column_stack(
        (
            radius * (1 - np.cos(theta)) + beyond * np.sin(angle),
            0 * s,
            np.minimum(s, start) + radius * np.sin(theta) + beyond * np.cos(angle),
        )
    )
    tangent = np.column_stack((np.sin(theta), 0 * s, np.cos(theta)))
    normal = np.column_stack((np.cos(theta), 0 * s, -np.sin(theta)))
    curvature = np.where((s > start) & (s < start + cfg.neck_length), 1 / radius, 0)
    return centers, tangent, normal, curvature


def beam_quadrature(cfg, end):
    """Piecewise Gauss quadrature split at the rest-curvature discontinuities."""
    if end <= cfg.clamp_length:
        return np.empty(0), np.empty(0)
    start = cfg.length - cfg.tip_length - cfg.neck_length
    breaks = np.unique(np.clip([cfg.clamp_length, start, start + cfg.neck_length, end], cfg.clamp_length, end))
    nodes, weights = np.polynomial.legendre.leggauss(24)
    points = np.concatenate([(a + b) / 2 + (b - a) / 2 * nodes for a, b in pairwise(breaks)])
    quadrature_weights = np.concatenate([(b - a) / 2 * weights for a, b in pairwise(breaks)])
    return points, quadrature_weights


def rotation_modes(cfg, arclength):
    """Infinitesimal spatial rotations and their arc derivatives [1/m, 1/m^2]."""
    s = np.atleast_1d(np.asarray(arclength, dtype=float))
    length = cfg.length - cfg.clamp_length
    xi = np.maximum((s - cfg.clamp_length) / length, 0)
    powers = np.array([2, 3, 2, 3])
    axes = np.array([[0, 1, 0], [0, 1, 0], [-1, 0, 0], [-1, 0, 0]])
    angle = powers * xi[:, None] ** (powers - 1) / length
    rate = powers * (powers - 1) * xi[:, None] ** (powers - 2) / length**2
    rate[s < cfg.clamp_length] = 0
    return angle[:, :, None] * axes[None], rate[:, :, None] * axes[None]


def centerline_modes(cfg, arclength):
    """Integrate u_c' = omega cross t_0, enforcing linear unshearability."""
    displacement = []
    for end in np.atleast_1d(arclength):
        nodes, weights = beam_quadrature(cfg, end)
        omega, _ = rotation_modes(cfg, nodes)
        tangent = centerline(cfg, nodes)[1]
        displacement.append(np.einsum("n,nmc->mc", weights, np.cross(omega, tangent[:, None])))
    return np.asarray(displacement)


def straw_sections(cfg):
    """Material coordinates for the fixed stem, curved neck, and short tip."""
    if cfg.tip_angle_degrees == 0:
        return np.r_[np.linspace(0, cfg.clamp_length, 6)[:-1], np.linspace(cfg.clamp_length, cfg.length, 44)]
    start = cfg.length - cfg.tip_length - cfg.neck_length
    return np.r_[
        np.linspace(0, cfg.clamp_length, 6)[:-1],
        np.linspace(cfg.clamp_length, start, 31)[:-1],
        np.linspace(start, start + cfg.neck_length, 13)[:-1],
        np.linspace(start + cfg.neck_length, cfg.length, 10),
    ]


def build_straw(cfg):
    if cfg.tip_angle_degrees == 0:
        return build_straight_straw(cfg)
    s, k = straw_sections(cfg), 40
    n = len(s)
    centers, tangent, normal, curvature = centerline(cfg, s)
    angle = np.arange(k) * 2 * np.pi / k
    binormal = np.array([0, 1, 0])
    radial_direction = np.cos(angle)[None, :, None] * normal[:, None] + np.sin(angle)[None, :, None] * binormal
    radial_offsets = []
    for radius in (cfg.outer_radius, cfg.inner_radius):
        radii = np.full(n, radius)
        radii[[0, -1]] += -0.00015 if radius == cfg.outer_radius else 0.00015
        radial_offsets.append(radii[:, None, None] * radial_direction)
    offsets = np.asarray(radial_offsets)
    vertices = (centers[None, :, None] + offsets).reshape(-1, 3)
    faces = []
    for side in range(2):
        for j in range(n - 1):
            for i in range(k):
                a, b = side * n * k + j * k + i, side * n * k + j * k + (i + 1) % k
                c, d = b + k, a + k
                faces.extend(((a, b, c), (a, c, d)) if side == 0 else ((a, c, b), (a, d, c)))
    for j in (0, n - 1):
        for i in range(k):
            a, b = j * k + i, j * k + (i + 1) % k
            c, d = b + n * k, a + n * k
            faces.extend(((a, c, b), (a, d, c)) if j == 0 else ((a, b, c), (a, c, d)))
    vertices, faces = orient_mesh(vertices, np.asarray(faces, dtype=np.int32))
    omega, _ = rotation_modes(cfg, s)
    displacement = centerline_modes(cfg, s)
    # Each real cross section moves by u_c + omega x r. The material centerline
    # and its directors are curved at rest, not transformed after simulation.
    raw = (displacement[None, :, None] + np.cross(omega[None, :, None], offsets[:, :, :, None])).reshape(-1, 4, 3)
    ds = np.r_[np.diff(s)[0] / 2, (s[2:] - s[:-2]) / 2, np.diff(s)[-1] / 2]
    ring_mass = cfg.density * np.pi * (cfg.outer_radius**2 - cfg.inner_radius**2) * ds
    jacobian = 1 - curvature[None, :, None] * np.einsum("snkc,nc->snk", offsets, normal)
    sample_mass = (ring_mass[None, :, None] / (2 * k) * jacobian).reshape(-1)
    nodes, weights = beam_quadrature(cfg, cfg.length)
    _, omega_prime = rotation_modes(cfg, nodes)
    tangent = centerline(cfg, nodes)[1]
    twist = np.einsum("nmc,nc->nm", omega_prime, tangent)
    moment = np.pi / 4 * (cfg.outer_radius**4 - cfg.inner_radius**4)
    ei = cfg.young_modulus * moment
    gj = cfg.young_modulus * moment / (1 + cfg.poisson_ratio)
    # For a stress-free curved rod, incremental material curvature is
    # R_0^T omega'. Split it into bending and torsion along the rest tangent.
    stiffness = ei * np.einsum("n,nic,njc->ij", weights, omega_prime, omega_prime)
    stiffness += (gj - ei) * np.einsum("n,ni,nj->ij", weights, twist, twist)
    basis, audit = diagonalize_basis(vertices, raw, sample_mass, stiffness, cfg, "curved_hollow_straw_Ritz")
    audit.update(
        method="Curved-rod rotation-based Rayleigh-Ritz; integrate u_c'=omega cross t_0, then mass/stiffness eigensolve",
        energy_model="Stress-free curved Kirchhoff rod: EI transverse |omega'|^2 + GJ tangent |omega'|^2",
        sample_mass_model="Annular cross-section mass on surface rings with curved-coordinate Jacobian; approximate rotary inertia",
        young_modulus_Pa=cfg.young_modulus,
        poisson_ratio=cfg.poisson_ratio,
        outer_diameter_m=2 * cfg.outer_radius,
        inner_diameter_m=2 * cfg.inner_radius,
        free_length_m=cfg.length - cfg.clamp_length,
        rest_tip_angle_deg=cfg.tip_angle_degrees,
        short_tip_length_m=cfg.tip_length,
        neck_arc_length_m=cfg.neck_length,
        neck_radius_m=cfg.neck_length / np.deg2rad(cfg.tip_angle_degrees),
        rest_tip_center_m=centers[-1].tolist(),
        rest_tip_tangent=tuple(centerline(cfg, [cfg.length])[1][0]),
        rest_geometry="Actual swept annular tube along straight stem, circular neck, and 30-degree short tip",
    )
    return vertices, faces, basis, audit


def project_rigid_fields(cfg, basis, com):
    """Choose a mass-orthogonal floating frame without changing the linear span."""
    x = basis.sample_points.astype(float)
    weights = np.repeat(basis.sample_mass.astype(float), 3)
    phi = basis.sample_phi.transpose(0, 2, 1).reshape(-1, 4).astype(float)
    rigid = np.zeros((len(x), 3, 6))
    rigid[:, :, :3] = np.eye(3)
    rigid[:, :, 3:] = np.stack([np.cross(axis, x - com) for axis in np.eye(3)], axis=-1)
    rigid = rigid.reshape(-1, 6)
    frame_mass = rigid.T @ (weights[:, None] * rigid)
    gauge = np.linalg.solve(frame_mass, rigid.T @ (weights[:, None] * phi))
    projected = phi - rigid @ gauge
    reduced_mass = projected.T @ (weights[:, None] * projected)
    result, modal_audit = diagonalize_basis(
        x,
        projected.reshape(len(x), 3, 4).transpose(0, 2, 1),
        basis.sample_mass,
        np.diag(basis.mode_stiffness),
        cfg,
        "floating_curved_straw",
    )
    new_phi = result.sample_phi.transpose(0, 2, 1).reshape(-1, 4).astype(float)
    transform = np.linalg.solve(reduced_mass, projected.T @ (weights[:, None] * new_phi))
    stiffness = transform.T @ np.diag(basis.mode_stiffness) @ transform
    cross = rigid.T @ (weights[:, None] * new_phi)
    normalized = cross / np.sqrt(np.diag(frame_mass)[:, None] * result.mode_mass[None, :])
    sections = straw_sections(cfg)
    centers = np.tile(np.repeat(centerline(cfg, sections)[0], 40, axis=0), (2, 1))
    omega = np.tile(np.repeat(rotation_modes(cfg, sections)[0], 40, axis=0), (2, 1, 1))
    center_modes = np.tile(np.repeat(centerline_modes(cfg, sections), 40, axis=0), (2, 1, 1))
    raw = (center_modes + np.cross(omega, (x - centers)[:, None])).transpose(0, 2, 1).reshape(-1, 4)
    old_transform = np.linalg.solve(raw.T @ (weights[:, None] * raw), raw.T @ (weights[:, None] * phi))
    old_psi = np.einsum("nmc,mk->nkc", omega, old_transform)
    result.sample_psi[:] = np.einsum("nmc,mk->nkc", old_psi - gauge[3:].T[None], transform)
    damping = transform.T @ np.diag(basis.mode_damping) @ transform
    audit = {
        "combined_span_max_error": float(np.abs(new_phi - projected @ transform).max()),
        "stiffness_transform_relative_error": float(
            np.linalg.norm(stiffness - np.diag(result.mode_stiffness)) / np.linalg.norm(stiffness)
        ),
        "normalized_rigid_modal_mass_coupling": float(np.abs(normalized).max()),
        "dense_transformed_damping_off_diagonal_fraction": float(
            np.linalg.norm(damping - np.diag(np.diag(damping))) / np.linalg.norm(damping)
        ),
        "root_phi_is_nonzero": bool(np.linalg.norm(result.sample_phi[x[:, 2] < cfg.clamp_length]) > 0),
        "root_psi_is_nonzero": bool(np.linalg.norm(result.sample_psi[x[:, 2] < cfg.clamp_length]) > 0),
        "full_stiffness_congruence_applied": True,
        "gauge_coefficients": gauge.tolist(),
        "modal_transform": transform.tolist(),
        "original_to_modes": old_transform.tolist(),
        "exact_transformed_original_damping": damping.tolist(),
        "damping_model": "Assumed 0.28 damping ratio in each free-body elastic mode; not a diagonal truncation of the original dense transformed damping",
        "modal_audit": modal_audit,
    }
    return result, audit


def projected_root_samples(point, com, audit):
    """Exact projected Phi/Psi within the original rigid lower section."""
    gauge = np.asarray(audit["gauge_coefficients"])
    transform = np.asarray(audit["modal_transform"])
    phi = -(gauge[:3].T + np.cross(gauge[3:].T, np.asarray(point) - com))
    return transform.T @ phi, -transform.T @ gauge[3:].T


@wp.kernel
def open_fixture(
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
    opening = ramp(clock[0], release, 0.25)
    speed = (ramp(clock[0] + dt, release, 0.25) - opening) / dt
    pose0[body] = wp.transform(wp.vec3(side * 0.065 * opening, 0.065 * opening, 0.012), wp.quat_identity())
    pose1[body] = pose0[body]
    velocity0[body] = wp.spatial_vector(wp.vec3(side * 0.065 * speed, 0.065 * speed, 0.0), wp.vec3(0.0))
    velocity1[body] = velocity0[body]


class Example(UtensilExample):
    """Curved floating-frame straw with contact bending and releasable root support."""

    slug, title, kind = "straw", "Rubber drinking straw", 0
    default_output = Path("output/straw")
    elastic_color = (0.035, 0.48, 0.43)
    camera: ClassVar[dict] = {"position": [0.10, -0.50, 0.38], "target": [-0.065, 0.0, 0.105], "fov": 32}
    description = "A hollow straw with a short 30-degree bent tip is deflected by a coral roller. Its support jaws open, releasing the entire straw to recoil and drop under gravity."
    model_description = "One finite-mass curved elastic tube with four mass-orthogonal curved-rod Ritz modes, a releasable fixture joint, rigid roller and colliding floor."
    driven = "Only the roller and visible support jaws are prescribed. The straw frame and all four modal coordinates are solved by Newton VBD; the support joint releases at 3.8 seconds."
    limitations: ClassVar[list[str]] = [
        "Generic procedural silicone straw; assumed 6 MPa linear elastic modulus, not calibrated material.",
        "Small-displacement curved Kirchhoff-rod ROM around the bent rest shape; no ovalization, crushing, buckling, self-contact, fluid, or fracture.",
        "Lower 35 mm is rigid within the ROM, but moves with the finite-mass straw. Its visible fixture restraint is a releasable fixed joint, not a frictional clamp model.",
        "Rotary inertia uses surface-ring mass lumping; contact uses the actual hollow surface vertices.",
        "Linear kinematics do not enforce finite-rotation inextensibility; measured edge strain bounds this run.",
        "Free-frame basis mass-projects rigid fields and transforms the full stiffness; this is a basis choice, not a runtime fix.",
        "Runtime inertial moments/couplings remain frozen at rest. Near-zero projected first moments suppress the observed boost defect, but do not restore general finite-deformation inertia; this is a bounded small-strain approximation.",
        "Damping assumes a 0.28 ratio in each free-body elastic mode. The original clamped-mode damping transforms to a dense matrix and is not retained.",
        "Floor contacts are generated within 1.5 mm to catch the drop at four substeps; collision margins remain zero and penetration uses the actual mesh.",
    ]
    solver_options: ClassVar[dict] = {
        "rigid_joint_linear_ke": 2e4,
        "rigid_joint_angular_ke": 100.0,
        "rigid_joint_linear_kd": 0.02,
        "rigid_joint_angular_kd": 0.001,
    }

    def __init__(self, viewer=None, cfg=None):
        self.cfg, self.viewer = cfg or Config(), viewer
        c = self.cfg
        self.vertices, self.faces, self.basis, self.audit = build_straw(c)
        self.root_anchor = np.array([0.0, 0.0, 0.020])
        self.root_phi, self.root_psi = np.zeros((4, 3)), np.zeros((4, 3))
        if c.floating_basis:
            com = np.average(self.vertices.astype(float), weights=self.basis.sample_mass, axis=0)
            self.basis, projection = project_rigid_fields(c, self.basis, com)
            self.audit["floating_frame"] = projection
            self.audit.update(projection["modal_audit"])
            self.audit["method"] = (
                "Curved Kirchhoff Ritz modes; remove six rigid fields by mass projection, then full mass/stiffness eigensolve"
            )
            self.root_phi, self.root_psi = projected_root_samples(self.root_anchor, com, projection)
            self.basis.add_sample(self.root_anchor, phi=self.root_phi, psi=self.root_psi)
        self.sections = straw_sections(c)
        material_s = np.tile(np.repeat(self.sections, 40), 2)
        self.clamped = self.vertices[:, 2] <= c.clamp_length + 1e-7
        self.tip = material_s > c.length - 1e-7
        self.tip_left = self.tip & (self.vertices[:, 1] > 0.001)
        self.tip_right = self.tip & (self.vertices[:, 1] < -0.001)
        gravity_force = -9.81 * np.einsum("n,nm->m", self.basis.sample_mass, self.basis.sample_phi[:, :, 2])
        self.gravity_equilibrium_q = gravity_force / self.basis.mode_stiffness
        self.no_contact_displacement_limit_m = max(0.0001, 2 * float(np.abs(self.gravity_equilibrium_q).sum()))
        if c.floating_basis:
            self.audit["free_frame_uniform_gravity_modal_force_N"] = gravity_force.tolist()
            self.audit["gravity_projection_note"] = (
                "Uniform gravity has zero free-frame elastic projection; the attached fixture still causes bending, measured in the no-roller control."
            )
        else:
            self.audit["linear_gravity_equilibrium_q"] = self.gravity_equilibrium_q.tolist()
            self.audit["no_contact_displacement_bound_m"] = self.no_contact_displacement_limit_m
        if c.tip_angle_degrees == 0:
            self.description = (
                "Straight-straw baseline: a hollow reusable straw is clamped, deflected by a roller, then released."
            )
            self.model_description = (
                "Previously validated straight tube with four mass-orthogonal polynomial bending modes."
            )
        self.visuals = []
        self.released = False
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81), up_axis="Z")
        builder.num_rigid_contacts_per_world = 8192
        mass = self.audit["mass_kg"]
        # The weak-axis inertia is a small difference of axial second moments.
        weights = np.asarray(self.basis.sample_mass[: len(self.vertices)], dtype=np.float64)
        mass = float(weights.sum())
        com = np.sum(self.vertices.astype(np.float64) * weights[:, None], axis=0) / mass
        r = self.vertices.astype(np.float64) - com
        inertia = np.eye(3) * np.sum(weights * np.sum(r * r, axis=1)) - (r * weights[:, None]).T @ r
        self.elastic_body = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0, 0, 0.012), wp.quat_identity()),
            mass=mass,
            com=wp.vec3(*com),
            inertia=wp.mat33(inertia),
            modal_basis=self.basis,
            lock_inertia=True,
            label="dynamic_hollow_silicone_straw",
        )
        builder.add_shape_mesh(
            self.elastic_body,
            mesh=newton.Mesh(self.vertices, self.faces.flatten()),
            cfg=shape_config(c),
            color=self.elastic_color,
        )
        self.driven_rotation = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2)
        self.driven_body = builder.add_body(
            xform=wp.transform(target_position(0.0, 0), self.driven_rotation),
            is_kinematic=True,
            label="prescribed_roller",
        )
        self.obstacle_radius = 0.012
        self.roller_shape = builder.add_shape_capsule(
            self.driven_body,
            radius=self.obstacle_radius,
            half_height=0.023,
            cfg=shape_config(c, c.contact_enabled),
            color=(0.85, 0.16, 0.09),
            label="visible_contact_roller",
        )
        v, f = capsule_mesh([0, 0, -0.023], [0, 0, 0.023], self.obstacle_radius)
        # Collision shape is already displayed by ViewerGL; this record supplies the interchange mesh.
        self.visuals.append(
            {
                "name": "contact_roller",
                "vertices": v,
                "faces": f,
                "body": self.driven_body,
                "color": (0.85, 0.16, 0.09),
                "roughness": 0.3,
                "metallic": 0.0,
            }
        )
        v, f = capsule_mesh([0, 0, -0.048], [0, 0, 0.048], 0.003)
        self.add_visual(builder, "roller_axle", v, f, (0.46, 0.48, 0.50), self.driven_body, metallic=0.8)
        self.fixture_bodies = []
        for side in (-1.0, 1.0):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(0, 0, 0.012), wp.quat_identity()),
                is_kinematic=True,
                label=f"retracting_fixture_{side}",
            )
            self.fixture_bodies.append(body)
            v, f = capsule_mesh([side * 0.0085, 0, 0.008], [side * 0.0085, 0, 0.030], 0.004, 40, 8)
            self.add_visual(builder, f"steel_clamp_jaw_{side}", v, f, (0.43, 0.47, 0.49), body, metallic=0.75)
            v, f = capsule_mesh([side * 0.009, 0, 0.020], [side * 0.033, 0, 0.020], 0.004, 40, 8)
            self.add_visual(builder, f"clamp_slide_{side}", v, f, (0.25, 0.29, 0.31), body, metallic=0.65)
            v, f = lathe(
                [(0.0001, -0.012), (0.014, -0.012), (0.015, -0.007), (0.015, -0.003), (0.012, 0), (0.0001, 0)], 64
            )
            v[:, 0] += side * 0.035
            self.add_visual(builder, f"fixture_foot_{side}", v, f, (0.30, 0.34, 0.36), body, metallic=0.65)
            v, f = capsule_mesh([side * 0.035, 0, 0], [side * 0.035, 0, 0.020], 0.005, 40, 8)
            self.add_visual(builder, f"fixture_post_{side}", v, f, (0.43, 0.47, 0.49), body, metallic=0.75)
        anchor = wp.transform(wp.vec3(0, 0, 0.020), wp.quat_identity())
        self.support_joint = builder.add_joint_fixed(
            parent=self.fixture_bodies[0],
            child=self.elastic_body,
            parent_xform=anchor,
            child_xform=anchor,
            label="releasable_straw_support",
        )
        floor_cfg = shape_config(c, c.diagnostic_floor_enabled)
        floor_cfg.gap = c.floor_contact_gap
        self.floor_shape = builder.add_ground_plane(cfg=floor_cfg)
        self.finish_setup(builder)
        endpoint = int(self.model.joint_child_elastic_endpoint.numpy()[self.support_joint])
        endpoint_phi = self.model.elastic_endpoint_phi.numpy().reshape(-1, 4, 3)[endpoint]
        endpoint_psi = self.model.elastic_endpoint_psi.numpy().reshape(-1, 4, 3)[endpoint]
        np.testing.assert_allclose(endpoint_phi, self.root_phi, atol=1e-6)
        np.testing.assert_allclose(endpoint_psi, self.root_psi, atol=1e-5)
        self.root_phi, self.root_psi = endpoint_phi.copy(), endpoint_psi.copy()

    def simulate(self):
        for _ in range(self.cfg.substeps):
            wp.launch(
                drive_frame,
                dim=1,
                inputs=[
                    self.clock,
                    self.dt,
                    self.kind,
                    self.driven_body,
                    -1,
                    -1,
                    self.driven_rotation,
                    self.state_0.body_q,
                    self.state_1.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_qd,
                    self.state_0.joint_q,
                    self.state_1.joint_q,
                    self.state_0.joint_qd,
                    self.state_1.joint_qd,
                ],
                device=self.model.device,
            )
            for side, body in zip((-1.0, 1.0), self.fixture_bodies, strict=True):
                wp.launch(
                    open_fixture,
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
            self.capacity_audit.after_collide()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.capacity_audit.after_solve()
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(tick, dim=1, inputs=[self.clock, self.dt], device=self.model.device)

    def step(self):
        if not self.released and self.sim_time >= self.cfg.release_time - 1e-8:
            enabled = self.model.joint_enabled.numpy()
            enabled[self.support_joint] = False
            self.model.joint_enabled.assign(enabled)
            self.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
            self.released = True
        super().step()

    def surface_distance(self, samples, poses):
        a, b = self.obstacle_endpoints(poses)
        self.roller_penetration = max(0.0, -float(capsule_sdf(samples, a, b, self.obstacle_radius).min()))
        self.floor_penetration = max(0.0, -float(samples[:, 2].min()))
        return np.array(
            [
                -max(
                    self.floor_penetration if self.cfg.diagnostic_floor_enabled else 0.0,
                    self.roller_penetration if self.cfg.contact_enabled else 0.0,
                )
            ]
        )

    def update_metrics(self):
        super().update_metrics()
        record = self.records[-1]
        poses = self.saved_body_q[-1]
        pose = poses[self.elastic_body]
        amplitudes = np.asarray(record["q"])
        anchor = self.root_anchor[None]
        root_local = self.root_anchor + amplitudes @ self.root_phi
        theta = amplitudes @ self.root_psi
        angle = float(np.linalg.norm(theta))
        rotation = wp.quat_from_axis_angle(wp.vec3(*(theta / angle)), angle) if angle > 1e-10 else wp.quat_identity()
        root_rotation = wp.quat(*pose[3:]) * rotation
        root_world = transform_vertices(root_local[None], pose)[0]
        root_pose = np.r_[root_world, np.asarray(root_rotation)]
        local = self.vertices + np.einsum("nmc,m->nc", self.phi, amplitudes)
        world = transform_vertices(local, pose)
        root_reference = transform_vertices(self.vertices - self.root_anchor, root_pose)
        bending = world - root_reference
        fixture = poses[self.fixture_bodies[0]]
        angular_error = wp.quat_inverse(wp.quat(*fixture[3:])) * root_rotation
        record["floating_frame_modal_displacement_m"] = record["max_deformation_m"]
        record["max_deformation_m"] = float(np.linalg.norm(bending, axis=1).max())
        record["max_clamp_displacement_m"] = (
            float(
                np.linalg.norm(
                    world[self.clamped] - transform_vertices(self.vertices[self.clamped], fixture), axis=1
                ).max()
            )
            if not self.released
            else 0.0
        )
        count = record["contacts"]
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:count]
        record.update(
            support_enabled=bool(self.model.joint_enabled.numpy()[self.support_joint]),
            support_joint_error_m=float(np.linalg.norm(root_world - transform_vertices(anchor, fixture)[0])),
            support_joint_angular_error_rad=float(
                2 * np.arctan2(np.linalg.norm(np.asarray(angular_error)[:3]), abs(float(angular_error[3])))
            ),
            root_modal_rotation_rad=angle,
            roller_contacts=int(np.sum((shape0 == self.roller_shape) | (shape1 == self.roller_shape))),
            floor_contacts=int(np.sum((shape0 == self.floor_shape) | (shape1 == self.floor_shape))),
            floor_penetration_m=self.floor_penetration,
            roller_penetration_m=self.roller_penetration,
            frame_position_m=pose[:3].tolist(),
            frame_com_m=transform_vertices(self.model.body_com.numpy()[self.elastic_body][None], pose)[0].tolist(),
            frame_speed_m_s=float(np.linalg.norm(self.saved_body_qd[-1][self.elastic_body, :3])),
            frame_angular_speed_rad_s=float(np.linalg.norm(self.saved_body_qd[-1][self.elastic_body, 3:])),
        )

    def obstacle_endpoints(self, poses):
        center = poses[self.driven_body, :3]
        return center + np.array([0, -0.023, 0]), center + np.array([0, 0.023, 0])

    def summary(self):
        result = super().summary()
        loading = [r for r in self.records if r["support_enabled"]]
        free = [r for r in self.records if not r["support_enabled"]]
        late = [r for r in free if r["time_s"] > 5.5]
        result.update(
            dynamic_release=True,
            driving_contact_metric="max_roller_contacts",
            control_interaction_disabled="Roller collision only; same fixture opening/release, roller poses, gravity, floor and material",
            driven_body_indices=[self.driven_body, *self.fixture_bodies],
            release_time_s=self.cfg.release_time,
            straw_mass_kg=self.audit["mass_kg"],
            max_roller_contacts=max(r["roller_contacts"] for r in self.records),
            max_floor_contacts=max(r["floor_contacts"] for r in self.records),
            max_support_joint_error_m=max(r["support_joint_error_m"] for r in loading),
            max_support_joint_angular_error_rad=max(r["support_joint_angular_error_rad"] for r in loading),
            max_root_modal_rotation_rad=max(r["root_modal_rotation_rad"] for r in self.records),
            max_floating_frame_modal_displacement_m=max(r["floating_frame_modal_displacement_m"] for r in self.records),
            deformation_definition="Maximum surface displacement relative to the actual deformed material-root pose; excludes the floating rigid frame",
            clamp_displacement_definition="Maximum held lower-section world displacement relative to the visible fixture; excludes released frames",
            loading_max_deformation_m=max(r["max_deformation_m"] for r in loading),
            max_free_frame_speed_m_s=max((r["frame_speed_m_s"] for r in free), default=0),
            max_free_frame_angular_speed_rad_s=max((r["frame_angular_speed_rad_s"] for r in free), default=0),
            free_body_com_drop_m=float(free[0]["frame_com_m"][2] - free[-1]["frame_com_m"][2]) if free else 0.0,
            max_free_frame_translation_m=max(
                (float(np.linalg.norm(np.array(r["frame_position_m"]) - [0, 0, 0.012])) for r in free), default=0
            ),
            post_release_min_deformation_m=min((r["max_deformation_m"] for r in free), default=None),
            recovery_residual_m=max((r["max_deformation_m"] for r in late), default=None),
            recovery_definition="Maximum root-relative bending after 5.5 s; grounded straw may still carry contact loads",
            penetration_samples="Dense straw skin against the exact roller capsule and floor plane; disabled roller excluded in the control",
        )
        result["rest_tip_angle_deg"] = self.cfg.tip_angle_degrees
        result["variant"] = "bent-tip-release" if self.cfg.tip_angle_degrees else "straight-release"
        return result

    def test_final(self):
        self.capacity_audit.check(len(self.records) * self.cfg.substeps)
        s = self.summary()
        assert s["frames"] >= 360 and s["graph_enabled"] and self.cfg.substeps == 4, s
        assert self.released and not self.model.joint_enabled.numpy()[self.support_joint]
        assert self.model.body_inv_mass.numpy()[self.elastic_body] > 0
        assert not self.model.body_flags.numpy()[self.elastic_body] & int(newton.BodyFlags.KINEMATIC)
        assert s["max_support_joint_error_m"] < 0.0005 and s["max_clamp_displacement_m"] < 0.0005, s
        assert s["max_support_joint_angular_error_rad"] < 0.005, s
        assert s["max_edge_strain"] < 0.05 and s["max_penetration_m"] < 0.001, s
        assert s["max_modal_solve_residual_ratio"] < 0.03 and s["max_deformation_m"] < 0.035, s
        assert s["max_free_frame_translation_m"] > 0.010 and s["max_free_frame_speed_m_s"] > 0.10, s
        assert s["max_floor_contacts"] > 0 and s["recovery_residual_m"] < 0.005, s
        assert s["max_free_frame_angular_speed_rad_s"] < 40 and s["max_root_modal_rotation_rad"] < 0.15, s
        if self.cfg.contact_enabled:
            assert 0.010 < s["loading_max_deformation_m"] < 0.025 and s["max_roller_contacts"] > 0, s
            assert s["free_body_com_drop_m"] > 0.07, s
            assert s["post_release_min_deformation_m"] < 0.003, s
        else:
            assert s["max_roller_contacts"] == 0 and s["loading_max_deformation_m"] < 0.0045, s

    def export(self, output, result):
        super().export(output, result)
        centers, tangents, normals, curvature = centerline(self.cfg, self.sections)
        np.savez_compressed(
            output / "rest_centerline.npz",
            arclength=self.sections,
            centers=centers,
            tangents=tangents,
            normals=normals,
            curvature=curvature,
            raw_centerline_modes=centerline_modes(self.cfg, self.sections),
            raw_rotation_modes=rotation_modes(self.cfg, self.sections)[0],
            centerline_phi=self.phi.reshape(2, len(self.sections), 40, 4, 3).mean(axis=(0, 2)),
        )
        path = output / "scene.json"
        scene = json.loads(path.read_text())
        scene["audit"]["rest_centerline"] = "rest_centerline.npz"
        scene["variant"] = result["variant"]
        scene["rest_tip_angle_deg"] = self.cfg.tip_angle_degrees
        scene["poster_frame"] = int(
            np.argmax([r["max_deformation_m"] if r["support_enabled"] else 0 for r in self.records])
        )
        scene["audit"]["mechanism"] = "mechanism.npz"
        if not self.cfg.contact_enabled:
            scene["description"] = (
                "Roller collision disabled; fixture release, gravity and floor collision remain enabled. "
                + self.description
            )
        path.write_text(json.dumps(scene, indent=2) + "\n")
        np.savez_compressed(
            output / "mechanism.npz",
            body_mass=self.model.body_mass.numpy(),
            body_com=self.model.body_com.numpy(),
            body_inertia=self.model.body_inertia.numpy(),
            body_flags=self.model.body_flags.numpy(),
            root_phi=self.root_phi,
            root_psi=self.root_psi,
            surface_psi=self.basis.sample_psi[: len(self.vertices)],
            support_joint=self.support_joint,
            support_enabled=[r["support_enabled"] for r in self.records],
            joint_parent=self.model.joint_parent.numpy(),
            joint_child=self.model.joint_child.numpy(),
            joint_parent_xform=self.model.joint_X_p.numpy(),
            joint_child_xform=self.model.joint_X_c.numpy(),
            driven_bodies=[self.driven_body, *self.fixture_bodies],
        )
        source = output / "source"
        source.mkdir(exist_ok=True)
        for name in ("straw.py", "spatula.py", "utensil_geometry.py", "test_utensils.py"):
            shutil.copy2(Path(__file__).with_name(name), source / name)


if __name__ == "__main__":
    run_cli(Example, Config, "straw")
