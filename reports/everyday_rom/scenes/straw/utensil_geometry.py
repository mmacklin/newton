# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Procedural utensil meshes, beam ROM construction, and auditable stepping.

Only Newton public APIs are used. Units are SI. The beam energy model is an
assumed engineering approximation, not a fitted silicone constitutive law.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp
from capacity_audit import CapacityAudit

import newton
import newton.solvers


@dataclass
class Config:
    substeps: int = 4
    iterations: int = 16
    contact_enabled: bool = True
    contact_ke: float = 2.0e4
    contact_kd: float = 0.03
    contact_gap: float = 0.00025
    friction: float = 0.30
    damping_ratio: float = 0.28
    density: float = 1100.0


def shape_config(cfg, collision=True):
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


def connect_rings(rings, cap=False):
    """Join ordered rings, with optional end fans; orient outward afterwards."""
    rings = np.asarray(rings, dtype=np.float64)
    n, k, _ = rings.shape
    vertices = rings.reshape(-1, 3).tolist()
    faces = []
    for j in range(n - 1):
        for i in range(k):
            a, b = j * k + i, j * k + (i + 1) % k
            c, d = b + k, a + k
            faces.extend(((a, b, c), (a, c, d)))
    if cap:
        for j, reverse in ((0, True), (n - 1, False)):
            center = len(vertices)
            vertices.append(rings[j].mean(axis=0).tolist())
            for i in range(k):
                a, b = j * k + i, j * k + (i + 1) % k
                faces.append((center, b, a) if reverse else (center, a, b))
    return orient_mesh(np.asarray(vertices), np.asarray(faces, dtype=np.int32))


def orient_mesh(vertices, faces):
    volume = (
        np.einsum("ij,ij->i", vertices[faces[:, 0]], np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum() / 6
    )
    if volume < 0:
        faces = faces[:, ::-1].copy()
    return vertices.astype(np.float32), faces.astype(np.int32)


def lathe(profile, segments=96):
    """Closed rotational profile (radius, height), without duplicated poles."""
    theta = np.arange(segments) * 2 * np.pi / segments
    rings = [np.column_stack((r * np.cos(theta), r * np.sin(theta), np.full(segments, z))) for r, z in profile]
    # Profiles use a tiny finite inner radius so all rings remain nondegenerate.
    rings.append(rings[0])
    v, f = connect_rings(rings)
    f[f >= (len(profile) * segments)] -= len(profile) * segments
    return orient_mesh(v[:-segments], f)


def capsule_mesh(a, b, radius, segments=48, hemispheres=10):
    """A smooth closed capsule matching Newton's capsule SDF [m]."""
    a, b = np.asarray(a), np.asarray(b)
    direction = (b - a) / np.linalg.norm(b - a)
    u = np.cross(direction, [0, 0, 1])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(direction, [0, 1, 0])
    u /= np.linalg.norm(u)
    w = np.cross(direction, u)
    theta = np.arange(segments) * 2 * np.pi / segments
    circle = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * w
    rings = []
    for center, angles in (
        (a, np.linspace(-np.pi / 2, 0, hemispheres + 1)[1:]),
        (b, np.linspace(0, np.pi / 2, hemispheres + 1)[:-1]),
    ):
        for angle in angles:
            rings.append(center + radius * np.sin(angle) * direction + radius * np.cos(angle) * circle)
    v, f = connect_rings(rings, cap=True)
    v[-2] = a - radius * direction
    v[-1] = b + radius * direction
    return orient_mesh(v, f)


def capsule_sdf(points, a, b, radius):
    d = np.asarray(b) - a
    t = np.clip((points - a) @ d / (d @ d), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * d), axis=1) - radius


def capsule_pose(a, b):
    a, b = np.asarray(a), np.asarray(b)
    axis = (b - a) / np.linalg.norm(b - a)
    return wp.transform(wp.vec3(*((a + b) / 2)), wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.vec3(*axis)))


def transform_vertices(vertices, pose):
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)
    return np.asarray(vertices) @ rotation.T + pose[:3]


def mesh_edges(faces):
    return np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)


def diagonalize_basis(vertices, raw_phi, sample_mass, stiffness, cfg, label):
    """Generalized eigensolve of the sampled mass and beam-energy stiffness."""
    p = raw_phi.transpose(0, 2, 1).reshape(-1, raw_phi.shape[1])
    mass = p.T @ (np.repeat(sample_mass, 3)[:, None] * p)
    vals, vecs = np.linalg.eigh(mass)
    assert vals.min() > 1e-12, vals
    whitening = vecs / np.sqrt(vals)[None, :]
    omega2, eigenvectors = np.linalg.eigh(whitening.T @ stiffness @ whitening)
    assert omega2.min() > 0, omega2
    transform = whitening @ eigenvectors
    normalized = p @ transform
    scale = np.linalg.norm(normalized.reshape(len(vertices), 3, -1), axis=1).max(axis=0)
    transform /= scale[None, :]
    phi = (p @ transform).reshape(len(vertices), 3, -1).transpose(0, 2, 1)
    masses = 1 / scale**2
    stiffnesses = masses * omega2
    basis = newton.ModalBasis(
        sample_points=vertices,
        sample_phi=phi,
        sample_mass=sample_mass,
        mode_mass=masses,
        mode_stiffness=stiffnesses,
        mode_damping=2 * cfg.damping_ratio * np.sqrt(masses * stiffnesses),
        label=label,
    )
    gram = transform.T @ mass @ transform
    audit = {
        "method": "Rayleigh-Ritz polynomial beam modes; generalized mass/stiffness eigensolve",
        "mass_kg": float(sample_mass.sum()),
        "mode_count": basis.mode_count,
        "frequency_hz": (np.sqrt(omega2) / (2 * np.pi)).tolist(),
        "mass_orthogonality_relative_error": float(np.linalg.norm(gram - np.diag(masses)) / np.linalg.norm(gram)),
        "sample_mass_model": "Cross-section volume lumped onto surface rings; approximate rotary inertia",
        "mode_mass_kg": masses.tolist(),
        "mode_stiffness_N_per_m": stiffnesses.tolist(),
    }
    return basis, audit


@wp.func
def ramp(t: float, start: float, duration: float):
    x = wp.clamp((t - start) / duration, 0.0, 1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


@wp.func
def target_position(t: float, kind: int):
    load = ramp(t, 0.6, 1.5) * (1.0 - ramp(t, 3.8, 1.0))
    if kind == 0:
        return wp.vec3(0.038 - 0.031 * load, 0.0, 0.167)
    sweep = ramp(t, 2.6, 1.0) * (1.0 - ramp(t, 3.8, 1.0))
    return wp.vec3(0.009 * sweep, 0.014 * sweep, 0.113 - 0.038 * load)


@wp.kernel
def drive_frame(
    clock: wp.array[float],
    dt: float,
    kind: int,
    body: int,
    q_start: int,
    qd_start: int,
    rotation: wp.quat,
    pose0: wp.array[wp.transform],
    pose1: wp.array[wp.transform],
    vel0: wp.array[wp.spatial_vector],
    vel1: wp.array[wp.spatial_vector],
    q0: wp.array[float],
    q1: wp.array[float],
    qd0: wp.array[float],
    qd1: wp.array[float],
):
    t = clock[0]
    pos = target_position(t, kind)
    v = (target_position(t + dt, kind) - pos) / dt
    pose0[body] = wp.transform(pos, rotation)
    pose1[body] = wp.transform(pos, rotation)
    vel0[body] = wp.spatial_vector(v, wp.vec3(0.0))
    vel1[body] = wp.spatial_vector(v, wp.vec3(0.0))
    # Only the seven floating-frame coordinates and six frame velocities are
    # prescribed. No modal coordinate or modal velocity is written here.
    if q_start >= 0:
        for j in range(3):
            q0[q_start + j] = pos[j]
            q1[q_start + j] = pos[j]
            qd0[qd_start + j] = v[j]
            qd1[qd_start + j] = v[j]
            qd0[qd_start + j + 3] = 0.0
            qd1[qd_start + j + 3] = 0.0
        for j in range(4):
            q0[q_start + 3 + j] = rotation[j]
            q1[q_start + 3 + j] = rotation[j]


@wp.kernel
def tick(clock: wp.array[float], dt: float):
    clock[0] += dt


class UtensilExample:
    """Shared frame capture and metrics for the two owned example modules."""

    def finish_setup(self, builder):
        builder.color()
        self.model = builder.finalize()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.pipeline = newton.CollisionPipeline(self.model, reduce_contacts=False, rigid_contact_max=16384)
        self.contacts, self.control = self.pipeline.contacts(), self.model.control()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.cfg.iterations,
            rigid_body_contact_buffer_size=4096,
            rigid_contact_k_start=self.cfg.contact_ke,
            rigid_compliant_alm=False,
            rigid_contact_hard=False,
            friction_epsilon=1e-5,
            elastic_body_relaxation=getattr(self.cfg, "elastic_relaxation", 0.6),
            elastic_solver_metrics=True,
            **getattr(self, "solver_options", {}),
        )
        elastic = int(self.model.body_elastic_index.numpy()[self.elastic_body])
        owner = int(self.model.elastic_joint.numpy()[elastic])
        self.frame_q_start = int(self.model.joint_q_start.numpy()[owner])
        self.frame_qd_start = int(self.model.joint_qd_start.numpy()[owner])
        self.q_start = self.frame_q_start + 7
        self.phi = (
            self.model.elastic_shape_vertex_phi.numpy()
            .reshape(-1, self.basis.mode_count, 3)[: len(self.vertices)]
            .copy()
        )
        np.testing.assert_allclose(self.phi, self.basis.sample_phi[: len(self.vertices)], atol=1e-5)
        np.testing.assert_allclose(
            self.model.elastic_shape_vertex_local.numpy()[: len(self.vertices)], self.vertices, atol=1e-7
        )
        self.edges = mesh_edges(self.faces)
        self.rest_lengths = np.linalg.norm(self.vertices[self.edges[:, 0]] - self.vertices[self.edges[:, 1]], axis=1)
        self.dt = 1 / (60 * self.cfg.substeps)
        self.clock = wp.zeros(1, dtype=float, device=self.model.device)
        self.capacity_audit = CapacityAudit(self.model, self.contacts, self.solver, self.clock, self.dt)
        self.sim_time = 0.0
        self.records, self.timings, self.saved_body_q, self.saved_joint_q = [], [], [], []
        self.saved_body_qd, self.saved_joint_qd, self.saved_clock = [], [], []
        self.graph = None
        if not self.model.device.is_cuda:
            raise RuntimeError("These report simulations require a CUDA device")
        if self.cfg.substeps % 2:
            raise ValueError("CUDA graph state swaps require an even substep count")
        start = time.perf_counter()
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph
        wp.synchronize()
        self.capture_seconds = time.perf_counter() - start
        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.show_elastic_strain = False

    def add_visual(self, builder, name, vertices, faces, color, body=-1, metallic=0.0, roughness=0.35):
        builder.add_shape_mesh(
            body,
            mesh=newton.Mesh(vertices, faces.flatten()),
            cfg=shape_config(self.cfg, False),
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
                "metallic": metallic,
                "roughness": roughness,
            }
        )

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
                    self.frame_q_start if self.kind == 1 else -1,
                    self.frame_qd_start if self.kind == 1 else -1,
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
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.capacity_audit.after_collide()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.capacity_audit.after_solve()
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(tick, dim=1, inputs=[self.clock, self.dt], device=self.model.device)

    def step(self):
        start = time.perf_counter()
        wp.capture_launch(self.graph)
        wp.synchronize()
        self.timings.append(time.perf_counter() - start)
        self.sim_time += 1 / 60
        self.update_metrics()

    def update_metrics(self):
        poses, q = self.state_0.body_q.numpy(), self.state_0.joint_q.numpy()
        qd, velocities = self.state_0.joint_qd.numpy(), self.state_0.body_qd.numpy()
        assert all(np.isfinite(a).all() for a in (poses, q, qd, velocities)), "Nonfinite Newton state"
        amplitudes = q[self.q_start : self.q_start + self.basis.mode_count]
        displacement = np.einsum("nmc,m->nc", self.phi, amplitudes)
        local = self.vertices + displacement
        world = transform_vertices(local, poses[self.elastic_body])
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
        distance = self.surface_distance(samples, poses)
        lengths = np.linalg.norm(local[self.edges[:, 0]] - local[self.edges[:, 1]], axis=1)
        metrics = self.solver.elastic_mode_solve_metrics()
        self.records.append(
            {
                "time_s": self.sim_time,
                "max_deformation_m": float(np.linalg.norm(displacement, axis=1).max()),
                "penetration_m": max(0.0, -float(distance.min())),
                "max_edge_strain": float(np.abs(lengths / self.rest_lengths - 1).max()),
                "max_clamp_displacement_m": float(np.linalg.norm(displacement[self.clamped], axis=1).max()),
                "tip_displacement_m": displacement[self.tip].mean(axis=0).tolist(),
                "twist_displacement_m": float(
                    displacement[self.tip_left, 2].mean() - displacement[self.tip_right, 2].mean()
                ),
                "contacts": int(self.contacts.rigid_contact_count.numpy()[0]),
                "modal_solve_residual_ratio": float(
                    metrics["solve_residual_norm"][0] / max(metrics["initial_residual_norm"][0], 1e-12)
                ),
                "modal_update_norm": float(metrics["update_norm"][0]),
                "modal_update_max": float(metrics["update_max"][0]),
                "q": amplitudes.tolist(),
            }
        )
        self.saved_body_q.append(poses.copy())
        self.saved_joint_q.append(q.copy())
        self.saved_body_qd.append(velocities.copy())
        self.saved_joint_qd.append(qd.copy())
        self.saved_clock.append(float(self.clock.numpy()[0]))

    def surface_distance(self, samples, poses):
        a, b = self.obstacle_endpoints(poses)
        return capsule_sdf(samples, a, b, self.obstacle_radius)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def summary(self):
        deformation = np.array([r["max_deformation_m"] for r in self.records])
        baseline = [r for r in self.records if 0.35 < r["time_s"] < 0.55]
        hold = [r for r in self.records if 2.2 < r["time_s"] < 2.55]
        release = [r for r in self.records if r["time_s"] > 5.5]
        baseline_q = np.mean([r["q"] for r in baseline], axis=0) if baseline else np.zeros(self.basis.mode_count)
        recovery = [
            np.linalg.norm(np.einsum("nmc,m->nc", self.phi, np.array(r["q"]) - baseline_q), axis=1).max()
            for r in release
        ]
        hold_tips = np.array([r["tip_displacement_m"] for r in hold])
        return {
            "slug": self.slug,
            "config": asdict(self.cfg),
            "basis": self.audit,
            "device": str(self.model.device),
            "graph_enabled": self.graph is not None,
            "frames": len(self.records),
            "duration_s": self.sim_time,
            "max_deformation_m": float(deformation.max()),
            "final_deformation_m": float(deformation[-1]),
            "max_displacement_m": float(deformation.max()),
            "final_displacement_m": float(deformation[-1]),
            "baseline_deformation_m": max((r["max_deformation_m"] for r in baseline), default=0),
            "recovery_residual_m": float(max(recovery)) if recovery else None,
            "recovery_definition": "Maximum post-5.5s surface displacement relative to mean pre-contact modal state at 0.35-0.55s",
            "max_penetration_m": max(r["penetration_m"] for r in self.records),
            "penetration_samples": "All vertices, triangle centroids, and edge midpoints against exact capsule SDF",
            "max_edge_strain": max(r["max_edge_strain"] for r in self.records),
            "max_clamp_displacement_m": max(r["max_clamp_displacement_m"] for r in self.records),
            "max_twist_displacement_m": max(abs(r["twist_displacement_m"]) for r in self.records),
            "max_contacts": max(r["contacts"] for r in self.records),
            "hold_contact_dropouts": sum(r["contacts"] == 0 for r in hold),
            "hold_tip_peak_to_peak_m": float(np.ptp(hold_tips, axis=0).max()) if hold else None,
            "max_frame_deformation_step_m": float(np.abs(np.diff(deformation)).max()) if len(deformation) > 1 else 0,
            "max_modal_solve_residual_ratio": max(r["modal_solve_residual_ratio"] for r in self.records),
            "max_modal_update_norm": max(r["modal_update_norm"] for r in self.records),
            "final_modal_update_norm": self.records[-1]["modal_update_norm"],
            "capture_seconds": self.capture_seconds,
            "warm_simulation_frame_ms": float(np.mean(self.timings[10:]) * 1000) if len(self.timings) > 10 else None,
            "timing_scope": "Synchronized CUDA graph, excludes first 10 frames, capture, geometry, metrics/readbacks, and export",
            "finite_state": True,
            "limitations": self.limitations,
        }

    def test_final(self):
        s = self.summary()
        assert s["graph_enabled"] and self.model.elastic_body_count == 1
        assert s["frames"] >= 360, "Run at least six seconds for contact and recovery assertions"
        assert s["max_clamp_displacement_m"] < 1e-7, s
        assert s["max_edge_strain"] < (0.12 if self.kind == 0 else 0.25), s
        assert s["recovery_residual_m"] < 0.0015, s
        assert s["max_frame_deformation_step_m"] < 0.003, s
        assert s["max_modal_solve_residual_ratio"] < 0.03, s
        if not self.cfg.contact_enabled:
            assert s["max_contacts"] == 0, s
            limit = getattr(self, "no_contact_displacement_limit_m", 0.0001 if self.kind == 0 else 0.008)
            assert s["max_deformation_m"] < limit, s
            return
        assert 0.010 < s["max_deformation_m"] < 0.045, s
        assert s["max_penetration_m"] < 0.0010, s
        assert s["hold_contact_dropouts"] == 0, s
        assert s["hold_tip_peak_to_peak_m"] < 0.0010, s
        if self.kind == 1:
            assert s["max_twist_displacement_m"] > 0.002, s

    def export(self, output, result):
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / "state.npz",
            body_q=self.saved_body_q,
            joint_q=self.saved_joint_q,
            body_qd=self.saved_body_qd,
            joint_qd=self.saved_joint_qd,
            times=[r["time_s"] for r in self.records],
            device_clock_s=self.saved_clock,
            elastic_body=self.elastic_body,
            modal_q_start=self.q_start,
        )
        np.savez_compressed(
            output / "basis.npz",
            rest_vertices=self.vertices,
            faces=self.faces,
            phi=self.phi,
            sample_mass=self.basis.sample_mass[: len(self.vertices)],
            mode_mass=self.basis.mode_mass,
            mode_stiffness=self.basis.mode_stiffness,
            mode_damping=self.basis.mode_damping,
            clamped=self.clamped,
        )
        arrays = {"times": np.array([r["time_s"] for r in self.records])}
        meshes = []
        modes = np.asarray(self.saved_joint_q)[:, self.q_start : self.q_start + self.basis.mode_count]
        local = self.vertices[None] + np.einsum("nmc,fm->fnc", self.phi, modes)
        elastic_world = np.stack(
            [transform_vertices(v, poses[self.elastic_body]) for v, poses in zip(local, self.saved_body_q, strict=True)]
        ).astype(np.float32)
        parts = [
            {
                "name": "elastic_" + self.slug,
                "vertices": elastic_world,
                "faces": self.faces,
                "color": self.elastic_color,
                "roughness": 0.38,
                "metallic": 0.0,
            }
        ]
        for visual in self.visuals:
            item = dict(visual)
            body = item.pop("body")
            if body >= 0:
                item["vertices"] = np.stack(
                    [transform_vertices(item["vertices"], poses[body]) for poses in self.saved_body_q]
                ).astype(np.float32)
            parts.append(item)
        for i, part in enumerate(parts):
            arrays[f"vertices_{i}"] = part.pop("vertices")
            arrays[f"faces_{i}"] = part.pop("faces")
            meshes.append(dict(part, vertices=f"vertices_{i}", faces=f"faces_{i}"))
        np.savez_compressed(output / "scene.npz", **arrays)
        scene = {
            "slug": self.slug,
            "title": self.title,
            "status": "validated prototype"
            if result.get("test_passed") and result.get("negative_control_passed")
            else "WIP",
            "fps": 60,
            "meshes": meshes,
            "camera": self.camera,
            "floor_z": 0.0,
            "description": self.description,
            "poster_frame": int(np.argmax([r["max_deformation_m"] for r in self.records])),
            "model": self.model_description,
            "limitations": self.limitations,
            "driven": self.driven,
            "mode_count": self.basis.mode_count,
            "parameters": asdict(self.cfg),
            "source_files": [self.slug + ".py", "utensil_geometry.py", "test_utensils.py"],
            "reproduce": reproduction_command(self.slug),
            "audit": {
                "state": "state.npz",
                "basis": "basis.npz",
                "metrics": "metrics.json",
                "elastic_mesh_index": 0,
                "elastic_body_index": self.elastic_body,
                "modal_q_start": self.q_start,
            },
        }
        if not self.cfg.contact_enabled:
            scene["title"] += " (no-contact negative control)"
            scene["description"] = "Collision disabled; visible intersections are expected. " + self.description
            scene["status"] = "negative-control diagnostic"
        (output / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
        result["surface_recovery_max_error_m"] = verify_export(output)
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        (output / "frame_metrics.json").write_text(json.dumps(self.records, indent=2) + "\n")


def reproduction_command(slug):
    worktree = Path(newton.__file__).resolve().parents[1]
    helpers = Path(__file__).resolve().parent
    return (
        f'env PYTHONPATH="{worktree}" '
        "OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/tmp/newton-everyday-utensils-warp "
        f'uv run --project "{worktree}" python "{helpers}/test_utensils.py" --simulation {slug}'
    )


def compare_negative_control(output):
    """Validate and merge separately locked contact/control experiments."""
    path = output / "metrics.json"
    result = json.loads(path.read_text())
    negative = json.loads((output / "negative-control" / "metrics.json").read_text())
    primary_tested = result.get("test_passed", False) and result.get("test_status") == "passed"
    control_tested = negative.get("test_passed", False) and negative.get("test_status") == "passed"
    for key, value in result["config"].items():
        if key != "contact_enabled":
            assert negative["config"][key] == value, (key, value)
    assert result["frames"] == negative["frames"]
    dynamic = result.get("dynamic_release", False)
    contact_metric = result.get("driving_contact_metric", "max_bowl_contacts") if dynamic else "max_contacts"
    assert negative[contact_metric] == 0
    assert result["config"]["contact_enabled"] and not negative["config"]["contact_enabled"]
    with (
        np.load(output / "state.npz", allow_pickle=False) as primary_state,
        np.load(output / "negative-control" / "state.npz", allow_pickle=False) as control_state,
    ):
        bodies = result["driven_body_indices"] if dynamic else slice(None)
        np.testing.assert_array_equal(primary_state["body_q"][:, bodies], control_state["body_q"][:, bodies])
        np.testing.assert_array_equal(primary_state["times"], control_state["times"])
    measure = "loading_max_deformation_m" if dynamic else "max_deformation_m"
    ratio = negative[measure] / result[measure]
    result["negative_control"] = {
        "interaction_disabled": "Elastic/obstacle collision only; identical trajectory, gravity, material, and solver",
        "max_deformation_m": negative["max_deformation_m"],
        "deformation_ratio": ratio,
        "contact_count": negative["max_contacts"],
        "test_status": negative["test_status"],
    }
    if dynamic:
        result["negative_control"].update(
            interaction_disabled=result.get(
                "control_interaction_disabled",
                "Bowl collision only; identical grip poses/release, gravity, floor, material and solver",
            ),
            comparison_window_s=[0.0, result["release_time_s"]],
            comparison_measure=measure,
            loading_max_deformation_m=negative[measure],
            disabled_interaction_contacts=negative[contact_metric],
            note="Floor impacts remain enabled; compare bending before release, not unmatched later impact maxima.",
        )
        if contact_metric == "max_bowl_contacts":
            result["negative_control"]["bowl_contacts"] = negative[contact_metric]
    result["negative_control_passed"] = bool(primary_tested and control_tested and ratio < 0.25)
    result["negative_control_status"] = "passed" if result["negative_control_passed"] else "failed"
    result["test_passed"] = bool(primary_tested and result["negative_control_passed"])
    result["test_status"] = "passed" if result["test_passed"] else "failed"
    result["negative_control"]["prescribed_poses_identical"] = True
    path.write_text(json.dumps(result, indent=2) + "\n")
    scene_path = output / "scene.json"
    scene = json.loads(scene_path.read_text())
    scene["status"] = "validated prototype" if result["test_passed"] else "WIP"
    scene["reproduce"] = reproduction_command(result["slug"])
    if result.get("variant") == "straight-baseline":
        scene["reproduce"] += " --straight"
    scene_path.write_text(json.dumps(scene, indent=2) + "\n")
    assert result["test_passed"], result["negative_control"]
    return result


def verify_export(output):
    """Independently recover every exported elastic frame from saved arrays."""
    with (
        np.load(output / "state.npz", allow_pickle=False) as state,
        np.load(output / "basis.npz", allow_pickle=False) as basis,
        np.load(output / "scene.npz", allow_pickle=False) as scene,
    ):
        start = int(state["modal_q_start"])
        body = int(state["elastic_body"])
        phi, rest = basis["phi"], basis["rest_vertices"]
        poses, surfaces = state["body_q"], scene["vertices_0"]
        count = phi.shape[1]
        error = 0.0
        for frame, q in enumerate(state["joint_q"]):
            local = rest + np.einsum("nmc,m->nc", phi, q[start : start + count])
            recovered = transform_vertices(local, poses[frame, body])
            error = max(error, float(np.abs(recovered - surfaces[frame]).max()))
        np.testing.assert_allclose(scene["times"], state["times"], atol=1e-10)
        assert error < 5e-8, error
        return error


def run_cli(example_class, config_class, slug):
    parser = argparse.ArgumentParser(description=example_class.__doc__)
    default_output = getattr(example_class, "default_output", Path("output") / slug)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--frames", type=int, default=390)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=config_class().iterations)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--no-contact", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Also run the full no-contact negative control")
    parser.add_argument("--no-test", action="store_true", help="Diagnostic run, marked WIP")
    if slug == "straw":
        parser.add_argument(
            "--straight", action="store_true", help="Original straight variant; writes under straight-baseline/"
        )
    args = parser.parse_args()
    if args.no_test and args.validate:
        parser.error("--no-test cannot be combined with --validate; validation requires both assertion suites")
    if slug == "straw" and args.straight:
        args.output /= "straight-baseline"
    wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", "/tmp/newton-everyday-utensils-warp")
    wp.init()
    wp.set_device(args.device)

    def run(contact, output):
        kwargs = {"tip_angle_degrees": 0.0} if slug == "straw" and args.straight else {}
        cfg = config_class(contact_enabled=contact, substeps=args.substeps, iterations=args.iterations, **kwargs)
        example = example_class(None, cfg)
        for frame in range(args.frames):
            example.step()
            if (frame + 1) % 60 == 0:
                r = example.records[-1]
                print(
                    f"{slug} contact={contact} frame={frame + 1} deformation_mm={r['max_deformation_m'] * 1000:.3f} "
                    f"penetration_mm={r['penetration_m'] * 1000:.3f} contacts={r['contacts']}",
                    flush=True,
                )
        result = example.summary()
        result["test_status"] = "not_run"
        result["test_passed"] = False
        result["negative_control_passed"] = False
        result["negative_control_status"] = (
            "pending; run with --validate" if contact else "this is the disabled-interaction run"
        )
        try:
            if not args.no_test:
                example.test_final()
                result["test_status"] = "passed"
                result["test_passed"] = True
        except AssertionError as error:
            result["test_status"] = "failed"
            result["test_error"] = str(error)
            raise
        finally:
            example.export(output, result)
            print(json.dumps(result, indent=2), flush=True)
        return example, result

    output = args.output / "negative-control" if args.no_contact else args.output
    run(not args.no_contact, output)
    if args.validate and not args.no_contact:
        run(False, args.output / "negative-control")
        result = compare_negative_control(args.output)
        ratio = result["negative_control"]["deformation_ratio"]
        print(f"{slug}: negative-control deformation ratio={ratio:.6f}", flush=True)
