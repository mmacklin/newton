# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Reduced Elastic Coriolis
#
# Demonstrates the floating-frame Coriolis coupling for reduced elastic
# bodies: two identical beams are clamped to a kinematic hub that spins at a
# constant rate about the beams' own axis (x). Each beam carries two
# equal-stiffness cantilever bending modes -- one deflecting in y, one in z.
# Both beams are deflected in z and released. They share the same modal mass,
# stiffness, and mode shapes; only the inertia coupling integrals differ.
#
#   - The "coupled" beam's basis carries per-sample masses, so as the plucked
#     z mode vibrates while spinning, the Coriolis term -2 omega . G qdot pumps
#     energy into the perpendicular y mode and the bending plane precesses
#     (the spinning-shaft whirl): in the inertial frame the swing plane stays
#     fixed while the hub turns under it.
#   - The "uncoupled" beam's basis has no sample mass (all coupling integrals
#     are zero), so its z vibration never feeds the y mode and its swing plane
#     simply rotates rigidly with the spinning hub.
#
# Each beam's frame sits at its clamp on the spin axis, so the frame origin
# only rotates (no translational acceleration), the spin is steady (no Euler
# term), and the on-axis transverse modes carry no centrifugal load -- leaving
# the Coriolis term as the only thing that couples the two modes.
#
# Command: python -m newton.examples basic_reduced_elastic_coriolis
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import beam_render_sample_points
from newton.examples.basic._reduced_elastic_contact import apply_kinematic_targets


def _cantilever_bending_modes(points: np.ndarray, length: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    s = np.clip(points[:, 0], 0.0, length)
    g = (s * s * (3.0 * length - s)) / (2.0 * length**3)
    slope = (3.0 * s * (2.0 * length - s)) / (2.0 * length**3)
    phi = np.zeros((points.shape[0], 2, 3), dtype=np.float32)
    phi[:, 0, 2] = g
    phi[:, 0, 0] = -points[:, 2] * slope
    phi[:, 1, 1] = g
    phi[:, 1, 0] = -points[:, 1] * slope
    return phi


def _find_free_joint_q_start(model: newton.Model, body: int) -> tuple[int, int]:
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    joint_q_start = model.joint_q_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    for j in range(len(joint_child)):
        if int(joint_child[j]) == body and int(joint_parent[j]) == -1:
            return int(joint_q_start[j]), int(joint_qd_start[j])
    raise RuntimeError(f"No free joint found for body {body}")


def _set_camera_from_bounds(viewer, bounds_min: np.ndarray, bounds_max: np.ndarray, offset_dir: np.ndarray):
    center = 0.5 * (bounds_min + bounds_max)
    extent = float(np.max(bounds_max - bounds_min))
    distance = max(extent, 1.0) / (2.0 * math.tan(math.radians(45.0) * 0.5)) * 1.35
    offset_dir = offset_dir / np.linalg.norm(offset_dir)
    pos = center + offset_dir * distance
    front = center - pos
    front /= np.linalg.norm(front)
    yaw = math.degrees(math.atan2(front[1], front[0]))
    pitch = math.degrees(math.asin(front[2]))
    viewer.set_camera(wp.vec3(*pos), pitch, yaw)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        self.length = 0.8
        self.hy = 0.035
        self.hz = 0.035
        self.base_height = 0.8
        self.spin_rate = 1.5
        self.pluck = 0.25
        self.beam_mass = 1.0

        centered_points = beam_render_sample_points(
            self.length,
            self.hy,
            self.hz,
            extra_points=((-0.5 * self.length, 0.0, 0.0), (0.5 * self.length, 0.0, 0.0)),
        )
        clamp_points = centered_points + np.array([0.5 * self.length, 0.0, 0.0], dtype=np.float32)
        phi = _cantilever_bending_modes(clamp_points, self.length)

        sample_mass = np.full(clamp_points.shape[0], self.beam_mass / clamp_points.shape[0], dtype=np.float32)
        modal_mass = np.einsum("s,smc,smc->m", sample_mass.astype(np.float64), phi, phi)
        mode_stiffness = [35.0, 35.0]
        mode_damping = [0.0, 0.0]

        coupled_basis = newton.ModalBasis(
            sample_points=clamp_points,
            sample_phi=phi,
            sample_mass=sample_mass,
            mode_stiffness=mode_stiffness,
            mode_damping=mode_damping,
            label="coriolis_coupled_basis",
        )
        uncoupled_basis = newton.ModalBasis(
            sample_points=clamp_points,
            sample_phi=phi,
            mode_mass=modal_mass,
            mode_stiffness=mode_stiffness,
            mode_damping=mode_damping,
            label="coriolis_uncoupled_basis",
        )

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_ground_plane()

        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False

        self.base = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, self.base_height), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05),
            is_kinematic=True,
            label="spinning_base",
        )
        builder.add_shape_box(self.base, hx=0.06, hy=0.06, hz=0.06, cfg=shape_cfg)

        half = 0.5 * self.length
        flip = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi)
        self.beams = {}
        for name, basis, orientation in (
            ("coupled", coupled_basis, wp.quat_identity()),
            ("uncoupled", uncoupled_basis, flip),
        ):
            beam = builder.add_body_elastic(
                xform=wp.transform(wp.vec3(0.0, 0.0, self.base_height), orientation),
                com=wp.vec3(half, 0.0, 0.0),
                mass=0.2,
                inertia=inertia,
                mode_q=[self.pluck, 0.0],
                modal_basis=basis,
                label=f"coriolis_{name}_beam",
            )
            builder.add_shape_box(
                beam,
                xform=wp.transform(wp.vec3(half, 0.0, 0.0), wp.quat_identity()),
                hx=half,
                hy=self.hy,
                hz=self.hz,
                cfg=shape_cfg,
            )
            builder.add_joint_fixed(
                parent=self.base,
                child=beam,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), orientation),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                label=f"clamp_{name}",
            )
            self.beams[name] = beam

        builder.color()

        self.model = builder.finalize()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        base_q_start, base_qd_start = _find_free_joint_q_start(self.model, self.base)
        self._base_q_starts = {self.base: base_q_start}
        self._base_qd_starts = {self.base: base_qd_start}

        elastic_joint = self.model.elastic_joint.numpy()
        body_elastic_index = self.model.body_elastic_index.numpy()
        joint_q_start = self.model.joint_q_start.numpy()
        self._q_index = {}
        for name, beam in self.beams.items():
            owner = int(elastic_joint[int(body_elastic_index[beam])])
            self._q_index[name] = int(joint_q_start[owner]) + 7

        self.max_abs_perp = dict.fromkeys(self.beams, 0.0)

        self.solver = newton.solvers.SolverVBD(self.model, iterations=32)

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = self.pluck
        reach = self.length + 0.2
        span = self.pluck + 0.2
        bounds_min = np.array([-reach, -span, self.base_height - span])
        bounds_max = np.array([reach, span, self.base_height + span])
        _set_camera_from_bounds(self.viewer, bounds_min, bounds_max, np.array([-0.4, -1.0, 0.35]))

    def _drive_targets(self, t: float):
        quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), self.spin_rate * t)
        return {self.base: (wp.vec3(0.0, 0.0, self.base_height), quat)}

    def _drive_velocities(self, t: float):
        return {self.base: (wp.vec3(0.0, 0.0, 0.0), wp.vec3(self.spin_rate, 0.0, 0.0))}

    def _perp_value(self, name: str) -> float:
        return float(self.state_0.joint_q.numpy()[self._q_index[name] + 1])

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            targets = self._drive_targets(t)
            velocities = self._drive_velocities(t)
            apply_kinematic_targets(self.state_0, self._base_q_starts, targets, velocities, self._base_qd_starts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        for name in self.beams:
            self.max_abs_perp[name] = max(self.max_abs_perp[name], abs(self._perp_value(name)))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if self.max_abs_perp["coupled"] < 3.0e-2:
            raise AssertionError(
                f"Coriolis did not pump the coupled beam's perpendicular mode: max |q_y| = {self.max_abs_perp['coupled']:.3e}"
            )
        if self.max_abs_perp["uncoupled"] > 3.0e-3:
            raise AssertionError(
                f"uncoupled beam's perpendicular mode was unexpectedly excited: max |q_y| = {self.max_abs_perp['uncoupled']:.3e}"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
