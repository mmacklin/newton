# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Basic Reduced Elastic Torsion
#
# Demonstrates a reduced elastic rectangular shaft attached through a revolute
# endpoint. The visible twist mode is built from a finite-rotation exemplar so
# the held 90 degree twist preserves volume much better than the small-angle
# torsion field u = theta cross r.
#
# Command: python -m newton.examples basic_reduced_elastic_torsion
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import (
    beam_render_sample_points,
    box_surface_mesh,
    elastic_shape_volume_ratio,
    finite_torsion_mode_fields,
)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.step_count = 0

        self.length = 1.0
        self.z = 0.18
        self.hold_time = 0.45
        self.target_tip_twist = math.radians(90.0)
        self.hy = 0.085
        self.hz = 0.05

        shaft_vertices, shaft_indices = box_surface_mesh(self.length, self.hy, self.hz)
        sample_points = beam_render_sample_points(
            self.length,
            self.hy,
            self.hz,
            extra_points=(
                (-0.5 * self.length, 0.0, 0.0),
                (0.5 * self.length, 0.0, 0.0),
            ),
        )

        linear_basis = newton.ModalGeneratorBeam(
            length=self.length,
            half_width_y=self.hy,
            half_width_z=self.hz,
            mode_specs=[
                {
                    "type": newton.ModalGeneratorBeam.Mode.TORSION,
                    "boundary": newton.ModalGeneratorBeam.Boundary.LINEAR,
                    "order": 1,
                }
            ],
            sample_count=49,
            density=500.0,
            shear_modulus=4.0e4,
            label="linear_torsion_properties",
        ).build(sample_points=sample_points)

        twist_phi, radial_phi = finite_torsion_mode_fields(sample_points, self.length, self.target_tip_twist)
        tip_twist_scale = self.target_tip_twist * self.target_tip_twist
        base_mass = float(linear_basis.mode_mass[0] * tip_twist_scale)
        base_stiffness = float(linear_basis.mode_stiffness[0] * tip_twist_scale)
        mode_mass = np.array([base_mass, base_mass], dtype=np.float32)
        mode_stiffness = np.array([base_stiffness, 4.0 * base_stiffness], dtype=np.float32)
        damping_ratio = 0.05
        mode_damping = np.array(
            [2.0 * damping_ratio * math.sqrt(float(mode_mass[i]) * float(mode_stiffness[i])) for i in range(2)],
            dtype=np.float32,
        )

        torsion_basis = newton.ModalBasis(
            sample_points=sample_points,
            sample_phi=np.stack((twist_phi, radial_phi), axis=1),
            mode_mass=mode_mass,
            mode_stiffness=mode_stiffness,
            mode_damping=mode_damping,
            label="finite_torsion_fixture_basis",
        )

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_ground_plane()

        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        self.body = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.5 * self.length, 0.0, self.z), wp.quat_identity()),
            mass=1.0,
            inertia=inertia,
            mode_q=[1.0, 1.0],
            modal_basis=torsion_basis,
            label="elastic_torsion_fixture",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False
        builder.add_shape_mesh(
            self.body,
            mesh=newton.Mesh(shaft_vertices, shaft_indices, compute_inertia=False),
            cfg=shape_cfg,
            label="elastic_torsion_shaft_mesh",
        )
        self.joint = builder.add_joint_revolute(
            parent=-1,
            child=self.body,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, self.z), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
            label="world_revolute_to_elastic_endpoint",
        )
        builder.color()

        self.model = builder.finalize()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        self.elastic_joint = int(self.model.elastic_joint.numpy()[0])
        self.elastic_q_start = int(self.model.joint_q_start.numpy()[self.elastic_joint])
        self.elastic_qd_start = int(self.model.joint_qd_start.numpy()[self.elastic_joint])
        self._joint_f = self.control.joint_f.numpy()
        self._sync_radial_mode()
        self.initial_volume_ratio = elastic_shape_volume_ratio(self.model, self.state_0)
        self.max_volume_ratio_error = abs(self.initial_volume_ratio - 1.0)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=16,
            rigid_joint_linear_k_start=1.0e5,
            rigid_joint_angular_k_start=1.0e3,
            rigid_joint_linear_ke=2.0e6,
            rigid_joint_angular_ke=1.0e5,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.52, -1.35, 0.72), -24.0, 84.0)

    def _mode_value(self) -> float:
        return float(self.state_0.joint_q.numpy()[self.elastic_q_start + 7])

    def _sync_radial_mode(self):
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        twist = float(joint_q[self.elastic_q_start + 7])
        twist_rate = float(joint_qd[self.elastic_qd_start + 6])
        joint_q[self.elastic_q_start + 8] = twist * twist
        joint_qd[self.elastic_qd_start + 7] = 2.0 * twist * twist_rate
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.assign(joint_qd)

    def _update_volume_metric(self):
        ratio = elastic_shape_volume_ratio(self.model, self.state_0)
        self.max_volume_ratio_error = max(self.max_volume_ratio_error, abs(ratio - 1.0))
        return ratio

    def step(self):
        if self.sim_time >= self.hold_time:
            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()
                self._joint_f[:] = 0.0
                self.control.joint_f.assign(self._joint_f)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
                self._sync_radial_mode()

        self.sim_time += self.frame_dt
        self.step_count += 1
        self._update_volume_metric()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if abs(self.initial_volume_ratio - 1.0) > 2.0e-3:
            raise AssertionError(f"initial finite-twist volume ratio is {self.initial_volume_ratio}")
        if self.max_volume_ratio_error > 0.08:
            raise AssertionError(f"torsion volume changed by {self.max_volume_ratio_error:.3f}")
        if self.step_count > 40 and abs(self._mode_value() - 1.0) < 1.0e-3:
            raise AssertionError("torsion release did not evolve from the initial twist")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
