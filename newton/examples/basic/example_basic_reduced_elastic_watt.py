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
# Example Basic Reduced Elastic Watt
#
# Demonstrates a Watt-style straight-line linkage with an elastic coupler and
# a guided midpoint follower.
# Driven DOF: the ground-to-left-rocker revolute joint target. The elastic
# modal DOFs are passive; no direct modal forces are applied in this example.
#
# Command: python -m newton.examples basic_reduced_elastic_watt
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import (
    beam_render_sample_points,
    elastic_shape_volume_ratio,
    joint_endpoint_world,
)


def _quat_from_angle_z(theta: float):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), theta)


def _angle_from_quat_z(q: np.ndarray) -> float:
    return 2.0 * math.atan2(float(q[2]), float(q[3]))


def _solve_fourbar(theta2: float, a: float, b: float, c: float, d: float) -> tuple[float, float]:
    k1 = d / a
    k2 = d / c
    k3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c)
    A = k1 - math.cos(theta2)
    B = -math.sin(theta2)
    C = k2 * math.cos(theta2) - k3
    denom = max(math.sqrt(A * A + B * B), 1.0e-8)
    theta4 = math.atan2(B, A) + math.acos(np.clip(C / denom, -1.0, 1.0))
    theta3 = math.atan2(c * math.sin(theta4) - a * math.sin(theta2), d + c * math.cos(theta4) - a * math.cos(theta2))
    return theta3, theta4


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 3
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.step_count = 0

        self.viewer = viewer
        self.args = args

        self.left_length = 0.42
        self.coupler_rest_length = 0.48
        self.right_length = 0.42
        self.ground_length = 0.76
        self.coupler_hy = 0.045
        self.coupler_hz = 0.03
        self.z = 0.18
        self.theta0 = 0.92
        self.initial_axial_q = 0.025
        self.initial_bend_q = 0.095
        self.max_mode_abs = 0.0
        self.max_joint_residual = 0.0
        self.max_rigid_closure_error = 0.0
        self.max_rail_error = 0.0
        self.max_volume_ratio_error = 0.0

        theta3, theta4 = _solve_fourbar(
            self.theta0,
            self.left_length,
            self.coupler_rest_length + self.initial_axial_q,
            self.right_length,
            self.ground_length,
        )
        left_ground = np.array([0.0, 0.0, self.z], dtype=float)
        right_ground = np.array([self.ground_length, 0.0, self.z], dtype=float)
        left_axis = np.array([math.cos(self.theta0), math.sin(self.theta0), 0.0], dtype=float)
        right_axis = np.array([math.cos(theta4), math.sin(theta4), 0.0], dtype=float)
        left_pin = left_ground + self.left_length * left_axis
        right_pin = right_ground + self.right_length * right_axis
        coupler_center = 0.5 * (left_pin + right_pin)

        sample_points = beam_render_sample_points(
            self.coupler_rest_length,
            self.coupler_hy,
            self.coupler_hz,
            extra_points=(
                (-0.5 * self.coupler_rest_length, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.5 * self.coupler_rest_length, 0.0, 0.0),
            ),
        )
        coupler_basis = newton.ModalGeneratorBeam(
            length=self.coupler_rest_length,
            half_width_y=self.coupler_hy,
            half_width_z=self.coupler_hz,
            mode_specs=[
                {"type": newton.ModalGeneratorBeam.Mode.AXIAL},
                {
                    "type": newton.ModalGeneratorBeam.Mode.BENDING_Y,
                    "boundary": newton.ModalGeneratorBeam.Boundary.PINNED_PINNED,
                    "order": 1,
                },
                {
                    "type": newton.ModalGeneratorBeam.Mode.BENDING_Y,
                    "boundary": newton.ModalGeneratorBeam.Boundary.PINNED_PINNED,
                    "order": 2,
                },
            ],
            density=320.0,
            young_modulus=2.0e6,
            damping_ratio=0.04,
            label="watt_coupler_basis",
        ).build(sample_points=sample_points)
        midpoint_def_local = np.array([0.0, self.initial_bend_q, 0.0], dtype=float)
        midpoint_def_world = np.array(
            [
                math.cos(theta3) * midpoint_def_local[0] - math.sin(theta3) * midpoint_def_local[1],
                math.sin(theta3) * midpoint_def_local[0] + math.cos(theta3) * midpoint_def_local[1],
                0.0,
            ],
            dtype=float,
        )
        follower_center = coupler_center + midpoint_def_world
        self.rail_x = float(follower_center[0])

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_ground_plane()

        inertia = wp.mat33(0.012, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.012)
        self.left_rocker = builder.add_body(
            xform=wp.transform(
                wp.vec3(*(left_ground + 0.5 * self.left_length * left_axis)), _quat_from_angle_z(self.theta0)
            ),
            mass=1.0,
            inertia=inertia,
            label="left_rocker",
        )
        self.coupler = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(*coupler_center), _quat_from_angle_z(theta3)),
            mass=0.8,
            inertia=inertia,
            mode_q=[self.initial_axial_q, self.initial_bend_q, 0.0],
            mode_stiffness=[float(coupler_basis.mode_stiffness[0]), 120.0, 320.0],
            mode_damping=[2.4, 0.65, 0.9],
            modal_basis=coupler_basis,
            label="elastic_watt_coupler",
        )
        self.right_rocker = builder.add_body(
            xform=wp.transform(
                wp.vec3(*(right_ground + 0.5 * self.right_length * right_axis)), _quat_from_angle_z(theta4)
            ),
            mass=1.0,
            inertia=inertia,
            label="right_rocker",
        )
        self.follower = builder.add_body(
            xform=wp.transform(wp.vec3(*follower_center), wp.quat_identity()),
            mass=0.5,
            inertia=inertia,
            label="midpoint_follower",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False
        builder.add_shape_box(self.left_rocker, hx=0.5 * self.left_length, hy=0.026, hz=0.024, cfg=shape_cfg)
        builder.add_shape_box(
            self.coupler, hx=0.5 * self.coupler_rest_length, hy=self.coupler_hy, hz=self.coupler_hz, cfg=shape_cfg
        )
        builder.add_shape_box(self.right_rocker, hx=0.5 * self.right_length, hy=0.026, hz=0.024, cfg=shape_cfg)
        builder.add_shape_box(self.follower, hx=0.04, hy=0.04, hz=0.045, cfg=shape_cfg)

        self.j_left_ground = builder.add_joint_revolute(
            parent=-1,
            child=self.left_rocker,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(*left_ground), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.left_length, 0.0, 0.0), wp.quat_identity()),
            target_pos=self.theta0,
            target_ke=80.0,
            target_kd=0.5,
            label="ground_to_left_rocker",
        )
        self.j_left_coupler = builder.add_joint_revolute(
            parent=self.left_rocker,
            child=self.coupler,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.5 * self.left_length, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.coupler_rest_length, 0.0, 0.0), wp.quat_identity()),
            label="left_rocker_to_elastic_coupler",
        )
        self.j_coupler_right = builder.add_joint_revolute(
            parent=self.coupler,
            child=self.right_rocker,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.5 * self.coupler_rest_length, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.5 * self.right_length, 0.0, 0.0), wp.quat_identity()),
            label="elastic_coupler_to_right_rocker",
        )
        self.j_right_ground = builder.add_joint_revolute(
            parent=-1,
            child=self.right_rocker,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(*right_ground), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.right_length, 0.0, 0.0), wp.quat_identity()),
            label="ground_to_right_rocker",
        )
        self.j_mid_follower = builder.add_joint_revolute(
            parent=self.coupler,
            child=self.follower,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            label="elastic_midpoint_to_follower",
        )
        self.j_follower_rail = builder.add_joint_prismatic(
            parent=-1,
            child=self.follower,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(self.rail_x, 0.0, self.z), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            label="vertical_follower_rail",
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
        self.drive_qd_start = int(self.model.joint_qd_start.numpy()[self.j_left_ground])
        self._target_pos = self.control.joint_target_pos.numpy()
        self._joint_f = self.control.joint_f.numpy()
        self.max_volume_ratio_error = abs(elastic_shape_volume_ratio(self.model, self.state_0) - 1.0)

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=24,
            rigid_joint_linear_k_start=1.0e5,
            rigid_joint_angular_k_start=1.0e4,
            rigid_joint_linear_ke=2.0e6,
            rigid_joint_angular_ke=5.0e5,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.39, -1.35, 0.78), -25.0, 88.0)

    def _mode_values(self) -> np.ndarray:
        return self.state_0.joint_q.numpy()[self.elastic_q_start + 7 : self.elastic_q_start + 10]

    def _joint_residuals(self) -> list[float]:
        joints = [
            self.j_left_ground,
            self.j_left_coupler,
            self.j_coupler_right,
            self.j_right_ground,
            self.j_mid_follower,
        ]
        return [
            float(
                np.linalg.norm(
                    joint_endpoint_world(self.model, self.state_0, joint, "parent")
                    - joint_endpoint_world(self.model, self.state_0, joint, "child")
                )
            )
            for joint in joints
        ]

    def _rigid_closure_error(self) -> float:
        theta2 = _angle_from_quat_z(self.state_0.body_q.numpy()[self.left_rocker, 3:7])
        b_eff = self.coupler_rest_length + float(self._mode_values()[0])
        _theta3, theta4 = _solve_fourbar(theta2, self.left_length, b_eff, self.right_length, self.ground_length)
        expected_right_pin = np.array(
            [
                self.ground_length + self.right_length * math.cos(theta4),
                self.right_length * math.sin(theta4),
                self.z,
            ],
            dtype=float,
        )
        actual_right_pin = joint_endpoint_world(self.model, self.state_0, self.j_coupler_right, "parent")
        return float(np.linalg.norm(actual_right_pin - expected_right_pin))

    def _set_controls(self, t: float):
        self._target_pos[self.drive_qd_start] = self.theta0 + 4.95 * math.sin(1.15 * t)
        self.control.joint_target_pos.assign(self._target_pos)

        self._joint_f[:] = 0.0
        self.control.joint_f.assign(self._joint_f)

    def _update_metrics(self):
        self.max_joint_residual = max(self.max_joint_residual, *self._joint_residuals())
        self.max_rigid_closure_error = max(self.max_rigid_closure_error, self._rigid_closure_error())
        follower_x = float(self.state_0.body_q.numpy()[self.follower, 0])
        self.max_rail_error = max(self.max_rail_error, abs(follower_x - self.rail_x))
        self.max_volume_ratio_error = max(
            self.max_volume_ratio_error, abs(elastic_shape_volume_ratio(self.model, self.state_0) - 1.0)
        )
        self.max_mode_abs = max(self.max_mode_abs, float(np.max(np.abs(self._mode_values()))))

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self._set_controls(self.sim_time + substep * self.sim_dt)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.step_count += 1
        self._update_metrics()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise AssertionError("body transforms contain non-finite values")
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if self.max_joint_residual > 3.0e-2:
            raise AssertionError(f"Watt linkage joint residual too large: {self.max_joint_residual}")
        if self.max_rigid_closure_error > 4.0e-2:
            raise AssertionError(f"Watt linkage rigid endpoint closure error too large: {self.max_rigid_closure_error}")
        if self.max_rail_error > 1.5e-2:
            raise AssertionError(f"midpoint follower left its vertical rail by {self.max_rail_error}")
        if self.max_mode_abs < 0.09:
            raise AssertionError(f"elastic coupler deformation was too small: {self.max_mode_abs}")
        if np.max(np.abs(self._mode_values())) > 0.4:
            raise AssertionError(f"elastic coupler mode amplitude is out of expected range: {self._mode_values()}")
        if self.max_volume_ratio_error > 0.32:
            raise AssertionError(f"elastic coupler volume changed by {self.max_volume_ratio_error:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
