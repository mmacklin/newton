# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Rolling Pulley
#
# Atwood machine: two weights connected by a tendon routed over a
# kinematic pulley that rotates under no-slip assumption.  The heavier
# weight descends, pulling the lighter weight up; the pulley spins
# proportionally to the cable displacement.
#
# Demonstrates kinematically-driven pulley rotation by tracking
# per-substep changes in tendon segment rest lengths.
#
# Command: python -m newton.examples tendon_rolling_pulley
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.examples.cable.cable import get_tendon_cable_lines, set_body_quat


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=Axis.Y, gravity=-9.81)

        self.pulley_radius = 0.15

        pulley = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 2.5, 0.0), q=wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_shape_cylinder(pulley, radius=self.pulley_radius, half_height=0.04)
        self.pulley_idx = pulley

        Dof = newton.ModelBuilder.JointDofConfig
        planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Y)]
        planar_ang = [Dof(axis=Axis.Z)]

        left = builder.add_link(
            xform=wp.transform(p=wp.vec3(-0.5, 1.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(left, hx=0.08, hy=0.08, hz=0.08)
        j1 = builder.add_joint_d6(
            parent=-1,
            child=left,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=wp.vec3(-0.5, 1.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        right = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.5, 1.0, 0.0), q=wp.quat_identity()),
            mass=3.0,
        )
        builder.add_shape_box(right, hx=0.12, hy=0.12, hz=0.12)
        j2 = builder.add_joint_d6(
            parent=-1,
            child=right,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=wp.vec3(0.5, 1.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        builder.add_articulation([j1])
        builder.add_articulation([j2])

        axis = (0.0, 0.0, 1.0)
        builder.add_tendon()
        builder.add_tendon_link(
            body=left,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.08, 0.0),
            axis=axis,
        )
        builder.add_tendon_link(
            body=pulley,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.pulley_radius,
            orientation=-1,
            mu=0.0,
            offset=(0.0, 0.0, 0.0),
            axis=axis,
            compliance=1.0e-5,
            damping=0.1,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=right,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.12, 0.0),
            axis=axis,
            compliance=1.0e-5,
            damping=0.1,
            rest_length=-1.0,
        )

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=8,
            joint_linear_relaxation=0.8,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.pulley_angle = 0.0

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, 1.5, 4.0), pitch=-5.0, yaw=-90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            rest_before = self.solver.tendon_seg_rest_length.numpy()[0]

            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            rest_after = self.solver.tendon_seg_rest_length.numpy()[0]
            d_rest = rest_after - rest_before
            self.pulley_angle -= d_rest / self.pulley_radius

            qz = np.sin(self.pulley_angle / 2.0)
            qw = np.cos(self.pulley_angle / 2.0)
            set_body_quat(self.state_0, self.pulley_idx, [0.0, 0.0, qz, qw])

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        # body 0 = pulley (kinematic), body 1 = left (light), body 2 = right (heavy)
        left_y = body_q[1][1]
        right_y = body_q[2][1]
        assert right_y < 1.0, f"Right (heavy) body should descend: y={right_y}"
        assert left_y > 1.0, f"Left (light) body should ascend: y={left_y}"

        assert np.isfinite(body_q).all(), "Non-finite values in body positions"
        assert abs(self.pulley_angle) > 0.1, f"Pulley should have rotated: angle={self.pulley_angle}"

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines("cable", starts, ends, colors=(0.9, 0.2, 0.2), width=0.008)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
