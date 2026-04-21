# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Pulley
#
# Demonstrates the tendon (cable joint) system: two weights connected by
# a tendon routed over a fixed pulley.  The heavier weight descends,
# pulling the lighter weight upward through the tendon constraint.
#
# This is the simplest test of the Cable Joints method (Müller et al.
# SCA 2018) extended with capstan friction.
#
# Command: python -m newton.examples tendon_pulley
#
###########################################################################

import warp as wp

import newton
import newton.examples
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType


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

        # -- Pulley (kinematic, fixed in space) --
        pulley_radius = 0.15
        pulley_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 2.0, 0.0), q=wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_shape_box(
            pulley_body,
            hx=pulley_radius,
            hy=0.02,
            hz=pulley_radius,
        )

        # -- Left weight (light, 1 kg) --
        # Positioned so cable from attachment to pulley tangent = rest length
        left_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.5, 1.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(left_body, hx=0.08, hy=0.08, hz=0.08)

        # -- Right weight (heavy, 3 kg) --
        right_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.5, 1.0, 0.0), q=wp.quat_identity()),
            mass=3.0,
        )
        builder.add_shape_box(right_body, hx=0.12, hy=0.12, hz=0.12)

        # -- Tendon: left_body -> pulley -> right_body --
        # Cable plane is the XY plane (normal = Z)
        cable_plane_axis = (0.0, 0.0, 1.0)

        builder.add_tendon()

        # link 0: attachment on left body (top center)
        builder.add_tendon_link(
            body=left_body,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.08, 0.0),
            axis=cable_plane_axis,
        )

        # link 1: rolling contact on pulley
        builder.add_tendon_link(
            body=pulley_body,
            link_type=int(TendonLinkType.ROLLING),
            radius=pulley_radius,
            orientation=1,
            mu=0.0,
            offset=(0.0, 0.0, 0.0),
            axis=cable_plane_axis,
            compliance=1.0e-5,
            damping=0.1,
            rest_length=-1.0,  # auto-compute
        )

        # link 2: attachment on right body (top center)
        builder.add_tendon_link(
            body=right_body,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.12, 0.0),
            axis=cable_plane_axis,
            compliance=1.0e-5,
            damping=0.1,
            rest_length=-1.0,  # auto-compute
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

        if viewer is not None:
            self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        # body indices: 0=pulley (fixed), 1=left, 2=right
        left_y = body_q[1][1]
        right_y = body_q[2][1]
        assert right_y < 1.0, f"Right body should descend: y={right_y}"
        assert left_y > 1.0, f"Left body should ascend: y={left_y}"

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()


if __name__ == "__main__":
    stage_path = "tendon_pulley.usd"
    viewer = newton.examples.create_viewer(stage_path=stage_path)
    example = Example(viewer, args=None)
    for _ in range(300):
        example.step()
        example.render()
