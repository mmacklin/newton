# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Compound Pulley
#
# Compound pulley system with two kinematic pulleys and two weights.
# The cable routes: left weight -> P1 (rolling) -> P2 (rolling) -> right
# weight.  Both pulleys rotate under the no-slip assumption, with
# orientation driven by per-substep rest-length changes.
#
# Command: python -m newton.examples tendon_compound_pulley
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

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

        self.r1 = 0.25
        self.r2 = 0.20

        p1 = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.5, 0.0, 4.0), q=wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
        )
        q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
        builder.add_shape_cylinder(p1, xform=wp.transform(q=q_cyl), radius=self.r1, half_height=0.05)
        self.p1_idx = p1

        p2 = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.5, 0.0, 3.5), q=wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_shape_cylinder(p2, xform=wp.transform(q=q_cyl), radius=self.r2, half_height=0.05)
        self.p2_idx = p2

        Dof = newton.ModelBuilder.JointDofConfig
        planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
        planar_ang = [Dof(axis=Axis.Y)]

        left = builder.add_link(
            xform=wp.transform(p=wp.vec3(-0.8, 0.0, 2.0), q=wp.quat_identity()),
            mass=1.5,
        )
        builder.add_shape_box(left, hx=0.09, hy=0.09, hz=0.09)
        j1 = builder.add_joint_d6(
            parent=-1,
            child=left,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=wp.vec3(-0.8, 0.0, 2.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        right = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.8, 0.0, 2.0), q=wp.quat_identity()),
            mass=4.0,
        )
        builder.add_shape_box(right, hx=0.13, hy=0.13, hz=0.13)
        j2 = builder.add_joint_d6(
            parent=-1,
            child=right,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=wp.vec3(0.8, 0.0, 2.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        builder.add_articulation([j1])
        builder.add_articulation([j2])

        axis = (0.0, 1.0, 0.0)
        builder.add_tendon()
        builder.add_tendon_link(
            body=left,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.09),
            axis=axis,
        )
        builder.add_tendon_link(
            body=p1,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.r1,
            orientation=1,
            mu=0.0,
            offset=(0.0, 0.0, 0.0),
            axis=axis,
            compliance=1.0e-5,
            damping=0.1,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=p2,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.r2,
            orientation=1,
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
            offset=(0.0, 0.0, 0.13),
            axis=axis,
            compliance=1.0e-5,
            damping=0.1,
            rest_length=-1.0,
        )

        builder.add_ground_plane()
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

        self.p1_angle = 0.0
        self.p2_angle = 0.0

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, -5.0, 2.5), pitch=5.0, yaw=90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            rest_before = self.solver.tendon_seg_rest_length.numpy().copy()

            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            rest_after = self.solver.tendon_seg_rest_length.numpy()

            # both pulleys rotate CW (negative y) when cable flows left-to-right
            d0 = rest_after[0] - rest_before[0]
            self.p1_angle -= d0 / self.r1

            d2 = rest_after[2] - rest_before[2]
            self.p2_angle += d2 / self.r2

            qy1 = np.sin(self.p1_angle / 2.0)
            qw1 = np.cos(self.p1_angle / 2.0)
            qy2 = np.sin(self.p2_angle / 2.0)
            qw2 = np.cos(self.p2_angle / 2.0)
            set_body_quat(self.state_0, self.p1_idx, [0.0, qy1, 0.0, qw1])
            set_body_quat(self.state_0, self.p2_idx, [0.0, qy2, 0.0, qw2])

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        if self.sim_time < self.frame_dt * 1.5:
            att_r = self.solver.tendon_seg_attachment_r.numpy()
            att_l = self.solver.tendon_seg_attachment_l.numpy()
            body_q = self.state_0.body_q.numpy()
            p1_z = body_q[self.p1_idx][2]
            p2_z = body_q[self.p2_idx][2]
            assert att_r[0][2] > p1_z, (
                f"Cable should wrap over P1: arrival tangent z={att_r[0][2]:.3f} <= center z={p1_z:.3f}"
            )
            assert att_l[2][2] > p2_z, (
                f"Cable should wrap over P2: departure tangent z={att_l[2][2]:.3f} <= center z={p2_z:.3f}"
            )

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in body positions"

        # body 0=P1, 1=P2, 2=left (light), 3=right (heavy)
        left_z = body_q[2][2]
        right_z = body_q[3][2]
        assert right_z < 2.0, f"Right (heavy) body should descend: z={right_z}"
        assert left_z > 2.0, f"Left (light) body should ascend: z={left_z}"

        # both pulleys should rotate the same direction (both negative = CW)
        assert self.p1_angle * self.p2_angle > 0, (
            f"Pulleys should rotate same direction: P1={self.p1_angle:.2f}, P2={self.p2_angle:.2f}"
        )

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines("cable", starts, ends, colors=(0.2, 0.7, 1.0), width=0.008)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
