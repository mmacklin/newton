# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Cable Machine
#
# Cable machine with three pulleys of varying sizes routing a single
# tendon from a light capsule weight to a heavy box weight.  A
# decorative sphere sits at ground level.  The box descends under
# gravity, pulling the capsule upward through the pulley chain.  All
# three pulleys rotate under the no-slip assumption, with orientation
# driven by per-substep rest-length changes.
#
# Demonstrates complex multi-pulley routing with diverse body shapes
# (capsules, boxes, cylinders, spheres).
#
# Command: python -m newton.examples tendon_cable_machine
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.examples.cable.cable import assert_tendon_total_length, get_tendon_cable_lines, set_body_quat


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

        self.r1 = 0.20
        self.r2 = 0.15
        self.r3 = 0.22

        p1 = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.6, 0.0, 3.8), q=wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
        )
        q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
        builder.add_shape_cylinder(p1, xform=wp.transform(q=q_cyl), radius=self.r1, half_height=0.05)
        self.p1_idx = p1

        p2 = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.5, 0.0, 4.4), q=wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_shape_cylinder(p2, xform=wp.transform(q=q_cyl), radius=self.r2, half_height=0.04)
        self.p2_idx = p2

        p3 = builder.add_body(
            xform=wp.transform(p=wp.vec3(1.5, 0.0, 3.6), q=wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_shape_cylinder(p3, xform=wp.transform(q=q_cyl), radius=self.r3, half_height=0.05)
        self.p3_idx = p3

        sphere_deco = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.5, 0.4, 1.15), q=wp.quat_identity()),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_shape_sphere(sphere_deco, radius=0.15)

        Dof = newton.ModelBuilder.JointDofConfig
        planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
        planar_ang = [Dof(axis=Axis.Y)]

        capsule_pos = wp.vec3(-0.9, 0.0, 2.2)
        q_vert = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
        left = builder.add_link(
            xform=wp.transform(p=capsule_pos, q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_capsule(
            left,
            xform=wp.transform(q=q_vert),
            radius=0.06,
            half_height=0.08,
        )
        j1 = builder.add_joint_d6(
            parent=-1,
            child=left,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=capsule_pos, q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        box_pos = wp.vec3(1.8, 0.0, 2.0)
        right = builder.add_link(
            xform=wp.transform(p=box_pos, q=wp.quat_identity()),
            mass=4.0,
        )
        builder.add_shape_box(right, hx=0.12, hy=0.15, hz=0.10)
        j2 = builder.add_joint_d6(
            parent=-1,
            child=right,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=box_pos, q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        builder.add_articulation([j1])
        builder.add_articulation([j2])

        axis = (0.0, 1.0, 0.0)
        builder.add_tendon()

        builder.add_tendon_link(
            body=left,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.14),
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
            body=p3,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.r3,
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
            offset=(0.0, 0.0, 0.15),
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
        self.p3_angle = 0.0

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.4, -7.0, 2.5), pitch=5.0, yaw=90.0)
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

            # In a series pulley chain the same cable passes through all
            # pulleys, so the linear displacement is the same everywhere.
            # Measure it from the first segment (capsule->P1).
            d_cable = rest_after[0] - rest_before[0]
            self.p1_angle -= d_cable / self.r1
            self.p2_angle -= d_cable / self.r2
            self.p3_angle -= d_cable / self.r3

            for idx, angle in [
                (self.p1_idx, self.p1_angle),
                (self.p2_idx, self.p2_angle),
                (self.p3_idx, self.p3_angle),
            ]:
                qy = np.sin(angle / 2.0)
                qw = np.cos(angle / 2.0)
                set_body_quat(self.state_0, idx, [0.0, qy, 0.0, qw])

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        assert_tendon_total_length(self)
        if self.sim_time < self.frame_dt * 1.5:
            att_r = self.solver.tendon_seg_attachment_r.numpy()
            att_l = self.solver.tendon_seg_attachment_l.numpy()
            body_q = self.state_0.body_q.numpy()
            p1_z = body_q[self.p1_idx][2]
            p3_z = body_q[self.p3_idx][2]
            assert att_r[0][2] > p1_z, (
                f"Cable should wrap over P1: arrival tangent z={att_r[0][2]:.3f} <= center z={p1_z:.3f}"
            )
            assert att_l[3][2] > p3_z, (
                f"Cable should wrap over P3: departure tangent z={att_l[3][2]:.3f} <= center z={p3_z:.3f}"
            )

    def test_final(self):
        assert_tendon_total_length(self)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in body positions"

        # body 0=P1, 1=P2, 2=P3, 3=sphere, 4=capsule(light), 5=box(heavy)
        capsule_z = body_q[4][2]
        box_z = body_q[5][2]
        assert box_z < 3.0, f"Box (heavy) should descend: z={box_z}"
        assert capsule_z > 1.5, f"Capsule (light) should ascend: z={capsule_z}"

        angles = [self.p1_angle, self.p2_angle, self.p3_angle]
        signs = [np.sign(a) for a in angles if abs(a) > 0.01]
        if len(signs) > 1:
            assert all(s == signs[0] for s in signs), (
                f"Pulleys should rotate same direction: "
                f"P1={self.p1_angle:.2f}, P2={self.p2_angle:.2f}, P3={self.p3_angle:.2f}"
            )

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines("cable", starts, ends, colors=(1.0, 0.6, 0.1), width=0.008)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
