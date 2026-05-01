# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Capstan Friction
#
# Three side-by-side asymmetric Atwood machines (3:1 mass ratio) with
# different capstan friction coefficients on the pulley:
#
#   Left:   mu = 0.0   (frictionless)
#   Center: mu = 0.005 (subcritical — visible partial grip)
#   Right:  mu = 10.0  (no-slip)
#
# Each pulley is a dynamic body on a hinge joint, free to rotate about Y.
# The Euler-Eytelwein capstan equation bounds the tension ratio across
# a frictional contact: T_tight / T_slack <= exp(mu * theta).
#
# For dynamic pulleys with mu>0 the cable grips the pulley (stick mode).
# The XPBD constraint solver couples the pulley's rotational inertia
# I/R^2 to the system through the shared body.  The capstan bound
# activates only when the required tension ratio exceeds exp(mu*theta).
# The pulleys use an explicit high inertia so the no-slip case has a
# visibly different weight trajectory from the freely sliding case.
# With dynamic pulleys, low finite friction spins the pulley only partially.
# The red tab on each pulley marks rim rotation; the frictionless pulley
# should translate the cable without spinning.
#
# Command: python -m newton.examples tendon_capstan_friction
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.examples.cable.cable import assert_tendon_total_length, get_tendon_cable_lines


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

        self.pulley_radius = 0.15
        pulley_mass = 20.0
        pulley_inertia_y = 0.10
        pulley_inertia = wp.mat33(
            0.06, 0.0, 0.0,
            0.0, pulley_inertia_y, 0.0,
            0.0, 0.0, 0.06,
        )
        self.mus = [0.0, 0.005, 10.0]
        self.x_offsets = [-1.5, 0.0, 1.5]

        mass_light = 1.0
        mass_heavy = 3.0

        q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))

        Dof = newton.ModelBuilder.JointDofConfig
        planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
        planar_ang = [Dof(axis=Axis.Y)]

        self.pulley_indices = []
        self.left_indices = []
        self.right_indices = []

        for mu, x_off in zip(self.mus, self.x_offsets, strict=True):
            pulley_pos = wp.vec3(x_off, 0.0, 3.5)
            pulley = builder.add_body(
                xform=wp.transform(p=pulley_pos, q=wp.quat_identity()),
                mass=pulley_mass,
                inertia=pulley_inertia,
                lock_inertia=True,
            )
            builder.add_shape_cylinder(
                pulley, xform=wp.transform(q=q_cyl),
                radius=self.pulley_radius, half_height=0.04,
            )
            marker_cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=False,
                has_particle_collision=False,
            )
            builder.add_shape_box(
                pulley,
                xform=wp.transform(p=wp.vec3(0.0, 0.0, self.pulley_radius + 0.025)),
                hx=0.035,
                hy=0.055,
                hz=0.012,
                cfg=marker_cfg,
                color=(0.95, 0.10, 0.06),
            )
            j_pulley = builder.add_joint_d6(
                parent=-1, child=pulley,
                linear_axes=[],
                angular_axes=[Dof(axis=Axis.Y)],
                parent_xform=wp.transform(p=pulley_pos),
                child_xform=wp.transform(),
            )
            builder.add_articulation([j_pulley])
            self.pulley_indices.append(pulley)

            left_pos = wp.vec3(x_off - 0.4, 0.0, 2.0)
            left = builder.add_link(
                xform=wp.transform(p=left_pos, q=wp.quat_identity()),
                mass=mass_light,
            )
            builder.add_shape_box(left, hx=0.06, hy=0.06, hz=0.06)
            j1 = builder.add_joint_d6(
                parent=-1, child=left,
                linear_axes=planar_lin, angular_axes=planar_ang,
                parent_xform=wp.transform(p=left_pos),
                child_xform=wp.transform(),
            )
            builder.add_articulation([j1])
            self.left_indices.append(left)

            right_pos = wp.vec3(x_off + 0.4, 0.0, 2.0)
            right = builder.add_link(
                xform=wp.transform(p=right_pos, q=wp.quat_identity()),
                mass=mass_heavy,
            )
            builder.add_shape_box(right, hx=0.09, hy=0.09, hz=0.09)
            j2 = builder.add_joint_d6(
                parent=-1, child=right,
                linear_axes=planar_lin, angular_axes=planar_ang,
                parent_xform=wp.transform(p=right_pos),
                child_xform=wp.transform(),
            )
            builder.add_articulation([j2])
            self.right_indices.append(right)

            axis = (0.0, 1.0, 0.0)
            builder.add_tendon()
            builder.add_tendon_link(
                body=left,
                link_type=int(TendonLinkType.ATTACHMENT),
                offset=(0.0, 0.0, 0.06),
                axis=axis,
            )
            builder.add_tendon_link(
                body=pulley,
                link_type=int(TendonLinkType.ROLLING),
                radius=self.pulley_radius,
                orientation=1,
                mu=mu,
                offset=(0.0, 0.0, 0.0),
                axis=axis,
                compliance=1.0e-5,
                damping=0.1,
                rest_length=-1.0,
            )
            builder.add_tendon_link(
                body=right,
                link_type=int(TendonLinkType.ATTACHMENT),
                offset=(0.0, 0.0, 0.06),
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

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, -6.0, 2.5), pitch=5.0, yaw=90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

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

    def test_post_step(self):
        assert_tendon_total_length(self)
        if self.sim_time < self.frame_dt * 1.5:
            att_r = self.solver.tendon_seg_attachment_r.numpy()
            att_l = self.solver.tendon_seg_attachment_l.numpy()
            body_q = self.state_0.body_q.numpy()
            for i, p_idx in enumerate(self.pulley_indices):
                pulley_z = body_q[p_idx][2]
                seg = i * 2
                assert att_r[seg][2] > pulley_z, (
                    f"Atwood {i}: arrival tangent z={att_r[seg][2]:.3f} "
                    f"<= center z={pulley_z:.3f}"
                )
                assert att_l[seg + 1][2] > pulley_z, (
                    f"Atwood {i}: departure tangent z={att_l[seg + 1][2]:.3f} "
                    f"<= center z={pulley_z:.3f}"
                )

    def test_final(self):
        assert_tendon_total_length(self)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in body positions"

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines(
                "cable", starts, ends,
                colors=(0.8, 0.5, 0.2), width=0.008,
            )
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
