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


def _get_tendon_cable_lines(solver, model, state):
    """Build line-segment arrays for tendon visualization including arc wraps."""
    att_l = solver.tendon_seg_attachment_l.numpy()
    att_r = solver.tendon_seg_attachment_r.numpy()

    starts_list = []
    ends_list = []
    for i in range(model.tendon_segment_count):
        starts_list.append(att_l[i])
        ends_list.append(att_r[i])

    tendon_start = model.tendon_start.numpy()
    link_type = model.tendon_link_type.numpy()
    link_body = model.tendon_link_body.numpy()
    link_offset = model.tendon_link_offset.numpy()
    link_axis = model.tendon_link_axis.numpy()
    body_q = state.body_q.numpy()

    seg = 0
    for t in range(model.tendon_count):
        start = tendon_start[t]
        end = tendon_start[t + 1]
        num_links = end - start
        for i in range(start + 1, end - 1):
            if link_type[i] == int(TendonLinkType.ROLLING):
                b = link_body[i]
                pose = body_q[b]
                p = pose[:3]
                q = pose[3:]
                off = link_offset[i]
                ax = link_axis[i]
                t2 = 2.0 * np.cross(q[:3], off)
                center = off + q[3] * t2 + np.cross(q[:3], t2) + p
                t2n = 2.0 * np.cross(q[:3], ax)
                normal = ax + q[3] * t2n + np.cross(q[:3], t2n)

                seg_left = seg + (i - start) - 1
                seg_right = seg + (i - start)
                pt_dep = att_r[seg_left]
                pt_arr = att_l[seg_right]

                r_dep = pt_dep - center
                r_arr = pt_arr - center
                cross_val = np.dot(np.cross(r_dep, r_arr), normal)
                dot_val = np.dot(r_dep, r_arr)
                total_angle = np.arctan2(cross_val, dot_val)
                if np.isnan(total_angle):
                    continue

                n_arc = max(8, int(abs(total_angle) / 0.2))
                for j in range(n_arc):
                    frac0 = j / n_arc
                    frac1 = (j + 1) / n_arc
                    angle0 = frac0 * total_angle
                    angle1 = frac1 * total_angle
                    c0, s0 = np.cos(angle0), np.sin(angle0)
                    p0 = center + r_dep * c0 + np.cross(normal, r_dep) * s0
                    c1, s1 = np.cos(angle1), np.sin(angle1)
                    p1 = center + r_dep * c1 + np.cross(normal, r_dep) * s1
                    starts_list.append(p0)
                    ends_list.append(p1)

        seg += num_links - 1

    starts = wp.array(np.array(starts_list, dtype=np.float32), dtype=wp.vec3)
    ends = wp.array(np.array(ends_list, dtype=np.float32), dtype=wp.vec3)
    return starts, ends


def _set_body_quat(state, body_idx, quat_xyzw):
    """Write a quaternion (x,y,z,w) into body_q for a kinematic body."""
    bq = state.body_q.numpy()
    bq[body_idx][3:7] = quat_xyzw
    state.body_q = wp.array(bq, dtype=wp.transform, device=state.body_q.device)


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
            _set_body_quat(self.state_0, self.pulley_idx, [0.0, 0.0, qz, qw])

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
            starts, ends = _get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines("cable", starts, ends, colors=(0.9, 0.2, 0.2), width=0.008)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
