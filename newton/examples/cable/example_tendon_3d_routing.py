# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon 3D Routing
#
# Right-angle pulley drive with three cylinders.  P1 and P3 have their
# axes along Y (wrapping in XZ) while P2 at 90 degrees has its axis
# along X (wrapping in YZ).  The cable routes over P1, under P2, over
# P3 with weights on both ends.  The inter-pulley cable segments run
# approximately vertically, along the intersection of adjacent wrapping
# planes.
#
# Command: python -m newton.examples tendon_3d_routing
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType


def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def _get_tendon_cable_lines(solver, model, state):
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

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

        self.r1 = 0.2
        self.r2 = 0.2
        self.r3 = 0.2

        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)

        # P1 at (-r, +r, +1), axis Y — wraps in XZ plane
        # Inner edge at x=0 aligns with P2 center; y=+r matches P2 tangent offset
        self.q_p1_init = np.array([-s, 0.0, 0.0, c])
        q_p1_wp = wp.quat(*self.q_p1_init.tolist())
        p1 = builder.add_body(
            xform=wp.transform(p=wp.vec3(-self.r1, self.r1, 1.0), q=q_p1_wp),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_shape_cylinder(p1, radius=self.r1, half_height=0.06)
        self.p1_idx = p1

        # P2 at (0, 0, -1), axis X — wraps in YZ plane
        self.q_p2_init = np.array([0.0, s, 0.0, c])
        q_p2_wp = wp.quat(*self.q_p2_init.tolist())
        p2 = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -1.0), q=q_p2_wp),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_shape_cylinder(p2, radius=self.r2, half_height=0.3)
        self.p2_idx = p2

        # P3 at (+r, -r, +1), axis Y — wraps in XZ plane
        # Inner edge at x=0 aligns with P2 center; y=-r matches P2 tangent offset
        self.q_p3_init = np.array([-s, 0.0, 0.0, c])
        q_p3_wp = wp.quat(*self.q_p3_init.tolist())
        p3 = builder.add_body(
            xform=wp.transform(p=wp.vec3(self.r3, -self.r3, 1.0), q=q_p3_wp),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_shape_cylinder(p3, radius=self.r3, half_height=0.06)
        self.p3_idx = p3

        Dof = newton.ModelBuilder.JointDofConfig
        free_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Y), Dof(axis=Axis.Z)]
        free_ang = [Dof(axis=Axis.X), Dof(axis=Axis.Y), Dof(axis=Axis.Z)]

        sphere_pos = wp.vec3(-2.0 * self.r1, self.r1, -2.0)
        left = builder.add_link(
            xform=wp.transform(p=sphere_pos, q=wp.quat_identity()),
            mass=1.5,
        )
        builder.add_shape_sphere(left, radius=0.10)
        j1 = builder.add_joint_d6(
            parent=-1,
            child=left,
            linear_axes=free_lin,
            angular_axes=free_ang,
            parent_xform=wp.transform(p=sphere_pos, q=wp.quat_identity()),
            child_xform=wp.transform(),
        )

        box_pos = wp.vec3(2.0 * self.r3, -self.r3, -2.0)
        right = builder.add_link(
            xform=wp.transform(p=box_pos, q=wp.quat_identity()),
            mass=3.5,
        )
        builder.add_shape_box(right, hx=0.12, hy=0.12, hz=0.12)
        j2 = builder.add_joint_d6(
            parent=-1,
            child=right,
            linear_axes=free_lin,
            angular_axes=free_ang,
            parent_xform=wp.transform(p=box_pos, q=wp.quat_identity()),
            child_xform=wp.transform(),
        )

        builder.add_articulation([j1])
        builder.add_articulation([j2])

        axis_z = (0.0, 0.0, 1.0)
        builder.add_tendon()

        builder.add_tendon_link(
            body=left,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.10),
            axis=(0.0, 1.0, 0.0),
        )
        builder.add_tendon_link(
            body=p1,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.r1,
            orientation=1,
            mu=0.0,
            offset=(0.0, 0.0, 0.0),
            axis=axis_z,
            compliance=1.0e-5,
            damping=0.1,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=p2,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.r2,
            orientation=-1,
            mu=0.0,
            offset=(0.0, 0.0, 0.0),
            axis=axis_z,
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
            axis=axis_z,
            compliance=1.0e-5,
            damping=0.1,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=right,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.12),
            axis=(0.0, 1.0, 0.0),
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

        self.p1_angle = 0.0
        self.p2_angle = 0.0
        self.p3_angle = 0.0

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(3.0, -5.2, 0.0), pitch=-5.0, yaw=120.0)
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

            d_cable = rest_after[0] - rest_before[0]
            self.p1_angle -= d_cable / self.r1
            self.p2_angle += d_cable / self.r2
            self.p3_angle -= d_cable / self.r3

            q_z1 = np.array([0.0, 0.0, np.sin(self.p1_angle / 2), np.cos(self.p1_angle / 2)])
            q_z2 = np.array([0.0, 0.0, np.sin(self.p2_angle / 2), np.cos(self.p2_angle / 2)])
            q_z3 = np.array([0.0, 0.0, np.sin(self.p3_angle / 2), np.cos(self.p3_angle / 2)])
            _set_body_quat(self.state_0, self.p1_idx, _quat_multiply(self.q_p1_init, q_z1))
            _set_body_quat(self.state_0, self.p2_idx, _quat_multiply(self.q_p2_init, q_z2))
            _set_body_quat(self.state_0, self.p3_idx, _quat_multiply(self.q_p3_init, q_z3))

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        if self.sim_time < self.frame_dt * 1.5:
            att_l = self.solver.tendon_seg_attachment_l.numpy()
            att_r = self.solver.tendon_seg_attachment_r.numpy()
            for i in range(self.model.tendon_segment_count):
                dx = abs(att_l[i][0] - att_r[i][0])
                dy = abs(att_l[i][1] - att_r[i][1])
                dz = abs(att_l[i][2] - att_r[i][2])
                assert dz > 0.1, f"Segment {i} has no vertical span: dz={dz}"
                assert dx < 0.02, f"Segment {i} not vertical in x: dx={dx}"
                assert dy < 0.02, f"Segment {i} not vertical in y: dy={dy}"

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in body positions"

        sphere_pos = body_q[3][:3]
        box_pos = body_q[4][:3]
        sphere_moved = np.linalg.norm(sphere_pos - np.array([-0.4, 0.2, -2.0])) > 0.1
        box_moved = np.linalg.norm(box_pos - np.array([0.4, -0.2, -2.0])) > 0.1
        assert sphere_moved, f"Sphere should have moved from start: {sphere_pos}"
        assert box_moved, f"Box should have moved from start: {box_pos}"

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
