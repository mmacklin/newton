# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon Rolling Pulley
#
# Atwood machine: two weights connected by a tendon routed over a
# dynamic pulley constrained by a hinge joint.  The heavier weight
# descends, pulling the lighter weight up; the pulley rotates freely
# under cable tension — no kinematic rotation tracking needed.
#
# The pulley has finite mass and inertia, so the XPBD constraint
# solve correctly transmits force between the two sides of the cable
# through the pulley body.
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
from newton.examples.cable.cable import assert_tendon_total_length, get_tendon_attachment_worlds, get_tendon_cable_lines


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

        self.pulley_radius = 0.22
        self.pulley_height = 2.55
        pulley_mass = 500.0
        pulley_half_height = 0.04
        contact_cfg = newton.ModelBuilder.ShapeConfig(
            has_shape_collision=True,
            collision_group=1,
            mu=0.7,
            margin=0.01,
            gap=0.02,
        )

        def cylinder_inertia(mass, radius, half_height):
            inertia_y = 0.5 * mass * radius * radius
            inertia_xz = (1.0 / 12.0) * mass * (3.0 * radius * radius + (2.0 * half_height) ** 2)
            return wp.mat33(
                inertia_xz,
                0.0,
                0.0,
                0.0,
                inertia_y,
                0.0,
                0.0,
                0.0,
                inertia_xz,
            )

        pulley = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.pulley_height), q=wp.quat_identity()),
            mass=pulley_mass,
            inertia=cylinder_inertia(pulley_mass, self.pulley_radius, pulley_half_height),
            lock_inertia=True,
        )
        q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
        builder.add_shape_cylinder(
            pulley,
            xform=wp.transform(q=q_cyl),
            radius=self.pulley_radius,
            half_height=pulley_half_height,
            cfg=contact_cfg,
        )
        self.pulley_idx = pulley

        Dof = newton.ModelBuilder.JointDofConfig

        # Hinge joint: pulley fixed in space, free to rotate about Y
        j_pulley = builder.add_joint_revolute(
            parent=-1,
            child=pulley,
            axis=Axis.Y,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, self.pulley_height), q=wp.quat_identity()),
            child_xform=wp.transform(),
        )

        planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
        planar_ang = [Dof(axis=Axis.Y)]

        self.left_idx = left = builder.add_link(
            xform=wp.transform(p=wp.vec3(-0.5, 0.0, 2.16), q=wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_box(left, hx=0.08, hy=0.08, hz=0.08, cfg=contact_cfg)
        j1 = builder.add_joint_d6(
            parent=-1,
            child=left,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=wp.vec3(-0.5, 0.0, 2.16), q=wp.quat_identity()),
            child_xform=wp.transform(),
        )

        self.right_idx = right = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.5, 0.0, 2.16), q=wp.quat_identity()),
            mass=2.0,
        )
        builder.add_shape_box(right, hx=0.10, hy=0.10, hz=0.10, cfg=contact_cfg)
        j2 = builder.add_joint_d6(
            parent=-1,
            child=right,
            linear_axes=planar_lin,
            angular_axes=planar_ang,
            parent_xform=wp.transform(p=wp.vec3(0.5, 0.0, 2.16), q=wp.quat_identity()),
            child_xform=wp.transform(),
        )

        builder.add_articulation([j_pulley])
        builder.add_articulation([j1])
        builder.add_articulation([j2])

        axis = (0.0, 1.0, 0.0)
        builder.add_tendon()
        builder.add_tendon_link(
            body=left,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.08),
            axis=axis,
        )
        builder.add_tendon_link(
            body=pulley,
            link_type=int(TendonLinkType.ROLLING),
            radius=self.pulley_radius,
            orientation=1,
            mu=10.0,
            offset=(0.0, 0.0, 0.0),
            axis=axis,
            compliance=1.0e-6,
            damping=5.0,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=right,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.0, 0.0, 0.10),
            axis=axis,
            compliance=1.0e-6,
            damping=5.0,
            rest_length=-1.0,
        )

        builder.add_ground_plane()
        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=48,
            joint_linear_relaxation=0.45,
            rigid_contact_relaxation=0.25,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self._initial_left_z = float(self.state_0.body_q.numpy()[self.left_idx][2])
        self._initial_right_z = float(self.state_0.body_q.numpy()[self.right_idx][2])
        self._pulley_theta = 0.0
        self._last_pulley_angle = None
        self._left_z_history = []
        self._right_z_history = []
        self._left_position_history = []
        self._right_position_history = []
        self._left_attachment_history = []
        self._left_pulley_distance_history = []
        self._pulley_rotation_history = []
        self._direction_validation_frames = 120

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, -4.0, 2.0), pitch=5.0, yaw=90.0)
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
        self._record_motion_sample()

    @staticmethod
    def _hinge_y_angle(q):
        return float(2.0 * np.arctan2(q[4], q[6]))

    @staticmethod
    def _angle_delta(prev_angle, angle):
        return float((angle - prev_angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _record_motion_sample(self):
        body_q = self.state_0.body_q.numpy()
        angle = self._hinge_y_angle(body_q[self.pulley_idx])
        if self._last_pulley_angle is not None:
            self._pulley_theta += self._angle_delta(self._last_pulley_angle, angle)
        self._last_pulley_angle = angle
        self._pulley_rotation_history.append(self._pulley_theta)
        self._left_z_history.append(float(body_q[self.left_idx][2]))
        self._right_z_history.append(float(body_q[self.right_idx][2]))
        self._left_position_history.append(np.array(body_q[self.left_idx][:3], dtype=np.float64))
        self._right_position_history.append(np.array(body_q[self.right_idx][:3], dtype=np.float64))
        att_l, _ = get_tendon_attachment_worlds(self.solver, self.model, self.state_0)
        self._left_attachment_history.append(np.array(att_l[0], dtype=np.float64))
        self._left_pulley_distance_history.append(
            float(np.linalg.norm(body_q[self.left_idx][:3] - body_q[self.pulley_idx][:3]))
        )

    def _assert_light_attachment_stays_below_pulley(self, attachment, body_q, label="Rolling pulley"):
        pulley_center = body_q[self.pulley_idx][:3]
        crown_limit = float(pulley_center[2] + self.pulley_radius + 0.04)
        side_limit = float(pulley_center[0] + self.pulley_radius + 0.04)
        body_center_z = float(body_q[self.left_idx][2])
        body_top_limit = float(pulley_center[2] + self.pulley_radius)
        assert body_center_z <= body_top_limit, (
            f"{label} light body center should not pass over the pulley crown: "
            f"body_z={body_center_z:.4f}, top_limit={body_top_limit:.4f}"
        )
        assert float(attachment[2]) <= crown_limit, (
            f"{label} light-side cable attachment should not crest over the pulley: "
            f"attachment_z={attachment[2]:.4f}, crown_limit={crown_limit:.4f}"
        )
        assert float(attachment[0]) <= side_limit, (
            f"{label} light weight should stay on the near side of the pulley: "
            f"attachment_x={attachment[0]:.4f}, side_limit={side_limit:.4f}"
        )

    def test_post_step(self):
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Rolling pulley produced non-finite body state"
        assert float(np.max(np.abs(body_q[:, :3]))) < 20.0, "Rolling pulley body state became unbounded"

        att_r = self.solver.tendon_seg_attachment_r.numpy()
        att_l = self.solver.tendon_seg_attachment_l.numpy()
        pulley_center = body_q[self.pulley_idx][:3]
        left_clearance = float(np.linalg.norm(att_l[0] - pulley_center) - self.pulley_radius)
        right_clearance = float(np.linalg.norm(att_r[1] - pulley_center) - self.pulley_radius)
        assert min(left_clearance, right_clearance) > 0.0, (
            f"Weight attachments should stay outside the pulley tangent singularity: "
            f"left={left_clearance:.4f}, right={right_clearance:.4f}"
        )
        if self._left_attachment_history:
            self._assert_light_attachment_stays_below_pulley(self._left_attachment_history[-1], body_q)

        if len(self._pulley_rotation_history) <= self._direction_validation_frames:
            assert_tendon_total_length(self, rel_tol=0.30)
        if self.sim_time < self.frame_dt * 1.5:
            pulley_z = body_q[self.pulley_idx][2]
            assert att_r[0][2] >= pulley_z - 1.0e-4, (
                f"Cable should wrap over pulley: left tangent z={att_r[0][2]:.3f} <= center z={pulley_z:.3f}"
            )
            assert att_l[1][2] >= pulley_z - 1.0e-4, (
                f"Cable should wrap over pulley: right tangent z={att_l[1][2]:.3f} <= center z={pulley_z:.3f}"
            )

    def test_final(self):
        if not self._pulley_rotation_history:
            self._record_motion_sample()

        sample = min(len(self._pulley_rotation_history) - 1, self._direction_validation_frames - 1)
        left_rise = self._left_z_history[sample] - self._initial_left_z
        right_drop = self._initial_right_z - self._right_z_history[sample]
        theta = self._pulley_rotation_history[sample]
        min_left_attachment_distance = min(
            np.linalg.norm(attachment - self.state_0.body_q.numpy()[self.pulley_idx][:3])
            for attachment in self._left_attachment_history
        )
        left_step = float(np.max(np.linalg.norm(np.diff(np.array(self._left_position_history), axis=0), axis=1)))
        right_step = float(np.max(np.linalg.norm(np.diff(np.array(self._right_position_history), axis=0), axis=1)))
        assert np.isfinite([left_rise, right_drop, theta]).all(), (
            f"Rolling pulley produced non-finite values inside the validated prefix: "
            f"left_rise={left_rise:.4f}, right_drop={right_drop:.4f}, theta={theta:.4f}"
        )
        assert left_rise > 0.05, f"Light side should rise over the validated prefix: dz={left_rise:.4f}"
        assert right_drop > 0.05, f"Heavy side should descend over the validated prefix: dz={right_drop:.4f}"
        assert theta > 0.5, f"High-friction pulley should rotate with cable travel: theta={theta:.4f}"
        assert min_left_attachment_distance < self.pulley_radius + 0.12, (
            f"Light-side cable attachment should reach the pulley contact neighborhood, not avoid it: "
            f"min_attachment_distance={min_left_attachment_distance:.4f}"
        )
        assert max(left_step, right_step) < 0.35, (
            f"Rolling pulley contact should stay bounded without frame jumps: "
            f"left_step={left_step:.4f}, right_step={right_step:.4f}"
        )
        body_q = self.state_0.body_q.numpy()
        body_top_limit = float(body_q[self.pulley_idx][2] + self.pulley_radius)
        max_left_z = max(self._left_z_history)
        assert max_left_z <= body_top_limit, (
            f"Rolling pulley light body center should stay below pulley crown over the full run: "
            f"max_z={max_left_z:.4f}, top_limit={body_top_limit:.4f}"
        )
        for attachment in self._left_attachment_history:
            self._assert_light_attachment_stays_below_pulley(attachment, body_q)

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
