# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""AVBD 3D Heavy Rope scene recreated with Newton VBD."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

TITLE = "Heavy Rope"
DESCRIPTION = "Rope with a large heavy terminal block."
REPORT_FRAMES = 180
STATUS_NOTE = "Reference parameters reproduced."


def _xform(
    position: tuple[float, float, float],
    rotation: wp.quat | None = None,
) -> wp.transform:
    if rotation is None:
        rotation = wp.quat_identity()
    return wp.transform(p=wp.vec3(float(position[0]), float(position[1]), float(position[2])), q=rotation)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.title = TITLE
        self.fps = 60
        self.frame_dt = 1.0 / 60.0
        self.sim_dt = self.frame_dt
        self.sim_substeps = 1
        self.sim_time = 0.0
        self.solver_iterations = 10
        self.collision_margin = 0.01
        self.penalty_min = 1.0
        self.penalty_max = 1.0e10
        self.position_limit = 500.0
        self.speed_limit = 500.0
        self.min_z_limit = -40.0
        self.rigid_joint_linear_ke = self.penalty_max
        self.rigid_joint_angular_ke = self.penalty_max
        self.rigid_joint_linear_kd = 0.0
        self.rigid_joint_angular_kd = 0.0
        self.rigid_avbd_contact_alpha = 0.99
        self.rigid_contact_stick_freeze_translation_eps = 0.0
        self.rigid_contact_stick_freeze_angular_eps = 0.0
        self.rigid_contact_k_start = self.penalty_min
        self.rigid_body_contact_buffer_size = 512
        self.rigid_body_serial_reverse = False
        self.soft_joint_hard = True
        self.soft_joint_indices: list[int] = []
        self.breakable_joint_indices: list[int] = []
        self.broken_joint_count = 0
        self.rigid_contact_history = True
        self.contact_matching = "latest"
        pass

        builder = newton.ModelBuilder(gravity=-10.0)
        builder.rigid_gap = self.collision_margin
        builder.default_shape_cfg.density = 1.0
        builder.default_shape_cfg.ke = self.penalty_max
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.mu = 0.5
        builder.default_shape_cfg.margin = self.collision_margin * 0.5
        builder.default_shape_cfg.gap = self.collision_margin

        self._build_scene(builder)
        builder.color()
        body_order = (
            range(builder.body_count - 1, -1, -1) if self.rigid_body_serial_reverse else range(builder.body_count)
        )
        builder.body_color_groups = [np.array([body], dtype=int) for body in body_order]
        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        rigid_contact_max = max(4096, self.model.shape_count * 64)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            contact_matching=self.contact_matching,
            contact_matching_pos_threshold=0.05,
            contact_matching_normal_dot_threshold=0.9,
            rigid_contact_max=rigid_contact_max,
        )
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.solver_iterations,
            friction_epsilon=0.0,
            rigid_avbd_alpha=0.99,
            rigid_avbd_joint_alpha=0.99,
            rigid_avbd_contact_alpha=self.rigid_avbd_contact_alpha,
            rigid_avbd_linear_beta=10000.0,
            rigid_avbd_angular_beta=100.0,
            rigid_avbd_gamma=0.999,
            rigid_contact_hard=True,
            rigid_contact_history=self.rigid_contact_history,
            rigid_contact_stick_motion_eps=1.0e-5,
            rigid_contact_stick_freeze_translation_eps=self.rigid_contact_stick_freeze_translation_eps,
            rigid_contact_stick_freeze_angular_eps=self.rigid_contact_stick_freeze_angular_eps,
            rigid_contact_k_start=self.rigid_contact_k_start,
            rigid_joint_linear_k_start=self.penalty_min,
            rigid_joint_angular_k_start=self.penalty_min,
            rigid_joint_linear_ke=self.rigid_joint_linear_ke,
            rigid_joint_angular_ke=self.rigid_joint_angular_ke,
            rigid_joint_linear_kd=self.rigid_joint_linear_kd,
            rigid_joint_angular_kd=self.rigid_joint_angular_kd,
            rigid_body_contact_buffer_size=self.rigid_body_contact_buffer_size,
        )
        if not self.soft_joint_hard:
            for joint_index in self.soft_joint_indices:
                self.solver.set_joint_constraint_mode(joint_index, hard=False)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.contacts = self.collision_pipeline.contacts()
        self.viewer.set_model(self.model)
        self.shadow_distance = 42.0
        self.viewer.set_camera(pos=wp.vec3(38.13091590, -27.13091590, 22.08293901), pitch=-24.0, yaw=135.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 55.0
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(wp.vec3(11.0, 0.0, 5.0))

        self.max_speed = 0.0
        self.final_speed = 0.0
        self.max_angular_speed = 0.0
        self.final_angular_speed = 0.0
        self.max_abs_position = 0.0
        self.final_max_abs_position = 0.0
        self.min_z = float("inf")
        self.max_z = float("-inf")
        self.final_min_z = float("inf")
        self.final_max_z = float("-inf")
        self.final_xy_radius = 0.0
        self.max_contacts = 0
        self.nan_detected = False
        self.extra_metrics: dict[str, float | int | bool] = {}

    def _shape_cfg(self, density: float, friction: float) -> newton.ModelBuilder.ShapeConfig:
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = density
        cfg.ke = self.penalty_max
        cfg.kd = 0.0
        cfg.mu = friction
        cfg.margin = self.collision_margin * 0.5
        cfg.gap = self.collision_margin
        return cfg

    def _build_scene(self, builder: newton.ModelBuilder) -> None:
        ground = builder.add_body(xform=_xform((0.0, 0.0, -20.0)), is_kinematic=True, label="ground")
        builder.add_shape_box(
            ground,
            hx=50.0,
            hy=50.0,
            hz=0.5,
            cfg=self._shape_cfg(0.0, 0.5),
            label="ground_shape",
        )
        prev = -1
        for i in range(20):
            is_last = i == 19
            size = 5.0 if is_last else 1.0
            x = float(i) + (2.5 if is_last else 0.0)
            body = builder.add_body(xform=_xform((x, 0.0, 10.0)), is_kinematic=i == 0, label=f"heavy_rope_{i}")
            builder.add_shape_box(
                body,
                hx=0.5 * size,
                hy=2.5 if is_last else 0.25,
                hz=2.5 if is_last else 0.25,
                cfg=self._shape_cfg(0.0 if i == 0 else 1.0, 0.5),
                label=f"heavy_rope_{i}_shape",
            )
            if prev >= 0:
                builder.add_joint_ball(
                    parent=prev,
                    child=body,
                    parent_xform=_xform((0.5, 0.0, 0.0)),
                    child_xform=_xform((-2.5 if is_last else -0.5, 0.0, 0.0)),
                    collision_filter_parent=True,
                    label=f"heavy_rope_joint_{i}",
                )
            prev = body

    def _update_metrics(self) -> None:
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(body_qd)):
            self.nan_detected = True
            return

        if body_q.size:
            pos = body_q[:, 0:3]
            self.max_abs_position = max(self.max_abs_position, float(np.max(np.abs(pos))))
            self.final_max_abs_position = float(np.max(np.abs(pos)))
            self.final_min_z = float(np.min(pos[:, 2]))
            self.final_max_z = float(np.max(pos[:, 2]))
            self.final_xy_radius = float(np.max(np.linalg.norm(pos[:, 0:2], axis=1)))
            self.min_z = min(self.min_z, self.final_min_z)
            self.max_z = max(self.max_z, self.final_max_z)
        if body_qd.size:
            self.final_speed = float(np.max(np.linalg.norm(body_qd[:, 0:3], axis=1)))
            self.final_angular_speed = float(np.max(np.linalg.norm(body_qd[:, 3:6], axis=1)))
            self.max_speed = max(self.max_speed, self.final_speed)
            self.max_angular_speed = max(self.max_angular_speed, self.final_angular_speed)
        if self.contacts.rigid_contact_count is not None:
            self.max_contacts = max(self.max_contacts, int(self.contacts.rigid_contact_count.numpy()[0]))

    def simulate(self) -> None:
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self._update_metrics()

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        if self.nan_detected:
            raise AssertionError(f"{self.title}: non-finite state detected")
        if self.max_abs_position > self.position_limit:
            raise AssertionError(
                f"{self.title}: max |position| {self.max_abs_position:.3f} exceeds {self.position_limit:.3f}"
            )
        if self.max_speed > self.speed_limit:
            raise AssertionError(f"{self.title}: max speed {self.max_speed:.3f} exceeds {self.speed_limit:.3f}")
        if self.min_z < self.min_z_limit:
            raise AssertionError(f"{self.title}: min z {self.min_z:.3f} below {self.min_z_limit:.3f}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=REPORT_FRAMES)
        return parser

    @property
    def metric_summary(self) -> dict[str, float | int | bool]:
        return {
            "max_speed": self.max_speed,
            "final_speed": self.final_speed,
            "max_angular_speed": self.max_angular_speed,
            "final_angular_speed": self.final_angular_speed,
            "max_abs_position": self.max_abs_position,
            "final_max_abs_position": self.final_max_abs_position,
            "min_z": self.min_z,
            "max_z": self.max_z,
            "final_min_z": self.final_min_z,
            "final_max_z": self.final_max_z,
            "final_xy_radius": self.final_xy_radius,
            "max_contacts": self.max_contacts,
            "broken_joint_count": self.broken_joint_count,
            "nan_detected": self.nan_detected,
            **self.extra_metrics,
        }


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
