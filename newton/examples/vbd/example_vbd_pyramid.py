# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""AVBD 3D Pyramid scene recreated with Newton VBD."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

TITLE = "Pyramid"
DESCRIPTION = "Sixteen-row brick pyramid settling under gravity."
REPORT_FRAMES = 180
STATUS_NOTE = "Reference parameters reproduced."


def _xform(
    position: tuple[float, float, float],
    rotation: wp.quat | None = None,
) -> wp.transform:
    if rotation is None:
        rotation = wp.quat_identity()
    return wp.transform(p=wp.vec3(float(position[0]), float(position[1]), float(position[2])), q=rotation)


def _fit_line_metrics(points: np.ndarray) -> tuple[float, float, float]:
    centered = points - np.mean(points, axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    distances = np.linalg.norm(np.cross(centered, axis), axis=1)
    tilt_xy_per_z = float(np.linalg.norm(axis[0:2]) / max(abs(axis[2]), 1.0e-8))
    return float(np.max(distances)), float(np.sqrt(np.mean(distances * distances))), tilt_xy_per_z


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
        self.speed_limit = 100.0
        self.min_z_limit = -80.0
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
        self.brick_bodies: list[int] = []
        self.brick_rows: list[list[int]] = []
        self.brick_initial_positions: list[tuple[float, float, float]] = []

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
        self.shadow_distance = 30.0
        self.viewer.set_camera(pos=wp.vec3(19.66853973, -19.66853973, 17.23819780), pitch=-22.0, yaw=135.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 55.0
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(wp.vec3(0.0, 0.0, 6.0))

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
        self.rigid_body_serial_reverse = True
        self.rigid_avbd_contact_alpha = 0.85
        self.rigid_contact_k_start = 1000.0

        ground = builder.add_body(xform=_xform((0.0, 0.0, -0.5)), is_kinematic=True, label="ground")
        builder.add_shape_box(
            ground,
            hx=50.0,
            hy=50.0,
            hz=0.5,
            cfg=self._shape_cfg(0.0, 0.5),
            label="ground_shape",
        )
        for y in range(16):
            row: list[int] = []
            for x in range(16 - y):
                position = (x * 1.01 + y * 0.5 - 8.0, 0.0, y * 0.85 + 0.5)
                body = builder.add_body(xform=_xform(position), label=f"brick_{y}_{x}")
                builder.add_shape_box(
                    body,
                    hx=0.5,
                    hy=0.25,
                    hz=0.25,
                    cfg=self._shape_cfg(1.0, 0.5),
                    label=f"brick_{y}_{x}_shape",
                )
                row.append(body)
                self.brick_bodies.append(body)
                self.brick_initial_positions.append(position)
            self.brick_rows.append(row)

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
        brick_pos = body_q[self.brick_bodies, 0:3]
        abs_y = float(np.max(np.abs(brick_pos[:, 1])))
        y_span = float(np.max(brick_pos[:, 1]) - np.min(brick_pos[:, 1]))
        row_y_span = 0.0
        for row in self.brick_rows:
            row_y = body_q[row, 1]
            row_y_span = max(row_y_span, float(np.max(row_y) - np.min(row_y)))
        self.extra_metrics["pyramid_observed_abs_y"] = max(
            abs_y, float(self.extra_metrics.get("pyramid_observed_abs_y", 0.0))
        )
        self.extra_metrics["pyramid_observed_lateral_span"] = max(
            y_span, float(self.extra_metrics.get("pyramid_observed_lateral_span", 0.0))
        )
        self.extra_metrics["pyramid_observed_row_lateral_span"] = max(
            row_y_span, float(self.extra_metrics.get("pyramid_observed_row_lateral_span", 0.0))
        )

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
        body_q = self.state_0.body_q.numpy()
        brick_pos = body_q[self.brick_bodies, 0:3]
        initial_pos = np.asarray(self.brick_initial_positions, dtype=np.float64)
        displacement = np.linalg.norm(brick_pos - initial_pos, axis=1)
        max_brick_displacement = float(np.max(displacement))
        mean_brick_displacement = float(np.mean(displacement))
        brick_final_max_z = float(np.max(brick_pos[:, 2]))
        brick_lateral_span = float(np.max(brick_pos[:, 1]) - np.min(brick_pos[:, 1]))
        brick_abs_y = float(np.max(np.abs(brick_pos[:, 1])))
        max_row_lateral_span = 0.0
        for row in self.brick_rows:
            row_y = body_q[row, 1]
            max_row_lateral_span = max(max_row_lateral_span, float(np.max(row_y) - np.min(row_y)))
        observed_abs_y = max(brick_abs_y, float(self.extra_metrics.get("pyramid_observed_abs_y", 0.0)))
        observed_lateral_span = max(
            brick_lateral_span, float(self.extra_metrics.get("pyramid_observed_lateral_span", 0.0))
        )
        observed_row_lateral_span = max(
            max_row_lateral_span, float(self.extra_metrics.get("pyramid_observed_row_lateral_span", 0.0))
        )

        self.extra_metrics["pyramid_max_brick_displacement"] = max_brick_displacement
        self.extra_metrics["pyramid_mean_brick_displacement"] = mean_brick_displacement
        self.extra_metrics["pyramid_final_max_z"] = brick_final_max_z
        self.extra_metrics["pyramid_lateral_span"] = brick_lateral_span
        self.extra_metrics["pyramid_abs_y"] = brick_abs_y
        self.extra_metrics["pyramid_row_lateral_span"] = max_row_lateral_span
        self.extra_metrics["pyramid_observed_abs_y"] = observed_abs_y
        self.extra_metrics["pyramid_observed_lateral_span"] = observed_lateral_span
        self.extra_metrics["pyramid_observed_row_lateral_span"] = observed_row_lateral_span

        if self.final_speed > 5.0:
            raise AssertionError(f"{self.title}: final speed {self.final_speed:.3f} exceeds 5.000")
        if self.final_angular_speed > 8.0:
            raise AssertionError(f"{self.title}: final angular speed {self.final_angular_speed:.3f} exceeds 8.000")
        if observed_abs_y > 0.10:
            raise AssertionError(
                f"{self.title}: brick centers left the source y=0 plane by {observed_abs_y:.3f} m; expected <= 0.100 m"
            )
        if observed_lateral_span > 0.16:
            raise AssertionError(
                f"{self.title}: brick y-span reached {observed_lateral_span:.3f} m; expected <= 0.160 m"
            )
        if observed_row_lateral_span > 0.12:
            raise AssertionError(
                f"{self.title}: one pyramid row spread {observed_row_lateral_span:.3f} m out of plane; "
                "expected <= 0.120 m"
            )
        if max_brick_displacement > 8.0:
            raise AssertionError(
                f"{self.title}: brick displacement {max_brick_displacement:.3f} m exceeds 8.000; "
                "pyramid did not stay compact"
            )
        if brick_final_max_z < 6.0:
            raise AssertionError(
                f"{self.title}: tallest brick center is {brick_final_max_z:.3f} m; expected the pile to retain visible height"
            )
        if brick_lateral_span > 4.0:
            raise AssertionError(
                f"{self.title}: brick lateral span {brick_lateral_span:.3f} m exceeds 4.000; pyramid scattered sideways"
            )
        if self.final_min_z < -2.0:
            raise AssertionError(f"{self.title}: final min z {self.final_min_z:.3f} below -2.000")
        if self.final_max_z > 12.0:
            raise AssertionError(
                f"{self.title}: final max z {self.final_max_z:.3f} exceeds 12.000; pyramid launched upward"
            )

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
