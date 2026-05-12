# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""AVBD 3D Soft Body scene recreated with Newton VBD."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

TITLE = "Soft Body"
DESCRIPTION = "Three 4x4x4 lattices of rigid cubes joined by finite-stiffness soft fixed joints."
REPORT_FRAMES = 120
STATUS_NOTE = "Uses source soft fixed-joint stiffness with 40 Newton VBD iterations to prevent lattice collapse."


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
        self.speed_limit = 20.0
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
        self.soft_bodies: list[list[list[list[int]]]] = []

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
        self.shadow_distance = 34.0
        self.viewer.set_camera(pos=wp.vec3(22.29101169, -22.29101169, 26.73662418), pitch=-22.0, yaw=135.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 55.0
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(wp.vec3(0.0, 0.0, 14.0))

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
        self.solver_iterations = 40
        self.soft_joint_hard = False
        self.rigid_joint_linear_ke = 1000.0
        self.rigid_joint_angular_ke = 250.0
        self.rigid_joint_linear_kd = 1.0
        self.rigid_joint_angular_kd = 1.0
        self.rigid_body_contact_buffer_size = 1024

        ground = builder.add_body(xform=_xform((0.0, 0.0, 0.0)), is_kinematic=True, label="ground")
        builder.add_shape_box(ground, hx=50.0, hy=50.0, hz=0.5, cfg=self._shape_cfg(0.0, 0.5), label="ground_shape")

        width = 4
        depth = 4
        height = 4
        size = 0.8
        half = size * 0.5
        base_z = 8.0
        stack_gap = 2.0

        for stack in range(3):
            bodies: list[list[list[int]]] = [
                [[0 for _z in range(height)] for _y in range(depth)] for _x in range(width)
            ]
            shapes: list[list[list[int]]] = [
                [[0 for _z in range(height)] for _y in range(depth)] for _x in range(width)
            ]
            stack_z = stack * (height * size + stack_gap)
            for x in range(width):
                for y in range(depth):
                    for z in range(height):
                        position = (
                            (x - (width - 1) * 0.5) * size,
                            (y - (depth - 1) * 0.5) * size,
                            base_z + stack_z + z * size,
                        )
                        body = builder.add_body(xform=_xform(position), label=f"soft_{stack}_{x}_{y}_{z}")
                        shape = builder.add_shape_box(
                            body,
                            hx=half,
                            hy=half,
                            hz=half,
                            cfg=self._shape_cfg(1.0, 0.5),
                            label=f"soft_{stack}_{x}_{y}_{z}_shape",
                        )
                        bodies[x][y][z] = body
                        shapes[x][y][z] = shape

            for x in range(1, width):
                for y in range(depth):
                    for z in range(height):
                        joint = builder.add_joint_fixed(
                            parent=bodies[x - 1][y][z],
                            child=bodies[x][y][z],
                            parent_xform=_xform((half, 0.0, 0.0)),
                            child_xform=_xform((-half, 0.0, 0.0)),
                            collision_filter_parent=True,
                            label=f"soft_x_{stack}_{x}_{y}_{z}",
                        )
                        self.soft_joint_indices.append(joint)
            for x in range(width):
                for y in range(1, depth):
                    for z in range(height):
                        joint = builder.add_joint_fixed(
                            parent=bodies[x][y - 1][z],
                            child=bodies[x][y][z],
                            parent_xform=_xform((0.0, half, 0.0)),
                            child_xform=_xform((0.0, -half, 0.0)),
                            collision_filter_parent=True,
                            label=f"soft_y_{stack}_{x}_{y}_{z}",
                        )
                        self.soft_joint_indices.append(joint)
            for x in range(width):
                for y in range(depth):
                    for z in range(1, height):
                        joint = builder.add_joint_fixed(
                            parent=bodies[x][y][z - 1],
                            child=bodies[x][y][z],
                            parent_xform=_xform((0.0, 0.0, half)),
                            child_xform=_xform((0.0, 0.0, -half)),
                            collision_filter_parent=True,
                            label=f"soft_z_{stack}_{x}_{y}_{z}",
                        )
                        self.soft_joint_indices.append(joint)

            for x in range(1, width):
                for y in range(depth):
                    for z in range(1, height):
                        builder.add_shape_collision_filter_pair(shapes[x - 1][y][z - 1], shapes[x][y][z])
                        builder.add_shape_collision_filter_pair(shapes[x][y][z - 1], shapes[x - 1][y][z])
            for x in range(width):
                for y in range(1, depth):
                    for z in range(1, height):
                        builder.add_shape_collision_filter_pair(shapes[x][y - 1][z - 1], shapes[x][y][z])
                        builder.add_shape_collision_filter_pair(shapes[x][y][z - 1], shapes[x][y - 1][z])
            for x in range(1, width):
                for y in range(1, depth):
                    for z in range(height):
                        builder.add_shape_collision_filter_pair(shapes[x - 1][y - 1][z], shapes[x][y][z])
                        builder.add_shape_collision_filter_pair(shapes[x][y - 1][z], shapes[x - 1][y][z])

            self.soft_bodies.append(bodies)

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
        body_q = self.state_0.body_q.numpy()
        min_stack_z_extent = float("inf")
        min_neighbor_distance = float("inf")
        for bodies in self.soft_bodies:
            stack_indices = [body for plane in bodies for row in plane for body in row]
            stack_pos = body_q[stack_indices, 0:3]
            z_extent = float(np.max(stack_pos[:, 2]) - np.min(stack_pos[:, 2]))
            min_stack_z_extent = min(min_stack_z_extent, z_extent)
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        p = body_q[bodies[x][y][z], 0:3]
                        for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                            nx = x + dx
                            ny = y + dy
                            nz = z + dz
                            if nx < 4 and ny < 4 and nz < 4:
                                q = body_q[bodies[nx][ny][nz], 0:3]
                                min_neighbor_distance = min(min_neighbor_distance, float(np.linalg.norm(q - p)))

        self.extra_metrics["soft_body_min_stack_z_extent"] = min_stack_z_extent
        self.extra_metrics["soft_body_min_neighbor_distance"] = min_neighbor_distance

        if self.final_speed > 1.0:
            raise AssertionError(f"{self.title}: final speed {self.final_speed:.3f} exceeds 1.000")
        if self.final_angular_speed > 2.0:
            raise AssertionError(f"{self.title}: final angular speed {self.final_angular_speed:.3f} exceeds 2.000")
        if min_stack_z_extent < 2.0:
            raise AssertionError(
                f"{self.title}: lattice z extent collapsed to {min_stack_z_extent:.3f} m; "
                "expected each 4x4x4 block to retain most of its height"
            )
        if min_neighbor_distance < 0.55:
            raise AssertionError(
                f"{self.title}: neighboring lattice bodies are {min_neighbor_distance:.3f} m apart; "
                "expected soft joints to prevent self-collapse"
            )
        if self.final_max_z > 22.0:
            raise AssertionError(
                f"{self.title}: final max z {self.final_max_z:.3f} exceeds 22.000; lattice is expanding"
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
