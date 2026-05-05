# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Basic Reduced Elastic Matrix ROM
#
# Demonstrates a Simscape-style reduced flexible body workflow. A small
# perforated bracket is converted from nodal mass, stiffness, and damping
# matrices into sampled modal shape functions with ModalGeneratorFEM. The left
# interface is fixed to a driven gripper frame; the right interface carries a
# rigid payload, so the matrix-derived modes bend under ordinary fixed joints.
#
# Command: python -m newton.examples basic_reduced_elastic_matrix_rom
#
###########################################################################

from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._reduced_elastic import elastic_shape_deformed_vertices, joint_endpoint_world


def _add_matrix_spring(stiffness: np.ndarray, node_a: int, node_b: int, stiffness_xyz: np.ndarray):
    for axis in range(3):
        value = float(stiffness_xyz[axis])
        ia = 3 * node_a + axis
        ib = 3 * node_b + axis
        stiffness[ia, ia] += value
        stiffness[ib, ib] += value
        stiffness[ia, ib] -= value
        stiffness[ib, ia] -= value


def _make_perforated_plate(
    length: float,
    width: float,
    thickness: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int, int]], np.ndarray, np.ndarray]:
    x = np.linspace(-0.5 * length, 0.5 * length, nx + 1, dtype=np.float32)
    y = np.linspace(-0.5 * width, 0.5 * width, ny + 1, dtype=np.float32)

    def is_cutout(cx: float, cy: float) -> bool:
        center_window = abs(cx) < 0.15 * length and abs(cy) < 0.22 * width
        left_window = -0.38 * length < cx < -0.18 * length and abs(cy) < 0.18 * width
        right_window = 0.20 * length < cx < 0.40 * length and abs(cy) < 0.18 * width
        return center_window or left_window or right_window

    active_cells: list[tuple[int, int]] = []
    used_grid_nodes: set[tuple[int, int]] = set()
    for i in range(nx):
        for j in range(ny):
            cx = 0.5 * (float(x[i]) + float(x[i + 1]))
            cy = 0.5 * (float(y[j]) + float(y[j + 1]))
            if is_cutout(cx, cy):
                continue
            active_cells.append((i, j))
            used_grid_nodes.update(((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)))

    node_lookup: dict[tuple[int, int], int] = {}
    nodes: list[tuple[float, float, float]] = []
    for key in sorted(used_grid_nodes):
        node_lookup[key] = len(nodes)
        nodes.append((float(x[key[0]]), float(y[key[1]]), 0.0))

    cells: list[tuple[int, int, int, int]] = []
    active_set = set(active_cells)
    for i, j in active_cells:
        cells.append(
            (node_lookup[(i, j)], node_lookup[(i + 1, j)], node_lookup[(i + 1, j + 1)], node_lookup[(i, j + 1)])
        )

    vertices: list[tuple[float, float, float]] = []
    vertex_node: list[int] = []
    top: dict[int, int] = {}
    bottom: dict[int, int] = {}
    for node, point in enumerate(nodes):
        px, py, pz = point
        bottom[node] = len(vertices)
        vertices.append((px, py, pz - 0.5 * thickness))
        vertex_node.append(node)
        top[node] = len(vertices)
        vertices.append((px, py, pz + 0.5 * thickness))
        vertex_node.append(node)

    indices: list[int] = []

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        indices.extend((a, b, c, a, c, d))

    for (i, j), (a, b, c, d) in zip(active_cells, cells, strict=True):
        add_quad(top[a], top[b], top[c], top[d])
        add_quad(bottom[a], bottom[d], bottom[c], bottom[b])

        edge_specs = (
            ((i, j - 1), a, b),
            ((i + 1, j), b, c),
            ((i, j + 1), d, c),
            ((i - 1, j), a, d),
        )
        for neighbor, n0, n1 in edge_specs:
            if neighbor not in active_set:
                add_quad(bottom[n0], bottom[n1], top[n1], top[n0])

    node_positions = np.asarray(nodes, dtype=np.float32)
    left_candidates = np.where(node_positions[:, 0] <= float(np.min(node_positions[:, 0])) + 1.0e-6)[0]
    right_candidates = np.where(node_positions[:, 0] >= float(np.max(node_positions[:, 0])) - 1.0e-6)[0]
    left_node = left_candidates[np.argmin(np.abs(node_positions[left_candidates, 1]))]
    right_node = right_candidates[np.argmin(np.abs(node_positions[right_candidates, 1]))]
    return (
        node_positions,
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(vertex_node, dtype=np.int32),
        cells,
        np.asarray([left_node], dtype=np.int32),
        np.asarray([right_node], dtype=np.int32),
    )


def _assemble_plate_matrices(
    nodes: np.ndarray,
    cells: list[tuple[int, int, int, int]],
    density: float,
    thickness: float,
    in_plane_stiffness: float,
    out_of_plane_stiffness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dof_count = 3 * int(nodes.shape[0])
    mass = np.zeros((dof_count, dof_count), dtype=np.float64)
    stiffness = np.zeros((dof_count, dof_count), dtype=np.float64)

    cell_area = 0.0
    nodal_area = np.zeros(nodes.shape[0], dtype=np.float64)
    edge_pairs: set[tuple[int, int]] = set()
    diagonal_pairs: set[tuple[int, int]] = set()
    for a, b, c, d in cells:
        p = nodes[[a, b, c, d]]
        area = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
        area += 0.5 * np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[0]))
        cell_area += float(area)
        for node in (a, b, c, d):
            nodal_area[node] += 0.25 * float(area)
        for pair in ((a, b), (b, c), (c, d), (d, a)):
            edge_pairs.add(tuple(sorted(pair)))
        for pair in ((a, c), (b, d)):
            diagonal_pairs.add(tuple(sorted(pair)))

    total_mass = max(float(density) * float(thickness) * cell_area, 1.0e-6)
    for node, area in enumerate(nodal_area):
        lumped = max(total_mass * area / max(cell_area, 1.0e-9), 1.0e-5)
        for axis in range(3):
            mass[3 * node + axis, 3 * node + axis] = lumped

    for node_a, node_b in sorted(edge_pairs):
        length = max(float(np.linalg.norm(nodes[node_b] - nodes[node_a])), 1.0e-6)
        _add_matrix_spring(
            stiffness,
            node_a,
            node_b,
            np.array([in_plane_stiffness / length, in_plane_stiffness / length, out_of_plane_stiffness / length]),
        )
    for node_a, node_b in sorted(diagonal_pairs):
        length = max(float(np.linalg.norm(nodes[node_b] - nodes[node_a])), 1.0e-6)
        _add_matrix_spring(
            stiffness,
            node_a,
            node_b,
            np.array(
                [
                    0.25 * in_plane_stiffness / length,
                    0.25 * in_plane_stiffness / length,
                    0.45 * out_of_plane_stiffness / length,
                ]
            ),
        )

    damping = 0.0015 * mass + 0.00015 * stiffness
    return mass, stiffness, damping


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 3
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.step_count = 0

        self.viewer = viewer
        self.args = args
        self.show_elastic_strain = True

        self.length = 0.95
        self.width = 0.36
        self.thickness = 0.028
        self.mode_count = 6
        self.max_joint_residual = 0.0
        self.max_mode_abs = 0.0
        self.mode_min = np.full(self.mode_count, np.inf, dtype=np.float32)
        self.mode_max = np.full(self.mode_count, -np.inf, dtype=np.float32)
        self.max_drive_target_abs = 0.0
        self.max_gripper_motion = 0.0
        self.max_tip_deflection = 0.0

        nodes, surface_vertices, surface_indices, surface_node_indices, cells, left_node_ids, right_node_ids = (
            _make_perforated_plate(self.length, self.width, self.thickness, nx=18, ny=8)
        )
        mass_matrix, stiffness_matrix, damping_matrix = _assemble_plate_matrices(
            nodes,
            cells,
            density=900.0,
            thickness=self.thickness,
            in_plane_stiffness=2400.0,
            out_of_plane_stiffness=36.0,
        )
        sample_points = np.vstack((surface_vertices, nodes)).astype(np.float32)
        sample_node_indices = np.concatenate((surface_node_indices, np.arange(nodes.shape[0], dtype=np.int32)))
        left_nodes = np.where(nodes[:, 0] <= float(np.min(nodes[:, 0])) + 1.0e-6)[0]
        matrix_generator = newton.ModalGeneratorFEM(
            node_positions=nodes,
            mass_matrix=mass_matrix,
            stiffness_matrix=stiffness_matrix,
            damping_matrix=damping_matrix,
            sample_points=sample_points,
            sample_node_indices=sample_node_indices,
            fixed_node_indices=left_nodes,
            mode_count=self.mode_count,
            label="perforated_bracket_matrix_rom",
        )
        matrix_basis = matrix_generator.build()
        self.modal_frequencies_hz = matrix_generator.frequencies

        self.left_local = nodes[int(left_node_ids[0])]
        self.right_local = nodes[int(right_node_ids[0])]
        self.initial_tip_local = np.array(self.right_local, dtype=float)
        self.panel_origin = np.array([0.0, 0.0, 0.78], dtype=float)
        self.gripper_origin = self.panel_origin + self.left_local
        self.payload_origin = self.panel_origin + self.right_local + np.array([0.0, 0.0, -0.13], dtype=float)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        inertia = wp.mat33(0.025, 0.0, 0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 0.025)
        self.gripper = builder.add_body(
            xform=wp.transform(wp.vec3(*self.gripper_origin), wp.quat_identity()),
            mass=4.0,
            inertia=inertia,
            is_kinematic=True,
            label="driven_gripper_frame",
        )
        self.j_drive = len(builder.joint_type) - 1
        self.panel = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(*self.panel_origin), wp.quat_identity()),
            mass=2.0,
            inertia=inertia,
            modal_basis=matrix_basis,
            label="matrix_rom_perforated_bracket",
        )
        self.payload = builder.add_body(
            xform=wp.transform(wp.vec3(*self.payload_origin), wp.quat_identity()),
            mass=1.8,
            inertia=wp.mat33(0.018, 0.0, 0.0, 0.0, 0.018, 0.0, 0.0, 0.0, 0.018),
            label="rigid_payload",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False
        builder.add_shape_box(self.gripper, hx=0.055, hy=0.13, hz=0.07, cfg=shape_cfg)
        builder.add_shape_box(
            self.gripper,
            xform=wp.transform(wp.vec3(-0.025, 0.0, -0.20), wp.quat_identity()),
            hx=0.045,
            hy=0.16,
            hz=0.20,
            cfg=shape_cfg,
            label="moving_gripper_clamp",
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(float(self.gripper_origin[0]), 0.0, 0.35), wp.quat_identity()),
            hx=0.035,
            hy=0.18,
            hz=0.35,
            cfg=shape_cfg,
        )
        builder.add_shape_mesh(
            self.panel,
            mesh=newton.Mesh(surface_vertices, surface_indices, compute_inertia=False),
            cfg=shape_cfg,
            label="matrix_rom_bracket_mesh",
        )
        builder.add_shape_box(self.payload, hx=0.08, hy=0.08, hz=0.055, cfg=shape_cfg)

        self.j_left = builder.add_joint_fixed(
            parent=self.gripper,
            child=self.panel,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(*self.left_local), wp.quat_identity()),
            label="gripper_to_matrix_rom_interface",
        )
        self.j_payload = builder.add_joint_fixed(
            parent=self.panel,
            child=self.payload,
            parent_xform=wp.transform(wp.vec3(*self.right_local), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.13), wp.quat_identity()),
            label="matrix_rom_interface_to_payload",
        )
        builder.color()

        self.model = builder.finalize()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        self.elastic_joint = int(self.model.elastic_joint.numpy()[0])
        self.elastic_q_start = int(self.model.joint_q_start.numpy()[self.elastic_joint])
        self.drive_q_start = int(self.model.joint_q_start.numpy()[self.j_drive])
        self.drive_qd_start = int(self.model.joint_qd_start.numpy()[self.j_drive])
        self._joint_f = self.control.joint_f.numpy()
        self.initial_tip_world = joint_endpoint_world(self.model, self.state_0, self.j_payload, "parent")
        self.initial_gripper_z = float(self.state_0.body_q.numpy()[self.gripper, 2])

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=28,
            rigid_joint_linear_k_start=1.0e5,
            rigid_joint_angular_k_start=1.0e4,
            rigid_joint_linear_ke=2.5e6,
            rigid_joint_angular_ke=7.5e5,
        )

        self.viewer.set_model(self.model)
        self.viewer.show_elastic_strain = True
        self.viewer.elastic_strain_color_max = 0.06
        self.viewer.set_camera(wp.vec3(0.18, -1.72, 0.92), -18.0, 80.0)

    def _mode_values(self) -> np.ndarray:
        return self.state_0.joint_q.numpy()[self.elastic_q_start + 7 : self.elastic_q_start + 7 + self.mode_count]

    @staticmethod
    def _smooth_trajectory(a: float, b: float, t0: float, t1: float, t: float) -> tuple[float, float]:
        if t <= t0:
            return a, 0.0
        if t >= t1:
            return b, 0.0
        s = (t - t0) / (t1 - t0)
        h = s * s * (3.0 - 2.0 * s)
        dh_dt = 6.0 * s * (1.0 - s) / (t1 - t0)
        delta = b - a
        return a + delta * h, delta * dh_dt

    def _drive_target(self, t: float) -> tuple[float, float]:
        keys = (
            (0.00, 0.00),
            (0.08, 0.22),
            (1.30, 0.22),
            (1.38, -0.18),
            (2.60, -0.18),
            (2.68, 0.14),
            (3.90, 0.14),
            (3.98, -0.08),
            (5.00, -0.08),
        )
        for (t0, a), (t1, b) in pairwise(keys):
            if t <= t1:
                return self._smooth_trajectory(a, b, t0, t1, t)
        return keys[-1][1], 0.0

    def _set_controls(self):
        target, target_vel = self._drive_target(self.sim_time)
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        joint_q[self.drive_q_start : self.drive_q_start + 3] = self.gripper_origin + np.array(
            [0.0, 0.0, target], dtype=np.float32
        )
        joint_q[self.drive_q_start + 3 : self.drive_q_start + 7] = [0.0, 0.0, 0.0, 1.0]
        joint_qd[self.drive_qd_start : self.drive_qd_start + 3] = [0.0, 0.0, target_vel]
        joint_qd[self.drive_qd_start + 3 : self.drive_qd_start + 6] = [0.0, 0.0, 0.0]
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )
        self.max_drive_target_abs = max(self.max_drive_target_abs, abs(target))
        self._joint_f[:] = 0.0
        self.control.joint_f.assign(self._joint_f)

    def _update_metrics(self):
        residuals = [
            float(
                np.linalg.norm(
                    joint_endpoint_world(self.model, self.state_0, joint, "parent")
                    - joint_endpoint_world(self.model, self.state_0, joint, "child")
                )
            )
            for joint in (self.j_left, self.j_payload)
        ]
        self.max_joint_residual = max(self.max_joint_residual, *residuals)
        modes = self._mode_values()
        self.max_mode_abs = max(self.max_mode_abs, float(np.max(np.abs(modes))))
        self.mode_min = np.minimum(self.mode_min, modes)
        self.mode_max = np.maximum(self.mode_max, modes)
        self.max_gripper_motion = max(
            self.max_gripper_motion, abs(float(self.state_0.body_q.numpy()[self.gripper, 2]) - self.initial_gripper_z)
        )
        tip_world = joint_endpoint_world(self.model, self.state_0, self.j_payload, "parent")
        self.max_tip_deflection = max(self.max_tip_deflection, abs(float(tip_world[2] - self.initial_tip_world[2])))

    def simulate(self):
        for _substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self._set_controls()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        self.simulate()
        self.step_count += 1
        self._update_metrics()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise AssertionError("body transforms contain non-finite values")
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("joint coordinates contain non-finite values")
        if self.model.elastic_mode_count.numpy()[0] != self.mode_count:
            raise AssertionError("matrix ROM example has the wrong number of retained modes")
        if np.any(self.modal_frequencies_hz <= 0.0):
            raise AssertionError(f"matrix ROM frequencies must be positive, got {self.modal_frequencies_hz}")
        if self.max_joint_residual > 2.5e-2:
            raise AssertionError(f"matrix ROM fixed-joint residual too large: {self.max_joint_residual}")
        if self.max_mode_abs < 8.0e-3:
            raise AssertionError(f"matrix ROM modes were not visibly excited: {self.max_mode_abs}")
        if self.max_mode_abs > 0.75:
            raise AssertionError(f"matrix ROM modal response is out of range: {self._mode_values()}")
        higher_mode_range = float(self.mode_max[2] - self.mode_min[2])
        if higher_mode_range < 1.0e-2:
            raise AssertionError(f"matrix ROM higher bending mode was not dynamically excited: {higher_mode_range}")
        if self.max_drive_target_abs < 0.11:
            raise AssertionError(f"matrix ROM base trajectory did not move enough: {self.max_drive_target_abs}")
        if self.max_gripper_motion < 0.10:
            raise AssertionError(f"matrix ROM gripper did not follow the base trajectory: {self.max_gripper_motion}")
        if self.max_tip_deflection < 0.025:
            raise AssertionError(f"matrix ROM tip did not move enough: {self.max_tip_deflection}")
        deformed = elastic_shape_deformed_vertices(self.model, self.state_0)
        if not np.isfinite(deformed).all():
            raise AssertionError("deformed matrix ROM vertices contain non-finite values")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
