# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.types import Axis
from ...geometry import GeoType, Mesh, ParticleFlags, ShapeFlags
from ...sim import BodyFlags, Contacts, Control, JointType, Model, State
from ..flags import SolverNotifyFlags
from ..solver import SolverBase


@dataclass
class _BodyMap:
    body_index: int
    ipc_index: int


class SolverIPC(SolverBase):
    """Incremental Potential Contact solver backed by libuipc.

    The Python package is distributed on PyPI as ``pyuipc`` and imported as
    ``uipc``. This solver currently supports a conservative subset of Newton's
    data model: triangle cloth, fixed particles, common rigid collision shapes,
    and free rigid bodies represented as libuipc affine bodies. Model positions,
    translations, linear velocities, gravity, and inertias are multiplied by
    ``length_scale`` before being passed to libuipc, then converted back to
    Newton units on output.
    """

    def __init__(
        self,
        model: Model,
        *,
        backend: str = "cuda",
        workspace: str | None = None,
        contact_d_hat: float = 0.01,
        contact_friction: float = 0.5,
        contact_resistance: float = 1.0e9,
        max_newton_iter: int = 32,
        min_newton_iter: int = 1,
        cloth_youngs: float | None = None,
        cloth_poisson: float = 0.4,
        cloth_density: float = 200.0,
        cloth_thickness: float = 0.001,
        cloth_bending_stiffness: float | None = None,
        rigid_kappa: float = 1.0e8,
        scene_config: dict[str, Any] | None = None,
        sync_state_in: bool = True,
        length_scale: float = 1.0,
    ):
        super().__init__(model)
        if length_scale <= 0.0:
            raise ValueError("length_scale must be positive.")
        self.uipc = self.import_uipc()
        self.backend = backend
        self.workspace = workspace or tempfile.mkdtemp(prefix="newton-ipc-")
        self.contact_d_hat = contact_d_hat
        self.contact_friction = contact_friction
        self.contact_resistance = contact_resistance
        self.max_newton_iter = max_newton_iter
        self.min_newton_iter = min_newton_iter
        self.cloth_youngs = cloth_youngs
        self.cloth_poisson = cloth_poisson
        self.cloth_density = cloth_density
        self.cloth_thickness = cloth_thickness
        self.cloth_bending_stiffness = cloth_bending_stiffness
        self.rigid_kappa = rigid_kappa
        self.scene_config = copy.deepcopy(scene_config) if scene_config is not None else None
        self.sync_state_in = sync_state_in
        self.length_scale = float(length_scale)

        self._needs_rebuild = True
        self._last_dt: float | None = None
        self._particle_indices = np.zeros(0, dtype=np.int32)
        self._body_maps: list[_BodyMap] = []
        self._unsupported_shapes: list[tuple[int, int]] = []

        self._engine = None
        self._world = None
        self._scene = None
        self._fem_accessor = None
        self._fem_state_geo = None
        self._fem_position_view = None
        self._fem_velocity_view = None
        self._abd_accessor = None
        self._abd_state_geo = None
        self._abd_transform_view = None
        self._abd_velocity_view = None

        self._validate_model()

    @staticmethod
    def import_uipc():
        try:
            import uipc  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "SolverIPC requires libuipc's Python package. Install Newton with the optional "
                "`ipc` extra, for example `pip install newton[ipc]` or `uv run --extra ipc ...`. "
                "The PyPI package is named `pyuipc` and currently provides CUDA-oriented wheels."
            ) from exc

        if hasattr(uipc, "Logger"):
            uipc.Logger.set_level(uipc.Logger.Level.Error)
        return uipc

    def _validate_model(self):
        model = self.model
        if not self.device.is_cuda:
            raise RuntimeError("SolverIPC currently requires a CUDA Warp device.")
        if model.world_count != 1:
            raise NotImplementedError("SolverIPC currently supports exactly one Newton world.")
        if model.tet_count:
            raise NotImplementedError("SolverIPC does not yet map Newton tetrahedral soft bodies.")
        if model.joint_count and model.joint_type is not None:
            joint_types = model.joint_type.numpy()
            if np.any(joint_types != int(JointType.FREE)):
                raise NotImplementedError(
                    "SolverIPC currently supports free rigid bodies but not constrained Newton joints/articulations."
                )
        if model.particle_count and not model.tri_count:
            raise NotImplementedError("SolverIPC currently supports cloth particles through Newton triangle elements.")

    def notify_model_changed(self, flags: int):
        rebuild_flags = (
            SolverNotifyFlags.MODEL_PROPERTIES
            | SolverNotifyFlags.BODY_PROPERTIES
            | SolverNotifyFlags.BODY_INERTIAL_PROPERTIES
            | SolverNotifyFlags.SHAPE_PROPERTIES
        )
        if flags & rebuild_flags:
            self._needs_rebuild = True

    def update_contacts(self, contacts: Contacts, state: State):
        raise NotImplementedError("SolverIPC uses libuipc contact generation and does not consume Newton Contacts.")

    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts | None, dt: float):
        if self._needs_rebuild or self._last_dt != float(dt):
            self._build_world(dt)

        if self.sync_state_in:
            self._copy_newton_to_ipc(state_in)

        self._world.advance()
        self._world.retrieve()

        self._copy_ipc_to_newton(state_in, state_out)

    def _scene_config_for_dt(self, dt: float):
        config = (
            self.uipc.core.Scene.default_config() if self.scene_config is None else copy.deepcopy(self.scene_config)
        )
        config["dt"] = float(dt)
        gravity = self.model.gravity.numpy()[0].astype(np.float64) * self.length_scale
        config["gravity"] = [[float(gravity[0])], [float(gravity[1])], [float(gravity[2])]]
        config["contact"]["d_hat"] = float(self.contact_d_hat)
        config["newton"]["max_iter"] = int(self.max_newton_iter)
        config["newton"]["min_iter"] = int(self.min_newton_iter)
        return config

    def _build_world(self, dt: float):
        uipc = self.uipc
        config = self._scene_config_for_dt(dt)

        os.makedirs(self.workspace, exist_ok=True)
        self._engine = uipc.core.Engine(self.backend, self.workspace)
        self._world = uipc.core.World(self._engine)
        self._scene = uipc.core.Scene(config)
        self._scene.contact_tabular().default_model(self.contact_friction, self.contact_resistance)
        default_contact = self._scene.contact_tabular().default_element()

        self._particle_indices = np.zeros(0, dtype=np.int32)
        self._body_maps = []
        self._unsupported_shapes = []

        self._add_cloth_geometry(default_contact)
        self._add_static_shape_geometry(default_contact)
        self._add_body_geometry(default_contact)

        self._world.init(self._scene)
        if not self._world.is_valid():
            raise RuntimeError(
                "libuipc rejected the SolverIPC scene during initialization. "
                "Check initial intersections, shell thickness, contact gap, and unit scaling."
            )
        self._init_state_accessors()
        self._needs_rebuild = False
        self._last_dt = float(dt)

    def _add_cloth_geometry(self, default_contact):
        model = self.model
        if model.tri_count == 0:
            return

        uipc = self.uipc
        tri_indices = model.tri_indices.numpy().reshape(-1, 3).astype(np.int32)
        particle_indices = np.unique(tri_indices.reshape(-1)).astype(np.int32)
        index_remap = np.full(model.particle_count, -1, dtype=np.int32)
        index_remap[particle_indices] = np.arange(particle_indices.size, dtype=np.int32)

        vertices = model.particle_q.numpy()[particle_indices].astype(np.float64) * self.length_scale
        triangles = index_remap[tri_indices].astype(np.int32)
        cloth_mesh = uipc.geometry.trimesh(vertices, triangles)
        uipc.geometry.label_surface(cloth_mesh)

        tri_materials = model.tri_materials.numpy() if model.tri_materials is not None else np.zeros((0, 5))
        if self.cloth_youngs is None and tri_materials.size:
            youngs = max(float(np.mean(tri_materials[:, 0])), 1.0)
        else:
            youngs = float(self.cloth_youngs or 1.0e4)
        moduli = uipc.constitution.ElasticModuli2D.youngs_poisson(youngs, self.cloth_poisson)
        uipc.constitution.NeoHookeanShell().apply_to(
            cloth_mesh,
            moduli,
            mass_density=self.cloth_density,
            thickness=self.cloth_thickness,
        )

        if model.edge_count:
            if self.cloth_bending_stiffness is None and model.edge_bending_properties is not None:
                bending = float(np.mean(model.edge_bending_properties.numpy()[:, 0]))
            else:
                bending = float(self.cloth_bending_stiffness if self.cloth_bending_stiffness is not None else 0.01)
            uipc.constitution.DiscreteShellBending().apply_to(cloth_mesh, bending_stiffness=max(bending, 0.0))

        if model.particle_flags is not None:
            fixed = ((model.particle_flags.numpy()[particle_indices] & int(ParticleFlags.ACTIVE)) == 0).astype(np.int32)
        else:
            fixed = np.zeros(particle_indices.shape[0], dtype=np.int32)
        if model.particle_inv_mass is not None:
            fixed = np.maximum(fixed, (model.particle_inv_mass.numpy()[particle_indices] == 0.0).astype(np.int32))
        is_fixed = cloth_mesh.vertices().find(uipc.builtin.is_fixed)
        if is_fixed is not None:
            uipc.view(is_fixed)[:] = fixed

        default_contact.apply_to(cloth_mesh)
        obj = self._scene.objects().create("newton_cloth")
        obj.geometries().create(cloth_mesh)
        self._particle_indices = particle_indices

    def _add_static_shape_geometry(self, default_contact):
        for shape_index in self.model.body_shapes.get(-1, []):
            shape_type = int(self.model.shape_type.numpy()[shape_index])
            if not self._shape_has_collision(shape_index):
                continue
            if shape_type == int(GeoType.PLANE):
                self._unsupported_shapes.append((shape_index, shape_type))
                continue

            mesh = self._shape_to_mesh(shape_index)
            if mesh is None:
                self._unsupported_shapes.append((shape_index, shape_type))
                continue

            mesh = self._world_space_static_mesh(shape_index, mesh)
            self._add_fixed_abd_mesh(f"static_shape_{shape_index}", mesh, default_contact)

    def _add_body_geometry(self, default_contact):
        model = self.model
        if model.body_count == 0:
            return

        for body_index in range(model.body_count):
            shape_indices = [s for s in model.body_shapes.get(body_index, []) if self._shape_has_collision(s)]
            mesh = self._body_mesh(shape_indices)

            if mesh is None:
                mass = float(model.body_mass.numpy()[body_index])
                inertia = model.body_inertia.numpy()[body_index].astype(np.float64) * (self.length_scale**2)
                com = model.body_com.numpy()[body_index].astype(np.float64) * self.length_scale
                mesh = self.uipc.constitution.AffineBodyConstitution().create_proxy(
                    self.rigid_kappa,
                    max(mass, 1.0e-6),
                    com,
                    inertia,
                    1.0,
                )
            else:
                vertices, triangles = mesh
                mesh = self.uipc.geometry.trimesh(vertices.astype(np.float64), triangles.astype(np.int32))
                self.uipc.geometry.label_surface(mesh)
                mass = float(model.body_mass.numpy()[body_index])
                inertia = model.body_inertia.numpy()[body_index].astype(np.float64) * (self.length_scale**2)
                com = model.body_com.numpy()[body_index].astype(np.float64) * self.length_scale
                abd_mass = self.uipc.geometry.affine_body.from_rigid_body(max(mass, 1.0e-6), com, inertia)
                self.uipc.constitution.AffineBodyConstitution().apply_to(mesh, self.rigid_kappa, abd_mass, 1.0)

            if self._body_is_fixed(body_index):
                fixed = mesh.instances().find(self.uipc.builtin.is_fixed)
                if fixed is not None:
                    self.uipc.view(fixed)[:] = 1

            default_contact.apply_to(mesh)
            obj = self._scene.objects().create(f"body_{body_index}")
            obj.geometries().create(mesh)
            self._body_maps.append(_BodyMap(body_index=body_index, ipc_index=len(self._body_maps)))

    def _add_static_plane(self, shape_index: int, default_contact):
        uipc = self.uipc
        transform = self.model.shape_transform.numpy()[shape_index]
        point = transform[:3].astype(np.float64) * self.length_scale
        normal = -_quat_rotate(transform[3:7], np.array([0.0, 0.0, 1.0], dtype=np.float64))
        plane = uipc.geometry.halfplane(point, normal)
        default_contact.apply_to(plane)
        obj = self._scene.objects().create(f"static_plane_{shape_index}")
        obj.geometries().create(plane)

    def _add_fixed_abd_mesh(self, name: str, mesh: tuple[np.ndarray, np.ndarray], default_contact):
        uipc = self.uipc
        vertices, triangles = mesh
        sc = uipc.geometry.trimesh(vertices.astype(np.float64), triangles.astype(np.int32))
        uipc.geometry.label_surface(sc)
        uipc.constitution.AffineBodyConstitution().apply_to(sc, self.rigid_kappa, mass_density=1000.0)
        fixed = sc.instances().find(uipc.builtin.is_fixed)
        if fixed is not None:
            uipc.view(fixed)[:] = 1
        default_contact.apply_to(sc)
        obj = self._scene.objects().create(name)
        obj.geometries().create(sc)

    def _shape_has_collision(self, shape_index: int) -> bool:
        flags = int(self.model.shape_flags.numpy()[shape_index])
        return bool(flags & (int(ShapeFlags.COLLIDE_PARTICLES) | int(ShapeFlags.COLLIDE_SHAPES)))

    def _body_is_fixed(self, body_index: int) -> bool:
        if self.model.body_inv_mass is not None and float(self.model.body_inv_mass.numpy()[body_index]) == 0.0:
            return True
        if self.model.body_flags is not None:
            return bool(int(self.model.body_flags.numpy()[body_index]) & int(BodyFlags.KINEMATIC))
        return False

    def _shape_to_mesh(self, shape_index: int) -> tuple[np.ndarray, np.ndarray] | None:
        shape_type = GeoType(int(self.model.shape_type.numpy()[shape_index]))
        scale = self.model.shape_scale.numpy()[shape_index].astype(np.float64)
        source = self.model.shape_source[shape_index] if self.model.shape_source else None

        if shape_type == GeoType.BOX:
            return _box_mesh(scale[0], scale[1], scale[2])
        elif shape_type == GeoType.SPHERE:
            mesh = Mesh.create_sphere(scale[0], num_latitudes=16, num_longitudes=16, compute_inertia=False)
        elif shape_type == GeoType.ELLIPSOID:
            mesh = Mesh.create_ellipsoid(
                scale[0], scale[1], scale[2], num_latitudes=16, num_longitudes=16, compute_inertia=False
            )
        elif shape_type == GeoType.CAPSULE:
            mesh = Mesh.create_capsule(scale[0], scale[1], up_axis=Axis.Z, segments=16, compute_inertia=False)
        elif shape_type == GeoType.CYLINDER:
            mesh = Mesh.create_cylinder(scale[0], scale[1], up_axis=Axis.Z, segments=16, compute_inertia=False)
        elif shape_type == GeoType.CONE:
            mesh = Mesh.create_cone(scale[0], scale[1], up_axis=Axis.Z, segments=16, compute_inertia=False)
        elif shape_type in (GeoType.MESH, GeoType.CONVEX_MESH) and source is not None:
            vertices = np.asarray(source.vertices, dtype=np.float64) * scale
            return vertices, np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
        else:
            return None

        return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)

    def _world_space_static_mesh(self, shape_index: int, mesh: tuple[np.ndarray, np.ndarray]):
        transform = self.model.shape_transform.numpy()[shape_index]
        vertices, triangles = mesh
        return _transform_points(vertices, transform) * self.length_scale, triangles

    def _body_mesh(self, shape_indices: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
        if not shape_indices:
            return None

        vertices_parts = []
        triangle_parts = []
        vertex_offset = 0
        shape_transforms = self.model.shape_transform.numpy()
        for shape_index in shape_indices:
            mesh = self._shape_to_mesh(shape_index)
            if mesh is None:
                self._unsupported_shapes.append((shape_index, int(self.model.shape_type.numpy()[shape_index])))
                continue

            vertices, triangles = mesh
            vertices = _transform_points(vertices, shape_transforms[shape_index]) * self.length_scale
            vertices_parts.append(vertices)
            triangle_parts.append(triangles + vertex_offset)
            vertex_offset += vertices.shape[0]

        if not vertices_parts:
            return None
        return np.vstack(vertices_parts), np.vstack(triangle_parts).astype(np.int32)

    def _init_state_accessors(self):
        uipc = self.uipc
        self._fem_accessor = self._world.features().find(uipc.core.FiniteElementStateAccessorFeature)
        if self._fem_accessor is not None and self._particle_indices.size:
            self._fem_state_geo = self._fem_accessor.create_geometry()
            pos_attr = self._fem_state_geo.vertices().create(uipc.builtin.position, uipc.Vector3.Zero())
            vel_attr = self._fem_state_geo.vertices().create(uipc.builtin.velocity, uipc.Vector3.Zero())
            self._fem_position_view = uipc.view(pos_attr)
            self._fem_velocity_view = uipc.view(vel_attr)

        self._abd_accessor = self._world.features().find(uipc.core.AffineBodyStateAccessorFeature)
        if self._abd_accessor is not None and self._body_maps:
            self._abd_state_geo = self._abd_accessor.create_geometry()
            transform_attr = self._abd_state_geo.instances().create(uipc.builtin.transform, uipc.Matrix4x4.Zero())
            velocity_attr = self._abd_state_geo.instances().create(uipc.builtin.velocity, uipc.Matrix4x4.Zero())
            self._abd_transform_view = uipc.view(transform_attr)
            self._abd_velocity_view = uipc.view(velocity_attr)

    def _copy_newton_to_ipc(self, state_in: State):
        if self._fem_accessor is not None and self._particle_indices.size:
            self._fem_accessor.copy_to(self._fem_state_geo)
            self._fem_position_view[:] = (
                state_in.particle_q.numpy()[self._particle_indices].astype(np.float64) * self.length_scale
            )[:, :, None]
            self._fem_velocity_view[:] = (
                state_in.particle_qd.numpy()[self._particle_indices].astype(np.float64) * self.length_scale
            )[:, :, None]
            self._fem_accessor.copy_from(self._fem_state_geo)

        if self._abd_accessor is not None and self._body_maps:
            self._abd_accessor.copy_to(self._abd_state_geo)
            body_q = state_in.body_q.numpy()
            body_qd = state_in.body_qd.numpy()
            for body_map in self._body_maps:
                self._abd_transform_view[body_map.ipc_index] = _transform_to_matrix(
                    body_q[body_map.body_index], self.length_scale
                )
                self._abd_velocity_view[body_map.ipc_index] = _spatial_velocity_to_matrix(
                    body_q[body_map.body_index], body_qd[body_map.body_index], self.length_scale
                )
            self._abd_accessor.copy_from(self._abd_state_geo)

    def _copy_ipc_to_newton(self, state_in: State, state_out: State):
        if self.model.particle_count:
            q = state_in.particle_q.numpy().copy()
            qd = state_in.particle_qd.numpy().copy()
            if self._fem_accessor is not None and self._particle_indices.size:
                self._fem_accessor.copy_to(self._fem_state_geo)
                q[self._particle_indices] = (
                    np.asarray(self._fem_position_view, dtype=np.float32).reshape(-1, 3) / self.length_scale
                )
                qd[self._particle_indices] = (
                    np.asarray(self._fem_velocity_view, dtype=np.float32).reshape(-1, 3) / self.length_scale
                )
            state_out.particle_q.assign(q)
            state_out.particle_qd.assign(qd)

        if self.model.body_count:
            body_q = state_in.body_q.numpy().copy()
            body_qd = state_in.body_qd.numpy().copy()
            if self._abd_accessor is not None and self._body_maps:
                self._abd_accessor.copy_to(self._abd_state_geo)
                for body_map in self._body_maps:
                    if self._body_is_fixed(body_map.body_index):
                        continue
                    transform = np.asarray(self._abd_transform_view[body_map.ipc_index], dtype=np.float64)
                    velocity = np.asarray(self._abd_velocity_view[body_map.ipc_index], dtype=np.float64)
                    body_q[body_map.body_index] = _matrix_to_transform(transform, self.length_scale)
                    body_qd[body_map.body_index] = _matrix_to_spatial_velocity(transform, velocity, self.length_scale)
            state_out.body_q.assign(body_q)
            state_out.body_qd.assign(body_qd)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return _quat_rotate_many(transform[3:7], points) + transform[:3]


def _box_mesh(hx: float, hy: float, hz: float) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 6, 2],
            [3, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return vertices, triangles


def _quat_rotate_many(quat: np.ndarray, points: np.ndarray) -> np.ndarray:
    rotation = _quat_to_matrix(quat)
    return points @ rotation.T


def _quat_rotate(quat: np.ndarray, point: np.ndarray) -> np.ndarray:
    return _quat_to_matrix(quat) @ point


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    m = matrix
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _transform_to_matrix(transform: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_to_matrix(transform[3:7])
    matrix[:3, 3] = transform[:3] * length_scale
    return matrix


def _matrix_to_transform(matrix: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    transform = np.zeros(7, dtype=np.float32)
    transform[:3] = (matrix[:3, 3] / length_scale).astype(np.float32)
    transform[3:7] = _matrix_to_quat(matrix[:3, :3]).astype(np.float32)
    return transform


def _spatial_velocity_to_matrix(
    transform: np.ndarray, spatial_velocity: np.ndarray, length_scale: float = 1.0
) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.float64)
    linear = spatial_velocity[:3].astype(np.float64) * length_scale
    angular = spatial_velocity[3:].astype(np.float64)
    rotation = _quat_to_matrix(transform[3:7])
    matrix[:3, :3] = _skew(angular) @ rotation
    matrix[:3, 3] = linear
    return matrix


def _matrix_to_spatial_velocity(transform: np.ndarray, velocity: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    rotation = transform[:3, :3]
    rotation_dot = velocity[:3, :3]
    angular_skew = rotation_dot @ rotation.T
    spatial = np.zeros(6, dtype=np.float32)
    spatial[:3] = (velocity[:3, 3] / length_scale).astype(np.float32)
    spatial[3:] = np.array(
        [angular_skew[2, 1], angular_skew[0, 2], angular_skew[1, 0]],
        dtype=np.float32,
    )
    return spatial


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
