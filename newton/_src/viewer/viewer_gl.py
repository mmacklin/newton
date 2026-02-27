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

from __future__ import annotations

import ctypes
import time

import numpy as np
import warp as wp

import newton as nt

from ..core.types import override
from ..utils.render import copy_rgb_frame_uint8
from .camera import Camera
from .gl.opengl import LinesGL, MeshGL, MeshInstancerGL, RendererGL
from .viewer import ViewerBase
from .viewer_gui import ViewerGui
from .wind import Wind


@wp.kernel
def _capsule_duplicate_vec3(in_values: wp.array(dtype=wp.vec3), out_values: wp.array(dtype=wp.vec3)):
    # Duplicate N values into 2N values (two caps per capsule).
    tid = wp.tid()
    out_values[tid] = in_values[tid // 2]


@wp.kernel
def _capsule_duplicate_vec4(in_values: wp.array(dtype=wp.vec4), out_values: wp.array(dtype=wp.vec4)):
    # Duplicate N values into 2N values (two caps per capsule).
    tid = wp.tid()
    out_values[tid] = in_values[tid // 2]


@wp.kernel
def _capsule_build_body_scales(
    shape_scale: wp.array(dtype=wp.vec3),
    shape_indices: wp.array(dtype=wp.int32),
    out_scales: wp.array(dtype=wp.vec3),
):
    # model.shape_scale stores capsule params as (radius, half_height, _unused).
    # ViewerGL instances scale meshes with a full (x, y, z) vector, so we expand to
    # (radius, radius, half_height) for the cylinder body.
    tid = wp.tid()
    s = shape_indices[tid]
    scale = shape_scale[s]
    r = scale[0]
    half_height = scale[1]
    out_scales[tid] = wp.vec3(r, r, half_height)


@wp.kernel
def _capsule_build_cap_xforms_and_scales(
    capsule_xforms: wp.array(dtype=wp.transform),
    capsule_scales: wp.array(dtype=wp.vec3),
    out_xforms: wp.array(dtype=wp.transform),
    out_scales: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = tid // 2
    # Each capsule has two caps; even tid is the +Z end, odd tid is the -Z end.
    is_plus_end = (tid % 2) == 0

    t = capsule_xforms[i]
    p = wp.transform_get_translation(t)
    q = wp.transform_get_rotation(t)

    r = capsule_scales[i][0]
    half_height = capsule_scales[i][2]
    offset_local = wp.vec3(0.0, 0.0, half_height if is_plus_end else -half_height)
    p2 = p + wp.quat_rotate(q, offset_local)

    out_xforms[tid] = wp.transform(p2, q)
    out_scales[tid] = wp.vec3(r, r, r)


@wp.kernel
def _compute_shape_vbo_xforms(
    shape_transform: wp.array(dtype=wp.transformf),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transformf),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_type: wp.array(dtype=int),
    shape_world: wp.array(dtype=int),
    world_offsets: wp.array(dtype=wp.vec3),
    write_indices: wp.array(dtype=int),
    out_world_xforms: wp.array(dtype=wp.transformf),
    out_vbo_xforms: wp.array(dtype=wp.mat44),
):
    """Process all model shapes, write mat44 to grouped output positions."""
    tid = wp.tid()
    out_idx = write_indices[tid]
    if out_idx < 0:
        return

    local_xform = shape_transform[tid]
    parent = shape_body[tid]

    if parent >= 0:
        xform = wp.transform_multiply(body_q[parent], local_xform)
    else:
        xform = local_xform

    if world_offsets:
        wi = shape_world[tid]
        if wi >= 0 and wi < world_offsets.shape[0]:
            p = wp.transform_get_translation(xform)
            xform = wp.transform(p + world_offsets[wi], wp.transform_get_rotation(xform))

    out_world_xforms[out_idx] = xform

    p = wp.transform_get_translation(xform)
    q = wp.transform_get_rotation(xform)
    R = wp.quat_to_matrix(q)

    # Only mesh/convex_mesh shapes use model scale; other primitives have
    # their dimensions baked into the geometry mesh, so scale is (1,1,1).
    geo = shape_type[tid]
    if geo == nt.GeoType.MESH or geo == nt.GeoType.CONVEX_MESH:
        s = shape_scale[tid]
    else:
        s = wp.vec3(1.0, 1.0, 1.0)

    out_vbo_xforms[out_idx] = wp.mat44(
        R[0, 0] * s[0],
        R[1, 0] * s[0],
        R[2, 0] * s[0],
        0.0,
        R[0, 1] * s[1],
        R[1, 1] * s[1],
        R[2, 1] * s[1],
        0.0,
        R[0, 2] * s[2],
        R[1, 2] * s[2],
        R[2, 2] * s[2],
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    )


class ViewerGL(ViewerBase):
    """
    OpenGL-based interactive viewer for Newton physics models.

    This class provides a graphical interface for visualizing and interacting with
    Newton models using OpenGL rendering. It supports real-time simulation control,
    camera navigation, object picking, wind effects, and a rich ImGui-based UI for
    model introspection and visualization options.

    Key Features:
        - Real-time 3D rendering of Newton models and simulation states.
        - Camera navigation with WASD/QE and mouse controls.
        - Object picking and manipulation via mouse.
        - Visualization toggles for joints, contacts, particles, springs, etc.
        - Wind force controls and visualization.
        - Performance statistics overlay (FPS, object counts, etc.).
        - Selection panel for introspecting and filtering model attributes.
        - Extensible logging of meshes, lines, points, and arrays for custom visualization.
    """

    def __init__(self, width=1920, height=1080, vsync=False, headless=False, paused=False):
        """
        Initialize the OpenGL viewer and UI.

        Args:
            width (int): Window width in pixels.
            height (int): Window height in pixels.
            vsync (bool): Enable vertical sync.
            headless (bool): Run in headless mode (no window).
            paused (bool): Start the viewer in a paused state.
        """
        super().__init__(paused=paused)

        # map from path to any object type
        self.objects = {}
        self.lines = {}
        self.renderer = RendererGL(vsync=vsync, screen_width=width, screen_height=height, headless=headless)
        self.renderer.set_title("Newton Viewer")

        self._packed_vbo_xforms = None

        self.renderer.register_key_press(self.on_key_press)
        self.renderer.register_key_release(self.on_key_release)
        self.renderer.register_mouse_press(self.on_mouse_press)
        self.renderer.register_mouse_release(self.on_mouse_release)
        self.renderer.register_mouse_drag(self.on_mouse_drag)
        self.renderer.register_mouse_scroll(self.on_mouse_scroll)
        self.renderer.register_resize(self.on_resize)

        # initialize viewer-local timer for per-frame integration
        self._last_time = time.perf_counter()

        # Only create GUI in non-headless mode to avoid OpenGL context dependency.
        if not headless:
            self.gui = ViewerGui(self, self.renderer.window)
            self.gui.register_ui_callback(self._render_gl_rendering_options, position="side")
            self.gui.register_ui_callback(self._render_wind_options, position="side")
        else:
            self.gui = None
        self._gizmo_log = None

        # a low resolution sphere mesh for point rendering
        self._point_mesh = None

        # Initialize PBO (Pixel Buffer Object) resources used in the `get_frame` method.
        self._pbo = None
        self._wp_pbo = None

        self.set_model(None)

    @property
    def ui(self):
        if self.gui is None:
            return None
        return self.gui.ui

    def _hash_geometry(self, geo_type: int, geo_scale, thickness: float, is_solid: bool, geo_src=None) -> int:
        # For capsules, ignore (radius, half_height) in the geometry hash so varying-length capsules batch together.
        # Capsule dimensions are stored per-shape in model.shape_scale as (radius, half_height, _unused) and
        # are remapped in set_model() to per-instance render scales (radius, radius, half_height).
        if geo_type == nt.GeoType.CAPSULE:
            geo_scale = (1.0, 1.0)
        return super()._hash_geometry(geo_type, geo_scale, thickness, is_solid, geo_src)

    def _invalidate_pbo(self):
        """Invalidate PBO resources, forcing reallocation on next get_frame() call."""
        if self._wp_pbo is not None:
            self._wp_pbo = None  # Let Python garbage collect the RegisteredGLBuffer
        if self._pbo is not None:
            gl = RendererGL.gl
            pbo_id = (gl.GLuint * 1)(self._pbo)
            gl.glDeleteBuffers(1, pbo_id)
            self._pbo = None

    # helper function to create a low resolution sphere mesh for point rendering
    def _create_point_mesh(self):
        """
        Create a low-resolution sphere mesh for point rendering.
        """
        mesh = nt.Mesh.create_sphere(1.0, num_latitudes=6, num_longitudes=6, compute_inertia=False)
        self._point_mesh = MeshGL(len(mesh.vertices), len(mesh.indices), self.device)

        points = wp.array(mesh.vertices, dtype=wp.vec3, device=self.device)
        normals = wp.array(mesh.normals, dtype=wp.vec3, device=self.device)
        uvs = wp.array(mesh.uvs, dtype=wp.vec2, device=self.device)
        indices = wp.array(mesh.indices, dtype=wp.int32, device=self.device)

        self._point_mesh.update(points, indices, normals, uvs)

    @override
    def log_gizmo(
        self,
        name,
        transform,
    ):
        # Store for this frame; call this every frame you want it drawn/active
        self._gizmo_log[name] = transform

    @override
    def set_model(self, model, max_worlds: int | None = None):
        """
        Set the Newton model to visualize.

        Args:
            model: The Newton model instance.
            max_worlds: Maximum number of worlds to render (None = all).
        """
        super().set_model(model, max_worlds=max_worlds)

        if self.model is not None:
            # For capsule batches, replace per-instance scales with (radius, radius, half_height)
            # so the capsule instancer path has the needed parameters.
            shape_scale = self.model.shape_scale
            if shape_scale.device != self.device:
                # Defensive: ensure inputs are on the launch device.
                shape_scale = wp.clone(shape_scale, device=self.device)

            def _ensure_indices_wp(model_shapes) -> wp.array:
                # Return shape indices as a Warp array on the viewer device
                if isinstance(model_shapes, wp.array):
                    if model_shapes.device == self.device:
                        return model_shapes
                    return wp.array(model_shapes.numpy().astype(np.int32), dtype=wp.int32, device=self.device)
                return wp.array(model_shapes, dtype=wp.int32, device=self.device)

            for batch in self._shape_instances.values():
                if batch.geo_type != nt.GeoType.CAPSULE:
                    continue

                shape_indices = _ensure_indices_wp(batch.model_shapes)
                num_shapes = len(shape_indices)
                out_scales = wp.empty(num_shapes, dtype=wp.vec3, device=self.device)
                if num_shapes == 0:
                    batch.scales = out_scales
                    continue
                wp.launch(
                    _capsule_build_body_scales,
                    dim=num_shapes,
                    inputs=[shape_scale, shape_indices],
                    outputs=[out_scales],
                    device=self.device,
                    record_tape=False,
                )
                batch.scales = out_scales

        self.wind = Wind(model)

        # Build packed arrays for batched GPU rendering of shape instances
        self._build_packed_vbo_arrays()

        fb_w, fb_h = self.renderer.window.get_framebuffer_size()
        self.camera = Camera(width=fb_w, height=fb_h, up_axis=model.up_axis if model else "Z")

    def _build_packed_vbo_arrays(self):
        """Build write-index + output arrays for batched shape transform computation.

        The kernel processes all model shapes (coalesced reads), uses a write-index
        array to scatter results into contiguous groups in the output buffer.
        """
        from .gl.opengl import MeshGL, MeshInstancerGL  # noqa: PLC0415

        if self.model is None:
            self._packed_groups = []
            return

        shape_count = self.model.shape_count
        device = self.device

        groups = []
        capsule_keys = set()
        total = 0

        for key, shapes in self._shape_instances.items():
            n = shapes.xforms.shape[0] if isinstance(shapes.xforms, wp.array) else len(shapes.xforms)
            if n == 0:
                continue
            if shapes.geo_type == nt.GeoType.CAPSULE:
                capsule_keys.add(key)
            groups.append((key, shapes, total, n))
            total += n

        self._capsule_keys = capsule_keys
        self._packed_groups = groups

        if total == 0:
            return

        # Write-index: maps model shape index → packed output position (-1 = skip)
        write_np = np.full(shape_count, -1, dtype=np.int32)
        # World xforms output (capsules read these for cap sphere computation)
        all_world_xforms = wp.empty(total, dtype=wp.transform, device=device)

        for _key, shapes, offset, n in groups:
            model_shapes = np.asarray(shapes.model_shapes, dtype=np.int32)
            write_np[model_shapes] = np.arange(offset, offset + n, dtype=np.int32)

            if _key in capsule_keys:
                shapes.world_xforms = all_world_xforms[offset : offset + n]

            if _key not in capsule_keys:
                if shapes.name not in self.objects:
                    if shapes.mesh in self.objects and isinstance(self.objects[shapes.mesh], MeshGL):
                        self.objects[shapes.name] = MeshInstancerGL(max(n, 1), self.objects[shapes.mesh])

        self._packed_write_indices = wp.array(write_np, dtype=int, device=device)
        self._packed_world_xforms = all_world_xforms
        self._packed_vbo_xforms = wp.empty(total, dtype=wp.mat44, device=device)
        self._packed_vbo_xforms_host = wp.empty(total, dtype=wp.mat44, device="cpu", pinned=True)

    @override
    def set_camera(self, pos: wp.vec3, pitch: float, yaw: float):
        """
        Set the camera position, pitch, and yaw.

        Args:
            pos: The camera position.
            pitch: The camera pitch.
            yaw: The camera yaw.
        """
        self.camera.pos = pos
        self.camera.pitch = pitch
        self.camera.yaw = yaw

    @override
    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array | None = None,
        uvs: wp.array | None = None,
        texture: np.ndarray | str | None = None,
        hidden=False,
        backface_culling=True,
    ):
        """
        Log a mesh for rendering.

        Args:
            name (str): Unique name for the mesh.
            points (wp.array): Vertex positions.
            indices (wp.array): Triangle indices.
            normals (wp.array, optional): Vertex normals.
            uvs (wp.array, optional): Vertex UVs.
            texture (np.ndarray | str, optional): Texture path/URL or image array (H, W, C).
            hidden (bool): Whether the mesh is hidden.
            backface_culling (bool): Enable backface culling.
        """
        assert isinstance(points, wp.array)
        assert isinstance(indices, wp.array)
        assert normals is None or isinstance(normals, wp.array)
        assert uvs is None or isinstance(uvs, wp.array)

        if name not in self.objects:
            self.objects[name] = MeshGL(
                len(points), len(indices), self.device, hidden=hidden, backface_culling=backface_culling
            )

        self.objects[name].update(points, indices, normals, uvs, texture)
        self.objects[name].hidden = hidden
        self.objects[name].backface_culling = backface_culling

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        """
        Log a batch of mesh instances for rendering.

        Args:
            name (str): Unique name for the instancer.
            mesh (str): Name of the base mesh.
            xforms: Array of transforms.
            scales: Array of scales.
            colors: Array of colors.
            materials: Array of materials.
            hidden: Whether the instances are hidden.
        """
        if mesh not in self.objects:
            raise RuntimeError(f"Path {mesh} not found")

        # check it is a mesh object
        if not isinstance(self.objects[mesh], MeshGL):
            raise RuntimeError(f"Path {mesh} is not a Mesh object")

        instancer = self.objects.get(name, None)
        transform_count = len(xforms) if xforms is not None else 0
        resized = False

        if instancer is None:
            capacity = max(transform_count, 1)
            instancer = MeshInstancerGL(capacity, self.objects[mesh])
            self.objects[name] = instancer
            resized = True
        elif transform_count > instancer.num_instances:
            new_capacity = max(transform_count, instancer.num_instances * 2)
            instancer = MeshInstancerGL(new_capacity, self.objects[mesh])
            self.objects[name] = instancer
            resized = True

        needs_update = resized or not hidden
        if needs_update:
            self.objects[name].update_from_transforms(xforms, scales, colors, materials)

        self.objects[name].hidden = hidden

    @override
    def log_capsules(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        """
        Render capsules using instanced cylinder bodies + instanced sphere end caps.

        This specialized path improves batching for varying-length capsules by reusing two
        prototype meshes (unit cylinder + unit sphere) and applying per-instance transforms/scales.

        Args:
            name (str): Unique name for the capsule instancer group.
            mesh: Capsule prototype mesh path from ViewerBase (unused in this backend).
            xforms: Capsule instance transforms (wp.transform), length N.
            scales: Capsule body instance scales, expected (radius, radius, half_height), length N.
            colors: Capsule instance colors (wp.vec3), length N or None (no update).
            materials: Capsule instance materials (wp.vec4), length N or None (no update).
            hidden (bool): Whether the instances are hidden.
        """
        # Render capsules via instanced cylinder body + instanced sphere caps.
        sphere_mesh = "/geometry/_capsule_instancer/sphere"
        cylinder_mesh = "/geometry/_capsule_instancer/cylinder"

        if sphere_mesh not in self.objects:
            self.log_geo(sphere_mesh, nt.GeoType.SPHERE, (1.0,), 0.0, True, hidden=True)
        if cylinder_mesh not in self.objects:
            self.log_geo(cylinder_mesh, nt.GeoType.CYLINDER, (1.0, 1.0), 0.0, True, hidden=True)

        # Cylinder body uses the capsule transforms and (radius, radius, half_height) scaling.
        cyl_name = f"{name}/capsule_cylinder"
        cap_name = f"{name}/capsule_caps"

        # If hidden, just hide the instancers (skip all per-frame cap buffer work).
        if hidden:
            self.log_instances(cyl_name, cylinder_mesh, None, None, None, None, hidden=True)
            self.log_instances(cap_name, sphere_mesh, None, None, None, None, hidden=True)
            return

        self.log_instances(cyl_name, cylinder_mesh, xforms, scales, colors, materials, hidden=hidden)

        # Sphere caps: two spheres per capsule, offset by ±half_height along local +Z.
        n = len(xforms) if xforms is not None else 0
        if n == 0:
            self.log_instances(cap_name, sphere_mesh, None, None, None, None, hidden=True)
            return

        cap_count = n * 2
        cap_xforms = wp.empty(cap_count, dtype=wp.transform, device=self.device)
        cap_scales = wp.empty(cap_count, dtype=wp.vec3, device=self.device)

        wp.launch(
            _capsule_build_cap_xforms_and_scales,
            dim=cap_count,
            inputs=[xforms, scales],
            outputs=[cap_xforms, cap_scales],
            device=self.device,
            record_tape=False,
        )

        cap_colors = None
        if colors is not None:
            cap_colors = wp.empty(cap_count, dtype=wp.vec3, device=self.device)
            wp.launch(
                _capsule_duplicate_vec3,
                dim=cap_count,
                inputs=[colors],
                outputs=[cap_colors],
                device=self.device,
                record_tape=False,
            )

        cap_materials = None
        if materials is not None:
            cap_materials = wp.empty(cap_count, dtype=wp.vec4, device=self.device)
            wp.launch(
                _capsule_duplicate_vec4,
                dim=cap_count,
                inputs=[materials],
                outputs=[cap_materials],
                device=self.device,
                record_tape=False,
            )

        self.log_instances(cap_name, sphere_mesh, cap_xforms, cap_scales, cap_colors, cap_materials, hidden=hidden)

    @override
    def log_lines(
        self,
        name,
        starts: wp.array,
        ends: wp.array,
        colors,
        width: float = 0.01,
        hidden=False,
    ):
        """
        Log line data for rendering.

        Args:
            name (str): Unique identifier for the line batch.
            starts (wp.array): Array of line start positions (shape: [N, 3]) or None for empty.
            ends (wp.array): Array of line end positions (shape: [N, 3]) or None for empty.
            colors: Array of line colors (shape: [N, 3]) or tuple/list of RGB or None for empty.
            width: The width of the lines (float)
            hidden (bool): Whether the lines are initially hidden.
        """
        # Handle empty logs by resetting the LinesGL object
        if starts is None or ends is None or colors is None:
            if name in self.lines:
                self.lines[name].update(None, None, None)
            return

        assert isinstance(starts, wp.array)
        assert isinstance(ends, wp.array)
        num_lines = len(starts)
        assert len(ends) == num_lines, "Number of line ends must match line begins"

        # Handle tuple/list colors by expanding to array (only if not already converted above)
        if isinstance(colors, tuple | list):
            if num_lines > 0:
                color_vec = wp.vec3(*colors)
                colors = wp.zeros(num_lines, dtype=wp.vec3, device=self.device)
                colors.fill_(color_vec)  # Efficiently fill on GPU
            else:
                # Handle zero lines case
                colors = wp.array([], dtype=wp.vec3, device=self.device)

        assert isinstance(colors, wp.array)
        assert len(colors) == num_lines, "Number of line colors must match line begins"

        # Create or resize LinesGL object based on current requirements
        if name not in self.lines:
            # Start with reasonable default size, will expand as needed
            max_lines = max(num_lines, 1000)  # Reasonable default
            self.lines[name] = LinesGL(max_lines, self.device, hidden=hidden)
        elif num_lines > self.lines[name].max_lines:
            # Need to recreate with larger capacity
            self.lines[name].destroy()
            max_lines = max(num_lines, self.lines[name].max_lines * 2)
            self.lines[name] = LinesGL(max_lines, self.device, hidden=hidden)

        self.lines[name].update(starts, ends, colors)

    @override
    def log_points(self, name, points, radii, colors, hidden=False):
        """
        Log a batch of points for rendering as spheres.

        Args:
            name (str): Unique name for the point batch.
            points: Array of point positions.
            radii: Array of point radius values.
            colors: Array of point colors.
            hidden (bool): Whether the points are hidden.
        """
        if self._point_mesh is None:
            self._create_point_mesh()

        num_points = len(points)
        if name not in self.objects:
            # Start with a reasonable default.
            initial_capacity = max(num_points, 256)
            self.objects[name] = MeshInstancerGL(initial_capacity, self._point_mesh)
        elif num_points > self.objects[name].num_instances:
            old = self.objects[name]
            new_capacity = max(num_points, old.num_instances * 2)
            self.objects[name] = MeshInstancerGL(new_capacity, self._point_mesh)

        self.objects[name].update_from_points(points, radii, colors)
        self.objects[name].hidden = hidden

    @override
    def log_array(self, name, array):
        """
        Log a generic array for visualization (not implemented).
        """
        pass

    @override
    def log_scalar(self, name, value):
        """
        Log a scalar value for visualization (not implemented).
        """
        pass

    @override
    def log_state(self, state):
        """
        Log the current simulation state for rendering.

        For shape instances on CUDA, uses a batched path: 2 kernel launches +
        1 D2H copy to a shared pinned buffer, then uploads slices per instancer.
        Everything else (capsules, SDF, particles, joints, …) uses the standard path.
        """
        self._last_state = state

        if self.model is None:
            return

        if self._packed_vbo_xforms is not None and self.device.is_cuda:
            # ---- Single kernel over all model shapes, scatter-write to grouped output ----
            wp.launch(
                _compute_shape_vbo_xforms,
                dim=self.model.shape_count,
                inputs=[
                    self.model.shape_transform,
                    self.model.shape_body,
                    state.body_q,
                    self.model.shape_scale,
                    self.model.shape_type,
                    self.model.shape_world,
                    self.world_offsets,
                    self._packed_write_indices,
                ],
                outputs=[self._packed_world_xforms, self._packed_vbo_xforms],
                device=self.device,
                record_tape=False,
            )
            wp.copy(self._packed_vbo_xforms_host, self._packed_vbo_xforms)
            wp.synchronize()  # copy is async (pinned destination), must sync before CPU read

            # ---- Upload pinned host slices to GL per instancer ----
            host_np = self._packed_vbo_xforms_host.numpy()

            for key, shapes, offset, count in self._packed_groups:
                visible = self._should_show_shape(shapes.flags, shapes.static)
                colors = shapes.colors if self.model_changed or shapes.colors_changed else None
                materials = shapes.materials if self.model_changed else None

                if key in self._capsule_keys:
                    self.log_capsules(
                        shapes.name,
                        shapes.mesh,
                        shapes.world_xforms,
                        shapes.scales,
                        colors,
                        materials,
                        hidden=not visible,
                    )
                else:
                    instancer = self.objects.get(shapes.name)
                    if instancer is not None:
                        instancer.hidden = not visible
                        instancer.update_from_pinned(
                            host_np[offset : offset + count],
                            count,
                            colors,
                            materials,
                        )

                shapes.colors_changed = False

            # ---- Non-shape rendering uses standard synchronous paths ----
            self._log_non_shape_state(state)
            self.model_changed = False
        else:
            # Fallback for CPU or when no packed data is available
            super().log_state(state)

        self._render_picking_line(state)

    def _render_picking_line(self, state):
        """
        Render a line from the mouse cursor to the actual picked point on the geometry.

        Args:
            state: The current simulation state.
        """
        if not self.picking_enabled or not self.picking.is_picking():
            # Clear the picking line if not picking
            self.log_lines("picking_line", None, None, None)
            return

        # Get the picked body index
        pick_body_idx = self.picking.pick_body.numpy()[0]
        if pick_body_idx < 0:
            self.log_lines("picking_line", None, None, None)
            return

        # Get the pick target and current picked point on geometry (in physics space)
        pick_state = self.picking.pick_state.numpy()
        pick_target = np.array([pick_state[8], pick_state[9], pick_state[10]], dtype=np.float32)
        picked_point = np.array([pick_state[11], pick_state[12], pick_state[13]], dtype=np.float32)

        # Apply world offset to convert from physics space to visual space
        if self.world_offsets is not None and self.world_offsets.shape[0] > 0:
            if self.model.body_world is not None:
                body_world_idx = self.model.body_world.numpy()[pick_body_idx]
                if body_world_idx >= 0 and body_world_idx < self.world_offsets.shape[0]:
                    world_offset = self.world_offsets.numpy()[body_world_idx]
                    pick_target = pick_target + world_offset
                    picked_point = picked_point + world_offset

        # Create line data
        starts = wp.array(
            [wp.vec3(picked_point[0], picked_point[1], picked_point[2])], dtype=wp.vec3, device=self.device
        )
        ends = wp.array([wp.vec3(pick_target[0], pick_target[1], pick_target[2])], dtype=wp.vec3, device=self.device)
        colors = wp.array([wp.vec3(0.0, 1.0, 1.0)], dtype=wp.vec3, device=self.device)

        # Render the line
        self.log_lines("picking_line", starts, ends, colors, hidden=False)

    @override
    def begin_frame(self, time):
        """
        Begin a new frame (calls parent implementation).

        Args:
            time: Current simulation time.
        """
        super().begin_frame(time)
        self._gizmo_log = {}

    @override
    def end_frame(self):
        """
        Finish rendering the current frame and process window events.

        This method first updates the renderer which will poll and process
        window events.  It is possible that the user closes the window during
        this event processing step, which would invalidate the underlying
        OpenGL context.  Trying to issue GL calls after the context has been
        destroyed results in a crash (access violation).  Therefore we check
        whether an exit was requested and early-out before touching GL if so.
        """
        self._update()

    @override
    def apply_forces(self, state):
        """
        Apply viewer-driven forces (picking, wind) to the model.

        Args:
            state: The current simulation state.
        """
        super().apply_forces(state)

        # Apply wind forces
        self.wind._apply_wind_force(state)

    def _update(self):
        """
        Internal update: process events, update camera, wind, render scene and UI.
        """
        self.renderer.update()

        # Integrate camera motion with viewer-owned timing
        now = time.perf_counter()
        dt = max(0.0, min(0.1, now - self._last_time))
        self._last_time = now
        self._update_camera(dt)

        self.wind.update(dt)

        # If the window was closed during event processing, skip rendering
        if self.renderer.has_exit():
            return

        # Render the scene and present it
        self.renderer.render(self.camera, self.objects, self.lines)

        # Always update FPS tracking, even if UI is hidden
        self._update_fps()

        if self.gui:
            self.gui.render_frame(update_fps=False)

        self.renderer.present()

    def get_frame(self, target_image: wp.array | None = None, render_ui: bool = False) -> wp.array:
        """
        Retrieve the last rendered frame.

        This method uses OpenGL Pixel Buffer Objects (PBO) and CUDA interoperability
        to transfer pixel data entirely on the GPU, avoiding expensive CPU-GPU transfers.

        Args:
            target_image (wp.array, optional):
                Optional pre-allocated Warp array with shape `(height, width, 3)`
                and dtype `wp.uint8`. If `None`, a new array will be created.
            render_ui (bool): Whether to render the UI.

        Returns:
            wp.array: GPU array containing RGB image data with shape `(height, width, 3)`
                and dtype `wp.uint8`. Origin is top-left (OpenGL's bottom-left is flipped).
        """

        gl = RendererGL.gl
        w, h = self.renderer._screen_width, self.renderer._screen_height

        # Lazy initialization of PBO (Pixel Buffer Object).
        if self._pbo is None:
            pbo_id = (gl.GLuint * 1)()
            gl.glGenBuffers(1, pbo_id)
            self._pbo = pbo_id[0]

            # Allocate PBO storage.
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._pbo)
            gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, gl.GLsizeiptr(w * h * 3), None, gl.GL_STREAM_READ)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

            # Register with CUDA.
            self._wp_pbo = wp.RegisteredGLBuffer(
                gl_buffer_id=int(self._pbo),
                device=self.device,
                flags=wp.RegisteredGLBuffer.READ_ONLY,
            )

            # Set alignment once.
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)

        # GPU-to-GPU readback into PBO.
        assert self.renderer._frame_fbo is not None
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.renderer._frame_fbo)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._pbo)

        if render_ui and self.gui:
            self.gui.render_frame(update_fps=False)

        gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Map PBO buffer and copy using RGB kernel.
        assert self._wp_pbo is not None
        buf = self._wp_pbo.map(dtype=wp.uint8, shape=(w * h * 3,))

        if target_image is None:
            target_image = wp.empty(
                shape=(h, w, 3),
                dtype=wp.uint8,  # pyright: ignore[reportArgumentType]
                device=self.device,
            )

        if target_image.shape != (h, w, 3):
            raise ValueError(f"Shape of `target_image` must be ({h}, {w}, 3), got {target_image.shape}")

        # Launch the RGB kernel.
        wp.launch(
            copy_rgb_frame_uint8,
            dim=(w, h),
            inputs=[buf, w, h],
            outputs=[target_image],
            device=self.device,
        )

        # Unmap the PBO buffer.
        self._wp_pbo.unmap()

        return target_image

    @override
    def is_running(self) -> bool:
        """
        Check if the viewer is still running.

        Returns:
            bool: True if the window is open, False if closed.
        """
        return not self.renderer.has_exit()

    @override
    def close(self):
        """
        Close the viewer and clean up resources.
        """
        if self.ui:
            self.ui.shutdown()
        self.renderer.close()

    @property
    def vsync(self) -> bool:
        """
        Get the current vsync state.

        Returns:
            bool: True if vsync is enabled, False otherwise.
        """
        return self.renderer.get_vsync()

    @vsync.setter
    def vsync(self, enabled: bool):
        """
        Set the vsync state.

        Args:
            enabled (bool): Enable or disable vsync.
        """
        self.renderer.set_vsync(enabled)

    @override
    def is_key_down(self, key):
        """
        Check if a key is currently pressed.

        Args:
            key: Either a string representing a character/key name, or an int
                 representing a pyglet key constant.

                 String examples: 'w', 'a', 's', 'd', 'space', 'escape', 'enter'
                 Int examples: pyglet.window.key.W, pyglet.window.key.SPACE

        Returns:
            bool: True if the key is currently pressed, False otherwise.
        """
        try:
            import pyglet
        except Exception:
            return False

        if isinstance(key, str):
            # Convert string to pyglet key constant
            key = key.lower()

            # Handle single characters
            if len(key) == 1 and key.isalpha():
                key_code = getattr(pyglet.window.key, key.upper(), None)
            elif len(key) == 1 and key.isdigit():
                key_code = getattr(pyglet.window.key, f"_{key}", None)
            else:
                # Handle special key names
                special_keys = {
                    "space": pyglet.window.key.SPACE,
                    "escape": pyglet.window.key.ESCAPE,
                    "esc": pyglet.window.key.ESCAPE,
                    "enter": pyglet.window.key.ENTER,
                    "return": pyglet.window.key.ENTER,
                    "tab": pyglet.window.key.TAB,
                    "shift": pyglet.window.key.LSHIFT,
                    "ctrl": pyglet.window.key.LCTRL,
                    "alt": pyglet.window.key.LALT,
                    "up": pyglet.window.key.UP,
                    "down": pyglet.window.key.DOWN,
                    "left": pyglet.window.key.LEFT,
                    "right": pyglet.window.key.RIGHT,
                    "backspace": pyglet.window.key.BACKSPACE,
                    "delete": pyglet.window.key.DELETE,
                }
                key_code = special_keys.get(key, None)

            if key_code is None:
                return False
        else:
            # Assume it's already a pyglet key constant
            key_code = key

        return self.renderer.is_key_down(key_code)

    # events

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """
        Handle mouse scroll for zooming (FOV adjustment).

        Args:
            x, y: Mouse position.
            scroll_x, scroll_y: Scroll deltas.
        """
        if self.gui and self.gui.should_ignore_mouse_input():
            return
        if self.gui:
            self.gui.adjust_camera_fov_from_scroll(scroll_y, scale=2.0)

    def _to_framebuffer_coords(self, x: float, y: float) -> tuple[float, float]:
        """Convert window coordinates to framebuffer coordinates."""
        if self.gui:
            return self.gui.map_window_to_target_coords(x, y, self.renderer.window)
        return float(x), float(y)

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Handle mouse press events (object picking).

        Args:
            x, y: Mouse position.
            button: Mouse button pressed.
            modifiers: Modifier keys.
        """
        if self.gui and self.gui.should_ignore_mouse_input():
            return

        import pyglet

        # Handle right-click for picking
        if button == pyglet.window.mouse.RIGHT and self.gui:
            self.gui.start_picking_from_screen(x, y, self._to_framebuffer_coords)

    def on_mouse_release(self, x, y, button, modifiers):
        """
        Handle mouse release events to stop dragging.

        Args:
            x, y: Mouse position.
            button: Mouse button released.
            modifiers: Modifier keys.
        """
        import pyglet

        if button == pyglet.window.mouse.RIGHT and self.gui:
            self.gui.release_picking()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """
        Handle mouse drag events for camera and picking.

        Args:
            x, y: Mouse position.
            dx, dy: Mouse movement deltas.
            buttons: Mouse buttons pressed.
            modifiers: Modifier keys.
        """
        import pyglet

        allow_active_pick_drag = bool(buttons & pyglet.window.mouse.RIGHT) and bool(
            self.gui and self.gui.is_pick_active()
        )
        if self.gui and self.gui.should_ignore_mouse_input(allow_active_pick_drag=allow_active_pick_drag):
            return

        if buttons & pyglet.window.mouse.LEFT and self.gui:
            self.gui.rotate_camera_from_drag(dx, dy, sensitivity=0.1)

        if buttons & pyglet.window.mouse.RIGHT and self.gui:
            self.gui.update_picking_from_screen(x, y, self._to_framebuffer_coords)

    def on_mouse_motion(self, x, y, dx, dy):
        """
        Handle mouse motion events (not used).
        """
        pass

    def on_key_press(self, symbol, modifiers):
        """
        Handle key press events for UI and simulation control.

        Args:
            symbol: Key symbol.
            modifiers: Modifier keys.
        """
        if self.gui and self.gui.should_ignore_keyboard_input():
            return

        try:
            import pyglet
        except Exception:
            return

        if symbol == pyglet.window.key.H:
            self.show_ui = not self.show_ui
        elif symbol == pyglet.window.key.SPACE:
            # Toggle pause with space key
            self._paused = not self._paused
        elif symbol == pyglet.window.key.F:
            # Frame camera around model bounds
            self._frame_camera_on_model()
        elif symbol == pyglet.window.key.ESCAPE:
            # Exit with Escape key
            self.renderer.close()

    def on_key_release(self, symbol, modifiers):
        """
        Handle key release events (not used).
        """
        pass

    def _frame_camera_on_model(self):
        """
        Frame the camera to show all visible objects in the scene.
        """
        if self.model is None:
            return

        # Compute bounds from all visible objects
        min_bounds = np.array([float("inf")] * 3)
        max_bounds = np.array([float("-inf")] * 3)
        found_objects = False

        # Check body positions if available
        if hasattr(self, "_last_state") and self._last_state is not None:
            if hasattr(self._last_state, "body_q") and self._last_state.body_q is not None:
                body_q = self._last_state.body_q.numpy()
                # body_q is an array of transforms (7 values: 3 pos + 4 quat)
                # Extract positions (first 3 values of each transform)
                for i in range(len(body_q)):
                    pos = body_q[i, :3]
                    min_bounds = np.minimum(min_bounds, pos)
                    max_bounds = np.maximum(max_bounds, pos)
                    found_objects = True

        # If no objects found, use default bounds
        if not found_objects:
            min_bounds = np.array([-5.0, -5.0, -5.0])
            max_bounds = np.array([5.0, 5.0, 5.0])

        # Calculate center and size of bounding box
        center = (min_bounds + max_bounds) * 0.5
        size = max_bounds - min_bounds
        max_extent = np.max(size)

        # Ensure minimum size to avoid camera being too close
        if max_extent < 1.0:
            max_extent = 1.0

        # Calculate camera distance based on field of view
        # Distance = extent / tan(fov/2) with some padding
        fov_rad = np.radians(self.camera.fov)
        padding = 1.5
        distance = max_extent / (2.0 * np.tan(fov_rad / 2.0)) * padding

        # Position camera at distance from current viewing direction, looking at center
        from pyglet.math import Vec3 as PyVec3

        front = self.camera.get_front()
        new_pos = PyVec3(
            center[0] - front.x * distance,
            center[1] - front.y * distance,
            center[2] - front.z * distance,
        )
        self.camera.pos = new_pos

    def _update_camera(self, dt: float):
        """
        Update the camera position and orientation based on user input.

        Args:
            dt (float): Time delta since last update.
        """
        if self.gui:
            self.gui.update_camera_from_keys(dt, self.renderer.is_key_down)
        fb_w, fb_h = self.renderer.window.get_framebuffer_size()
        self.camera.update_screen_size(fb_w, fb_h)

    def on_resize(self, width, height):
        """
        Handle window resize events.

        Args:
            width (int): New window width.
            height (int): New window height.
        """
        fb_w, fb_h = self.renderer.window.get_framebuffer_size()
        self.camera.update_screen_size(fb_w, fb_h)
        self._invalidate_pbo()

        if self.ui:
            self.ui.resize(width, height)

    def _render_gl_rendering_options(self, imgui):
        """GL-specific rendering options UI callback."""
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("Rendering Options"):
            imgui.separator()

            changed, vsync = imgui.checkbox("VSync", self.vsync)
            if changed:
                self.vsync = vsync

            _changed, self.renderer.draw_sky = imgui.checkbox("Sky", self.renderer.draw_sky)
            _changed, self.renderer.draw_shadows = imgui.checkbox("Shadows", self.renderer.draw_shadows)
            _changed, self.renderer.draw_wireframe = imgui.checkbox("Wireframe", self.renderer.draw_wireframe)
            _changed, self.renderer._light_color = imgui.color_edit3("Light Color", self.renderer._light_color)
            _changed, self.renderer.sky_upper = imgui.color_edit3("Sky Color", self.renderer.sky_upper)
            _changed, self.renderer.sky_lower = imgui.color_edit3("Ground Color", self.renderer.sky_lower)

    def _render_wind_options(self, imgui):
        """Wind control UI callback."""
        imgui.set_next_item_open(False, imgui.Cond_.once)
        if imgui.collapsing_header("Wind"):
            imgui.separator()
            changed, amplitude = imgui.slider_float("Wind Amplitude", self.wind.amplitude, -2.0, 2.0, "%.2f")
            if changed:
                self.wind.amplitude = amplitude

            changed, period = imgui.slider_float("Wind Period", self.wind.period, 1.0, 30.0, "%.2f")
            if changed:
                self.wind.period = period

            changed, frequency = imgui.slider_float("Wind Frequency", self.wind.frequency, 0.1, 5.0, "%.2f")
            if changed:
                self.wind.frequency = frequency

            direction = [self.wind.direction[0], self.wind.direction[1], self.wind.direction[2]]
            changed, direction = imgui.slider_float3("Wind Direction", direction, -1.0, 1.0, "%.2f")
            if changed:
                self.wind.direction = direction
