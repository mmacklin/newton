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
import math
import os
from time import perf_counter

import numpy as np
import warp as wp

from ..core.types import override

try:
    from pxr import Gf, UsdGeom
except ImportError:
    Gf = UsdGeom = None

from .viewer_usd import ViewerUSD


class ViewerRTX(ViewerUSD):
    """Real-time ray-traced viewer using NVIDIA OVRTX.

    Builds a USD scene during the first simulation frame using the ViewerUSD
    base class, serializes it to disk, then creates an OVRTX renderer for
    real-time path-traced rendering.  Subsequent frames update rigid-body
    transforms (and deforming-mesh vertices) via the OVRTX attribute API
    and present the rendered image in a pyglet / OpenGL window.
    """

    _PHASE_BUILD = 0
    _PHASE_RENDER = 1

    def __init__(self, width=1280, height=720, fps=60, up_axis="Z", num_frames=None, scaling=1.0, headless=False):
        os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

        try:
            import ovrtx  # noqa: F401
        except ImportError as e:
            raise ImportError("ovrtx package is required for ViewerRTX. Install with: pip install ovrtx") from e

        if UsdGeom is None:
            raise ImportError("usd-core package is required for ViewerRTX. Install with: pip install usd-core")

        self._tmp_usd_path = os.path.abspath("_tmp_rtx_scene.usda")

        super().__init__(
            output_path=self._tmp_usd_path,
            fps=fps,
            up_axis=up_axis,
            num_frames=num_frames,
            scaling=scaling,
        )

        self._width = width
        self._height = height
        self._headless = headless

        # OVRTX renderer (created at end of first frame)
        self._rtx = None
        self._phase = self._PHASE_BUILD

        # Pyglet window state
        self._window = None
        self._pyglet_app = None
        self._should_close = False
        self._camera_dirty = True

        # Instance prim paths collected during build phase, keyed by instancer name.
        # Iteration order (insertion order) defines the layout inside the binding.
        self._instance_prim_paths: dict[str, list[str]] = {}
        self._all_instance_paths: list[str] = []
        self._transform_binding = None

        # Mesh prim paths for deforming-mesh point updates
        self._mesh_prim_paths: dict[str, str] = {}

        # Per-frame pending data (cleared each begin_frame)
        self._pending_xforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._pending_mesh_points: dict[str, np.ndarray] = {}

        # Camera state (matches ViewerGL Camera conventions)
        self._up_axis_idx = {"X": 0, "Y": 1, "Z": 2}.get(up_axis.strip().upper(), 2)
        self._camera_pos = np.array([10.0, 0.0, 2.0], dtype=np.float64)
        self._camera_pitch = 0.0
        self._camera_yaw = -180.0
        self._camera_fov = 45.0
        self._camera_near = 0.01
        self._camera_far = 1000.0
        self._camera_prim_path = "/World/Camera"
        self._render_product_path = "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"

        # Input / timing state
        self._paused = False
        self._keys_down: set[int] = set()
        self._cam_speed = 4.0
        self._last_perf_time: float | None = None

        # Window is deferred until _init_ovrtx to avoid pyglet/Warp
        # kernel compilation deadlock on Windows.

    # ------------------------------------------------------------------ window

    def _init_window(self):
        """Create a pyglet window with GL texture + shader for fast framebuffer blitting."""
        import ctypes  # noqa: PLC0415

        import pyglet

        pyglet.options["debug_gl"] = False
        from pyglet import gl

        self._window = pyglet.window.Window(
            width=self._width,
            height=self._height,
            caption="Newton RTX Viewer",
            resizable=True,
            visible=not self._headless,
        )
        self._pyglet_app = pyglet.app

        # ---- GL texture + shader for zero-copy blit --------------------------
        self._window.switch_to()

        tex_id = (gl.GLuint * 1)()
        gl.glGenTextures(1, tex_id)
        self._gl_texture = tex_id[0]
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._gl_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            self._width,
            self._height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Compile fullscreen-triangle shader (sRGB→linear + Y-flip in fragment)
        _VS = b"""#version 330
out vec2 uv;
void main() {
    uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
\x00"""
        _FS = b"""#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 fragColor;
void main() {
    vec4 c = texture(tex, vec2(uv.x, 1.0 - uv.y));
    fragColor = vec4(pow(c.rgb, vec3(2.2)), c.a);
}
\x00"""

        def _compile_shader(src, stype):
            s = gl.glCreateShader(stype)
            src_p = ctypes.c_char_p(src)
            src_pp = (ctypes.c_char_p * 1)(src_p)
            gl.glShaderSource(s, 1, ctypes.cast(src_pp, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))), None)
            gl.glCompileShader(s)
            return s

        vs = _compile_shader(_VS, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_FS, gl.GL_FRAGMENT_SHADER)
        self._gl_program = gl.glCreateProgram()
        gl.glAttachShader(self._gl_program, vs)
        gl.glAttachShader(self._gl_program, fs)
        gl.glLinkProgram(self._gl_program)
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)

        # Empty VAO required by core profile for the fullscreen triangle
        vao = (gl.GLuint * 1)()
        gl.glGenVertexArrays(1, vao)
        self._gl_vao = vao[0]

        # ---- input callbacks ------------------------------------------------
        @self._window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                self._camera_yaw -= dx * 0.2
                self._camera_pitch += dy * 0.2
                self._camera_pitch = max(-89.0, min(89.0, self._camera_pitch))
                self._camera_dirty = True

        @self._window.event
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            self._camera_fov = max(15.0, min(90.0, self._camera_fov - scroll_y * 2.0))
            self._camera_dirty = True

        @self._window.event
        def on_key_press(symbol, modifiers):
            self._keys_down.add(symbol)
            if symbol == pyglet.window.key.ESCAPE:
                self._window.close()
            elif symbol == pyglet.window.key.SPACE:
                self._paused = not self._paused

        @self._window.event
        def on_key_release(symbol, modifiers):
            self._keys_down.discard(symbol)

        @self._window.event
        def on_close():
            self._should_close = True

    # ------------------------------------------------------------------ camera

    def _get_camera_front(self):
        pitch = max(-89.0, min(89.0, self._camera_pitch))
        pr, yr = np.deg2rad(pitch), np.deg2rad(self._camera_yaw)
        cp, sp = np.cos(pr), np.sin(pr)
        cy, sy = np.cos(yr), np.sin(yr)
        if self._up_axis_idx == 2:  # Z-up
            return np.array([cy * cp, sy * cp, sp], dtype=np.float64)
        if self._up_axis_idx == 0:  # X-up
            return np.array([sp, cy * cp, sy * cp], dtype=np.float64)
        return np.array([cy * cp, sp, sy * cp], dtype=np.float64)  # Y-up

    def _get_world_up(self):
        up = np.zeros(3, dtype=np.float64)
        up[self._up_axis_idx] = 1.0
        return up

    def _compute_camera_matrix(self):
        """Return a 4x4 row-major world-transform for the camera prim (USD convention)."""
        fwd = self._get_camera_front()
        fwd /= np.linalg.norm(fwd) + 1e-12
        world_up = self._get_world_up()
        right = np.cross(fwd, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.cross(fwd, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            rn = np.linalg.norm(right)
        right /= rn
        up = np.cross(right, fwd)
        up /= np.linalg.norm(up)

        mat = np.eye(4, dtype=np.float64)
        mat[0, :3] = right
        mat[1, :3] = up
        mat[2, :3] = -fwd  # USD cameras look along local -Z
        mat[3, :3] = self._camera_pos
        return mat

    def _update_camera_movement(self, dt: float):
        import pyglet

        fwd = self._get_camera_front()
        world_up = self._get_world_up()
        right = np.cross(fwd, world_up)
        rn = np.linalg.norm(right)
        if rn > 1e-6:
            right /= rn

        fwd_horiz = fwd - world_up * np.dot(fwd, world_up)
        fn = np.linalg.norm(fwd_horiz)
        if fn > 1e-6:
            fwd_horiz /= fn

        desired = np.zeros(3, dtype=np.float64)
        key = pyglet.window.key
        if key.W in self._keys_down or key.UP in self._keys_down:
            desired += fwd_horiz
        if key.S in self._keys_down or key.DOWN in self._keys_down:
            desired -= fwd_horiz
        if key.A in self._keys_down or key.LEFT in self._keys_down:
            desired -= right
        if key.D in self._keys_down or key.RIGHT in self._keys_down:
            desired += right
        if key.E in self._keys_down:
            desired += world_up
        if key.Q in self._keys_down:
            desired -= world_up

        dn = np.linalg.norm(desired)
        if dn > 1e-6:
            desired = desired / dn * self._cam_speed
            self._camera_pos += desired * dt
            self._camera_dirty = True

    # -------------------------------------------------------- USD scene helpers

    def _add_camera_lights_and_render_product(self):
        """Insert camera, lights, and RenderProduct into the stage before serialisation."""
        from pxr import Sdf, UsdLux

        # ---- Camera ----------------------------------------------------------
        cam = UsdGeom.Camera.Define(self.stage, self._camera_prim_path)

        aspect = self._width / max(self._height, 1)
        h_aperture = 20.955
        v_aperture = h_aperture / aspect
        focal_length = h_aperture / (2.0 * math.tan(math.radians(self._camera_fov) / 2.0))

        cam.GetFocalLengthAttr().Set(focal_length)
        cam.GetHorizontalApertureAttr().Set(h_aperture)
        cam.GetVerticalApertureAttr().Set(v_aperture)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(self._camera_near, self._camera_far))

        xform = UsdGeom.Xform(cam.GetPrim())
        xform.ClearXformOpOrder()
        mat_op = xform.AddTransformOp()
        cam_mat = self._compute_camera_matrix()
        gf_mat = Gf.Matrix4d(*cam_mat.flatten().tolist())
        mat_op.Set(gf_mat)

        # ---- Lights ----------------------------------------------------------
        dome = UsdLux.DomeLight.Define(self.stage, "/root/_RTXDomeLight")
        dome.GetIntensityAttr().Set(1000.0)

        distant = UsdLux.DistantLight.Define(self.stage, "/root/_RTXDistantLight")
        distant.GetIntensityAttr().Set(3000.0)
        distant.GetAngleAttr().Set(0.53)
        dx = UsdGeom.Xform(distant.GetPrim())
        dx.ClearXformOpOrder()
        rot = dx.AddRotateXYZOp()
        if self._up_axis_idx == 2:
            rot.Set(Gf.Vec3f(-45.0, 30.0, 0.0))
        else:
            rot.Set(Gf.Vec3f(-45.0, 0.0, 30.0))

        # ---- Render hierarchy (must match Kit convention for OVRTX) ------------
        # Structure: /Render/OmniverseKit/HydraTextures/<product>
        #            /Render/Vars/LdrColor
        #            /Render/OmniverseGlobalRenderSettings
        self.stage.DefinePrim("/Render")
        self.stage.DefinePrim("/Render/OmniverseKit")
        self.stage.DefinePrim("/Render/OmniverseKit/HydraTextures")

        rp = self.stage.DefinePrim(self._render_product_path, "RenderProduct")
        rp.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.Create(
                prependedItems=[
                    "OmniRtxSettingsCommonAdvancedAPI_1",
                    "OmniRtxSettingsRtAdvancedAPI_1",
                    "OmniRtxSettingsPtAdvancedAPI_1",
                    "OmniRtxPostColorGradingAPI_1",
                    "OmniRtxPostChromaticAberrationAPI_1",
                    "OmniRtxPostBloomPhysicalAPI_1",
                    "OmniRtxPostMatteObjectAPI_1",
                    "OmniRtxPostCompositingAPI_1",
                    "OmniRtxPostDofAPI_1",
                    "OmniRtxPostMotionBlurAPI_1",
                    "OmniRtxPostTvNoiseAPI_1",
                    "OmniRtxPostTonemapIrayReinhardAPI_1",
                    "OmniRtxPostDebugSettingsAPI_1",
                    "OmniRtxDebugSettingsAPI_1",
                ]
            ),
        )
        rp.CreateRelationship("camera").SetTargets([Sdf.Path(self._camera_prim_path)])
        rp.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2, custom=False).Set(Gf.Vec2i(self._width, self._height))

        # RenderVar lives at /Render/Vars/LdrColor (NOT nested under the product)
        rv_path = "/Render/Vars/LdrColor"
        rv = self.stage.DefinePrim(rv_path, "RenderVar")
        rv.CreateAttribute("sourceName", Sdf.ValueTypeNames.String, custom=False).Set("LdrColor")
        rp.CreateRelationship("orderedVars").SetTargets([Sdf.Path(rv_path)])

        # ---- RTX render settings on the RenderProduct -------------------------
        rp.CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set("RealTimePathTracing")
        rp.CreateAttribute("omni:rtx:ambientOcclusion:denoiserMode", Sdf.ValueTypeNames.Token).Set("none")
        rp.CreateAttribute("omni:rtx:background:source:texture:textureMode", Sdf.ValueTypeNames.Token).Set(
            "repeatMirrored"
        )
        rp.CreateAttribute("omni:rtx:background:source:type", Sdf.ValueTypeNames.Token).Set("domeLight")
        rp.CreateAttribute("omni:rtx:debug:view:pixelDebug:enableFixedTextPos", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateAttribute("omni:rtx:directLighting:sampledLighting:denoisingTechnique", Sdf.ValueTypeNames.Token).Set(
            "None"
        )
        rp.CreateAttribute("omni:rtx:dlss:frameGeneration", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateAttribute("omni:rtx:indirectDiffuse:denoiser:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:post:aa:limitedOps", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:post:registeredCompositing:invertColorCorrection", Sdf.ValueTypeNames.Bool).Set(
            True
        )
        rp.CreateAttribute("omni:rtx:post:registeredCompositing:invertToneMap", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateAttribute("omni:rtx:pt:maxSamplesPerLaunch", Sdf.ValueTypeNames.Int).Set(2073600)
        rp.CreateAttribute("omni:rtx:pt:mgpu:maxPixelsPerRegionExponent", Sdf.ValueTypeNames.Int).Set(12)
        rp.CreateAttribute("omni:rtx:reflections:denoiser:enabled", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:rt:ambientLight:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.1))
        rp.CreateAttribute("omni:rtx:rt:demoire", Sdf.ValueTypeNames.Bool).Set(False)
        rp.CreateAttribute("omni:rtx:rt:lightcache:spatialCache:dontResolveConflicts", Sdf.ValueTypeNames.Bool).Set(
            True
        )
        rp.CreateAttribute("omni:rtx:rt:sss:samples", Sdf.ValueTypeNames.Int).Set(1)
        rp.CreateAttribute("omni:rtx:rtpt:maxVolumeBounces", Sdf.ValueTypeNames.Int).Set(15)
        rp.CreateAttribute("omni:rtx:rtpt:modulatingRoughnessThreshold", Sdf.ValueTypeNames.Float).Set(0.08)
        rp.CreateAttribute("omni:rtx:scene:hydra:mdlMaterialWarmup", Sdf.ValueTypeNames.Bool).Set(True)
        rp.CreateAttribute("omni:rtx:viewTile:limit", Sdf.ValueTypeNames.UInt).Set(4294967295)

        # Disable the quality convergence loop to minimize step() latency
        rp.CreateAttribute("omni:rtx:quality", Sdf.ValueTypeNames.Int, custom=False).Set(0)
        rp.CreateAttribute("omni:rtx:waitForEvents", Sdf.ValueTypeNames.TokenArray).Set([])

        # ---- RenderSettings --------------------------------------------------
        rs = self.stage.DefinePrim("/Render/OmniverseGlobalRenderSettings", "RenderSettings")
        rs.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.Create(
                prependedItems=[
                    "OmniRtxSettingsGlobalRtAdvancedAPI_1",
                    "OmniRtxSettingsGlobalPtAdvancedAPI_1",
                ]
            ),
        )
        rs.CreateRelationship("products").SetTargets([Sdf.Path(self._render_product_path)])

    # ------------------------------------------------------------- OVRTX init

    def _init_ovrtx(self):
        """Serialise the USD stage, create the OVRTX renderer and load the scene."""
        import ovrtx

        self._add_camera_lights_and_render_product()
        self.stage.GetRootLayer().Save()

        config = ovrtx.RendererConfig()
        config.log_level = "error"
        self._rtx = ovrtx.Renderer(config=config)
        self._rtx.add_usd(self._tmp_usd_path)

        # Flat prim-path list for a single transform binding
        self._all_instance_paths = []
        for paths in self._instance_prim_paths.values():
            self._all_instance_paths.extend(paths)

        if self._all_instance_paths:
            from ovrtx import PrimMode, Semantic

            self._transform_binding = self._rtx.bind_attribute(
                prim_paths=self._all_instance_paths,
                attribute_name="omni:xform",
                semantic=Semantic.XFORM_MAT4x4,
                prim_mode=PrimMode.MUST_EXIST,
            )

        # Create the presentation window now that all Warp kernels have been
        # compiled.  Doing this earlier causes a deadlock on Windows because
        # the Win32 message pump and Warp's JIT compilation fight for the
        # main thread.
        self._init_window()

        self._phase = self._PHASE_RENDER

    # ------------------------------------------------ ViewerUSD overrides

    @override
    def set_model(self, model, max_worlds=None):
        super().set_model(model, max_worlds=max_worlds)
        if model is not None:
            axis_idx = (
                model.up_axis
                if isinstance(model.up_axis, int)
                else {"X": 0, "Y": 1, "Z": 2}.get(str(model.up_axis).upper(), 2)
            )
            self._up_axis_idx = axis_idx
            if axis_idx == 0:
                self._camera_pos = np.array([2.0, 0.0, 10.0], dtype=np.float64)
            elif axis_idx == 2:
                self._camera_pos = np.array([10.0, 0.0, 2.0], dtype=np.float64)
            else:
                self._camera_pos = np.array([0.0, 2.0, 10.0], dtype=np.float64)

    @override
    def set_camera(self, pos, pitch: float, yaw: float):
        if hasattr(pos, "__iter__"):
            self._camera_pos = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
        self._camera_pitch = pitch
        self._camera_yaw = yaw
        self._camera_dirty = True

    @override
    def begin_frame(self, time):
        self._t_begin_frame = perf_counter()
        super().begin_frame(time)
        self._pending_xforms.clear()
        self._pending_mesh_points.clear()

        if self._window and not self._headless:
            try:
                self._window.switch_to()
                self._window.dispatch_events()
            except Exception:
                pass

        now = perf_counter()
        if self._last_perf_time is not None:
            dt = min(now - self._last_perf_time, 0.1)
            self._update_camera_movement(dt)
        self._last_perf_time = now
        self._t_after_begin = perf_counter()

    @override
    def end_frame(self):
        if self._phase == self._PHASE_BUILD:
            self._init_ovrtx()
        elif self._phase == self._PHASE_RENDER:
            t0 = perf_counter()
            self._update_ovrtx_camera()
            t1 = perf_counter()
            self._update_ovrtx_transforms()
            t2 = perf_counter()
            self._update_ovrtx_mesh_points()
            t3 = perf_counter()
            self._render_and_display()
            t4 = perf_counter()

            self._perf_frame_count = getattr(self, "_perf_frame_count", 0) + 1
            if self._perf_frame_count % 60 == 0:
                sim_ms = 1e3 * (t0 - self._t_after_begin)
                begin_ms = 1e3 * (self._t_after_begin - self._t_begin_frame)
                print(
                    f"[perf] begin={begin_ms:.1f}ms  sim+log={sim_ms:.1f}ms  "
                    f"cam={1e3 * (t1 - t0):.1f}ms  xforms={1e3 * (t2 - t1):.1f}ms  "
                    f"mesh_pts={1e3 * (t3 - t2):.1f}ms  render+blit={1e3 * (t4 - t3):.1f}ms  "
                    f"total={1e3 * (t4 - self._t_begin_frame):.1f}ms",
                    flush=True,
                )

    @override
    def log_mesh(
        self, name, points, indices, normals=None, uvs=None, texture=None, hidden=False, backface_culling=True
    ):
        if self._phase == self._PHASE_BUILD:
            super().log_mesh(name, points, indices, normals, uvs, texture, hidden, backface_culling)
            self._mesh_prim_paths[name] = self._get_path(name)
        elif name in self._mesh_prim_paths:
            pts = (
                points.numpy().astype(np.float32)
                if isinstance(points, wp.array)
                else np.asarray(points, dtype=np.float32)
            )
            self._pending_mesh_points[name] = pts

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        if self._phase == self._PHASE_BUILD:
            super().log_instances(name, mesh, xforms, scales, colors, materials, hidden)
            if xforms is not None:
                count = len(xforms)
                paths = [self._get_path(name) + f"/instance_{i}" for i in range(count)]
                self._instance_prim_paths[name] = paths
        else:
            if xforms is not None:
                xf = xforms.numpy() if isinstance(xforms, wp.array) else np.asarray(xforms)
                if scales is not None:
                    sc = scales.numpy() if isinstance(scales, wp.array) else np.asarray(scales)
                else:
                    sc = np.ones((len(xf), 3), dtype=np.float32)
                self._pending_xforms[name] = (xf, sc)

    @override
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        if self._phase == self._PHASE_BUILD:
            super().log_lines(name, starts, ends, colors, width, hidden)

    @override
    def log_points(self, name, points, radii, colors, hidden=False):
        if self._phase == self._PHASE_BUILD:
            super().log_points(name, points, radii, colors, hidden)
            self._mesh_prim_paths[name] = self._get_path(name)
        elif name in self._mesh_prim_paths:
            pts = (
                points.numpy().astype(np.float32)
                if isinstance(points, wp.array)
                else np.asarray(points, dtype=np.float32)
            )
            self._pending_mesh_points[name] = pts

    # --------------------------------------------------------- OVRTX updates

    def _update_ovrtx_camera(self):
        if self._rtx is None or not self._camera_dirty:
            return
        import ovrtx.math

        mat = self._compute_camera_matrix()
        cam_mat = ovrtx.math.Matrix4d()
        for i in range(4):
            for j in range(4):
                cam_mat[i][j] = mat[i, j]

        from ovrtx import Semantic

        self._rtx.write_attribute(
            prim_paths=[self._camera_prim_path],
            attribute_name="omni:xform",
            tensor=cam_mat.to_dltensor(),
            semantic=Semantic.XFORM_MAT4x4,
        )
        self._camera_dirty = False

    def _update_ovrtx_transforms(self):
        if not self._transform_binding or not self._pending_xforms:
            return

        from ovrtx import Device

        t_map0 = perf_counter()
        with self._transform_binding.map(device=Device.CPU) as mapping:
            t_map1 = perf_counter()
            matrices = np.from_dlpack(mapping.tensor)  # (N, 4, 4) float64

            total_updated = 0
            offset = 0
            for name, paths in self._instance_prim_paths.items():
                count = len(paths)
                if name in self._pending_xforms:
                    xf, sc = self._pending_xforms[name]
                    n = min(count, len(xf))
                    for i in range(n):
                        matrices[offset + i] = self._xform_to_mat44(xf[i][:3], xf[i][3:7], sc[i])
                    total_updated += n
                offset += count
            t_compute = perf_counter()

        t_unmap = perf_counter()
        pfc = getattr(self, "_perf_frame_count", 0)
        if pfc % 60 == 0:
            print(
                f"  [xforms] map={1e3 * (t_map1 - t_map0):.2f}ms  "
                f"compute({total_updated} prims)={1e3 * (t_compute - t_map1):.2f}ms  "
                f"unmap={1e3 * (t_unmap - t_compute):.2f}ms",
                flush=True,
            )

    @staticmethod
    def _make_point3f_dltensor(points_np):
        """Create a DLTensor with float3 (lanes=3) dtype from an (N,3) float32 array.

        OVRTX Fabric stores 'points' as point3f[] where each element is 12 bytes
        (float32 x 3 lanes). A plain DLTensor.from_dlpack on a (N,3) float32 array
        produces scalar float32 elements (4 bytes), causing an element-size mismatch.
        """
        from ovrtx._src.dlpack import DLTensor

        flat = np.ascontiguousarray(points_np, dtype=np.float32).reshape(-1)
        dl = DLTensor.from_dlpack(flat)
        n = len(flat) // 3
        dl.dtype.lanes = 3
        dl.ndim = 1
        shape_arr = (ctypes.c_int64 * 1)(n)
        dl.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
        dl._point3f_shape = shape_arr  # prevent GC
        return dl

    def _update_ovrtx_mesh_points(self):
        if self._rtx is None or not self._pending_mesh_points:
            return

        for mesh_name, points_np in self._pending_mesh_points.items():
            prim_path = self._mesh_prim_paths.get(mesh_name)
            if prim_path is None:
                continue
            dl = self._make_point3f_dltensor(points_np)
            self._rtx.write_array_attribute(
                prim_paths=[prim_path],
                attribute_name="points",
                tensors=[dl],
            )

    @staticmethod
    def _xform_to_mat44(pos, quat, scale):
        """Convert (pos, quaternion_xyzw, scale) to 4x4 row-major matrix (float64).

        Matches the USD GfMatrix4d / OVRTX fabric convention where the
        translation lives in the last row and basis vectors in the first three.
        """
        x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        mat = np.empty((4, 4), dtype=np.float64)
        mat[0, 0] = (1.0 - 2.0 * (yy + zz)) * sx
        mat[0, 1] = (2.0 * (xy + wz)) * sx
        mat[0, 2] = (2.0 * (xz - wy)) * sx
        mat[0, 3] = 0.0
        mat[1, 0] = (2.0 * (xy - wz)) * sy
        mat[1, 1] = (1.0 - 2.0 * (xx + zz)) * sy
        mat[1, 2] = (2.0 * (yz + wx)) * sy
        mat[1, 3] = 0.0
        mat[2, 0] = (2.0 * (xz + wy)) * sz
        mat[2, 1] = (2.0 * (yz - wx)) * sz
        mat[2, 2] = (1.0 - 2.0 * (xx + yy)) * sz
        mat[2, 3] = 0.0
        mat[3, 0] = float(pos[0])
        mat[3, 1] = float(pos[1])
        mat[3, 2] = float(pos[2])
        mat[3, 3] = 1.0
        return mat

    # ------------------------------------------------------- render + display

    def _render_and_display(self):
        if self._rtx is None or self._should_close:
            return
        from ovrtx import Device

        t_step0 = perf_counter()
        products = self._rtx.step(
            render_products={self._render_product_path},
            delta_time=1.0 / self.fps,
        )
        t_step1 = perf_counter()

        for _pname, product in products.items():
            for frame in product.frames:
                if "LdrColor" in frame.render_vars:
                    t_map0 = perf_counter()
                    with frame.render_vars["LdrColor"].map(device=Device.CPU) as mapping:
                        t_map1 = perf_counter()
                        pixels = mapping.tensor.numpy()
                        t_numpy = perf_counter()
                        self._blit_to_window(pixels)
                        t_blit = perf_counter()
                    t_unmap = perf_counter()

                    pfc = getattr(self, "_perf_frame_count", 0)
                    if pfc % 60 == 0:
                        print(
                            f"  [render] step={1e3 * (t_step1 - t_step0):.1f}ms  "
                            f"fb_map={1e3 * (t_map1 - t_map0):.2f}ms  "
                            f"numpy={1e3 * (t_numpy - t_map1):.2f}ms  "
                            f"blit={1e3 * (t_blit - t_numpy):.2f}ms  "
                            f"fb_unmap={1e3 * (t_unmap - t_blit):.2f}ms",
                            flush=True,
                        )
                    return

    def _blit_to_window(self, pixels: np.ndarray):
        """Upload *pixels* to a GL texture and draw a fullscreen triangle (GPU sRGB + flip)."""
        import ctypes  # noqa: PLC0415

        from pyglet import gl

        if self._window is None or self._window.context is None:
            return

        h, w = pixels.shape[:2]
        if not pixels.flags["C_CONTIGUOUS"]:
            pixels = np.ascontiguousarray(pixels)

        self._window.switch_to()
        fb_w, fb_h = self._window.get_framebuffer_size()
        gl.glViewport(0, 0, fb_w, fb_h)

        # Upload pixels directly from numpy buffer (no copy)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._gl_texture)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            w,
            h,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        )

        # Draw fullscreen triangle with sRGB→linear shader
        gl.glUseProgram(self._gl_program)
        gl.glBindVertexArray(self._gl_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self._window.flip()

    # ----------------------------------------------------------- viewer API

    @override
    def is_running(self) -> bool:
        if self._should_close:
            return False
        if self.num_frames is not None:
            return self._frame_count < self.num_frames
        return True

    @override
    def is_paused(self) -> bool:
        return self._paused

    @override
    def close(self):
        if self._transform_binding is not None:
            self._transform_binding.unbind()
            self._transform_binding = None

        self._rtx = None

        if self._window is not None:
            if not self._headless:
                try:
                    self._pyglet_app.event_loop.dispatch_event("on_exit")
                    self._pyglet_app.platform_event_loop.stop()
                except Exception:
                    pass
            self._window.close()
            self._window = None
