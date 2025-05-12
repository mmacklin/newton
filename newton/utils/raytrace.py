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

import math

import numpy as np

import warp as wp
# import warp.sim # Replaced by Newton imports
from newton.core.model import Model
from newton.core.builder import ModelBuilder
from newton.core.types import Mesh
from newton.core.state import State
import pyglet # Added for the renderer class

# Define geometry types
GEO_SPHERE = wp.constant(0)
GEO_BOX = wp.constant(1)
GEO_CAPSULE = wp.constant(2)
GEO_CYLINDER = wp.constant(3)
GEO_CONE = wp.constant(4)
GEO_MESH = wp.constant(5)
GEO_SDF = wp.constant(6)
GEO_PLANE = wp.constant(7)
GEO_NONE = wp.constant(8)


# --- HitResult Struct ---
@wp.struct
class HitResult:
    t: float
    normal: wp.vec3  # Normal in shape's local coordinate system
    hit: bool
    shape_idx_hit: int  # Index of the shape that was hit


# --- Helper Kernels / Functions ---


@wp.func
def ray_sphere_intersect(
    ray_origin: wp.vec3, ray_dir: wp.vec3, sphere_radius: float
) -> HitResult:
    result = HitResult()
    result.t = -1.0
    result.hit = False
    result.normal = wp.vec3()
    result.shape_idx_hit = -1

    oc = ray_origin
    a = wp.dot(ray_dir, ray_dir)
    b = 2.0 * wp.dot(oc, ray_dir)
    c = wp.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4.0 * a * c

    hit_t_candidate = -1.0
    if discriminant >= 0.0:
        sqrt_discriminant = wp.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2.0 * a)
        t2 = (-b + sqrt_discriminant) / (2.0 * a)

        if t1 > 1.0e-5 and t2 > 1.0e-5:
            hit_t_candidate = wp.min(t1, t2)
        elif t1 > 1.0e-5:
            hit_t_candidate = t1
        elif t2 > 1.0e-5:
            hit_t_candidate = t2

    if hit_t_candidate > 0.0:
        result.t = hit_t_candidate
        result.hit = True
        hit_point_local = ray_origin + ray_dir * hit_t_candidate
        result.normal = wp.normalize(hit_point_local)

    return result


@wp.func
def ray_box_intersect(
    ray_origin: wp.vec3, ray_dir: wp.vec3, box_half_extents: wp.vec3
) -> HitResult:
    result = HitResult()
    result.t = -1.0
    result.hit = False
    result.normal = wp.vec3()
    result.shape_idx_hit = -1

    inv_dir_x = 0.0
    if wp.abs(ray_dir[0]) > 1.0e-6:
        inv_dir_x = 1.0 / ray_dir[0]
    else:
        inv_dir_x = 1.0e6 * wp.sign(ray_dir[0])

    inv_dir_y = 0.0
    if wp.abs(ray_dir[1]) > 1.0e-6:
        inv_dir_y = 1.0 / ray_dir[1]
    else:
        inv_dir_y = 1.0e6 * wp.sign(ray_dir[1])

    inv_dir_z = 0.0
    if wp.abs(ray_dir[2]) > 1.0e-6:
        inv_dir_z = 1.0 / ray_dir[2]
    else:
        inv_dir_z = 1.0e6 * wp.sign(ray_dir[2])

    inv_dir = wp.vec3(inv_dir_x, inv_dir_y, inv_dir_z)

    t_min_bound = wp.cw_mul(-box_half_extents - ray_origin, inv_dir)
    t_max_bound = wp.cw_mul(box_half_extents - ray_origin, inv_dir)

    tmin_val = wp.min(t_min_bound, t_max_bound)
    tmax_val = wp.max(t_min_bound, t_max_bound)

    t_enter = wp.max(wp.max(tmin_val[0], tmin_val[1]), tmin_val[2])
    t_exit = wp.min(wp.min(tmax_val[0], tmax_val[1]), tmax_val[2])

    hit_t_candidate = -1.0
    if t_enter < t_exit and t_exit > 1.0e-5:
        if t_enter > 1.0e-5:
            hit_t_candidate = t_enter
        else:  # Ray origin inside box
            hit_t_candidate = t_exit

    if hit_t_candidate > 0.0:
        result.t = hit_t_candidate
        result.hit = True
        p = ray_origin + ray_dir * hit_t_candidate
        eps = 1.0e-4

        if wp.abs(p[0] - box_half_extents[0]) < eps:
            result.normal = wp.vec3(1.0, 0.0, 0.0)
        elif wp.abs(p[0] + box_half_extents[0]) < eps:
            result.normal = wp.vec3(-1.0, 0.0, 0.0)
        elif wp.abs(p[1] - box_half_extents[1]) < eps:
            result.normal = wp.vec3(0.0, 1.0, 0.0)
        elif wp.abs(p[1] + box_half_extents[1]) < eps:
            result.normal = wp.vec3(0.0, -1.0, 0.0)
        elif wp.abs(p[2] - box_half_extents[2]) < eps:
            result.normal = wp.vec3(0.0, 0.0, 1.0)
        elif wp.abs(p[2] + box_half_extents[2]) < eps:
            result.normal = wp.vec3(0.0, 0.0, -1.0)

    return result


@wp.func
def ray_plane_intersect(ray_origin_local: wp.vec3,
                        ray_dir_local_normalized: wp.vec3,
                        plane_width: float,
                        plane_length: float) -> HitResult:
    result = HitResult()
    result.t = -1.0
    result.hit = False
    result.normal = wp.vec3(0.0, 1.0, 0.0)
    result.shape_idx_hit = -1

    if wp.abs(ray_dir_local_normalized[1]) > 1.0e-6:
        t_candidate = -ray_origin_local[1] / ray_dir_local_normalized[1]

        if t_candidate > 1.0e-5:
            hit_point_local = ray_origin_local + \
                              ray_dir_local_normalized * t_candidate

            on_finite_plane = True
            if plane_width > 0.0:
                if wp.abs(hit_point_local[0]) > plane_width / 2.0:
                    on_finite_plane = False

            if plane_length > 0.0:
                if wp.abs(hit_point_local[2]) > plane_length / 2.0:
                    on_finite_plane = False

            if on_finite_plane:
                result.t = t_candidate
                result.hit = True
    return result


# --- Capsule Intersection ---
@wp.func
def ray_capsule_intersect(ray_origin: wp.vec3, ray_dir_normalized: wp.vec3,
                          radius: float, half_height: float) -> HitResult:
    res = HitResult()
    res.t = -1.0
    res.hit = False
    res.normal = wp.vec3()
    res.shape_idx_hit = -1

    min_overall_t = 1.0e6

    a_cyl = ray_dir_normalized[0] * ray_dir_normalized[0] + \
            ray_dir_normalized[2] * ray_dir_normalized[2]
    b_cyl = 2.0 * (ray_origin[0] * ray_dir_normalized[0] + \
                   ray_origin[2] * ray_dir_normalized[2])
    c_cyl = ray_origin[0] * ray_origin[0] + \
            ray_origin[2] * ray_origin[2] - radius * radius

    discriminant_cyl = b_cyl * b_cyl - 4.0 * a_cyl * c_cyl

    if discriminant_cyl >= 0.0:
        sqrt_disc_cyl = wp.sqrt(discriminant_cyl)
        t0_cyl = (-b_cyl - sqrt_disc_cyl) / (2.0 * a_cyl)
        t1_cyl = (-b_cyl + sqrt_disc_cyl) / (2.0 * a_cyl)

        if t0_cyl > 1.0e-5:
            p_hit_cyl = ray_origin + ray_dir_normalized * t0_cyl
            if wp.abs(p_hit_cyl[1]) <= half_height:
                if t0_cyl < min_overall_t:
                    min_overall_t = t0_cyl
                    res.t = t0_cyl
                    res.normal = wp.normalize(
                        wp.vec3(p_hit_cyl[0], 0.0, p_hit_cyl[2])
                    )
                    res.hit = True

        if t1_cyl > 1.0e-5:
            p_hit_cyl = ray_origin + ray_dir_normalized * t1_cyl
            if wp.abs(p_hit_cyl[1]) <= half_height:
                if t1_cyl < min_overall_t:
                    min_overall_t = t1_cyl
                    res.t = t1_cyl
                    res.normal = wp.normalize(
                        wp.vec3(p_hit_cyl[0], 0.0, p_hit_cyl[2])
                    )
                    res.hit = True

    cap_center_top = wp.vec3(0.0, half_height, 0.0)
    hit_cap_top_res = ray_sphere_intersect(ray_origin - cap_center_top,
                                           ray_dir_normalized, radius)
    if hit_cap_top_res.hit and hit_cap_top_res.t > 1.0e-5:
        if hit_cap_top_res.t < min_overall_t:
            hit_point_on_cap_surface_local_to_cap = \
                (ray_origin - cap_center_top) + \
                ray_dir_normalized * hit_cap_top_res.t
            if hit_point_on_cap_surface_local_to_cap[1] >= -1.0e-5:
                min_overall_t = hit_cap_top_res.t
                res.t = hit_cap_top_res.t
                res.normal = hit_cap_top_res.normal
                res.hit = True

    cap_center_bottom = wp.vec3(0.0, -half_height, 0.0)
    hit_cap_bottom_res = ray_sphere_intersect(ray_origin - cap_center_bottom,
                                              ray_dir_normalized, radius)
    if hit_cap_bottom_res.hit and hit_cap_bottom_res.t > 1.0e-5:
        if hit_cap_bottom_res.t < min_overall_t:
            hit_point_on_cap_surface_local_to_cap = \
                (ray_origin - cap_center_bottom) + \
                ray_dir_normalized * hit_cap_bottom_res.t
            if hit_point_on_cap_surface_local_to_cap[1] <= 1.0e-5:
                min_overall_t = hit_cap_bottom_res.t
                res.t = hit_cap_bottom_res.t
                res.normal = hit_cap_bottom_res.normal
                res.hit = True

    return res


@wp.func
def cast_ray_against_scene(
    ray_origin_world: wp.vec3,
    ray_dir_world: wp.vec3,
    max_t_to_check: float,
    model_shape_geo_type: wp.array(dtype=wp.int32),
    model_shape_geo_scale: wp.array(dtype=wp.vec3),
    model_shape_geo_source: wp.array(dtype=wp.uint64),
    model_shape_transform_model: wp.array(dtype=wp.transform),
    model_shape_body_idx: wp.array(dtype=wp.int32),
    state_body_q_world: wp.array(dtype=wp.transform),
    num_shapes: int
) -> HitResult:

    closest_hit_result = HitResult()
    closest_hit_result.t = max_t_to_check
    closest_hit_result.hit = False
    closest_hit_result.shape_idx_hit = -1
    closest_hit_result.normal = wp.vec3() # Initialize

    for shape_idx in range(num_shapes):
        shape_type = model_shape_geo_type[shape_idx]
        shape_scale_vec = model_shape_geo_scale[shape_idx]
        shape_mesh_id = model_shape_geo_source[shape_idx]
        shape_tf_model = model_shape_transform_model[shape_idx]
        body_idx = model_shape_body_idx[shape_idx]

        current_shape_tf_world = shape_tf_model
        if body_idx >= 0:
            if body_idx < state_body_q_world.shape[0]:
                body_tf_world = state_body_q_world[body_idx]
                current_shape_tf_world = body_tf_world * shape_tf_model
            else:
                continue

        ray_origin_local = wp.transform_point(
            wp.transform_inverse(current_shape_tf_world), ray_origin_world)
        ray_dir_local = wp.transform_vector(
            wp.transform_inverse(current_shape_tf_world), ray_dir_world)
        ray_dir_local_normalized = wp.normalize(ray_dir_local)

        current_shape_hit_result = HitResult() 
        current_shape_hit_result.hit = False
        current_shape_hit_result.t = -1.0 
        # Normal is set by intersection functions if hit occurs

        if shape_type == GEO_SPHERE:
            current_shape_hit_result = ray_sphere_intersect(ray_origin_local,
                                                        ray_dir_local_normalized,
                                                        shape_scale_vec[0])
        elif shape_type == GEO_BOX:
            current_shape_hit_result = ray_box_intersect(ray_origin_local,
                                                       ray_dir_local_normalized,
                                                       shape_scale_vec)
        elif shape_type == GEO_CAPSULE:
            current_shape_hit_result = ray_capsule_intersect(ray_origin_local,
                                                           ray_dir_local_normalized,
                                                           shape_scale_vec[0],
                                                           shape_scale_vec[1])
        elif shape_type == GEO_PLANE:
            current_shape_hit_result = ray_plane_intersect(ray_origin_local,
                                                         ray_dir_local_normalized,
                                                         shape_scale_vec[0],
                                                         shape_scale_vec[1])
        elif shape_type == GEO_MESH:
            mesh_query = wp.mesh_query_ray(shape_mesh_id,
                                           ray_origin_local,
                                           ray_dir_local_normalized,
                                           closest_hit_result.t)
            if mesh_query.result:
                current_shape_hit_result.t = mesh_query.t
                current_shape_hit_result.normal = mesh_query.normal
                current_shape_hit_result.hit = True

        if current_shape_hit_result.hit and \
           current_shape_hit_result.t > 1.0e-5 and \
           current_shape_hit_result.t < closest_hit_result.t:
            closest_hit_result.t = current_shape_hit_result.t
            closest_hit_result.hit = True
            closest_hit_result.normal = current_shape_hit_result.normal
            closest_hit_result.shape_idx_hit = shape_idx

    return closest_hit_result


# --- Main Raytracing Kernel ---
@wp.kernel
def raytrace_kernel(
    model_shape_geo_type: wp.array(dtype=wp.int32),
    model_shape_geo_scale: wp.array(dtype=wp.vec3),
    model_shape_geo_source: wp.array(dtype=wp.uint64),
    model_shape_transform_model: wp.array(dtype=wp.transform),
    model_shape_body_idx: wp.array(dtype=wp.int32),
    state_body_q_world: wp.array(dtype=wp.transform),
    num_shapes: int,
    cam_pos_world: wp.vec3,
    cam_forward: wp.vec3,
    cam_right: wp.vec3,
    cam_up: wp.vec3,
    img_width: int,
    img_height: int,
    fov_rad: float,
    light_pos_world: wp.vec3,
    pixels: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    px = tid % img_width
    py = tid // img_width

    aspect_ratio = float(img_width) / float(img_height)

    u = (float(px) + 0.5) / float(img_width) - 0.5
    v = (float(py) + 0.5) / float(img_height) - 0.5

    ray_dir_cam_space_x = u * 2.0 * wp.tan(fov_rad / 2.0)
    ray_dir_cam_space_y = v * 2.0 * wp.tan(fov_rad / 2.0) / aspect_ratio
    ray_dir_cam_space_z = -1.0

    ray_dir_local_cam = wp.normalize(
        wp.vec3(ray_dir_cam_space_x, ray_dir_cam_space_y, ray_dir_cam_space_z)
    )

    cam_orientation_mat = wp.mat33(
        cam_right[0], cam_up[0], -cam_forward[0],
        cam_right[1], cam_up[1], -cam_forward[1],
        cam_right[2], cam_up[2], -cam_forward[2]
    )

    primary_ray_dir_world = cam_orientation_mat * ray_dir_local_cam
    primary_ray_origin_world = cam_pos_world

    hit_color = wp.vec3(0.1, 0.1, 0.3)

    primary_hit_result = cast_ray_against_scene(
        primary_ray_origin_world, primary_ray_dir_world, 1.0e6,
        model_shape_geo_type, model_shape_geo_scale, model_shape_geo_source,
        model_shape_transform_model, model_shape_body_idx, 
        state_body_q_world, num_shapes
    )

    if primary_hit_result.hit:
        hit_shape_idx = primary_hit_result.shape_idx_hit
        hit_shape_tf_model = model_shape_transform_model[hit_shape_idx]
        hit_body_idx = model_shape_body_idx[hit_shape_idx]

        world_transform_of_hit_shape = hit_shape_tf_model
        if hit_body_idx >= 0:
            if hit_body_idx < state_body_q_world.shape[0]:
                hit_body_tf_world = state_body_q_world[hit_body_idx]
                world_transform_of_hit_shape = hit_body_tf_world * \
                                               hit_shape_tf_model

        world_normal_of_hit = wp.normalize(
            wp.transform_vector(world_transform_of_hit_shape, 
                                primary_hit_result.normal)
        )
        hit_point_world = primary_ray_origin_world + \
                          primary_ray_dir_world * primary_hit_result.t

        # Determine base color based on shape type
        object_base_color = wp.vec3(0.7, 0.7, 0.7)  # Default gray
        hit_shape_type = model_shape_geo_type[hit_shape_idx]

        if hit_shape_type == GEO_PLANE:
            object_base_color = wp.vec3(0.3, 0.6, 0.3)  # Greenish
        elif hit_shape_type == GEO_SPHERE:
            object_base_color = wp.vec3(0.8, 0.2, 0.2)  # Reddish
        elif hit_shape_type == GEO_BOX:
            object_base_color = wp.vec3(0.2, 0.3, 0.8)  # Bluish
        elif hit_shape_type == GEO_CAPSULE:
            object_base_color = wp.vec3(0.8, 0.8, 0.2)  # Yellowish
        elif hit_shape_type == GEO_MESH:
            object_base_color = wp.vec3(0.6, 0.2, 0.8)  # Purplish
        elif hit_shape_type == GEO_CYLINDER:
            object_base_color = wp.vec3(0.2, 0.8, 0.8)  # Cyanish
        elif hit_shape_type == GEO_CONE:
            object_base_color = wp.vec3(0.8, 0.5, 0.2)  # Orangish

        in_shadow = False
        shadow_ray_origin = hit_point_world + world_normal_of_hit * 1.0e-4
        shadow_ray_dir_to_light = wp.normalize(light_pos_world - shadow_ray_origin)
        max_t_for_shadow_ray = wp.length(
            light_pos_world - shadow_ray_origin) - 1.0e-3

        if max_t_for_shadow_ray > 1.0e-4:
            shadow_hit_result = cast_ray_against_scene(
                shadow_ray_origin, shadow_ray_dir_to_light, 
                max_t_for_shadow_ray,
                model_shape_geo_type, model_shape_geo_scale, 
                model_shape_geo_source, model_shape_transform_model, 
                model_shape_body_idx, state_body_q_world, num_shapes
            )
            if shadow_hit_result.hit:
                # Check if the hit for shadow is not the original shape itself, 
                # or if it is, ensure it's a valid occlusion (not just exiting the same shape)
                # This simple check assumes occluder is different or significantly closer.
                if shadow_hit_result.shape_idx_hit != primary_hit_result.shape_idx_hit or \
                   shadow_hit_result.t < (max_t_for_shadow_ray - 1.0e-4): #Ensure it's not the light source itself.
                    in_shadow = True

        diffuse_intensity = wp.max(0.0,
            wp.dot(world_normal_of_hit, shadow_ray_dir_to_light))
        ambient_intensity = 0.2

        if in_shadow:
            hit_color = object_base_color * ambient_intensity
        else:
            effective_diffuse = diffuse_intensity * (1.0 - ambient_intensity)
            hit_color = object_base_color * (ambient_intensity + effective_diffuse)
            hit_color = wp.min(hit_color, wp.vec3(1.0, 1.0, 1.0))

    pixels[tid] = hit_color


# --- Python Wrapper & Example ---


def setup_camera_vectors(cam_pos, look_at, up_vector):
    cam_forward = wp.normalize(look_at - cam_pos)
    cam_right = wp.normalize(wp.cross(cam_forward, up_vector))
    cam_up = wp.normalize(wp.cross(cam_right, cam_forward))
    return cam_pos, cam_forward, cam_right, cam_up


def render_model_shapes(
    model: Model,
    state: State,
    cam_pos_np: np.ndarray,
    cam_look_at_np: np.ndarray,
    cam_up_np: np.ndarray,
    img_width: int,
    img_height: int,
    fov_deg: float = 60.0,
    light_pos_np: np.ndarray = np.array([5.0, 5.0, 5.0]),
    device: str = "cpu",
):
    wp.init()

    effective_device = device

    cam_pos_wp = wp.vec3(cam_pos_np[0], cam_pos_np[1], cam_pos_np[2])
    cam_look_at_wp = wp.vec3(cam_look_at_np[0],
                                cam_look_at_np[1],
                                cam_look_at_np[2])
    cam_world_up_wp = wp.vec3(cam_up_np[0], cam_up_np[1], cam_up_np[2])
    light_pos_wp = wp.vec3(light_pos_np[0], light_pos_np[1], light_pos_np[2])

    cam_pos_vec, cam_forward_vec, cam_right_vec, cam_up_vec = setup_camera_vectors(
        cam_pos_wp, cam_look_at_wp, cam_world_up_wp
    )
    fov_rad_wp = math.radians(fov_deg)

    body_q_for_kernel = state.body_q
    if state.body_q is None or state.body_q.shape[0] == 0:
        body_q_for_kernel = wp.empty(shape=(0,), dtype=wp.transform,
                                     device=effective_device)

    pixels_arr = wp.zeros(img_width * img_height, dtype=wp.vec3,
                          device=effective_device)

    with wp.ScopedDevice(effective_device):
        wp.launch(
            kernel=raytrace_kernel,
            dim=img_width * img_height,
            inputs=[
                model.shape_geo.type,
                model.shape_geo.scale,
                model.shape_geo.source,
                model.shape_transform,
                model.shape_body,
                body_q_for_kernel,
                model.shape_count,
                cam_pos_vec,
                cam_forward_vec,
                cam_right_vec,
                cam_up_vec,
                img_width,
                img_height,
                fov_rad_wp,
                light_pos_wp,
                pixels_arr,
            ],
        )
    wp.synchronize()
    return pixels_arr.numpy().reshape((img_height, img_width, 3))


class RaytraceRendererPyglet:
    def __init__(self, model: Model, image_width: int, image_height: int, title_prefix="Newton Raytraced"):
        self.model = model
        self.image_width = image_width
        self.image_height = image_height

        self.window = pyglet.window.Window(
            width=self.image_width,
            height=self.image_height,
            caption=f"{title_prefix} - Model: {model.name if hasattr(model, 'name') else 'Untitled'}"
        )
        self.pyglet_image_data = None

        @self.window.event
        def on_close():
            pyglet.app.exit()

        # Default camera and light settings
        self.cam_pos_arr = np.array([-2.5, 1.5, 3.0])
        self.cam_look_at_arr = np.array([0.0, 0.5, 0.0])
        self.cam_up_arr = np.array([0.0, 1.0, 0.0])
        self.light_pos_arr = np.array([4.0, 5.0, 3.0])
        self.fov_deg = 50.0

    def set_camera(self, pos_np: np.ndarray, look_at_np: np.ndarray, up_np: np.ndarray, fov_deg: float = None):
        self.cam_pos_arr = pos_np
        self.cam_look_at_arr = look_at_np
        self.cam_up_arr = up_np
        if fov_deg is not None:
            self.fov_deg = fov_deg

    def set_light_pos(self, light_pos_np: np.ndarray):
        self.light_pos_arr = light_pos_np

    def _update_image_data(self, state: State):
        render_device_str = "cpu"
        if self.model.device.is_cuda:
            render_device_str = "cuda"
        elif self.model.device.is_cpu:
            render_device_str = "cpu"
        else:
            print(
                f"Warning: Model device {self.model.device} is not CUDA/CPU. "
                f"Raytracer defaulting to cpu."
            )

        pixels_output_flat = render_model_shapes(
            self.model, state,
            self.cam_pos_arr, self.cam_look_at_arr, self.cam_up_arr,
            self.image_width, self.image_height,
            fov_deg=self.fov_deg,
            light_pos_np=self.light_pos_arr,
            device=render_device_str
        )
        pixels_output_numpy = pixels_output_flat.reshape((self.image_height, self.image_width, 3))

        pixels_uint8 = (np.clip(pixels_output_numpy, 0, 1) * 255).astype(np.uint8)
        image_data_bytes = pixels_uint8.tobytes()

        self.pyglet_image_data = pyglet.image.ImageData(
            self.image_width,
            self.image_height,
            'RGB',
            image_data_bytes,
            pitch=self.image_width * 3
        )

    def render_frame(self, state: State):
        if self.window.has_exit:
            return

        self._update_image_data(state)

        self.window.clear()
        if self.pyglet_image_data:
            self.pyglet_image_data.blit(0, 0)
        self.window.flip()

    def dispatch_events(self):
        if not self.window.has_exit:
            pyglet.app.platform_event_loop.dispatch_posted_events()
            self.window.dispatch_events()

    def has_exit(self) -> bool:
        return self.window.has_exit

    def close_window(self):
        if not self.window.has_exit:
            self.window.close()
        # pyglet.app.exit() # Should be called by application after main loop


if __name__ == "__main__":
    # This section is for basic testing of the raytracer library itself.
    print("Newton Raytracer Library (newton.utils.raytrace)")
    print("Contains raytracing functions and Pyglet renderer for Newton models.")
    print("This file is not intended to be run directly as a full example.")

    # For a full example, see newton/examples/example_quadruped_raytraced.py

    # --- Example of how to use RaytraceRendererPyglet for a quick self-test ---
    # from newton.core.model import ModelBuilder, Mesh 
    # wp.init()
    # builder = ModelBuilder()
    # builder.set_ground_plane(offset=0.0)
    # builder.add_shape_sphere(body=-1, pos=(0,0.5,-2), rot=wp.quat(), radius=0.5)
    # builder.add_shape_box(body=-1, pos=(1.5, 0.5, -2), rot=wp.quat(), hx=0.3,hy=0.3,hz=0.3)
    # test_model = builder.finalize(device="cpu")
    # test_state = test_model.state()

    # renderer = RaytraceRendererPyglet(test_model, 320, 240, "Raytracer Self-Test")
    # renderer.set_camera(
    #     pos_np=np.array([0.0, 1.0, 2.0]),
    #     look_at_np=np.array([0.0, 0.5, -2.0]),
    #     up_np=np.array([0.0, 1.0, 0.0]),
    #     fov_deg=60.0
    # )
    # renderer.set_light_pos(np.array([3.0, 3.0, 1.0]))

    # print("\nRunning Pyglet self-test render loop...")
    # while not renderer.has_exit():
    #     pyglet.clock.tick()
    #     renderer.dispatch_events()
    #     if renderer.has_exit():
    #         break
    #     renderer.render_frame(test_state) # Render the static scene
    # pyglet.app.exit()
    # print("Pyglet self-test finished.")
