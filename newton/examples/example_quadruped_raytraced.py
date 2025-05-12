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
# Example Sim Quadruped Raytraced
#
# Shows how to set up a simulation of a rigid-body quadruped articulation
# from a URDF using the newton.ModelBuilder() and render it using a
# custom raytracer with Pyglet.
# Note this example does not include a trained policy.
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.collision
import newton.core.articulation
import newton.examples
import newton.utils
from newton.utils.raytrace import render_model_shapes
import pyglet


class Example:
    def __init__(self, num_envs=1):  # Default to 1 env for raytracing
        # --- Articulation Setup ---
        articulation_builder = newton.ModelBuilder()
        base_xform = wp.transform(
            [0.0, 0.7, 0.0],  # Initial position offset
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)
        )
        newton.utils.parse_urdf(
            newton.examples.get_asset("quadruped.urdf"),
            articulation_builder,
            xform=base_xform,
            floating=True,
            density=1000,
            armature=0.01,
            stiffness=200.0,  # This will set joint_target_ke
            damping=10.0,     # This will set joint_target_kd
            contact_ke=1.0e4,
            contact_kd=1.0e2,
            contact_kf=1.0e2,
            contact_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
        )

        # Set desired initial joint positions and mode for the base articulation
        # The quadruped URDF has 12 DoFs (3 per leg)
        if len(articulation_builder.joint_q) >= 12:
            initial_pose = [
                0.2, 0.4, -0.8, -0.2, 0.4, -0.8,
                0.2, 0.4, -0.8, -0.2, 0.4, -0.8
            ]
            articulation_builder.joint_q[-12:] = initial_pose
            articulation_builder.joint_axis_mode = (
                [newton.JOINT_MODE_TARGET_POSITION] *
                len(articulation_builder.joint_axis_mode)
            )
            articulation_builder.joint_act[-12:] = initial_pose
            # Ensure target_ke and target_kd are set for PD control
            # parse_urdf with stiffness/damping should handle this.
            # If specific values are needed per joint they can be set here.
            # For example:
            # for i in range(len(articulation_builder.joint_target_ke)):
            #    if articulation_builder.joint_axis_mode[i] == \
            #            newton.JOINT_MODE_TARGET_POSITION:
            #        articulation_builder.joint_target_ke[i] = 200.0
            #        articulation_builder.joint_target_kd[i] = 20.0

        # --- Main Scene Builder ---
        builder = newton.ModelBuilder()

        self.sim_time = 0.0
        fps = 30  # Target FPS for simulation steps and rendering
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_envs = num_envs

        # --- Camera and Light Setup for Raytracer ---
        self.cam_pos_arr = np.array([-2.5, 1.5, 3.0])
        self.cam_look_at_arr = np.array([0.0, 0.5, 0.0])
        self.cam_up_arr = np.array([0.0, 1.0, 0.0])
        self.light_pos_arr = np.array([4.0, 5.0, 3.0])
        self.image_width = 400
        self.image_height = 300
        self.field_of_view = 50.0

        if self.num_envs > 1:
            # Adjust camera to try to view multiple environments
            # This is a simple adjustment and might need tuning
            # Based on compute_env_offsets placing items with approx 2.0 spacing
            grid_side_len = math.ceil(math.sqrt(self.num_envs))
            # Center x/z offset
            scene_center_offset = (grid_side_len - 1) * 2.0 / 2.0
            self.cam_look_at_arr[0] += scene_center_offset
            self.cam_look_at_arr[2] += scene_center_offset / 2.0  # Pan a bit
            self.cam_pos_arr[0] += scene_center_offset
            self.cam_pos_arr[2] += scene_center_offset / 2.0
            self.cam_pos_arr[1] += (grid_side_len - 1) * 0.5  # Move up a bit

        # --- Environment Instantiation ---
        offsets = newton.examples.compute_env_offsets(self.num_envs)
        for i in range(self.num_envs):
            env_xform = wp.transform(offsets[i], wp.quat_identity())
            builder.add_builder(articulation_builder, xform=env_xform)

        np.set_printoptions(suppress=True)
        self.model = builder.finalize()
        self.model.ground = True

        self.solver = newton.solvers.XPBDSolver(self.model)

        # --- Pyglet Window Setup ---
        self.window = pyglet.window.Window(
            width=self.image_width,
            height=self.image_height,
            caption=f"Newton Quadruped Raytraced (Pyglet) - Envs: {self.num_envs}"
        )
        self.pyglet_image_data = None

        @self.window.event
        def on_close():
            pyglet.app.exit()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.core.articulation.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd,
            None, self.state_0
        )

        self.use_cuda_graph = (
            wp.get_device().is_cuda and
            wp.is_mempool_enabled(wp.get_device())
        )
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.collision.collide(self.model, self.state_0)
            self.solver.step(
                self.model, self.state_0, self.state_1,
                self.control, None, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step_simulation"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.window.has_exit:
            return

        with wp.ScopedTimer("render_raytrace_frame"):
            # Determine device for raytracer (prefers CUDA if model is on CUDA)
            render_device_str = "cpu"
            if self.model.device.is_cuda:
                render_device_str = "cuda"
            elif self.model.device.is_cpu:
                 render_device_str = "cpu"
            else:
                # Fallback if model device is not explicitly CUDA or CPU (e.g. Taichi).
                # Defaulting to CPU, but user might need to specify for other backends.
                print(
                    f"Warning: Model device {self.model.device} is not explicitly CUDA/CPU. "
                    f"Raytracer defaulting to cpu. May be slow or incompatible."
                )

            pixels_output_numpy = render_model_shapes(
                self.model, self.state_0,
                self.cam_pos_arr, self.cam_look_at_arr, self.cam_up_arr,
                self.image_width, self.image_height,
                fov_deg=self.field_of_view,
                light_pos_np=self.light_pos_arr,
                device=render_device_str
            )

        with wp.ScopedTimer("render_pyglet_update"):
            pixels_uint8 = (np.clip(pixels_output_numpy, 0, 1) * 255).astype(np.uint8)
            # Flip vertically for Pyglet (origin is bottom-left)
            pixels_uint8_flipped = np.flipud(pixels_uint8)
            image_data_bytes = pixels_uint8_flipped.tobytes()

            self.pyglet_image_data = pyglet.image.ImageData(
                self.image_width,
                self.image_height,
                'RGB',  # Raytracer produces RGB
                image_data_bytes,
                pitch=self.image_width * 3  # Bytes per row
            )
            # Drawing is handled in the main loop's on_draw or direct draw call


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override the default Warp device."
    )
    parser.add_argument(
        "--num_frames", type=int, default=300,
        help="Total number of frames to simulate."
    )
    parser.add_argument(
        "--num_envs", type=int, default=1,
        help="Total number of simulated environments (quadrupeds)."
    )

    args = parser.parse_known_args()[0]

    wp.init()

    with wp.ScopedDevice(args.device):
        print(f"Running example on device: {wp.get_device()}")
        example = Example(num_envs=args.num_envs)

        frame_num = 0
        while frame_num < args.num_frames and not example.window.has_exit:
            pyglet.clock.tick()  # Process clock and other scheduled events

            # Process window events (like close button). crucial for responsiveness.
            example.window.dispatch_events()

            if example.window.has_exit: # Check again after dispatching
                print("Pyglet window closed by user.")
                break

            example.step()
            example.render() # This updates example.pyglet_image_data

            example.window.clear()
            if example.pyglet_image_data:
                example.pyglet_image_data.blit(0, 0)
            example.window.flip()

            frame_num += 1
            if (frame_num % 10 == 0):
                print(f"Simulated frame {frame_num}/{args.num_frames}")

        if not example.window.has_exit:
            print("\nSimulation finished. Close Pyglet window to exit.")
            # Keep window open and responsive until user closes it
            while not example.window.has_exit:
                pyglet.clock.tick()
                example.window.dispatch_events()
                # Redraw the last frame or a static message
                example.window.clear()
                if example.pyglet_image_data:
                    example.pyglet_image_data.blit(0, 0)
                else:  # Should not happen if render was called at least once
                    pass  # Or draw a placeholder / waiting message
                example.window.flip()
                # A small sleep can reduce CPU usage when idle here,
                # but tick() should handle it.
        else:
            print("\nSimulation loop exited because window was closed.")

    pyglet.app.exit()
    print("Exiting application.") 