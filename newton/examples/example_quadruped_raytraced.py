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
from newton.utils.raytrace import RaytraceRendererPyglet
import pyglet

import os

# Modify the Example class definition:
class Example:
    def __init__(self, num_envs=1, dump_frames=False):
        ...
        self.dump_frames = dump_frames
        self.frame_num = 0
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

        # --- Main Scene Builder ---
        builder = newton.ModelBuilder()

        self.sim_time = 0.0
        fps = 30
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.num_envs = num_envs

        # --- Image dimensions (can be class attributes or passed to renderer) ---
        self.image_width = 1280
        self.image_height = 720

        # --- Environment Instantiation ---
        offsets = newton.examples.compute_env_offsets(self.num_envs)
        for i in range(self.num_envs):
            env_xform = wp.transform(offsets[i], wp.quat_identity())
            builder.add_builder(articulation_builder, xform=env_xform)

        np.set_printoptions(suppress=True)
        self.model = builder.finalize()
        self.model.ground = True

        self.solver = newton.solvers.XPBDSolver(self.model)

        # --- Pyglet Raytrace Renderer Setup ---
        self.renderer = RaytraceRendererPyglet(
            self.model,
            self.image_width,
            self.image_height,
            title_prefix=f"Quadruped (Pyglet Envs: {self.num_envs})"
        )

        # --- Adjust Camera for multiple environments if necessary ---
        cam_pos_arr = np.array([-2.5, 1.5, 3.0])
        cam_look_at_arr = np.array([0.0, 0.5, 0.0])
        cam_up_arr = np.array([0.0, 1.0, 0.0])
        fov_deg = 50.0

        if self.num_envs > 1:
            grid_side_len = math.ceil(math.sqrt(self.num_envs))
            # Determine rough center of the environment grid
            scene_center_offset = (grid_side_len - 1) * 2.0 / 2.0
            cam_look_at_arr[0] += scene_center_offset
            cam_look_at_arr[2] += scene_center_offset / 2.0
            cam_pos_arr[0] += scene_center_offset
            cam_pos_arr[2] += scene_center_offset / 2.0
            cam_pos_arr[1] += (grid_side_len - 1) * 0.5
        
        self.renderer.set_camera(cam_pos_arr, cam_look_at_arr, cam_up_arr, fov_deg)
        self.renderer.set_light_pos(np.array([4.0, 5.0, 3.0])) # Default light

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
        if self.renderer and not self.renderer.has_exit():
            with wp.ScopedTimer("render_frame_call"):
                self.renderer.render_frame(self.state_0)

                if self.dump_frames:
                    buffer = pyglet.image.get_buffer_manager().get_color_buffer()
                    os.makedirs("frames", exist_ok=True)
                    filename = f"frames/frame_{self.frame_num:04d}.png"
                    buffer.save(filename)
                    print(f"Saved frame to {filename}")
                    self.frame_num += 1                 


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

    parser.add_argument(
        "--dump_frames", action="store_true",
        help="Dump each rendered frame as a PNG to disk."
    )

    args = parser.parse_known_args()[0]

    

    wp.init()

    with wp.ScopedDevice(args.device):
        print(f"Running example on device: {wp.get_device()}")
        example = Example(num_envs=args.num_envs, dump_frames=args.dump_frames)

        frame_num = 0
        # Main loop using Pyglet window status
        while frame_num < args.num_frames and not example.renderer.has_exit():
            pyglet.clock.tick() 
            example.renderer.dispatch_events()

            if example.renderer.has_exit():
                print("Pyglet window closed by user.")
                break

            example.step()
            example.render() # This now calls renderer.render_frame()

            frame_num += 1
            if (frame_num % 10 == 0):
                print(f"Simulated frame {frame_num}/{args.num_frames}")

        if not example.renderer.has_exit():
            print("\nSimulation finished. Close Pyglet window to exit.")
            # Keep window open and responsive until user closes it
            while not example.renderer.has_exit():
                pyglet.clock.tick()
                example.renderer.dispatch_events()
                # Re-render the last state or just ensure window stays active
                example.renderer.render_frame(example.state_0) 
                # A small sleep can reduce CPU usage if nothing is changing
                # pyglet.clock.sleep(0.01) # Optional: if CPU usage is high when idle
        else:
            print("\nSimulation loop exited because window was closed.")

    if example.renderer: # Ensure renderer exists before trying to close
        example.renderer.close_window() # Attempt to close pyglet window if not already
    pyglet.app.exit() # Ensure pyglet app itself exits
    print("Exiting application.") 