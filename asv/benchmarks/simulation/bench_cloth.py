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

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if
from pxr import Usd

wp.config.quiet = True

import newton
import newton.examples
import newton.usd
from newton import ParticleFlags
from newton.examples.cloth.example_cloth_franka import Example as ExampleClothManipulation
from newton.examples.cloth.example_cloth_twist import Example as ExampleClothTwist
from newton.viewer import ViewerNull


@wp.kernel
def _lift_pinned_particles(
    pin_indices: wp.array(dtype=wp.int32),
    pin_rest_q: wp.array(dtype=wp.vec3),
    pick_time: wp.array(dtype=float),
    dt: float,
    lift_duration: float,
    lift_height: float,
    x_splay: float,
    q_0: wp.array(dtype=wp.vec3),
    qd_0: wp.array(dtype=wp.vec3),
    q_1: wp.array(dtype=wp.vec3),
    qd_1: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_idx = pin_indices[tid]
    rest_q = pin_rest_q[tid]

    next_pick_time = pick_time[0] + dt
    alpha = wp.min(next_pick_time / lift_duration, 1.0)
    smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    side = wp.where(rest_q[0] < 0.0, -1.0, 1.0)
    target_q = rest_q + wp.vec3(side * x_splay * smooth_alpha, 0.0, lift_height * smooth_alpha)
    target_qd = (target_q - q_0[particle_idx]) / dt

    q_0[particle_idx] = target_q
    qd_0[particle_idx] = target_qd
    q_1[particle_idx] = target_q
    qd_1[particle_idx] = target_qd

    if tid == 0:
        pick_time[0] = next_pick_time


class _PinnedShirtPickup:
    def __init__(self, num_frames, solver="vbd"):
        model_scale = 0.01
        self.num_frames = num_frames
        self.solver_name = solver
        self.sim_substeps = 2 if self.solver_name == "ipc" else 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.lift_height = 0.45
        self.x_splay = 0.08

        # libuipc's shell/contact Python bindings do not expose Newton-style kd
        # parameters, so keep VBD damping disabled for this solver comparison.
        self.cloth_tri_kd = 0.0
        self.cloth_edge_kd = 0.0
        self.cloth_contact_kd = 0.0
        self.shape_contact_kd = 0.0

        self.cloth_particle_radius = 0.008
        self.cloth_body_contact_margin = 0.008
        self.particle_self_contact_radius = 0.002
        # VBD's self-contact initialization also caps each particle's
        # per-substep displacement by margin * relaxation * 0.5. A margin equal
        # to the contact radius looks overly viscous in this meter-scale scene.
        self.particle_self_contact_margin = 0.00205
        self.particle_conservative_bound_relaxation = 0.85

        scene = newton.ModelBuilder(gravity=-9.81)
        scene.add_shape_box(
            -1,
            wp.transform(wp.vec3(0.0, -0.5, 0.1), wp.quat_identity()),
            hx=0.4,
            hy=0.4,
            hz=0.1,
        )

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
        shirt_mesh = newton.usd.get_mesh(usd_prim)

        scene.add_cloth_mesh(
            vertices=[wp.vec3(v) for v in shirt_mesh.vertices],
            indices=shirt_mesh.indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            pos=wp.vec3(0.0, 0.7, 0.45),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.04,
            scale=model_scale,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=self.cloth_tri_kd,
            edge_ke=5.0,
            edge_kd=self.cloth_edge_kd,
            particle_radius=self.cloth_particle_radius,
        )
        scene.color()
        scene.add_ground_plane()

        self.model = scene.finalize(requires_grad=False)
        self.model.edge_rest_angle.zero_()
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = self.cloth_contact_kd
        self.model.soft_contact_mu = 0.25

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[...] = 5.0e4
        shape_kd[...] = self.shape_contact_kd
        shape_mu[...] = 1.5
        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.device
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.pin_indices_np = self._select_pin_indices()
        flags = self.model.particle_flags.numpy()
        for particle_idx in self.pin_indices_np:
            flags[particle_idx] = flags[particle_idx] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags, dtype=self.model.particle_flags.dtype, device=self.model.device)

        pin_rest_q_np = self.state_0.particle_q.numpy()[self.pin_indices_np]
        self.pin_indices = wp.array(self.pin_indices_np, dtype=wp.int32, device=self.model.device)
        self.pin_rest_q = wp.array(pin_rest_q_np, dtype=wp.vec3, device=self.model.device)
        self.pick_time = wp.zeros(1, dtype=float, device=self.model.device)

        if self.solver_name == "vbd":
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                soft_contact_margin=self.cloth_body_contact_margin,
            )
            self.contacts = self.collision_pipeline.contacts()
            self.cloth_solver = newton.solvers.SolverVBD(
                self.model,
                iterations=self.iterations,
                particle_self_contact_radius=self.particle_self_contact_radius,
                particle_self_contact_margin=self.particle_self_contact_margin,
                particle_conservative_bound_relaxation=self.particle_conservative_bound_relaxation,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.005,
                particle_enable_self_contact=True,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                particle_collision_detection_interval=-1,
                rigid_contact_k_start=self.model.soft_contact_ke,
            )
        elif self.solver_name == "ipc":
            self.collision_pipeline = None
            self.contacts = None
            self.cloth_solver = newton.solvers.SolverIPC(
                self.model,
                contact_d_hat=0.0005,
                contact_friction=self.model.soft_contact_mu,
                contact_resistance=1.0e9,
                max_newton_iter=8,
                cloth_youngs=2.0e4,
                cloth_density=200.0,
                cloth_thickness=0.0002,
                cloth_bending_stiffness=200.0,
            )
        else:
            raise ValueError(f"Unsupported solver: {self.solver_name}")

    def _select_pin_indices(self):
        particle_q = self.model.particle_q.numpy()
        top_z = particle_q[:, 2] >= np.quantile(particle_q[:, 2], 0.95)
        top_indices = np.flatnonzero(top_z)
        return np.array(
            [
                top_indices[np.argmin(particle_q[top_indices, 0])],
                top_indices[np.argmax(particle_q[top_indices, 0])],
            ],
            dtype=np.int32,
        )

    def step(self):
        if self.solver_name == "vbd":
            self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            wp.launch(
                _lift_pinned_particles,
                dim=self.pin_indices.shape[0],
                inputs=[
                    self.pin_indices,
                    self.pin_rest_q,
                    self.pick_time,
                    self.sim_dt,
                    1.0,
                    self.lift_height,
                    self.x_splay,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.state_1.particle_q,
                    self.state_1.particle_qd,
                ],
                device=self.model.device,
            )

            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0


class FastExampleClothManipulation:
    timeout = 300
    repeat = 3
    number = 1

    def setup(self):
        self.num_frames = 30
        if hasattr(newton.examples, "default_args"):
            args = newton.examples.default_args()
        else:
            args = None
        self.example = ExampleClothManipulation(ViewerNull(num_frames=self.num_frames), args)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        newton.examples.run(self.example, args=None)

        wp.synchronize_device()


class FastExampleClothPinnedPickup:
    timeout = 300
    repeat = 3
    number = 1

    def setup(self):
        self.num_frames = 60
        self.example = _PinnedShirtPickup(self.num_frames, solver="vbd")

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()

        wp.synchronize_device()


class FastExampleClothPinnedPickupIPC:
    timeout = 300
    repeat = 3
    number = 1

    def setup(self):
        self.num_frames = 60
        self.example = _PinnedShirtPickup(self.num_frames, solver="ipc")

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()

        wp.synchronize_device()


class FastExampleClothTwist:
    repeat = 5
    number = 1

    def setup(self):
        self.num_frames = 100
        if hasattr(newton.examples, "default_args"):
            args = newton.examples.default_args()
        else:
            args = None
        self.example = ExampleClothTwist(ViewerNull(num_frames=self.num_frames), args)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        newton.examples.run(self.example, None)

        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleClothManipulation": FastExampleClothManipulation,
        "FastExampleClothPinnedPickup": FastExampleClothPinnedPickup,
        "FastExampleClothPinnedPickupIPC": FastExampleClothPinnedPickupIPC,
        "FastExampleClothTwist": FastExampleClothTwist,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--bench", default=None, action="append", choices=benchmark_list.keys(), help="Run a single benchmark."
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
