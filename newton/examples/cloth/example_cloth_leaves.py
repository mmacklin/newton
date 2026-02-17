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
# Example Sim Cloth Leaves
#
# Demonstrates falling maple leaves simulated as cloth meshes with
# aerodynamic lift and drag forces. Each leaf is a procedurally
# generated triangle mesh instanced at random positions and
# orientations above a ground plane.
#
# A custom Warp kernel computes per-triangle aerodynamic drag and
# lift as external forces, creating the characteristic tumbling and
# gliding of falling leaves.
#
# Command:
#   python -m newton.examples cloth_leaves
#   python -m newton.examples cloth_leaves --solver xpbd
#   python -m newton.examples cloth_leaves --solver vbd
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def perturb_particle_velocities(
    particle_qd: wp.array(dtype=wp.vec3),
    seed: int,
    noise: float,
):
    """Add small random velocity perturbation to each particle."""
    tid = wp.tid()
    state = wp.rand_init(seed, tid)
    vx = (wp.randf(state) - 0.5) * 2.0 * noise
    vy = (wp.randf(state) - 0.5) * 2.0 * noise
    vz = (wp.randf(state) - 0.5) * 2.0 * noise
    particle_qd[tid] = particle_qd[tid] + wp.vec3(vx, vy, vz)


@wp.kernel
def eval_aerodynamic_forces(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=int),
    k_drag: float,
    k_lift: float,
    particle_f: wp.array(dtype=wp.vec3),
):
    """Compute per-triangle aerodynamic drag and lift.

    Drag opposes motion proportional to projected triangle area
    times v^2. Lift acts along the triangle normal proportional
    to angle of attack times v^2. Forces are distributed equally
    to the three triangle vertices.
    """
    tid = wp.tid()

    i = tri_indices[tid, 0]
    j = tri_indices[tid, 1]
    k = tri_indices[tid, 2]

    x0 = particle_q[i]
    x1 = particle_q[j]
    x2 = particle_q[k]

    v0 = particle_qd[i]
    v1 = particle_qd[j]
    v2 = particle_qd[k]

    # Triangle normal and area
    n = wp.cross(x1 - x0, x2 - x0)
    area = wp.length(n) * 0.5
    n = wp.normalize(n)

    # Centroid velocity
    vmid = (v0 + v1 + v2) / 3.0
    speed_sq = wp.dot(vmid, vmid)

    if speed_sq > 1.0e-12:
        vdir = wp.normalize(vmid)

        # Drag: opposes motion, projected area * v^2
        f_drag = vmid * (k_drag * area * wp.abs(wp.dot(n, vmid)))

        # Lift: normal force, angle of attack * v^2
        cos_a = wp.clamp(wp.dot(n, vdir), -1.0, 1.0)
        aoa = wp.HALF_PI - wp.acos(cos_a)
        f_lift = n * (k_lift * area * aoa) * speed_sq

        # Distribute 1/3 of total aero force per vertex
        f_aero = -(f_drag + f_lift) / 3.0

        wp.atomic_add(particle_f, i, f_aero)
        wp.atomic_add(particle_f, j, f_aero)
        wp.atomic_add(particle_f, k, f_aero)


def create_leaf_mesh(radius=0.10, segments=12, noise=0.005):
    """Create a circular disc mesh with per-vertex noise.

    A small amount of noise is added to each outer vertex to break
    the symmetry, producing more interesting aerodynamic behaviour.

    Args:
        radius: Radius in metres.
        segments: Number of segments around the circumference.
        noise: Magnitude of random per-vertex displacement (metres).

    Returns:
        Tuple of (vertices, indices) where vertices is a list of
        ``wp.vec3`` and indices is a ``numpy.int32`` array.
    """
    rng = np.random.default_rng(42)

    vertices = []
    # Vertex 0: center
    vertices.append(wp.vec3(0.0, 0.0, 0.0))
    # Outer ring with per-vertex noise
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x = radius * math.cos(angle) + rng.uniform(-noise, noise)
        y = radius * math.sin(angle) + rng.uniform(-noise, noise)
        vertices.append(wp.vec3(x, y, 0.0))

    # Fan triangulation
    indices = []
    for i in range(segments):
        i1 = 1 + i
        i2 = 1 + (i + 1) % segments
        indices.extend([0, i1, i2])

    return vertices, np.array(indices, dtype=np.int32)


def create_maple_leaf_mesh(radius=0.15, segments=12, noise=0.005):
    """Create a simplified maple-leaf disc mesh.

    Starts from the same fan-triangulated disc as
    :func:`create_leaf_mesh` but modulates the outer radius with
    five lobes to approximate a maple-leaf silhouette.  A small
    amount of noise is added to each outer vertex.

    Args:
        radius: Approximate tip-to-centre distance (metres).
        segments: Number of segments around the circumference.
        noise: Magnitude of random per-vertex displacement (metres).

    Returns:
        Tuple of (vertices, indices).
    """
    rng = np.random.default_rng(42)

    # Five lobes at these angles (radians), pointing "up" (+y)
    lobe_angles = np.radians([90.0, 162.0, 234.0, 306.0, 18.0])
    lobe_scales = [1.0, 0.75, 0.6, 0.6, 0.75]
    lobe_width = 0.55  # cosine half-width in radians
    base_scale = 0.45  # radius fraction between lobes

    vertices = []
    vertices.append(wp.vec3(0.0, 0.0, 0.0))  # centre

    for i in range(segments):
        angle = 2.0 * math.pi * i / segments

        # Start with the base (sinus) radius
        r_scale = base_scale
        # Add each lobe as a cosine bump
        for lc, ls in zip(lobe_angles, lobe_scales, strict=True):
            da = (angle - lc + math.pi) % (2.0 * math.pi) - math.pi
            if abs(da) < lobe_width:
                bump = ls * math.cos(da / lobe_width * (math.pi / 2.0))
                r_scale = max(r_scale, base_scale + bump * (1.0 - base_scale))

        r = radius * r_scale
        x = r * math.cos(angle) + rng.uniform(-noise, noise)
        y = r * math.sin(angle) + rng.uniform(-noise, noise)
        vertices.append(wp.vec3(x, y, 0.0))

    # Fan triangulation
    indices = []
    for i in range(segments):
        i1 = 1 + i
        i2 = 1 + (i + 1) % segments
        indices.extend([0, i1, i2])

    return vertices, np.array(indices, dtype=np.int32)


class Example:
    def __init__(
        self,
        viewer,
        args=None,
        solver_type="xpbd",
        num_leaves=32,
    ):
        self.solver_type = solver_type
        self.tri_drag = 0.01
        self.tri_lift = 500.0

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        if self.solver_type == "semi_implicit":
            self.sim_substeps = 32
            self.iterations = 1
        elif self.solver_type == "xpbd":
            self.sim_substeps = 8
            self.iterations = 4
        else:
            self.sim_substeps = 10
            self.iterations = 10

        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        rng = np.random.default_rng(7)

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1.0e2
        builder.default_shape_cfg.kd = 1.0e0
        builder.default_shape_cfg.mu = 0.5

        leaf_verts, leaf_indices = create_maple_leaf_mesh()

        for _ in range(num_leaves):
            px = rng.uniform(-1.5, 1.5)
            py = rng.uniform(-1.5, 1.5)
            pz = rng.uniform(2.0, 6.0)

            # Tilt ~30 deg from horizontal so the leaf has
            # angle-of-attack when falling straight down.
            tilt_axis = rng.normal(size=3)
            tilt_axis[2] = 0.0  # keep tilt axis horizontal
            tilt_axis /= np.linalg.norm(tilt_axis) + 1.0e-8
            tilt_angle = rng.uniform(-1.2, 1.2)  # 12-34 deg
            rot = wp.quat_from_axis_angle(
                wp.vec3(tilt_axis[0], tilt_axis[1], tilt_axis[2]),
                tilt_angle,
            )

            cloth_params = {
                "pos": wp.vec3(px, py, pz),
                "rot": rot,
                "scale": 1.0,
                "vel": wp.vec3(
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(-0.0, 0.5),
                ),
                "vertices": leaf_verts,
                "indices": leaf_indices,
                "density": 10.0,
                "tri_drag": 0.0,
                "tri_lift": 0.0,
                "edge_ke": 20.0,
                "edge_kd": 0.01,
                "particle_radius": 0.02,
            }

            if self.solver_type == "xpbd":
                cloth_params.update(
                    add_springs=True,
                    spring_ke=1.0e3,
                    spring_kd=1.0e0,
                )
            elif self.solver_type == "semi_implicit":
                cloth_params.update(
                    tri_ke=1.0e2,
                    tri_ka=1.0e2,
                    tri_kd=1.0e0,
                    edge_ke=10.0,
                    edge_kd=0.01,
                )
            else:
                cloth_params.update(
                    tri_ke=1.0e2,
                    tri_ka=1.0e2,
                    tri_kd=1.0e0,
                    edge_ke=0.1,
                    edge_kd=0.05,
                )

            builder.add_cloth_mesh(**cloth_params)

        if self.solver_type == "vbd":
            builder.color(include_bending=True)

        if self.solver_type == "semi_implicit":
            ground_cfg = builder.default_shape_cfg.copy()
            ground_cfg.ke = 1.0e2
            ground_cfg.kd = 1.0e0
            builder.add_ground_plane(cfg=ground_cfg)
        else:
            builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 0.5

        # Disable particle-particle contact grid — leaf vertices are
        # closer together than particle_radius so the penalty forces
        # between them would be explosive.
        self.model.particle_grid = None

        if self.solver_type == "semi_implicit":
            self.solver = newton.solvers.SolverSemiImplicit(
                model=self.model,
                enable_tri_contact=False,
            )
        elif self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=self.iterations,
            )
        else:
            self.solver = newton.solvers.SolverVBD(
                self.model,
                self.iterations,
                particle_enable_self_contact=False,
            )

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase_mode=newton.BroadPhaseMode.NXN,
            soft_contact_margin=0.05,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        # Add per-vertex velocity noise to break symmetry
        wp.launch(
            kernel=perturb_particle_velocities,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_qd, 42, 0.3],
            device=self.model.device,
        )

        # if self.solver_type == "vbd":
        self.capture()
        # else:
        #     self.graph = None

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)

            # Aerodynamic drag and lift (custom kernel for all solvers)
            wp.launch(
                kernel=eval_aerodynamic_forces,
                dim=self.model.tri_count,
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.model.tri_indices,
                    self.tri_drag,
                    self.tri_lift,
                ],
                outputs=[
                    self.state_0.particle_f,
                ],
                device=self.model.device,
            )

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            (self.state_0, self.state_1) = (
                self.state_1,
                self.state_0,
            )

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        choices=["semi_implicit", "xpbd", "vbd"],
        default="semi_implicit",
        help="Solver type (default: xpbd).",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=32,
        help="Number of leaves (default: 32).",
    )

    viewer, args = newton.examples.init(parser)

    example = Example(
        viewer=viewer,
        args=args,
        solver_type=args.solver,
        num_leaves=args.num_leaves,
    )

    newton.examples.run(example, args)
