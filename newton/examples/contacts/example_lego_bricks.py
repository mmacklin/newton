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
# Example LEGO Bricks
#
# Demonstrates frictional locking of LEGO bricks using mesh + SDF
# collision with compliant contact.  A 2x2 brick is dropped onto a 2x4
# brick at true-to-life dimensions, pushed down so the studs lock via
# friction, and then the assembly is picked up by a simple parallel
# gripper (two kinematic boxes).
#
# All geometry is generated procedurally at real LEGO dimensions
# (scaled uniformly for simulation stability, as in the nut/bolt
# example).  No fixed joints or other artificial constraints are used --
# the locking is purely from frictional contact.
#
# Each brick is a single mesh (hollow shell + stud cylinders) with
# consistent outward-facing normals, so the SDF correctly resolves
# interior cavity vs. solid material.
#
# Command: python -m newton.examples lego_bricks
#          python -m newton.examples lego_bricks --solver xpbd
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

# ---------------------------------------------------------------------------
# LEGO dimensions (metres, true-to-life) -- uniformly scaled for sim
# ---------------------------------------------------------------------------
SCENE_SCALE = 1.0

PITCH = 0.008 * SCENE_SCALE
STUD_RADIUS = 0.0024 * SCENE_SCALE
STUD_HEIGHT = 0.0017 * SCENE_SCALE
BODY_HEIGHT = 0.0096 * SCENE_SCALE
WALL_THICKNESS = 0.0012 * SCENE_SCALE
TOP_THICKNESS = 0.001 * SCENE_SCALE
TUBE_OUTER_RADIUS = 0.003255 * SCENE_SCALE
TUBE_HEIGHT = BODY_HEIGHT - TOP_THICKNESS

CYLINDER_SEGMENTS = 16
SDF_RESOLUTION = 128
SDF_NARROW_BAND = 0.02 * SCENE_SCALE
SDF_MARGIN = 0.02 * SCENE_SCALE

# Interference fit: inflate each brick's collision surface to model the
# clutch-power overlap.  Two bricks' margins are summed, giving ~1mm
# total overlap -- exaggerated vs. real (~0.2mm) for a visible lock.
BRICK_MARGIN = 0.0001 * SCENE_SCALE


# ---------------------------------------------------------------------------
# Mesh generation helpers
# ---------------------------------------------------------------------------


def _cylinder_mesh(radius, height, segments, cx=0.0, cy=0.0, cz=0.0):
    """Closed cylinder: bottom cap, side quads, top cap.  Outward normals."""
    n = segments
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    verts = np.zeros((2 * n + 2, 3), dtype=np.float32)
    verts[:n, 0] = cx + radius * cos_a
    verts[:n, 1] = cy + radius * sin_a
    verts[:n, 2] = cz
    verts[n : 2 * n, 0] = cx + radius * cos_a
    verts[n : 2 * n, 1] = cy + radius * sin_a
    verts[n : 2 * n, 2] = cz + height
    verts[2 * n] = [cx, cy, cz]
    verts[2 * n + 1] = [cx, cy, cz + height]

    bc, tc = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        # side quads (outward radial)
        faces.append([i, n + j, n + i])
        faces.append([i, j, n + j])
        # bottom cap (normal -z)
        faces.append([bc, j, i])
        # top cap (normal +z)
        faces.append([tc, n + i, n + j])
    return verts, np.array(faces, dtype=np.int32)


def _combine_meshes(mesh_list):
    all_v, all_f, off = [], [], 0
    for v, f in mesh_list:
        all_v.append(v)
        all_f.append(f + off)
        off += len(v)
    return np.vstack(all_v).astype(np.float32), np.vstack(all_f).astype(np.int32)


def make_shell_mesh(nx, ny):
    """Watertight hollow box shell for an *nx* x *ny* LEGO brick.

    Origin at the centre-bottom (z = 0).  Inner cavity is open at the
    bottom and sealed by a top plate.
    """
    ox = nx * PITCH / 2.0
    oy = ny * PITCH / 2.0
    inx = ox - WALL_THICKNESS
    iny = oy - WALL_THICKNESS
    H = BODY_HEIGHT
    T = TOP_THICKNESS

    # Vertex layout:
    #   0-3:  outer ring at z=0      4-7:   outer ring at z=H
    #   8-11: inner ring at z=0      12-15: inner ring at z=H-T
    v = np.array(
        [
            [-ox, -oy, 0],  # 0
            [+ox, -oy, 0],  # 1
            [+ox, +oy, 0],  # 2
            [-ox, +oy, 0],  # 3
            [-ox, -oy, H],  # 4
            [+ox, -oy, H],  # 5
            [+ox, +oy, H],  # 6
            [-ox, +oy, H],  # 7
            [-inx, -iny, 0],  # 8
            [+inx, -iny, 0],  # 9
            [+inx, +iny, 0],  # 10
            [-inx, +iny, 0],  # 11
            [-inx, -iny, H - T],  # 12
            [+inx, -iny, H - T],  # 13
            [+inx, +iny, H - T],  # 14
            [-inx, +iny, H - T],  # 15
        ],
        dtype=np.float32,
    )
    # Every face wound so that (v1-v0) x (v2-v0) points OUTWARD from solid.
    f = np.array(
        [
            # outer top  (normal +z)
            [4, 5, 6],
            [4, 6, 7],
            # outer sides
            [0, 1, 5],
            [0, 5, 4],  # -y face  (normal -y)
            [2, 3, 7],
            [2, 7, 6],  # +y face  (normal +y)
            [3, 0, 4],
            [3, 4, 7],  # -x face  (normal -x)
            [1, 2, 6],
            [1, 6, 5],  # +x face  (normal +x)
            # bottom rim  (normal -z)
            [0, 8, 9],
            [0, 9, 1],
            [1, 9, 10],
            [1, 10, 2],
            [2, 10, 11],
            [2, 11, 3],
            [3, 11, 8],
            [3, 8, 0],
            # inner walls  (normals point into cavity)
            [9, 8, 12],
            [9, 12, 13],  # -y wall  (normal +y)
            [11, 10, 14],
            [11, 14, 15],  # +y wall  (normal -y)
            [8, 11, 15],
            [8, 15, 12],  # -x wall  (normal +x)
            [10, 9, 13],
            [10, 13, 14],  # +x wall  (normal -x)
            # inner ceiling  (normal -z, into cavity)
            [12, 15, 14],
            [12, 14, 13],
        ],
        dtype=np.int32,
    )
    return v, f


def make_lego_brick(nx, ny):
    """Full LEGO brick mesh (shell + studs + interior tubes) for an *nx* x *ny* brick.

    Each sub-component (shell, stud cylinders, tube cylinders) is a closed
    surface with consistent outward normals.  The combined mesh relies on the
    winding-number sign convention used by ``wp.mesh_query_point``.
    """
    shell_v, shell_f = make_shell_mesh(nx, ny)

    stud_meshes = []
    for i in range(nx):
        for j in range(ny):
            sx = (i - (nx - 1) / 2.0) * PITCH
            sy = (j - (ny - 1) / 2.0) * PITCH
            stud_meshes.append(
                _cylinder_mesh(STUD_RADIUS, STUD_HEIGHT, CYLINDER_SEGMENTS, cx=sx, cy=sy, cz=BODY_HEIGHT)
            )

    # Interior tubes: for 2-wide bricks, one tube between each adjacent
    # pair of stud columns, centred on the y-axis.  These grip the studs
    # of the brick below to create the clutch mechanism.
    tube_meshes = []
    if ny == 2:
        for i in range(nx - 1):
            tx = (i - (nx - 2) / 2.0) * PITCH
            tube_meshes.append(_cylinder_mesh(TUBE_OUTER_RADIUS, TUBE_HEIGHT, CYLINDER_SEGMENTS, cx=tx, cy=0.0, cz=0.0))

    return _combine_meshes([(shell_v, shell_f), *stud_meshes, *tube_meshes])


def _build_mesh_with_sdf(verts, faces, color):
    mesh = newton.Mesh(verts, faces.flatten(), color=color)
    mesh.build_sdf(
        max_resolution=SDF_RESOLUTION,
        narrow_band_range=(-SDF_NARROW_BAND, SDF_NARROW_BAND),
        margin=SDF_MARGIN,
    )
    return mesh


@wp.kernel
def _add_forces(src: wp.array(dtype=wp.spatial_vector), dst: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    dst[i] = dst[i] + src[i]


@wp.kernel
def _set_kinematic_bodies(
    body_q: wp.array(dtype=wp.transform),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    finger_l: int,
    finger_r: int,
    pusher: int,
    jq_start_l: int,
    jq_start_r: int,
    jq_start_p: int,
    jqd_start_l: int,
    jqd_start_r: int,
    jqd_start_p: int,
    params: wp.array(dtype=wp.float32),
):
    # params: [gripper_y, gripper_z, pusher_z]
    y = params[0]
    z = params[1]
    pz = params[2]

    body_q[finger_l] = wp.transform(wp.vec3(0.0, -y, z), wp.quat_identity())
    body_q[finger_r] = wp.transform(wp.vec3(0.0, y, z), wp.quat_identity())
    body_q[pusher] = wp.transform(wp.vec3(0.0, 0.0, pz), wp.quat_identity())

    # Also update joint_q so MuJoCo's solver sees the correct positions
    # Free joint_q layout: [px, py, pz, qx, qy, qz, qw]
    joint_q[jq_start_l + 0] = 0.0
    joint_q[jq_start_l + 1] = -y
    joint_q[jq_start_l + 2] = z
    joint_q[jq_start_l + 3] = 0.0
    joint_q[jq_start_l + 4] = 0.0
    joint_q[jq_start_l + 5] = 0.0
    joint_q[jq_start_l + 6] = 1.0

    joint_q[jq_start_r + 0] = 0.0
    joint_q[jq_start_r + 1] = y
    joint_q[jq_start_r + 2] = z
    joint_q[jq_start_r + 3] = 0.0
    joint_q[jq_start_r + 4] = 0.0
    joint_q[jq_start_r + 5] = 0.0
    joint_q[jq_start_r + 6] = 1.0

    joint_q[jq_start_p + 0] = 0.0
    joint_q[jq_start_p + 1] = 0.0
    joint_q[jq_start_p + 2] = pz
    joint_q[jq_start_p + 3] = 0.0
    joint_q[jq_start_p + 4] = 0.0
    joint_q[jq_start_p + 5] = 0.0
    joint_q[jq_start_p + 6] = 1.0

    # Zero joint velocities for all kinematic bodies
    for i in range(6):
        joint_qd[jqd_start_l + i] = 0.0
        joint_qd[jqd_start_r + i] = 0.0
        joint_qd[jqd_start_p + i] = 0.0


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.sim_time = 0.0

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        solver_type = getattr(args, "solver", None) or "xpbd"
        self.solver_type = solver_type

        # -- phase timing (seconds) ----------------------------------------
        self.t_push_start = 0.05
        self.t_push_end = 1.0
        self.t_grip_start = 1.2
        self.t_grip_closed = 2.0
        self.t_lift_start = 2.2

        # gentle downward push ~2x gravity for the 2x2 brick (~2-3 g)
        self.push_force = -0.05

        # -- generate brick meshes -----------------------------------------
        print("Generating LEGO brick meshes …")

        v_2x4, f_2x4 = make_lego_brick(4, 2)
        mesh_2x4 = _build_mesh_with_sdf(v_2x4, f_2x4, color=(0.8, 0.1, 0.1))

        v_2x2, f_2x2 = make_lego_brick(2, 2)
        mesh_2x2 = _build_mesh_with_sdf(v_2x2, f_2x2, color=(0.2, 0.4, 0.8))

        # -- scene ----------------------------------------------------------
        print("Building scene …")
        builder = newton.ModelBuilder()

        if solver_type == "mujoco":
            contact_ke, contact_kd = 1e4, 2e2
        else:
            contact_ke, contact_kd = 10.0, 5.0

        contact_gap = 0.005 * SCENE_SCALE

        brick_cfg = newton.ModelBuilder.ShapeConfig(
            density=1050.0,
            ke=contact_ke,
            kd=contact_kd,
            mu=0.8,
            margin=BRICK_MARGIN,
            gap=contact_gap,
        )
        gripper_density = 1.0 if solver_type == "mujoco" else 0.0
        gripper_cfg = newton.ModelBuilder.ShapeConfig(
            density=gripper_density,
            ke=contact_ke,
            kd=contact_kd,
            mu=0.8,
            gap=contact_gap,
        )
        ground_cfg = newton.ModelBuilder.ShapeConfig(ke=contact_ke, kd=contact_kd, mu=0.5, gap=contact_gap)
        builder.add_ground_plane(cfg=ground_cfg)

        # 2x4 brick
        self.body_2x4 = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.001 * SCENE_SCALE), wp.quat_identity()),
            label="brick_2x4",
        )
        builder.add_shape_mesh(self.body_2x4, mesh=mesh_2x4, cfg=brick_cfg)

        # 2x2 brick -- start well above the 2x4 studs so there's no initial overlap
        drop_gap = 0.01 * SCENE_SCALE
        self.body_2x2 = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, BODY_HEIGHT + drop_gap), wp.quat_identity()),
            label="brick_2x2",
        )
        builder.add_shape_mesh(self.body_2x2, mesh=mesh_2x2, cfg=brick_cfg)

        # pusher cube (kinematic, slides vertically above the bricks)
        pusher_hx = PITCH * 1.5
        pusher_hy = PITCH * 0.75
        pusher_hz = 0.003 * SCENE_SCALE
        self.pusher_rest_z = (BODY_HEIGHT * 2 + STUD_HEIGHT + 0.015) * SCENE_SCALE
        pusher_density = 1.0 if solver_type == "mujoco" else 0.0
        pusher_cfg = newton.ModelBuilder.ShapeConfig(
            density=pusher_density,
            ke=contact_ke,
            kd=contact_kd,
            mu=0.3,
            gap=contact_gap,
        )
        self.body_pusher = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, self.pusher_rest_z), wp.quat_identity()),
            label="pusher",
        )
        builder.add_shape_box(self.body_pusher, hx=pusher_hx, hy=pusher_hy, hz=pusher_hz, cfg=pusher_cfg)

        # gripper fingers (approach along ±Y, grip the full 2-brick stack)
        brick_hy = PITCH  # brick outer half-width in Y
        finger_hx = 0.025 * SCENE_SCALE
        finger_hy = 0.005 * SCENE_SCALE
        finger_hz = BODY_HEIGHT + STUD_HEIGHT
        self.finger_cz = BODY_HEIGHT * 0.5

        self.gripper_open_y = 0.050 * SCENE_SCALE
        self.gripper_closed_y = brick_hy + finger_hy - 0.0005 * SCENE_SCALE

        self.body_finger_l = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, -self.gripper_open_y, self.finger_cz), wp.quat_identity()),
            label="finger_left",
        )
        builder.add_shape_box(self.body_finger_l, hx=finger_hx, hy=finger_hy, hz=finger_hz, cfg=gripper_cfg)

        self.body_finger_r = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, self.gripper_open_y, self.finger_cz), wp.quat_identity()),
            label="finger_right",
        )
        builder.add_shape_box(self.body_finger_r, hx=finger_hx, hy=finger_hy, hz=finger_hz, cfg=gripper_cfg)

        # -- finalize -------------------------------------------------------
        if solver_type == "vbd":
            builder.color()

        self.model = builder.finalize()
        self.model.rigid_contact_max = 256

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=256,
            broad_phase="nxn",
        )

        if solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,
                nconmax=512,
                njmax=512,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                iterations=15,
                ls_iterations=100,
                impratio=1.0,
            )
        elif solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model,
                iterations=30,
                rigid_avbd_beta=1.0e3,
                rigid_avbd_gamma=0.8,
                rigid_contact_k_start=1.0,
            )
        else:
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=16,
                rigid_contact_relaxation=0.7,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Look up joint_q_start / joint_qd_start for kinematic bodies
        joint_child_np = self.model.joint_child.numpy()
        jq_start_np = self.model.joint_q_start.numpy()
        jqd_start_np = self.model.joint_qd_start.numpy()
        self.jq_start_l = self.jq_start_r = self.jq_start_p = 0
        self.jqd_start_l = self.jqd_start_r = self.jqd_start_p = 0
        for j in range(self.model.joint_count):
            if joint_child_np[j] == self.body_finger_l:
                self.jq_start_l = int(jq_start_np[j])
                self.jqd_start_l = int(jqd_start_np[j])
            elif joint_child_np[j] == self.body_finger_r:
                self.jq_start_r = int(jq_start_np[j])
                self.jqd_start_r = int(jqd_start_np[j])
            elif joint_child_np[j] == self.body_pusher:
                self.jq_start_p = int(jq_start_np[j])
                self.jqd_start_p = int(jqd_start_np[j])

        # GPU-side arrays updated each frame before graph launch
        n_bodies = self.model.body_count
        push_f = np.zeros((n_bodies, 6), dtype=np.float32)
        push_f[self.body_2x2][2] = self.push_force
        self._push_force_base = wp.array(push_f, dtype=wp.spatial_vector)
        self._push_force = wp.zeros(n_bodies, dtype=wp.spatial_vector)

        self._kin_params = wp.array(
            np.array([self.gripper_open_y, self.finger_cz, self.pusher_rest_z], dtype=np.float32),
            dtype=wp.float32,
        )
        self._kin_params_host = wp.array(
            np.array([self.gripper_open_y, self.finger_cz, self.pusher_rest_z], dtype=np.float32),
            dtype=wp.float32,
            device="cpu",
        )

        # Slider state (initialised to open position, pusher high)
        self.gripper_y = self.gripper_open_y
        self.gripper_z = self.finger_cz
        self.pusher_z = self.pusher_rest_z
        self.push_active = False
        self.auto_mode = False

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "picking"):
            self.viewer.picking.pick_stiffness = 5.0
            self.viewer.picking.pick_damping = 1.0
            ps = self.viewer.picking.pick_state.numpy()
            ps[6] = 5.0
            ps[7] = 1.0
            self.viewer.picking.pick_state = wp.array(ps, dtype=float, device=self.model.device)
        cam_dist = 0.12 * SCENE_SCALE
        self.viewer.set_camera(pos=wp.vec3(cam_dist, -cam_dist, cam_dist * 0.6), pitch=-25.0, yaw=135.0)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    # -- simulation ---------------------------------------------------------

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.launch(
                _set_kinematic_bodies,
                dim=1,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.joint_q,
                    self.state_0.joint_qd,
                    self.body_finger_l,
                    self.body_finger_r,
                    self.body_pusher,
                    self.jq_start_l,
                    self.jq_start_r,
                    self.jq_start_p,
                    self.jqd_start_l,
                    self.jqd_start_r,
                    self.jqd_start_p,
                    self._kin_params,
                ],
            )

            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                _add_forces,
                dim=self._push_force.shape[0],
                inputs=[self._push_force, self.state_0.body_f],
            )

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _compute_gripper_yz(self):
        """Compute gripper y and z positions from sim_time (CPU-side)."""
        t = self.sim_time
        if t < self.t_grip_start:
            y = self.gripper_open_y
        elif t < self.t_grip_closed:
            s = (t - self.t_grip_start) / (self.t_grip_closed - self.t_grip_start)
            s = max(0.0, min(1.0, s))
            s = s * s * (3.0 - 2.0 * s)
            y = self.gripper_open_y + (self.gripper_closed_y - self.gripper_open_y) * s
        else:
            y = self.gripper_closed_y

        z_off = max(0.0, t - self.t_lift_start) * 0.03 * SCENE_SCALE
        z = self.finger_cz + z_off
        return float(y), float(z)

    # -- GUI ----------------------------------------------------------------

    def gui(self, ui):
        _, self.auto_mode = ui.checkbox("Auto sequence", self.auto_mode)
        _, self.pusher_z = ui.slider_float("Pusher Z", self.pusher_z, BODY_HEIGHT * 0.5, self.pusher_rest_z)
        if not self.auto_mode:
            _, self.gripper_y = ui.slider_float(
                "Gripper Y", self.gripper_y, self.gripper_closed_y * 0.8, self.gripper_open_y
            )
            _, self.gripper_z = ui.slider_float("Gripper Z", self.gripper_z, 0.0, self.finger_cz + 0.2 * SCENE_SCALE)
            _, self.push_active = ui.checkbox("Push down", self.push_active)

        if self.solver_type == "vbd":
            changed, v = ui.slider_float("VBD beta (log10)", np.log10(self.solver.avbd_beta), 1.0, 8.0)
            if changed:
                self.solver.avbd_beta = 10.0**v
            changed, v = ui.slider_float(
                "VBD k_start (log10)", np.log10(max(self.solver.k_start_body_contact, 1.0)), 0.0, 6.0
            )
            if changed:
                self.solver.k_start_body_contact = 10.0**v
            changed, v = ui.slider_float("VBD gamma", self.solver.avbd_gamma, 0.0, 1.0)
            if changed:
                self.solver.avbd_gamma = v
            changed, v = ui.slider_float("VBD iterations", float(self.solver.iterations), 1.0, 60.0)
            if changed:
                self.solver.iterations = int(v)

    # -- step / render ------------------------------------------------------

    def step(self):
        if self.auto_mode:
            y, z = self._compute_gripper_yz()
            push_on = self.t_push_start < self.sim_time < self.t_push_end
        else:
            y, z = self.gripper_y, self.gripper_z
            push_on = self.push_active

        host = self._kin_params_host.numpy()
        host[0] = y
        host[1] = z
        host[2] = self.pusher_z
        wp.copy(self._kin_params, self._kin_params_host)

        if push_on:
            wp.copy(self._push_force, self._push_force_base)
        else:
            self._push_force.zero_()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        z_2x4 = body_q[self.body_2x4][2]
        z_2x2 = body_q[self.body_2x2][2]

        assert z_2x4 > -0.01 * SCENE_SCALE, f"2x4 brick fell through ground: z={z_2x4:.4f}"
        assert z_2x2 > z_2x4, f"2x2 should be above 2x4: z_2x2={z_2x2:.4f}, z_2x4={z_2x4:.4f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver",
        type=str,
        default="xpbd",
        choices=["xpbd", "mujoco", "vbd"],
        help="Solver type.",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
