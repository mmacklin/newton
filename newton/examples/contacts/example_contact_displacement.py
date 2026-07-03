# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Contact Displacement
#
# Demonstrates micro-scale surface geometry on a triangle mesh driven by a
# user-owned GPU heightfield, with ZERO engine changes:
#
#   1. The sheet is a plain GeoType.MESH shape (with per-vertex UVs).
#   2. After model.collide(), a user-side displace_contacts() kernel offsets
#      the sheet-side contact points along the interpolated smooth normal by
#      a height sampled from an fp32 heightfield at the contact UV.
#      Solvers recompute contact depth from the (body-frame) contact points
#      every substep, so the displacement is solver-agnostic.
#   3. The heightfield is a plain wp.array2d the user can write on the GPU
#      at any time - here a wear kernel removes material under a sanding
#      pad (Archard-style: removal ~ slip speed * dwell), sanding the sheet
#      from ~4 um down to a ~0.2 um residual roughness.
#
# Scenario: a PD-driven sanding pad rasters over a metal sheet. Surface
# roughness (Rq), height under the tool, and removed volume are measured
# every frame. The micro-scale surface is visualized live on an exaggerated
# inspection panel next to the physical scene, and an interactive HTML
# report (plotly) is written at the end.
#
# Command: uv run -m newton.examples contact_displacement
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

SHEET_SIZE = 0.2  # sheet side length [m]
GRID_N = 65  # sheet mesh vertices per side
HF_RES = 256  # heightfield resolution (square)
ROUGHNESS_RMS = 4.0e-6  # initial RMS roughness [m] (4 um)
FLOOR_FRAC = 0.05  # residual roughness fraction (0.2 um)
VIZ_EXAGGERATION = 2000.0  # vertical exaggeration of the inspection panel
VIZ_OFFSET_X = 0.32  # inspection panel offset from the sheet [m]

PAD_HALF = 0.02  # sanding pad half extents (x, y) [m]
PAD_HALF_Z = 0.005  # sanding pad half thickness [m]
PRESS_FORCE = 10.0  # downward press force [N]
MAX_VZ = 0.05  # tool vertical speed clamp [m/s] (the sheet mesh is an open
# surface with no thickness; this keeps force-press transients from
# tunneling the pad through it in a single frame)

RASTER_MARGIN = 0.022  # keep the pad this far from the sheet edge [m]
RASTER_STRIPES = 11  # number of raster stripes
K_WEAR = 1.2e-3  # wear rate coefficient [m removed per m slid]
WEAR_SIGMA_UV = 0.05  # gaussian stamp sigma in UV space (~pad half width)
WEAR_GATE = 1.0e-3  # max contact separation that still causes wear [m]


# ---------------------------------------------------------------------------
# GPU functions
# ---------------------------------------------------------------------------


@wp.func
def sample_height_bilinear(heightfield: wp.array2d[wp.float32], u: float, v: float) -> float:
    """Bilinearly sample the heightfield over UV in [0, 1]^2 (u -> cols, v -> rows)."""
    nrow = heightfield.shape[0]
    ncol = heightfield.shape[1]
    cf = wp.clamp(u, 0.0, 1.0) * wp.float32(ncol - 1)
    rf = wp.clamp(v, 0.0, 1.0) * wp.float32(nrow - 1)
    c0 = wp.min(wp.int32(cf), ncol - 2)
    r0 = wp.min(wp.int32(rf), nrow - 2)
    fc = cf - wp.float32(c0)
    fr = rf - wp.float32(r0)
    h00 = heightfield[r0, c0]
    h01 = heightfield[r0, c0 + 1]
    h10 = heightfield[r0 + 1, c0]
    h11 = heightfield[r0 + 1, c0 + 1]
    return wp.lerp(wp.lerp(h00, h01, fc), wp.lerp(h10, h11, fc), fr)


@wp.func
def query_sheet_uv_normal(
    mesh_id: wp.uint64,
    uvs: wp.array[wp.vec2],
    normals: wp.array[wp.vec3],
    p_local: wp.vec3,
    max_dist: float,
):
    """Closest-point query on the base mesh -> (hit, uv, smooth normal).

    Warp convention (validated at startup): attributes interpolate as
    u*a[i0] + v*a[i1] + (1-u-v)*a[i2].
    """
    q = wp.mesh_query_point_no_sign(mesh_id, p_local, max_dist)
    uv = wp.vec2(0.0, 0.0)
    n = wp.vec3(0.0, 0.0, 1.0)
    if q.result:
        mesh = wp.mesh_get(mesh_id)
        i0 = mesh.indices[q.face * 3 + 0]
        i1 = mesh.indices[q.face * 3 + 1]
        i2 = mesh.indices[q.face * 3 + 2]
        w2 = 1.0 - q.u - q.v
        uv = q.u * uvs[i0] + q.v * uvs[i1] + w2 * uvs[i2]
        n = wp.normalize(q.u * normals[i0] + q.v * normals[i1] + w2 * normals[i2])
    return q.result, uv, n


@wp.kernel
def displace_contacts_kernel(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    sheet_shape: wp.int32,
    X_ws: wp.transform,
    mesh_id: wp.uint64,
    uvs: wp.array[wp.vec2],
    normals: wp.array[wp.vec3],
    heightfield: wp.array2d[wp.float32],
    max_query_dist: float,
):
    """Offset sheet-side contact points by h(uv) along the smooth surface normal.

    Contact points are stored in the body frame; the sheet is static (body -1)
    so its body frame is the world frame. Solvers recompute contact depth from
    these points every substep, so this correction is solver-agnostic.
    """
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    side = int(-1)
    if contact_shape0[tid] == sheet_shape:
        side = 0
    elif contact_shape1[tid] == sheet_shape:
        side = 1
    else:
        return

    p_w = contact_point0[tid]
    if side == 1:
        p_w = contact_point1[tid]

    p_local = wp.transform_point(wp.transform_inverse(X_ws), p_w)
    hit, uv, n_local = query_sheet_uv_normal(mesh_id, uvs, normals, p_local, max_query_dist)
    if not hit:
        return

    h = sample_height_bilinear(heightfield, uv[0], uv[1])
    disp = wp.transform_vector(X_ws, n_local) * h

    if side == 0:
        contact_point0[tid] = p_w + disp
    else:
        contact_point1[tid] = p_w + disp


@wp.kernel
def apply_wear_kernel(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[wp.float32],
    contact_margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    sheet_shape: wp.int32,
    tool_body: wp.int32,
    X_ws: wp.transform,
    mesh_id: wp.uint64,
    uvs: wp.array[wp.vec2],
    normals: wp.array[wp.vec3],
    heightfield: wp.array2d[wp.float32],
    max_query_dist: float,
    k_wear: float,
    sigma_uv: float,
    stamp_radius: wp.int32,
    gate: float,
    dt: float,
):
    """Archard-style material removal: dh ~ k * slip_speed * dt, stamped with a
    gaussian footprint around each active sheet contact. Runs entirely on GPU."""
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    side = int(-1)
    if contact_shape0[tid] == sheet_shape:
        side = 0
    elif contact_shape1[tid] == sheet_shape:
        side = 1
    else:
        return

    # world positions of both contact points (sheet is static -> body frame == world)
    X_tool = body_q[tool_body]
    if side == 0:
        p_sheet_w = contact_point0[tid]
        p_other_w = wp.transform_point(X_tool, contact_point1[tid])
        d = wp.dot(contact_normal[tid], p_other_w - p_sheet_w)
    else:
        p_sheet_w = contact_point1[tid]
        p_other_w = wp.transform_point(X_tool, contact_point0[tid])
        d = wp.dot(contact_normal[tid], p_sheet_w - p_other_w)
    d = d - (contact_margin0[tid] + contact_margin1[tid])
    if d > gate:
        return

    # tangential slip speed of the tool at the contact
    n = contact_normal[tid]
    v_lin = wp.spatial_top(body_qd[tool_body])
    v_ang = wp.spatial_bottom(body_qd[tool_body])
    r = p_sheet_w - wp.transform_get_translation(X_tool)
    v_c = v_lin + wp.cross(v_ang, r)
    v_t = v_c - n * wp.dot(n, v_c)
    slip = wp.length(v_t)
    if slip < 1.0e-6:
        return

    # contact UV
    p_local = wp.transform_point(wp.transform_inverse(X_ws), p_sheet_w)
    hit, uv, _n_local = query_sheet_uv_normal(mesh_id, uvs, normals, p_local, max_query_dist)
    if not hit:
        return

    nrow = heightfield.shape[0]
    ncol = heightfield.shape[1]
    c0 = wp.int32(wp.clamp(uv[0], 0.0, 1.0) * wp.float32(ncol - 1))
    r0 = wp.int32(wp.clamp(uv[1], 0.0, 1.0) * wp.float32(nrow - 1))

    dh0 = k_wear * slip * dt
    inv_2s2 = 1.0 / (2.0 * sigma_uv * sigma_uv)
    du = 1.0 / wp.float32(ncol - 1)
    dv = 1.0 / wp.float32(nrow - 1)

    for i in range(-stamp_radius, stamp_radius + 1):
        for j in range(-stamp_radius, stamp_radius + 1):
            rr = r0 + i
            cc = c0 + j
            if rr < 0 or rr >= nrow or cc < 0 or cc >= ncol:
                continue
            r2 = (wp.float32(i) * dv) * (wp.float32(i) * dv) + (wp.float32(j) * du) * (wp.float32(j) * du)
            w = wp.exp(-r2 * inv_2s2)
            wp.atomic_add(heightfield, rr, cc, -dh0 * w)


@wp.kernel
def clamp_heightfield_kernel(
    heightfield: wp.array2d[wp.float32],
    floor: wp.array2d[wp.float32],
):
    i, j = wp.tid()
    heightfield[i, j] = wp.max(heightfield[i, j], floor[i, j])


@wp.func
def raster_target(t: float, x0: float, x1: float, y0: float, dy: float, n_stripes: int, speed: float) -> wp.vec2:
    """Boustrophedon raster path over the sheet."""
    stripe_len = x1 - x0
    t_stripe = stripe_len / speed
    s = t / t_stripe
    row = wp.int32(wp.floor(s))
    frac = s - wp.float32(row)
    if row >= n_stripes:
        row = n_stripes - 1
        frac = 1.0
    x = x0 + frac * stripe_len
    if row % 2 == 1:
        x = x1 - frac * stripe_len
    y = y0 + wp.float32(row) * dy
    return wp.vec2(x, y)


@wp.kernel
def drive_tool_kernel(
    time: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    tool_body: wp.int32,
    x0: float,
    x1: float,
    y0: float,
    dy: float,
    n_stripes: wp.int32,
    speed: float,
    press_force: float,
    kp: float,
    kd: float,
    kp_ang: float,
    kd_ang: float,
    max_vz: float,
    dt: float,
):
    """PD raster drive in xy, constant press force in -z, angular PD to stay level.

    Runs single-threaded per substep; also advances the GPU clock (graph-capture safe).
    """
    t = time[0]
    tgt = raster_target(t, x0, x1, y0, dy, n_stripes, speed)

    X = body_q[tool_body]
    p = wp.transform_get_translation(X)
    q = wp.transform_get_rotation(X)
    v = wp.spatial_top(body_qd[tool_body])
    omega = wp.spatial_bottom(body_qd[tool_body])

    # clamp vertical speed so press transients cannot tunnel through the
    # (zero-thickness) sheet mesh within a single collision frame
    if wp.abs(v[2]) > max_vz:
        v = wp.vec3(v[0], v[1], wp.clamp(v[2], -max_vz, max_vz))
        body_qd[tool_body] = wp.spatial_vector(v, omega)

    f = wp.vec3(
        kp * (tgt[0] - p[0]) - kd * v[0],
        kp * (tgt[1] - p[1]) - kd * v[1],
        -press_force,
    )

    # small-angle attitude PD toward identity: rotvec ~ 2 * sign(qw) * (qx, qy, qz)
    sgn = wp.where(q[3] < 0.0, -1.0, 1.0)
    tau = wp.vec3(q[0], q[1], q[2]) * (-2.0 * kp_ang * sgn) - omega * kd_ang

    body_f[tool_body] = body_f[tool_body] + wp.spatial_vector(f, tau)
    time[0] = t + dt


@wp.kernel
def update_surface_points_kernel(
    heightfield: wp.array2d[wp.float32],
    origin: wp.vec3,
    size: float,
    exaggeration: float,
    points: wp.array[wp.vec3],
):
    i, j = wp.tid()  # i -> rows (v/y), j -> cols (u/x)
    nrow = heightfield.shape[0]
    ncol = heightfield.shape[1]
    x = origin[0] + size * (wp.float32(j) / wp.float32(ncol - 1) - 0.5)
    y = origin[1] + size * (wp.float32(i) / wp.float32(nrow - 1) - 0.5)
    z = origin[2] + heightfield[i, j] * exaggeration
    points[i * ncol + j] = wp.vec3(x, y, z)


@wp.kernel
def measure_kernel(
    heightfield: wp.array2d[wp.float32],
    body_q: wp.array[wp.transform],
    tool_body: wp.int32,
    X_ws: wp.transform,
    sheet_size: float,
    stats: wp.array[wp.float32],  # [sum, sum_sq, max, h_under_tool]
):
    i, j = wp.tid()
    h = heightfield[i, j]
    wp.atomic_add(stats, 0, h)
    wp.atomic_add(stats, 1, h * h)
    wp.atomic_max(stats, 2, h)
    if i == 0 and j == 0:
        p_tool = wp.transform_get_translation(body_q[tool_body])
        p_local = wp.transform_point(wp.transform_inverse(X_ws), p_tool)
        u = p_local[0] / sheet_size + 0.5
        v = p_local[1] / sheet_size + 0.5
        stats[3] = sample_height_bilinear(heightfield, u, v)


@wp.kernel
def validate_interpolation_kernel(
    mesh_id: wp.uint64,
    uvs: wp.array[wp.vec2],
    normals: wp.array[wp.vec3],
    heightfield: wp.array2d[wp.float32],
    query_points: wp.array[wp.vec3],
    out_pos_err: wp.array[wp.float32],
    out_h: wp.array[wp.float32],
):
    """Validate the barycentric convention (u*a0 + v*a1 + (1-u-v)*a2) against
    wp.mesh_eval_position, and sample h(uv) for a CPU-side bilinear cross-check."""
    tid = wp.tid()
    p = query_points[tid]
    q = wp.mesh_query_point_no_sign(mesh_id, p, 1.0)
    if not q.result:
        out_pos_err[tid] = 1.0e10
        return
    mesh = wp.mesh_get(mesh_id)
    i0 = mesh.indices[q.face * 3 + 0]
    i1 = mesh.indices[q.face * 3 + 1]
    i2 = mesh.indices[q.face * 3 + 2]
    w2 = 1.0 - q.u - q.v
    p_interp = q.u * mesh.points[i0] + q.v * mesh.points[i1] + w2 * mesh.points[i2]
    p_ref = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    out_pos_err[tid] = wp.length(p_interp - p_ref)
    uv = q.u * uvs[i0] + q.v * uvs[i1] + w2 * uvs[i2]
    out_h[tid] = sample_height_bilinear(heightfield, uv[0], uv[1])


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame_count = 0
        self.num_frames = getattr(args, "num_frames", 900) if args is not None else 900

        self.viewer = viewer

        # ---------------- sheet mesh (plain GeoType.MESH with UVs) ----------------
        n = GRID_N
        lin = np.linspace(-0.5 * SHEET_SIZE, 0.5 * SHEET_SIZE, n, dtype=np.float32)
        xx, yy = np.meshgrid(lin, lin, indexing="xy")
        vertices = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3)
        uvs = np.stack([xx / SHEET_SIZE + 0.5, yy / SHEET_SIZE + 0.5], axis=-1).reshape(-1, 2).astype(np.float32)

        quads_i, quads_j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
        v00 = (quads_i * n + quads_j).ravel()
        v01 = v00 + 1
        v10 = v00 + n
        v11 = v10 + 1
        # CCW when viewed from +z
        indices = np.concatenate(
            [np.stack([v00, v01, v11], axis=-1), np.stack([v00, v11, v10], axis=-1)], axis=0
        ).astype(np.int32)

        sheet_mesh = newton.Mesh(vertices, indices.flatten(), uvs=uvs, compute_inertia=False)

        # area-weighted smooth vertex normals (trivially +z here; computed for generality)
        normals = np.zeros_like(vertices)
        tri = vertices[indices]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        for k in range(3):
            np.add.at(normals, indices[:, k], fn)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        self.sheet_uvs = wp.array(uvs, dtype=wp.vec2)
        self.sheet_normals = wp.array(normals.astype(np.float32), dtype=wp.vec3)

        # ---------------- heightfield: user-owned fp32 GPU buffer ----------------
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((HF_RES, HF_RES)).astype(np.float32)
        # band-limit with a gaussian low-pass in frequency space
        fy = np.fft.fftfreq(HF_RES)[:, None]
        fx = np.fft.fftfreq(HF_RES)[None, :]
        sigma_f = 0.03  # cycles/texel; larger -> finer asperities
        lp = np.exp(-(fx**2 + fy**2) / (2.0 * sigma_f**2))
        smooth = np.real(np.fft.ifft2(np.fft.fft2(noise) * lp))
        smooth -= smooth.mean()
        # folded gaussian: h >= 0 asperity ridges with a small mean offset
        folded = np.abs(smooth)
        h0 = (folded * (ROUGHNESS_RMS / folded.std())).astype(np.float32)  # RMS about mean = 4 um

        self.heightfield = wp.array(h0, dtype=wp.float32)  # 2D fp32 buffer, GPU-writable
        self.heightfield_floor = wp.array(h0 * FLOOR_FRAC, dtype=wp.float32)
        self.h_initial = h0.copy()
        self.initial_sum = float(h0.sum())

        # ---------------- model ----------------
        builder = newton.ModelBuilder()

        sheet_cfg = newton.ModelBuilder.ShapeConfig(mu=0.3)
        self.sheet_shape = builder.add_shape_mesh(body=-1, mesh=sheet_mesh, cfg=sheet_cfg)

        pad_cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.3)
        start = wp.vec3(-0.5 * SHEET_SIZE + RASTER_MARGIN, -0.5 * SHEET_SIZE + RASTER_MARGIN, PAD_HALF_Z + 1.0e-4)
        self.tool_body = builder.add_body(xform=wp.transform(p=start, q=wp.quat_identity()))
        builder.add_shape_box(body=self.tool_body, hx=PAD_HALF, hy=PAD_HALF, hz=PAD_HALF_Z, cfg=pad_cfg)

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
        self.contacts = self.model.contacts()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # sheet shape -> world transform (static body: body frame == world frame)
        self.X_ws = wp.transform(*self.model.shape_transform.numpy()[self.sheet_shape])
        self.sheet_mesh_id = wp.uint64(int(self.model.shape_source_ptr.numpy()[self.sheet_shape]))
        self.max_query_dist = 0.01

        # raster parameters
        x0 = -0.5 * SHEET_SIZE + RASTER_MARGIN
        x1 = 0.5 * SHEET_SIZE - RASTER_MARGIN
        stripe_len = x1 - x0
        total_time = self.num_frames * self.frame_dt
        # finish the raster ~1 s before the end, but cap the speed so the PD
        # drive can track (short runs then simply cover less of the sheet)
        speed = min(0.25, RASTER_STRIPES * stripe_len / max(total_time - 1.0, 0.5))
        self.raster = {
            "x0": x0,
            "x1": x1,
            "y0": x0,
            "dy": (x1 - x0) / (RASTER_STRIPES - 1),
            "n_stripes": RASTER_STRIPES,
            "speed": speed,
        }
        self.raster_duration = RASTER_STRIPES * stripe_len / speed

        self.time_wp = wp.zeros(1, dtype=wp.float32)
        self.stats_wp = wp.zeros(4, dtype=wp.float32)
        self.wear_stamp_radius = int(2.0 * WEAR_SIGMA_UV * (HF_RES - 1))

        # measurement history (CPU side, observational only)
        self.history = {
            "t": [],
            "rms": [],
            "max": [],
            "h_tool": [],
            "removed_volume": [],
            "tool_x": [],
            "tool_y": [],
        }
        self.snapshots = []
        self.snapshot_times = []
        self.snapshot_every = max(1, self.num_frames // 40)

        # exaggerated inspection panel (render-only mesh, updated per frame on GPU)
        self.viz_origin = wp.vec3(VIZ_OFFSET_X, 0.0, 0.0)
        self.viz_points = wp.zeros(HF_RES * HF_RES, dtype=wp.vec3)
        viz_i, viz_j = np.meshgrid(np.arange(HF_RES - 1), np.arange(HF_RES - 1), indexing="ij")
        w00 = (viz_i * HF_RES + viz_j).ravel()
        w01 = w00 + 1
        w10 = w00 + HF_RES
        w11 = w10 + 1
        viz_idx = np.concatenate(
            [np.stack([w00, w01, w11], axis=-1), np.stack([w00, w11, w10], axis=-1)], axis=0
        ).astype(np.int32)
        self.viz_indices = wp.array(viz_idx.flatten(), dtype=wp.int32)

        self.viewer.set_model(self.model)
        # frame both the physical sheet and the exaggerated inspection panel
        self.viewer.set_camera(pos=wp.vec3(0.16, -0.42, 0.28), pitch=-32.0, yaw=90.0)

        self._validate_conventions()
        self.capture()

    # ------------------------------------------------------------------
    def _validate_conventions(self):
        """Assert the barycentric attribute-interpolation convention and the
        bilinear heightfield sampling against CPU references."""
        rng = np.random.default_rng(7)
        pts = np.zeros((16, 3), dtype=np.float32)
        pts[:, 0] = rng.uniform(-0.4, 0.4, 16) * SHEET_SIZE
        pts[:, 1] = rng.uniform(-0.4, 0.4, 16) * SHEET_SIZE
        pts[:, 2] = rng.uniform(0.0, 0.001, 16)
        query_points = wp.array(pts, dtype=wp.vec3)
        pos_err = wp.zeros(16, dtype=wp.float32)
        h_gpu = wp.zeros(16, dtype=wp.float32)
        wp.launch(
            validate_interpolation_kernel,
            dim=16,
            inputs=[
                self.sheet_mesh_id,
                self.sheet_uvs,
                self.sheet_normals,
                self.heightfield,
                query_points,
                pos_err,
                h_gpu,
            ],
        )
        max_err = float(pos_err.numpy().max())
        assert max_err < 1.0e-5, f"barycentric convention mismatch: err={max_err:.3e}"

        # CPU bilinear reference at the (flat sheet) query xy -> uv
        u = pts[:, 0] / SHEET_SIZE + 0.5
        v = pts[:, 1] / SHEET_SIZE + 0.5
        cf = np.clip(u, 0, 1) * (HF_RES - 1)
        rf = np.clip(v, 0, 1) * (HF_RES - 1)
        c0 = np.minimum(cf.astype(int), HF_RES - 2)
        r0 = np.minimum(rf.astype(int), HF_RES - 2)
        fc, fr = cf - c0, rf - r0
        h = self.h_initial
        h_ref = (
            h[r0, c0] * (1 - fc) * (1 - fr)
            + h[r0, c0 + 1] * fc * (1 - fr)
            + h[r0 + 1, c0] * (1 - fc) * fr
            + h[r0 + 1, c0 + 1] * fc * fr
        )
        h_err = float(np.abs(h_gpu.numpy() - h_ref).max())
        assert h_err < 1.0e-9, f"bilinear sampling mismatch: err={h_err:.3e}"

    # ------------------------------------------------------------------
    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)

        # --- user-side contact displacement: sample the live GPU heightfield ---
        wp.launch(
            displace_contacts_kernel,
            dim=self.contacts.rigid_contact_max,
            inputs=[
                self.contacts.rigid_contact_count,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                self.contacts.rigid_contact_point0,
                self.contacts.rigid_contact_point1,
                self.sheet_shape,
                self.X_ws,
                self.sheet_mesh_id,
                self.sheet_uvs,
                self.sheet_normals,
                self.heightfield,
                self.max_query_dist,
            ],
        )

        # --- material removal under the pad (writes the heightfield in place) ---
        wp.launch(
            apply_wear_kernel,
            dim=self.contacts.rigid_contact_max,
            inputs=[
                self.contacts.rigid_contact_count,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                self.contacts.rigid_contact_point0,
                self.contacts.rigid_contact_point1,
                self.contacts.rigid_contact_normal,
                self.contacts.rigid_contact_margin0,
                self.contacts.rigid_contact_margin1,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.sheet_shape,
                self.tool_body,
                self.X_ws,
                self.sheet_mesh_id,
                self.sheet_uvs,
                self.sheet_normals,
                self.heightfield,
                self.max_query_dist,
                K_WEAR,
                WEAR_SIGMA_UV,
                self.wear_stamp_radius,
                WEAR_GATE,
                self.frame_dt,
            ],
        )
        wp.launch(
            clamp_heightfield_kernel,
            dim=(HF_RES, HF_RES),
            inputs=[self.heightfield, self.heightfield_floor],
        )

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                drive_tool_kernel,
                dim=1,
                inputs=[
                    self.time_wp,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.tool_body,
                    self.raster["x0"],
                    self.raster["x1"],
                    self.raster["y0"],
                    self.raster["dy"],
                    self.raster["n_stripes"],
                    self.raster["speed"],
                    PRESS_FORCE,
                    3000.0,  # kp
                    40.0,  # kd
                    5.0,  # kp_ang
                    0.1,  # kd_ang
                    MAX_VZ,
                    self.sim_dt,
                ],
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame_count += 1
        self._measure()

    # ------------------------------------------------------------------
    def _measure(self):
        """Per-frame observational readback (kept out of the captured graph)."""
        self.stats_wp.zero_()
        wp.launch(
            measure_kernel,
            dim=(HF_RES, HF_RES),
            inputs=[
                self.heightfield,
                self.state_0.body_q,
                self.tool_body,
                self.X_ws,
                SHEET_SIZE,
                self.stats_wp,
            ],
        )
        s = self.stats_wp.numpy()
        count = HF_RES * HF_RES
        mean = s[0] / count
        rms = float(np.sqrt(max(s[1] / count - mean * mean, 0.0)))
        texel_area = (SHEET_SIZE / (HF_RES - 1)) ** 2
        removed = (self.initial_sum - float(s[0])) * texel_area

        tool_q = self.state_0.body_q.numpy()[self.tool_body]
        self.history["t"].append(self.sim_time)
        self.history["rms"].append(rms)
        self.history["max"].append(float(s[2]))
        self.history["h_tool"].append(float(s[3]))
        self.history["removed_volume"].append(removed)
        self.history["tool_x"].append(float(tool_q[0]))
        self.history["tool_y"].append(float(tool_q[1]))

        if self.frame_count % self.snapshot_every == 0 or self.frame_count == self.num_frames:
            self.snapshots.append(self.heightfield.numpy()[::4, ::4].copy())
            self.snapshot_times.append(self.sim_time)

    # ------------------------------------------------------------------
    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        wp.launch(
            update_surface_points_kernel,
            dim=(HF_RES, HF_RES),
            inputs=[self.heightfield, self.viz_origin, SHEET_SIZE, VIZ_EXAGGERATION, self.viz_points],
        )
        self.viewer.log_mesh(
            "/surface_inspection",
            self.viz_points,
            self.viz_indices,
            color=(0.72, 0.75, 0.79),
            roughness=0.35,
            metallic=0.9,
            backface_culling=False,
        )
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    def test_final(self):
        rms0 = ROUGHNESS_RMS
        rms = self.history["rms"][-1]
        removed = self.history["removed_volume"][-1]
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "NaN/Inf in body transforms"
        assert removed > 0.0, "no material was removed"
        # tool should still be on the sheet
        z = float(body_q[self.tool_body, 2])
        assert -0.01 < z < 0.05, f"tool left the sheet: z={z:.4f}"
        # only expect global convergence if the raster actually finished
        if self.sim_time > self.raster_duration:
            assert rms < 0.5 * rms0, f"roughness did not converge: rms={rms * 1e6:.3f} um"

    # ------------------------------------------------------------------
    def save_report(self, path):
        import json  # noqa: PLC0415
        import os  # noqa: PLC0415

        os.makedirs(path, exist_ok=True)
        um = 1.0e6
        data = {
            "t": [round(v, 4) for v in self.history["t"]],
            "rms_um": [round(v * um, 5) for v in self.history["rms"]],
            "max_um": [round(v * um, 5) for v in self.history["max"]],
            "h_tool_um": [round(v * um, 5) for v in self.history["h_tool"]],
            "removed_mm3": [round(v * 1e9, 6) for v in self.history["removed_volume"]],
            "tool_x": [round(v, 5) for v in self.history["tool_x"]],
            "tool_y": [round(v, 5) for v in self.history["tool_y"]],
            "target_um": ROUGHNESS_RMS * FLOOR_FRAC * um,
            "initial_um": ROUGHNESS_RMS * um,
            "sheet_size": SHEET_SIZE,
            "snap_t": [round(v, 3) for v in self.snapshot_times],
            "snapshots_um": [np.round(s * um, 4).tolist() for s in self.snapshots],
            "surface_initial_um": np.round(self.h_initial[::2, ::2] * um, 4).tolist(),
            "surface_final_um": np.round(self.heightfield.numpy()[::2, ::2] * um, 4).tolist(),
        }
        html = _REPORT_TEMPLATE.replace("__DATA__", json.dumps(data))
        out = os.path.join(path, "index.html")
        with open(out, "w") as f:
            f.write(html)
        print(f"Report written to {out}")


_REPORT_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sanding Simulation — Contact Displacement</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
 body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 24px; background: #111; color: #ddd; }
 h1 { font-weight: 600; } h1 small { color: #888; font-weight: 400; font-size: 60%; }
 .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
 .panel { background: #1a1a1e; border-radius: 8px; padding: 8px; }
 .wide { grid-column: 1 / -1; }
</style>
</head>
<body>
<h1>Sanding simulation <small>mesh + GPU heightfield contact displacement</small></h1>
<div class="grid">
 <div class="panel" id="rms"></div>
 <div class="panel" id="htool"></div>
 <div class="panel" id="removed"></div>
 <div class="panel" id="path"></div>
 <div class="panel wide" id="slider"></div>
 <div class="panel" id="surf0"></div>
 <div class="panel" id="surf1"></div>
</div>
<script>
const D = __DATA__;
const dark = { paper_bgcolor: "#1a1a1e", plot_bgcolor: "#1a1a1e", font: { color: "#ccc" }, margin: { t: 40 } };

Plotly.newPlot("rms", [
  { x: D.t, y: D.rms_um, name: "Rq (RMS)", line: { color: "#4fc3f7" } },
  { x: D.t, y: D.max_um, name: "max asperity", line: { color: "#ffb74d" } },
  { x: [D.t[0], D.t[D.t.length-1]], y: [D.target_um, D.target_um], name: "target",
    line: { color: "#81c784", dash: "dash" } }
], { ...dark, title: "Surface roughness [\\u00b5m]", yaxis: { type: "log" }, xaxis: { title: "time [s]" } });

Plotly.newPlot("htool", [
  { x: D.t, y: D.h_tool_um, name: "h under tool", line: { color: "#ce93d8" } }
], { ...dark, title: "Surface height under tool [\\u00b5m]", xaxis: { title: "time [s]" } });

Plotly.newPlot("removed", [
  { x: D.t, y: D.removed_mm3, line: { color: "#e57373" } }
], { ...dark, title: "Removed volume [mm\\u00b3]", xaxis: { title: "time [s]" } });

Plotly.newPlot("path", [
  { x: D.tool_x, y: D.tool_y, mode: "lines", line: { color: "#4dd0e1", width: 1 } }
], { ...dark, title: "Tool raster path [m]", xaxis: { scaleanchor: "y" } });

const frames = D.snapshots_um.map((z, i) => ({ name: String(i), data: [{ z }] }));
Plotly.newPlot("slider", [{ z: D.snapshots_um[0], type: "heatmap", colorscale: "Viridis",
  zmin: 0, zmax: D.initial_um * 3, colorbar: { title: "\\u00b5m" } }],
 { ...dark, title: "Heightfield over time [\\u00b5m]",
   sliders: [{ steps: D.snap_t.map((t, i) => ({ label: t.toFixed(1) + "s", method: "animate",
     args: [[String(i)], { mode: "immediate", frame: { duration: 0, redraw: true } }] })) }] },
).then(() => Plotly.addFrames("slider", frames));

function surf(div, z, title) {
  Plotly.newPlot(div, [{ z, type: "surface", colorscale: "Viridis", showscale: false }],
    { ...dark, title, scene: { aspectratio: { x: 1, y: 1, z: 0.25 },
      zaxis: { range: [0, D.initial_um * 4], title: "\\u00b5m" },
      bgcolor: "#1a1a1e" } });
}
surf("surf0", D.surface_initial_um, "Initial surface (z in \\u00b5m)");
surf("surf1", D.surface_final_um, "Final surface (z in \\u00b5m)");
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import os

    parser = newton.examples.create_parser()
    parser.add_argument(
        "--report-path",
        type=str,
        default="~/reports/sanding-sim",
        help="Directory for the HTML report ('none' to disable).",
    )
    parser.set_defaults(num_frames=900)
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)
    newton.examples.run(example, args)

    if args.report_path and args.report_path.lower() != "none":
        example.save_report(os.path.expanduser(args.report_path))
