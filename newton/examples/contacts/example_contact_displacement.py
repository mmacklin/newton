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
# Scenario: a PD-driven sanding pad rasters over a curved sheet-metal
# panel (a cylindrical arch -- the curved base geometry is exactly what the
# mesh representation buys over a plain heightfield shape). The tool
# orientation and press direction track the local surface normal. Surface
# roughness (Ra, plus Rq), height under the tool, and removed volume are measured
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
PANEL_RADIUS = 0.2  # cylinder radius of the curved panel (axis along x) [m]
GRID_N = 129  # sheet mesh vertices per side (1.6 mm triangles keep the
# faceting sagitta ~1.5 um, below the 4 um asperity scale)
HF_RES = 256  # heightfield resolution (square)
ROUGHNESS_RA = 4.0e-6  # initial Ra (arithmetic mean) roughness [m] (4 um);
# surface-finish specs quote Ra, so the field is normalized to it
FLOOR_FRAC = 0.05  # residual roughness fraction (0.2 um Ra)
VIZ_EXAGGERATION = 2000.0  # vertical exaggeration of the inspection panel
VIZ_OFFSET = wp.vec3(0.0, 0.35, 0.0)  # inspection panel offset from the sheet [m]

PAD_HALF = 0.015  # sanding pad half extents (x, y) [m] (3 cm pad: chord
# sagitta on the R=0.2 panel is ~0.6 mm, so the rigid pad conforms well)
PAD_HALF_Z = 0.005  # sanding pad half thickness [m]
PRESS_FORCE = 10.0  # press force along the surface normal [N]
MAX_VZ = 0.08  # tool vertical speed clamp [m/s] (the sheet mesh is an open
# surface with no thickness; this keeps force-press transients from
# tunneling the pad through it in a single frame)

LENS_RES = 128  # displacement-lens grid resolution
LENS_SIZE = 0.06  # displacement-lens patch size [m]
LENS_EXAGGERATION = 500.0  # default lens height exaggeration
LENS_LIFT = 5.0e-4  # lens offset along the surface normal to avoid z-fighting [m]

ZOOM_PATCH_SIZE = 0.08  # microsurface patch size in zoom mode [m]
ZOOM_EXAGGERATION = 300.0  # height exaggeration in zoom mode (applied to both
# the microsurface and the pad's clearance, so the pad visibly rides the bumps)

RASTER_MARGIN = 0.008  # commanded pad-center distance from the sheet edge [m]
# (the pad lags the PD target by a few mm under friction, so the command
# overshoots slightly to keep the footprint covering the full sheet)
RASTER_STRIPES = 13  # number of raster stripes
K_WEAR = 2.0e-3  # wear rate coefficient [m removed per m slid]
ORBITAL_SPEED = 0.15  # intrinsic abrasive speed of the (orbital) sander [m/s];
# keeps removal going through raster turnarounds where the feed speed
# passes through zero
WEAR_GATE = 1.0e-3  # max contact separation that still causes wear [m]
WEAR_FOOTPRINT_MARGIN = 1.0e-3  # mask margin around the pad extents [m]
WEAR_WINDOW = 65  # wear kernel launch window [texels], covers the rotated footprint


# ---------------------------------------------------------------------------
# GPU functions
# ---------------------------------------------------------------------------


@wp.func
def panel_height(y: float, radius: float, z_offset: float) -> float:
    """Height of the cylindrical-arch panel (axis along x): z = sqrt(R^2 - y^2) + z_offset."""
    return wp.sqrt(radius * radius - y * y) + z_offset


@wp.func
def panel_normal(y: float, radius: float) -> wp.vec3:
    """Outward unit normal of the cylindrical-arch panel at lateral position y."""
    return wp.vec3(0.0, y / radius, wp.sqrt(radius * radius - y * y) / radius)


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


# live-tunable control parameters, read by kernels inside the captured CUDA
# graph and written from the GUI: [press_force, wear_rate, wear_enabled]
CTRL_PRESS = 0
CTRL_WEAR_RATE = 1
CTRL_WEAR_ENABLED = 2


@wp.kernel
def contact_gate_kernel(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[wp.float32],
    contact_margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    sheet_shape: wp.int32,
    tool_body: wp.int32,
    min_separation: wp.array[wp.float32],
):
    """Minimum displaced contact separation between pad and sheet -> gate for wear."""
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
    wp.atomic_min(min_separation, 0, d)


@wp.kernel
def reset_gate_kernel(min_separation: wp.array[wp.float32]):
    min_separation[0] = 1.0e6


@wp.kernel
def apply_wear_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    tool_body: wp.int32,
    X_ws: wp.transform,
    heightfield: wp.array2d[wp.float32],
    min_separation: wp.array[wp.float32],
    ctrl: wp.array[wp.float32],
    gate: float,
    sheet_size: float,
    radius: float,
    z_offset: float,
    pad_half: float,
    footprint_margin: float,
    orbital_speed: float,
    dt: float,
):
    """Archard-style material removal, one thread per texel in a window around
    the pad: while the pad is in contact (min displaced separation below the
    gate), every texel inside the pad footprint (tested in the tool frame, so
    abrasion happens strictly under the pad) is worn by dh = k * |v_slip| * dt.
    Runs entirely on GPU."""
    if min_separation[0] > gate:
        return

    i, j = wp.tid()  # window centered on the pad center uv
    nrow = heightfield.shape[0]
    ncol = heightfield.shape[1]

    X_tool = body_q[tool_body]
    p_c = wp.transform_point(wp.transform_inverse(X_ws), wp.transform_get_translation(X_tool))
    c0 = wp.int32((wp.clamp(p_c[0] / sheet_size + 0.5, 0.0, 1.0)) * wp.float32(ncol - 1))
    r0 = wp.int32((wp.clamp(p_c[1] / sheet_size + 0.5, 0.0, 1.0)) * wp.float32(nrow - 1))

    half_w = 32  # window half extent [texels]; covers the rotated footprint
    rr = r0 + i - half_w
    cc = c0 + j - half_w
    if rr < 0 or rr >= nrow or cc < 0 or cc >= ncol:
        return

    # footprint mask in the tool frame
    du = 1.0 / wp.float32(ncol - 1)
    dv = 1.0 / wp.float32(nrow - 1)
    tx = (wp.float32(cc) * du - 0.5) * sheet_size
    ty = (wp.float32(rr) * dv - 0.5) * sheet_size
    p_texel = wp.transform_point(X_ws, wp.vec3(tx, ty, panel_height(ty, radius, z_offset)))
    p_pad = wp.transform_point(wp.transform_inverse(X_tool), p_texel)
    if wp.abs(p_pad[0]) > pad_half + footprint_margin or wp.abs(p_pad[1]) > pad_half + footprint_margin:
        return

    # tangential slip speed of the pad at the texel; the orbital term is the
    # sander's intrinsic abrasive speed, independent of the feed motion
    n = panel_normal(ty, radius)
    v_lin = wp.spatial_top(body_qd[tool_body])
    v_ang = wp.spatial_bottom(body_qd[tool_body])
    v_c = v_lin + wp.cross(v_ang, p_texel - wp.transform_get_translation(X_tool))
    v_t = v_c - n * wp.dot(n, v_c)
    slip = wp.length(v_t) + orbital_speed

    dh = ctrl[CTRL_WEAR_RATE] * ctrl[CTRL_WEAR_ENABLED] * slip * dt
    heightfield[rr, cc] = heightfield[rr, cc] - dh


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
    ctrl: wp.array[wp.float32],
    radius: float,
    kp: float,
    kd: float,
    align_rate: float,
    align_blend: float,
    max_vz: float,
    dt: float,
):
    """PD raster drive in xy, press force along the local surface normal, and
    velocity-level attitude steering aligning the pad with the curved panel.

    Attitude is steered at the velocity level (blending omega toward a target
    angular rate) rather than with torque PD: the pad's tiny inertia
    (~6e-6 kg m^2) makes any usefully stiff torque PD unstable under explicit
    substep integration.

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
    # (zero-thickness) sheet mesh in a single collision frame. The clamp is
    # asymmetric: XPBD's positional projection re-emits penetration as upward
    # velocity, and an uncapped rebound sustains mm-scale porpoising that
    # buries the um-scale asperity response.
    v = wp.vec3(v[0], v[1], wp.clamp(v[2], -max_vz, 0.25 * max_vz))

    # attitude steering toward the surface-aligned target orientation:
    # rotating the pad +z axis onto the panel normal is a rotation about x
    # by -asin(y/R)
    q_tgt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.asin(tgt[1] / radius))
    q_d = q_tgt * wp.quat_inverse(q)
    sgn = wp.where(q_d[3] < 0.0, -1.0, 1.0)
    rotvec = wp.vec3(q_d[0], q_d[1], q_d[2]) * (2.0 * sgn)
    omega_des = rotvec * align_rate
    ang_speed = wp.length(omega_des)
    if ang_speed > 4.0:
        omega_des = omega_des * (4.0 / ang_speed)
    omega = omega + (omega_des - omega) * align_blend

    body_qd[tool_body] = wp.spatial_vector(v, omega)

    # press along the local surface normal of the curved panel; strong
    # vertical damping removes residual bounce energy
    n = panel_normal(tgt[1], radius)
    f = (
        wp.vec3(
            kp * (tgt[0] - p[0]) - kd * v[0],
            kp * (tgt[1] - p[1]) - kd * v[1],
            -60.0 * v[2],
        )
        - n * ctrl[CTRL_PRESS]
    )

    body_f[tool_body] = body_f[tool_body] + wp.spatial_vector(f, wp.vec3(0.0, 0.0, 0.0))
    time[0] = t + dt


@wp.kernel
def update_surface_points_kernel(
    heightfield: wp.array2d[wp.float32],
    origin: wp.vec3,
    size: float,
    radius: float,
    z_offset: float,
    exaggeration: float,
    points: wp.array[wp.vec3],
):
    """Render-only inspection panel: true curved base + exaggerated asperities
    offset along the local surface normal."""
    i, j = wp.tid()  # i -> rows (v/y), j -> cols (u/x)
    nrow = heightfield.shape[0]
    ncol = heightfield.shape[1]
    x = size * (wp.float32(j) / wp.float32(ncol - 1) - 0.5)
    y = size * (wp.float32(i) / wp.float32(nrow - 1) - 0.5)
    base = wp.vec3(x, y, panel_height(y, radius, z_offset))
    n = panel_normal(y, radius)
    points[i * ncol + j] = origin + base + n * (heightfield[i, j] * exaggeration)


@wp.kernel
def update_lens_kernel(
    heightfield: wp.array2d[wp.float32],
    body_q: wp.array[wp.transform],
    tool_body: wp.int32,
    X_ws: wp.transform,
    sheet_size: float,
    radius: float,
    z_offset: float,
    res: wp.int32,
    lens_size: float,
    exaggeration: float,
    lift: float,
    inv_h_scale: float,
    points: wp.array[wp.vec3],
    uvs_out: wp.array[wp.vec2],
):
    """Displacement lens: a patch following the pad that renders the displaced
    collision surface base(x, y) + n * h(uv) with adjustable exaggeration.
    Vertex UVs index a 1D colormap strip by normalized height."""
    i, j = wp.tid()  # i -> y, j -> x

    p_tool = wp.transform_point(wp.transform_inverse(X_ws), wp.transform_get_translation(body_q[tool_body]))
    half = 0.5 * lens_size
    # bias the window ahead of the raster (+y) so it spans both the freshly
    # sanded trail under the pad and the approaching unsanded asperities
    cx = wp.clamp(p_tool[0], -0.5 * sheet_size + half, 0.5 * sheet_size - half)
    cy = wp.clamp(p_tool[1] + 0.3 * lens_size, -0.5 * sheet_size + half, 0.5 * sheet_size - half)

    x = cx + lens_size * (wp.float32(j) / wp.float32(res - 1) - 0.5)
    y = cy + lens_size * (wp.float32(i) / wp.float32(res - 1) - 0.5)
    h = sample_height_bilinear(heightfield, x / sheet_size + 0.5, y / sheet_size + 0.5)

    base = wp.vec3(x, y, panel_height(y, radius, z_offset))
    n = panel_normal(y, radius)
    points[i * res + j] = wp.transform_point(X_ws, base + n * (h * exaggeration + lift))
    # colormap strip lookup by normalized height (inset to avoid edge bleed)
    u_tex = wp.clamp(h * inv_h_scale, 0.0, 1.0) * 0.99 + 0.005
    uvs_out[i * res + j] = wp.vec2(u_tex, 0.5)


@wp.kernel
def update_pad_proxy_kernel(
    body_q: wp.array[wp.transform],
    tool_body: wp.int32,
    X_ws: wp.transform,
    radius: float,
    exaggeration: float,
    ride_height: float,
    verts_local: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
):
    """Zoom-mode pad proxy: the pad drawn at true size but rigidly lifted by
    the exaggerated clearance, so it rides the exaggerated microsurface
    consistently (the physical clearance is um-scale and invisible)."""
    tid = wp.tid()
    X_tool = body_q[tool_body]
    p_w = wp.transform_point(X_tool, verts_local[tid])
    p_c = wp.transform_point(wp.transform_inverse(X_ws), wp.transform_get_translation(X_tool))
    n = panel_normal(p_c[1], radius)
    lift = (exaggeration - 1.0) * wp.max(ride_height, 0.0)
    points[tid] = p_w + wp.transform_vector(X_ws, n) * lift


@wp.kernel
def heightfield_to_image_kernel(
    heightfield: wp.array2d[wp.float32],
    inv_scale: float,
    image: wp.array2d[wp.float32],
):
    i, j = wp.tid()
    image[i, j] = wp.clamp(heightfield[i, j] * inv_scale, 0.0, 1.0)


@wp.kernel
def measure_kernel(
    heightfield: wp.array2d[wp.float32],
    body_q: wp.array[wp.transform],
    tool_body: wp.int32,
    X_ws: wp.transform,
    sheet_size: float,
    radius: float,
    z_offset: float,
    pad_half: float,
    pad_half_z: float,
    stats: wp.array[wp.float32],  # [sum, sum_sq, max, h_under_tool, sum_abs_dev, ride_height]
):
    i, j = wp.tid()
    h = heightfield[i, j]
    wp.atomic_add(stats, 0, h)
    wp.atomic_add(stats, 1, h * h)
    wp.atomic_max(stats, 2, h)
    if i == 0 and j == 0:
        X_tool = body_q[tool_body]
        X_sw = wp.transform_inverse(X_ws)
        p_local = wp.transform_point(X_sw, wp.transform_get_translation(X_tool))
        u = p_local[0] / sheet_size + 0.5
        v = p_local[1] / sheet_size + 0.5
        stats[3] = sample_height_bilinear(heightfield, u, v)
        # pad ride height: minimum normal clearance of the pad's bottom face
        # above the undisplaced base surface, sampled over the footprint.
        # This is the height of the asperity currently carrying the pad
        # (center-point clearance would instead measure friction-induced
        # tilt, which rises mm-scale via the 15 mm lever arm).
        ride = 1.0e6
        for a in range(7):
            for b in range(7):
                px = (wp.float32(a) / 3.0 - 1.0) * pad_half
                py = (wp.float32(b) / 3.0 - 1.0) * pad_half
                p_pad = wp.transform_point(X_tool, wp.vec3(px, py, -pad_half_z))
                p_s = wp.transform_point(X_sw, p_pad)
                base = wp.vec3(p_s[0], p_s[1], panel_height(p_s[1], radius, z_offset))
                n = panel_normal(p_s[1], radius)
                ride = wp.min(ride, wp.dot(n, p_s - base))
        stats[5] = ride


@wp.kernel
def measure_deviation_kernel(
    heightfield: wp.array2d[wp.float32],
    mean: float,
    stats: wp.array[wp.float32],  # [.., .., .., .., sum_abs_dev, ..]
):
    """Second measurement pass: Ra needs the mean line from the first pass."""
    i, j = wp.tid()
    wp.atomic_add(stats, 4, wp.abs(heightfield[i, j] - mean))


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

        # optional frame-accurate video capture via ViewerGL.get_frame()
        # (includes imgui overlays when a UI is present, i.e. non-headless)
        self.video_path = getattr(args, "capture", None) if args is not None else None
        self._video_writer = None

        # ------- curved sheet mesh (plain GeoType.MESH with UVs) -------
        # cylindrical arch: axis along x, z = sqrt(R^2 - y^2) + z_offset,
        # crown at y = 0 and edges at z = 0. The curved base geometry is
        # what the mesh representation provides over a plain heightfield.
        self.panel_z_offset = -float(np.sqrt(PANEL_RADIUS**2 - (0.5 * SHEET_SIZE) ** 2))
        n = GRID_N
        lin = np.linspace(-0.5 * SHEET_SIZE, 0.5 * SHEET_SIZE, n, dtype=np.float32)
        xx, yy = np.meshgrid(lin, lin, indexing="xy")
        zz = np.sqrt(PANEL_RADIUS**2 - yy**2) + self.panel_z_offset
        vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
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

        # area-weighted smooth vertex normals
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
        # folded gaussian: h >= 0 asperity ridges with a small mean offset,
        # normalized so Ra (mean absolute deviation from the mean line) = 4 um
        folded = np.abs(smooth)
        ra_folded = np.abs(folded - folded.mean()).mean()
        h0 = (folded * (ROUGHNESS_RA / ra_folded)).astype(np.float32)

        self.heightfield = wp.array(h0, dtype=wp.float32)  # 2D fp32 buffer, GPU-writable
        self.heightfield_floor = wp.array(h0 * FLOOR_FRAC, dtype=wp.float32)
        self.h_initial = h0.copy()
        self.initial_sum = float(h0.sum())

        # ---------------- model ----------------
        builder = newton.ModelBuilder()

        sheet_cfg = newton.ModelBuilder.ShapeConfig(mu=0.3)
        self.sheet_shape = builder.add_shape_mesh(body=-1, mesh=sheet_mesh, cfg=sheet_cfg)

        pad_cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.3)
        # start pose: on the curved surface at the raster origin, aligned with
        # the local surface normal
        x_start = -0.5 * SHEET_SIZE + RASTER_MARGIN
        y_start = -0.5 * SHEET_SIZE + RASTER_MARGIN
        n_start = np.array([0.0, y_start / PANEL_RADIUS, np.sqrt(PANEL_RADIUS**2 - y_start**2) / PANEL_RADIUS])
        p_start = np.array(
            [x_start, y_start, np.sqrt(PANEL_RADIUS**2 - y_start**2) + self.panel_z_offset]
        ) + n_start * (PAD_HALF_Z + 1.0e-4)
        q_start = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -float(np.arcsin(y_start / PANEL_RADIUS)))
        self.tool_body = builder.add_body(xform=wp.transform(p=wp.vec3(*p_start), q=q_start))
        builder.add_shape_box(body=self.tool_body, hx=PAD_HALF, hy=PAD_HALF, hz=PAD_HALF_Z, cfg=pad_cfg)

        self.model = builder.finalize()

        # give the tool the mass/inertia of the full sander body rigidly
        # attached to the pad: the bare pad's inertia (~6e-6 kg m^2) makes it
        # spin violently from single contact impulses
        m_tool, i_tool = 1.0, 1.0e-3
        body_mass = self.model.body_mass.numpy()
        body_inertia = self.model.body_inertia.numpy()
        body_mass[self.tool_body] = m_tool
        body_inertia[self.tool_body] = np.eye(3) * i_tool
        self.model.body_mass.assign(body_mass)
        self.model.body_inv_mass.assign(1.0 / body_mass)
        self.model.body_inertia.assign(body_inertia)
        inv_inertia = body_inertia.copy()
        inv_inertia[self.tool_body] = np.eye(3) / i_tool
        self.model.body_inv_inertia.assign(inv_inertia)

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
        self.stats_wp = wp.zeros(6, dtype=wp.float32)
        self.gate_wp = wp.full(1, 1.0e6, dtype=wp.float32)

        # GUI-tunable controls, read live by kernels inside the captured graph
        self.press_force = PRESS_FORCE
        self.wear_rate = K_WEAR
        self.wear_enabled = True
        self.ctrl_wp = wp.array(np.array([PRESS_FORCE, K_WEAR, 1.0], dtype=np.float32), dtype=wp.float32)
        self.viz_exaggeration = VIZ_EXAGGERATION

        # live heightfield image for the viewer (normalized to initial max)
        self.hf_image = wp.zeros((HF_RES, HF_RES), dtype=wp.float32)
        self.hf_image_inv_scale = 1.0 / max(float(h0.max()), 1.0e-9)

        # measurement history (CPU side, observational only)
        self.history = {
            "t": [],
            "ra": [],
            "rms": [],
            "max": [],
            "h_tool": [],
            "ride": [],
            "removed_volume": [],
            "tool_x": [],
            "tool_y": [],
        }
        self.snapshots = []
        self.snapshot_times = []
        self.snapshot_every = max(1, self.num_frames // 40)

        # exaggerated inspection panel (render-only mesh, updated per frame on GPU)
        self.viz_origin = VIZ_OFFSET
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

        # displacement lens: patch following the pad, showing the displaced
        # collision surface (base + n * h) with adjustable exaggeration.
        # In zoom mode it becomes the primary microsurface view; in the
        # default view it is off (toggle in the GUI panel).
        self.zoom_mode = bool(getattr(args, "zoom", False)) if args is not None else False
        if self.zoom_mode:
            self.lens_enabled = True
            self.lens_exaggeration = ZOOM_EXAGGERATION
            self.lens_size = ZOOM_PATCH_SIZE
            self.lens_lift = 0.0
        else:
            self.lens_enabled = False
            self.lens_exaggeration = LENS_EXAGGERATION
            self.lens_size = LENS_SIZE
            self.lens_lift = LENS_LIFT
        self.lens_points = wp.zeros(LENS_RES * LENS_RES, dtype=wp.vec3)
        self.lens_uvs = wp.zeros(LENS_RES * LENS_RES, dtype=wp.vec2)
        li, lj = np.meshgrid(np.arange(LENS_RES - 1), np.arange(LENS_RES - 1), indexing="ij")
        l00 = (li * LENS_RES + lj).ravel()
        l01 = l00 + 1
        l10 = l00 + LENS_RES
        l11 = l10 + 1
        lens_idx = np.concatenate(
            [np.stack([l00, l01, l11], axis=-1), np.stack([l00, l11, l10], axis=-1)], axis=0
        ).astype(np.int32)
        self.lens_indices = wp.array(lens_idx.flatten(), dtype=wp.int32)
        # 1D colormap strip (viridis-like), indexed by normalized height via UVs
        anchors = np.array(
            [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], dtype=np.float32
        )
        t = np.linspace(0.0, 1.0, 256) * (len(anchors) - 1)
        idx = np.minimum(t.astype(int), len(anchors) - 2)
        frac = (t - idx)[:, None]
        strip = anchors[idx] * (1.0 - frac) + anchors[idx + 1] * frac
        self.lens_texture = strip.astype(np.uint8).reshape(1, 256, 3)

        # zoom-mode pad proxy: box drawn per-face (24 verts) for crisp normals
        hx, hy, hz = PAD_HALF, PAD_HALF, PAD_HALF_Z
        corners = np.array(
            [[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float32
        )
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        pverts: list = []
        pidx: list = []
        for q in quads:
            b = len(pverts)
            pverts += [corners[k] for k in q]
            pidx += [b, b + 1, b + 2, b, b + 2, b + 3]
        self.proxy_verts_local = wp.array(np.array(pverts, dtype=np.float32), dtype=wp.vec3)
        self.proxy_points = wp.zeros(len(pverts), dtype=wp.vec3)
        self.proxy_indices = wp.array(np.array(pidx, dtype=np.int32), dtype=wp.int32)

        self.viewer.set_model(self.model)
        if self.zoom_mode:
            # hide the real model shapes: the physical pad sits um above the
            # base surface and would render inside the exaggerated microsurface
            if hasattr(self.viewer, "show_visual"):
                self.viewer.show_visual = False
            self.viewer.set_camera(pos=wp.vec3(-0.07, -0.2, 0.09), pitch=-28.0, yaw=90.0)
        else:
            # frame both the physical sheet and the exaggerated inspection panel;
            # scene sits left of viewport center so the docked image window
            # (which auto-centers) does not occlude the pad's raster path
            self.viewer.set_camera(pos=wp.vec3(0.14, -0.42, 0.3), pitch=-28.0, yaw=90.0)

        self._validate_conventions()
        self.capture()

    # ------------------------------------------------------------------
    def _validate_conventions(self):
        """Assert the barycentric attribute-interpolation convention and the
        bilinear heightfield sampling against CPU references."""
        # query at mesh vertices, offset slightly along the vertex normal: for
        # the convex panel the closest surface point is then the vertex itself,
        # so the interpolated UV has an exact CPU reference
        rng = np.random.default_rng(7)
        vertices = np.asarray(self.model.shape_source[self.sheet_shape].vertices)
        normals = self.sheet_normals.numpy()
        sel = rng.integers(GRID_N + 1, len(vertices) - GRID_N - 1, 16)  # skip boundary
        pts = (vertices[sel] + normals[sel] * 2.0e-4).astype(np.float32)
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

        # CPU bilinear reference at the queried vertices' (projected) uv
        u = vertices[sel, 0] / SHEET_SIZE + 0.5
        v = vertices[sel, 1] / SHEET_SIZE + 0.5
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
        assert h_err < 1.0e-8, f"bilinear sampling mismatch: err={h_err:.3e}"

    # ------------------------------------------------------------------
    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        # Collision runs per substep: at 60 Hz frames the pad slides ~2 mm
        # between contact updates and skips off stale contact planes on the
        # curved panel; per-substep contacts keep it riding the um-scale
        # displaced surface continuously.
        for _ in range(self.sim_substeps):
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
            wp.launch(reset_gate_kernel, dim=1, inputs=[self.gate_wp])
            wp.launch(
                contact_gate_kernel,
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
                    self.sheet_shape,
                    self.tool_body,
                    self.gate_wp,
                ],
            )
            wp.launch(
                apply_wear_kernel,
                dim=(WEAR_WINDOW, WEAR_WINDOW),
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.tool_body,
                    self.X_ws,
                    self.heightfield,
                    self.gate_wp,
                    self.ctrl_wp,
                    WEAR_GATE,
                    SHEET_SIZE,
                    PANEL_RADIUS,
                    self.panel_z_offset,
                    PAD_HALF,
                    WEAR_FOOTPRINT_MARGIN,
                    ORBITAL_SPEED,
                    self.sim_dt,
                ],
            )
            wp.launch(
                clamp_heightfield_kernel,
                dim=(HF_RES, HF_RES),
                inputs=[self.heightfield, self.heightfield_floor],
            )

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
                    self.ctrl_wp,
                    PANEL_RADIUS,
                    3000.0,  # kp
                    40.0,  # kd
                    10.0,  # align_rate [1/s]
                    0.3,  # align_blend per substep
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

        # overlay plots (no-op on non-GL viewers)
        h = self.history
        self.viewer.log_scalar("Ra roughness [um]", h["ra"][-1] * 1.0e6)
        self.viewer.log_scalar("Rq roughness [um]", h["rms"][-1] * 1.0e6)
        self.viewer.log_scalar("Max asperity [um]", h["max"][-1] * 1.0e6)
        self.viewer.log_scalar("Height under tool [um]", h["h_tool"][-1] * 1.0e6)
        self.viewer.log_scalar("Pad ride height [um]", h["ride"][-1] * 1.0e6)
        self.viewer.log_scalar("Removed volume [mm3]", h["removed_volume"][-1] * 1.0e9)

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
                PANEL_RADIUS,
                self.panel_z_offset,
                PAD_HALF,
                PAD_HALF_Z,
                self.stats_wp,
            ],
        )
        s = self.stats_wp.numpy()
        count = HF_RES * HF_RES
        mean = s[0] / count
        # second pass: Ra (arithmetic mean deviation) about the mean line
        wp.launch(
            measure_deviation_kernel,
            dim=(HF_RES, HF_RES),
            inputs=[self.heightfield, float(mean), self.stats_wp],
        )
        ra = float(self.stats_wp.numpy()[4]) / count
        rms = float(np.sqrt(max(s[1] / count - mean * mean, 0.0)))
        texel_area = (SHEET_SIZE / (HF_RES - 1)) ** 2
        removed = (self.initial_sum - float(s[0])) * texel_area

        tool_q = self.state_0.body_q.numpy()[self.tool_body]
        self.history["t"].append(self.sim_time)
        self.history["ra"].append(ra)
        self.history["rms"].append(rms)
        self.history["max"].append(float(s[2]))
        self.history["h_tool"].append(float(s[3]))
        self.history["ride"].append(float(s[5]))
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
        if not self.zoom_mode:
            self.viewer.log_contacts(self.contacts, self.state_0)
            wp.launch(
                update_surface_points_kernel,
                dim=(HF_RES, HF_RES),
                inputs=[
                    self.heightfield,
                    self.viz_origin,
                    SHEET_SIZE,
                    PANEL_RADIUS,
                    self.panel_z_offset,
                    self.viz_exaggeration,
                    self.viz_points,
                ],
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
        else:
            # zoom mode: follow the pad closely from behind the raster
            if self.history["t"]:
                tx = self.history["tool_x"][-1]
                ty = self.history["tool_y"][-1]
                bz = float(np.sqrt(PANEL_RADIUS**2 - min(ty * ty, 0.99 * PANEL_RADIUS**2))) + self.panel_z_offset
                self.viewer.set_camera(
                    pos=wp.vec3(tx, ty - 0.13, bz + 0.065),
                    pitch=-27.0,
                    yaw=90.0,
                )
            # pad proxy riding the exaggerated microsurface; the displayed
            # clearance is smoothed and clamped so stripe-transition spikes
            # don't launch the proxy visually
            ride_raw = self.history["ride"][-1] if self.history["ride"] else 0.0
            self._ride_smooth = 0.8 * getattr(self, "_ride_smooth", 0.0) + 0.2 * min(max(ride_raw, 0.0), 2.5e-5)
            ride = self._ride_smooth
            wp.launch(
                update_pad_proxy_kernel,
                dim=len(self.proxy_points),
                inputs=[
                    self.state_0.body_q,
                    self.tool_body,
                    self.X_ws,
                    PANEL_RADIUS,
                    self.lens_exaggeration,
                    float(ride),
                    self.proxy_verts_local,
                    self.proxy_points,
                ],
            )
            self.viewer.log_mesh(
                "/pad_proxy",
                self.proxy_points,
                self.proxy_indices,
                color=(0.55, 0.8, 0.95),
                roughness=0.35,
                metallic=0.2,
                backface_culling=False,
            )
        # displacement lens: displaced collision surface under the pad
        if self.lens_enabled:
            wp.launch(
                update_lens_kernel,
                dim=(LENS_RES, LENS_RES),
                inputs=[
                    self.heightfield,
                    self.state_0.body_q,
                    self.tool_body,
                    self.X_ws,
                    SHEET_SIZE,
                    PANEL_RADIUS,
                    self.panel_z_offset,
                    LENS_RES,
                    self.lens_size,
                    self.lens_exaggeration,
                    self.lens_lift,
                    self.hf_image_inv_scale,
                    self.lens_points,
                    self.lens_uvs,
                ],
            )
        self.viewer.log_mesh(
            "/displacement_lens",
            self.lens_points,
            self.lens_indices,
            uvs=self.lens_uvs,
            texture=self.lens_texture,
            hidden=not self.lens_enabled,
            backface_culling=False,
            color=(1.0, 1.0, 1.0),
            roughness=0.6,
            metallic=0.0,
        )
        # the GL shader gates the albedo texture on Material.w (texture_enable),
        # which log_mesh does not set -- enable it directly on the mesh object
        lens_obj = getattr(self.viewer, "objects", {}).get("/displacement_lens")
        if lens_obj is not None and hasattr(lens_obj, "material"):
            r, m, c, _t = lens_obj.material
            lens_obj.material = (r, m, c, 1.0)
        # live heightfield image window (GL only; skipped in zoom mode where
        # it would occlude the close-up)
        if not self.zoom_mode:
            wp.launch(
                heightfield_to_image_kernel,
                dim=(HF_RES, HF_RES),
                inputs=[self.heightfield, self.hf_image_inv_scale, self.hf_image],
            )
            self.viewer.log_image("heightfield", self.hf_image)
        self.viewer.end_frame()

        # frame-accurate mp4 capture from the GL framebuffer (PBO readback);
        # render_ui bakes the imgui overlays in when a UI exists
        if self.video_path and hasattr(self.viewer, "get_frame"):
            if self._video_writer is None:
                import imageio.v2 as imageio  # noqa: PLC0415

                self._video_writer = imageio.get_writer(
                    self.video_path, fps=self.fps, codec="libx264", quality=8, macro_block_size=1
                )
            has_ui = getattr(self.viewer, "gui", None) is not None
            frame = self.viewer.get_frame(render_ui=has_ui).numpy()
            # crop to even dimensions for x264/yuv420p
            frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
            self._video_writer.append_data(frame)
            if self.frame_count >= self.num_frames:
                self._video_writer.close()
                self._video_writer = None
                print(f"Video captured to {self.video_path}")
                self.video_path = None

    # ------------------------------------------------------------------
    def gui(self, ui):
        """Custom side-panel (auto-registered by newton.examples.run)."""
        h = self.history
        if h["t"]:
            ui.text(f"Ra:          {h['ra'][-1] * 1e6:7.3f} um")
            ui.text(f"Rq:          {h['rms'][-1] * 1e6:7.3f} um")
            ui.text(f"target Ra:   {ROUGHNESS_RA * FLOOR_FRAC * 1e6:7.3f} um")
            ui.text(f"max asperity:{h['max'][-1] * 1e6:7.3f} um")
            ui.text(f"under tool:  {h['h_tool'][-1] * 1e6:7.3f} um")
            ui.text(f"removed:     {h['removed_volume'][-1] * 1e9:7.2f} mm3")
            progress = min(self.sim_time / self.raster_duration, 1.0)
            ui.text(f"raster:      {100.0 * progress:5.1f} %")
        ui.separator()

        changed = False
        c, self.press_force = ui.slider_float("Press force [N]", self.press_force, 0.0, 30.0)
        changed |= c
        c, self.wear_rate = ui.slider_float("Wear rate [mm/m]", self.wear_rate * 1.0e3, 0.0, 5.0)
        self.wear_rate *= 1.0e-3
        changed |= c
        c, self.wear_enabled = ui.checkbox("Wear enabled", self.wear_enabled)
        changed |= c
        if changed:
            self.ctrl_wp.assign(
                np.array(
                    [self.press_force, self.wear_rate, 1.0 if self.wear_enabled else 0.0],
                    dtype=np.float32,
                )
            )

        _, self.viz_exaggeration = ui.slider_float("Panel exaggeration", self.viz_exaggeration, 100.0, 8000.0)
        ui.separator()
        _, self.lens_enabled = ui.checkbox("Displacement lens", self.lens_enabled)
        _, self.lens_exaggeration = ui.slider_float("Lens exaggeration", self.lens_exaggeration, 1.0, 2000.0)

    # ------------------------------------------------------------------
    def test_final(self):
        ra0 = ROUGHNESS_RA
        ra = self.history["ra"][-1]
        removed = self.history["removed_volume"][-1]
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "NaN/Inf in body transforms"
        assert removed > 0.0, "no material was removed"
        # tool should still be on the sheet
        z = float(body_q[self.tool_body, 2])
        assert -0.01 < z < 0.05, f"tool left the sheet: z={z:.4f}"
        # only expect global convergence if the raster actually finished
        if self.sim_time > self.raster_duration:
            assert ra < 0.5 * ra0, f"roughness did not converge: Ra={ra * 1e6:.3f} um"

    # ------------------------------------------------------------------
    def save_report(self, path):
        import json  # noqa: PLC0415
        import os  # noqa: PLC0415

        os.makedirs(path, exist_ok=True)
        um = 1.0e6
        data = {
            "t": [round(v, 4) for v in self.history["t"]],
            "ra_um": [round(v * um, 5) for v in self.history["ra"]],
            "rms_um": [round(v * um, 5) for v in self.history["rms"]],
            "max_um": [round(v * um, 5) for v in self.history["max"]],
            "h_tool_um": [round(v * um, 5) for v in self.history["h_tool"]],
            "ride_um": [round(v * um, 5) for v in self.history["ride"]],
            "removed_mm3": [round(v * 1e9, 6) for v in self.history["removed_volume"]],
            "tool_x": [round(v, 5) for v in self.history["tool_x"]],
            "tool_y": [round(v, 5) for v in self.history["tool_y"]],
            "target_um": ROUGHNESS_RA * FLOOR_FRAC * um,
            "initial_um": ROUGHNESS_RA * um,
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
 .prose { padding: 20px 28px; line-height: 1.55; max-width: 1100px; }
 .prose h2 { color: #9fc9ff; font-size: 1.15em; margin-top: 1.4em; }
 .prose code { background: #26262c; padding: 1px 5px; border-radius: 4px; font-size: 0.92em; }
 .prose pre { background: #101014; border: 1px solid #2a2a30; border-radius: 6px; padding: 14px;
              overflow-x: auto; font-size: 0.85em; line-height: 1.45; }
 .prose .flow { color: #ffd54f; font-family: monospace; background: #101014; display: block;
                padding: 10px 14px; border-radius: 6px; border: 1px solid #2a2a30; }
</style>
</head>
<body>
<h1>Sanding simulation <small>mesh + GPU heightfield contact displacement</small></h1>
<div class="grid">
 <div class="panel wide" id="video" style="display:none; text-align:center;">
  <video src="sanding.mp4" controls loop muted playsinline style="max-width:100%; border-radius:6px;"
         onloadeddata="document.getElementById('video').style.display='block'"></video>
 </div>
 <div class="panel wide" id="videozoom" style="display:none; text-align:center;">
  <p style="color:#888; margin:4px;">Close-up: pad riding the displaced microsurface
  (heights and pad clearance exaggerated &times;300, run with <code>--zoom</code>)</p>
  <video src="sanding-zoom.mp4" controls loop muted playsinline style="max-width:100%; border-radius:6px;"
         onloadeddata="document.getElementById('videozoom').style.display='block'"></video>
 </div>
 <div class="panel wide prose">
  <h2>Approach: heightfield-displaced contacts on a mesh, with zero engine changes</h2>
  <p>
  The workpiece is a plain <code>GeoType.MESH</code> shape &mdash; a cylindrical-arch sheet-metal panel
  (R&nbsp;=&nbsp;0.2&nbsp;m, 129&sup2; grid, per-vertex UVs) &mdash; and the micro-scale surface state is a
  user-owned <code>wp.array2d[float32]</code> heightfield (256&sup2;, asperity height in meters,
  UV-parameterized). Newton's collision pipeline never sees the heightfield. Instead, the example
  post-processes the <code>Contacts</code> structure between collision detection and the solver step:
  </p>
  <span class="flow">per substep (480 Hz):
  model.collide(state, contacts)
  &rarr; displace_contacts_kernel(contacts, heightfield)   # user code
  &rarr; apply_wear_kernel(contacts, heightfield)           # user code, writes heightfield in place
  &rarr; solver.step(state, contacts)                       # unmodified XPBD</span>
  <p>
  The whole loop is recorded into a single CUDA graph; there is no CPU synchronization anywhere in the
  simulation path. Collision runs <em>per substep</em>: at frame rate the pad slides ~2&nbsp;mm between
  contact updates and skips off stale contact planes on the curved panel, while per-substep contacts keep
  it riding the &micro;m-scale displaced surface continuously (measured pad ride height &asymp; footprint
  peak asperity height, decaying as the surface is sanded).
  </p>

  <h2>Contact displacement kernel</h2>
  <p>
  <code>Contacts.rigid_contact_point0/1</code> are <em>body-frame</em> points, and Newton solvers
  recompute the separation every substep as
  <code>d = dot(n, p1<sub>w</sub> &minus; p0<sub>w</sub>) &minus; (margin0 + margin1)</code>. So shifting the
  sheet-side contact point is sufficient to displace the effective surface &mdash; depth, contact position,
  and friction anchors all follow, for <em>any</em> solver that consumes the pipeline contacts. Per contact:
  </p>
  <pre>face, u, v   = wp.mesh_query_point_no_sign(mesh_id, p_local, max_dist)   # base-mesh BVH
uv           = u*uvs[i0] + v*uvs[i1] + (1-u-v)*uvs[i2]     # barycentric attribute interp
n_smooth     = normalize(u*n[i0] + v*n[i1] + (1-u-v)*n[i2])
h            = bilinear(heightfield, uv)                    # live fp32 buffer, [m]
contact_point += h * n_smooth                               # body frame</pre>
  <p>
  The base-mesh BVH is built once and never refit: displacements (&le;&nbsp;tens of &micro;m) are far below
  the contact-margin band, so candidate generation against the undisplaced mesh is conservative. The
  barycentric convention (<code>u&middot;a0 + v&middot;a1 + (1&minus;u&minus;v)&middot;a2</code>) and the bilinear
  sampler are validated at startup against <code>wp.mesh_eval_position</code> and a CPU reference, with
  queries taken at mesh vertices offset along vertex normals (exact on the convex panel). Because the
  heightfield is sampled at contact-generation time, writing the array from any kernel changes collision
  on the next frame &mdash; no rebuild, no notification, no CPU round trip. One base mesh (one BVH, one
  UV/normal table) can serve thousands of shapes with per-shape heightfields, e.g. RL environments.
  </p>

  <h2>Material removal (sanding)</h2>
  <p>
  An Archard-style wear kernel runs on the same contact list: for each active sheet contact
  (displaced separation below a gate), it removes <code>dh = k&nbsp;&middot;&nbsp;|v<sub>slip</sub>|&nbsp;&middot;&nbsp;dt</code>,
  stamped in UV space with a Gaussian falloff and masked to the pad footprint (each texel is tested in
  the tool frame, so abrasion happens strictly under the pad) via <code>wp.atomic_add</code>, then clamps to a residual
  floor field (5% of the initial asperities &rarr; Ra&nbsp;0.2&nbsp;&micro;m). Slip is the tangential velocity of the
  tool at the contact point. Press force acts along the local surface normal and the pad orientation
  tracks the surface-aligned target (velocity-level steering &mdash; torque PD on the pad's tiny inertia is
  unstable under explicit substepping).
  </p>

  <h2>Measurement and capture</h2>
  <p>
  Roughness is measured per frame outside the captured graph: pass&nbsp;1 accumulates
  <code>&Sigma;h, &Sigma;h&sup2;, max</code> (mean, Rq); pass&nbsp;2 accumulates <code>&Sigma;|h &minus; mean|</code>
  for Ra, the metric finish specs are quoted in. The video is captured frame-accurately from the GL
  framebuffer via <code>ViewerGL.get_frame(render_ui=True)</code> (PBO readback, CUDA&ndash;GL interop),
  which bakes the imgui overlays into the frames. Source:
  <code>newton/examples/contacts/example_contact_displacement.py</code>, branch
  <code>horde/mesh-heightfield-shape</code>.
  </p>
 </div>
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
  { x: D.t, y: D.ra_um, name: "Ra", line: { color: "#4fc3f7" } },
  { x: D.t, y: D.rms_um, name: "Rq (RMS)", line: { color: "#7986cb" } },
  { x: D.t, y: D.max_um, name: "max asperity", line: { color: "#ffb74d" } },
  { x: [D.t[0], D.t[D.t.length-1]], y: [D.target_um, D.target_um], name: "target Ra",
    line: { color: "#81c784", dash: "dash" } }
], { ...dark, title: "Surface roughness [\\u00b5m]", yaxis: { type: "log" }, xaxis: { title: "time [s]" } });

Plotly.newPlot("htool", [
  { x: D.t, y: D.h_tool_um, name: "h under tool", line: { color: "#ce93d8" } },
  { x: D.t, y: D.ride_um, name: "pad ride height", line: { color: "#80cbc4" } }
], { ...dark, title: "Pad response: surface height vs ride height [\\u00b5m]", xaxis: { title: "time [s]" } });

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
    parser.add_argument(
        "--capture",
        type=str,
        default=None,
        help="Capture an mp4 of the run via ViewerGL.get_frame() (requires imageio; "
        "includes the imgui overlays when the viewer is not headless).",
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Close-up mode: follow-cam on the pad riding the exaggerated "
        "microsurface (real shapes hidden; pad drawn with its clearance "
        "exaggerated consistently with the surface).",
    )
    parser.set_defaults(num_frames=900)
    viewer, args = newton.examples.init(parser)

    if args.capture:
        args.capture = os.path.expanduser(args.capture)

    example = Example(viewer, args)
    newton.examples.run(example, args)

    if args.report_path and args.report_path.lower() != "none":
        example.save_report(os.path.expanduser(args.report_path))
