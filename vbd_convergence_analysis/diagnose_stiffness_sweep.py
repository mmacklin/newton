#!/usr/bin/env python3
"""Quick sweep: vary ke/kf and min_mass to find where VBD converges.

For each (ke, min_mass) combo, runs 100 frames then 1 substep with 20 iterations
and reports convergence behavior.
"""
import sys, os, time
import numpy as np

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

import warp as wp
wp.init()

import newton
from newton import ModelBuilder
from newton.solvers import SolverVBD
from vbd_convergence_analysis.analysis_common import (
    DENSITY, EDGE_KD, EDGE_KE, FRAME_DT, FRICTION_EPSILON, GRAVITY,
    LATERAL_OFFSET_RANGE, LOW_DROP_HEIGHT_RANGE, PARTICLE_RADIUS,
    SELF_CONTACT_MARGIN, SELF_CONTACT_RADIUS, SELF_CONTACT_REST_EXCLUSION_RADIUS,
    SOFT_CONTACT_MARGIN, SIM_SUBSTEPS, TRI_KA, TRI_KD, TRI_KE,
    load_shirt_mesh_vertices,
)


def run_test(ke, min_mass, seed=42, warmup_frames=100, test_iters=20):
    rng = np.random.default_rng(seed)
    drop_height = rng.uniform(*LOW_DROP_HEIGHT_RANGE)
    rot_angles = rng.uniform(-np.pi, np.pi, size=3)
    lateral_offset = rng.uniform(*LATERAL_OFFSET_RANGE, size=2)

    scene = ModelBuilder(gravity=GRAVITY)
    vertices, indices, _ = load_shirt_mesh_vertices()

    qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(rot_angles[0]))
    qy = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(rot_angles[1]))
    qz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rot_angles[2]))
    rot = wp.mul(qz, wp.mul(qy, qx))

    verts_np = np.array([[v[0], v[1], v[2]] for v in vertices], dtype=np.float32)
    from scipy.spatial.transform import Rotation as R
    r = R.from_quat([rot[0], rot[1], rot[2], rot[3]])
    rotated = r.apply(verts_np)
    z_offset = drop_height - float(rotated[:, 2].min())
    pos = wp.vec3(float(lateral_offset[0]), float(lateral_offset[1]), float(z_offset))

    scene.add_cloth_mesh(
        vertices=vertices, indices=indices,
        rot=rot, pos=pos, vel=wp.vec3(0.0, 0.0, 0.0),
        density=DENSITY, scale=1.0,
        tri_ke=TRI_KE, tri_ka=TRI_KA, tri_kd=TRI_KD,
        edge_ke=EDGE_KE, edge_kd=EDGE_KD,
        particle_radius=PARTICLE_RADIUS,
    )
    scene.color()
    scene.add_ground_plane()
    model = scene.finalize(requires_grad=False)

    model.soft_contact_ke = ke
    model.soft_contact_kd = 1e-2
    model.soft_contact_kf = ke  # kf = ke
    model.soft_contact_mu = 0.25

    # Clamp masses
    mass_np = model.particle_mass.numpy()
    below = mass_np < min_mass
    n_clamped = below.sum()
    if below.any():
        mass_np[below] = min_mass
        model.particle_mass.assign(wp.array(mass_np, dtype=wp.float32))
        inv_mass_np = np.where(mass_np > 0, 1.0 / mass_np, 0.0)
        model.particle_inv_mass.assign(wp.array(inv_mass_np, dtype=wp.float32))

    solver = SolverVBD(
        model, iterations=10,
        particle_tri_material_model="neohookean",
        particle_enable_self_contact=True,
        particle_self_contact_radius=SELF_CONTACT_RADIUS,
        particle_self_contact_margin=SELF_CONTACT_MARGIN,
        particle_topological_contact_filter_threshold=1,
        particle_rest_shape_contact_exclusion_radius=SELF_CONTACT_REST_EXCLUSION_RADIUS,
        particle_vertex_contact_buffer_size=512,
        particle_edge_contact_buffer_size=512,
        particle_collision_detection_interval=-1,
        friction_epsilon=FRICTION_EPSILON,
    )
    solver.avbd_beta = 0.0
    solver.k_start_body_contact = 2500.0

    s0, s1 = model.state(), model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=SOFT_CONTACT_MARGIN)
    contacts = pipeline.contacts()
    dt = FRAME_DT / SIM_SUBSTEPS

    # Warmup
    for _ in range(warmup_frames):
        for _ in range(SIM_SUBSTEPS):
            s0.clear_forces(); s1.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0

    pos_np = s0.particle_q.numpy()
    has_nan = bool(np.any(np.isnan(pos_np)))
    if has_nan:
        return dict(ke=ke, min_mass=min_mass, nan=True, n_clamped=n_clamped)

    # Test convergence with more iterations
    solver.iterations = test_iters
    solver.track_convergence = True
    solver.reset_convergence_data()

    s0.clear_forces(); s1.clear_forces()
    pipeline.collide(s0, contacts)
    solver.step(s0, s1, control, contacts, dt)

    conv = solver.get_convergence_data()
    residuals = []
    if conv and len(conv) > 0:
        for ir in conv[0].get("iteration_residuals", []):
            residuals.append(ir.get("rms_force_residual", 0))

    # Compute inertia vs contact stiffness ratio
    h2 = dt * dt
    inertia_stiffness = min_mass / h2

    # RMS velocity in last 10 frames
    solver.iterations = 10
    solver.track_convergence = False
    vels = []
    for _ in range(10):
        pos_before = s0.particle_q.numpy().copy()
        for _ in range(SIM_SUBSTEPS):
            s0.clear_forces(); s1.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
        pos_after = s0.particle_q.numpy()
        delta = pos_after - pos_before
        rms = float(np.sqrt(np.mean(np.sum(delta**2, axis=1))) / FRAME_DT)
        vels.append(rms)

    return dict(
        ke=ke, min_mass=min_mass, nan=False,
        n_clamped=int(n_clamped),
        inertia=float(inertia_stiffness),
        residuals=residuals,
        first_res=residuals[0] if residuals else float("nan"),
        last_res=residuals[-1] if residuals else float("nan"),
        reduction=residuals[-1] / residuals[0] if residuals and residuals[0] > 0 else float("nan"),
        med_vel=float(np.median(vels)),
        max_vel=float(np.max(vels)),
    )


if __name__ == "__main__":
    print("VBD Stiffness/Mass Sweep")
    print("=" * 100)

    # Sweep: ke and min_mass
    ke_values = [1e3, 1e4, 1e5]
    mass_values = [1e-3, 1e-2, 1e-1]

    print(f"{'ke':>8} {'min_m':>8} {'inertia':>10} {'clamped':>8} {'first_r':>10} {'last_r':>10} "
          f"{'ratio':>8} {'med_vel':>10} {'max_vel':>10} {'NaN':>5}")
    print("-" * 100)

    for ke in ke_values:
        for mm in mass_values:
            t0 = time.perf_counter()
            r = run_test(ke, mm)
            elapsed = time.perf_counter() - t0
            if r["nan"]:
                print(f"{ke:>8.0e} {mm:>8.0e} {'--':>10} {r['n_clamped']:>8} "
                      f"{'NaN':>10} {'NaN':>10} {'NaN':>8} {'NaN':>10} {'NaN':>10} {'YES':>5}")
            else:
                print(f"{ke:>8.0e} {mm:>8.0e} {r['inertia']:>10.1f} {r['n_clamped']:>8} "
                      f"{r['first_res']:>10.3f} {r['last_res']:>10.3f} "
                      f"{r['reduction']:>8.4f} {r['med_vel']:>10.5f} {r['max_vel']:>10.5f} {'NO':>5}")
            sys.stdout.flush()

    print("\nDone")
