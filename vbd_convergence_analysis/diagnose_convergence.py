#!/usr/bin/env python3
"""Diagnose VBD convergence barriers in the resting phase of a t-shirt simulation.

Runs a single scenario (seed 42, All Fixes IPC, ke=kf=1e5, buffers 64/80) for 120 frames,
then performs detailed convergence analysis at frame 100:
  1. Per-iteration force residuals over 40 iterations
  2. Per-vertex force residual analysis (top-20 worst vertices)
  3. Contact buffer overflow check
  4. Stiffness/Hessian analysis for worst vertices
  5. Iteration scaling test (10, 20, 40, 80 iterations)
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import time

import numpy as np

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

# ---------------------------------------------------------------------------
# Patch friction model to IPC if needed (worktree may have primal_dual)
# ---------------------------------------------------------------------------

PARTICLE_KERNEL = os.path.join(
    WORKTREE, "newton", "_src", "solvers", "vbd", "particle_vbd_kernels.py"
)
RIGID_KERNEL = os.path.join(
    WORKTREE, "newton", "_src", "solvers", "vbd", "rigid_vbd_kernels.py"
)
WARP_CACHE = os.path.expanduser("~/.cache/warp/1.12.0")

_originals: dict[str, str] = {}


def _ensure_ipc_friction():
    """Patch both kernel files to use IPC friction model if they aren't already."""
    global _originals
    for path in (PARTICLE_KERNEL, RIGID_KERNEL):
        with open(path, "r") as f:
            content = f.read()
        if 'VBD_FRICTION_MODEL = "ipc"' in content:
            continue  # already IPC
        _originals[path] = content
        patched = content.replace(
            'VBD_FRICTION_MODEL = "primal_dual"',
            'VBD_FRICTION_MODEL = "ipc"',
        )
        if patched == content:
            raise RuntimeError(f"Could not find VBD_FRICTION_MODEL in {path}")
        with open(path, "w") as f:
            f.write(patched)
        print(f"  Patched {os.path.basename(path)} -> IPC friction")
    # Clear warp cache so patched kernels are recompiled
    if _originals and os.path.isdir(WARP_CACHE):
        shutil.rmtree(WARP_CACHE)
        print(f"  Cleared warp cache: {WARP_CACHE}")


def _restore_friction():
    """Restore original kernel files after script completes."""
    for path, content in _originals.items():
        with open(path, "w") as f:
            f.write(content)
        print(f"  Restored {os.path.basename(path)}", flush=True)


_ensure_ipc_friction()

import warp as wp  # noqa: E402 (must import after patching)
wp.init()

import newton  # noqa: E402
from newton import ModelBuilder  # noqa: E402
from newton.solvers import SolverVBD  # noqa: E402
from vbd_convergence_analysis.analysis_common import (  # noqa: E402
    DENSITY,
    EDGE_KD,
    EDGE_KE,
    FRAME_DT,
    FRICTION_EPSILON,
    GRAVITY,
    LATERAL_OFFSET_RANGE,
    LOW_DROP_HEIGHT_RANGE,
    PARTICLE_RADIUS,
    SELF_CONTACT_MARGIN,
    SELF_CONTACT_RADIUS,
    SELF_CONTACT_REST_EXCLUSION_RADIUS,
    SIM_SUBSTEPS,
    SOFT_CONTACT_MARGIN,
    TRI_KA,
    TRI_KD,
    TRI_KE,
    load_shirt_mesh_vertices,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
NUM_FRAMES = 120
ANALYSIS_FRAME = 100
SOFT_CONTACT_KE = 1e5
SOFT_CONTACT_KF = 1e5
SOFT_CONTACT_KD = 1e-2
SOFT_CONTACT_MU = 0.25
AVBD_BETA = 0.0
CONTACT_KE_BODY = 2500.0
VERTEX_BUFFER = 512
EDGE_BUFFER = 512
MIN_PARTICLE_MASS = 1e-3  # 1 gram minimum per vertex


# ---------------------------------------------------------------------------
# Build scenario (matches ablation study worker pattern)
# ---------------------------------------------------------------------------


def build_scenario(seed: int, iterations: int = 10):
    """Build an All-Fixes-IPC scenario with configurable iteration count."""
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

    # Compute z offset so lowest point is at drop_height above ground
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

    model.soft_contact_ke = SOFT_CONTACT_KE
    model.soft_contact_kd = SOFT_CONTACT_KD
    model.soft_contact_kf = SOFT_CONTACT_KF
    model.soft_contact_mu = SOFT_CONTACT_MU

    # Clamp particle masses to a sensible minimum
    mass_np = model.particle_mass.numpy()
    below_min = mass_np < MIN_PARTICLE_MASS
    if below_min.any():
        mass_np[below_min] = MIN_PARTICLE_MASS
        model.particle_mass.assign(wp.array(mass_np, dtype=wp.float32))
        inv_mass_np = np.where(mass_np > 0, 1.0 / mass_np, 0.0)
        model.particle_inv_mass.assign(wp.array(inv_mass_np, dtype=wp.float32))

    solver = SolverVBD(
        model, iterations=iterations,
        particle_tri_material_model="neohookean",
        particle_enable_self_contact=True,
        particle_self_contact_radius=SELF_CONTACT_RADIUS,
        particle_self_contact_margin=SELF_CONTACT_MARGIN,
        particle_topological_contact_filter_threshold=1,
        particle_rest_shape_contact_exclusion_radius=SELF_CONTACT_REST_EXCLUSION_RADIUS,
        particle_vertex_contact_buffer_size=VERTEX_BUFFER,
        particle_edge_contact_buffer_size=EDGE_BUFFER,
        particle_collision_detection_interval=-1,
        friction_epsilon=FRICTION_EPSILON,
    )
    solver.avbd_beta = AVBD_BETA
    solver.k_start_body_contact = CONTACT_KE_BODY
    solver.track_convergence = False

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=SOFT_CONTACT_MARGIN)
    contacts = pipeline.contacts()

    return dict(
        model=model, solver=solver,
        state_0=state_0, state_1=state_1,
        control=control, contacts=contacts,
        collision_pipeline=pipeline,
        dt=FRAME_DT / SIM_SUBSTEPS,
    )


# ---------------------------------------------------------------------------
# Utility: simulate N frames
# ---------------------------------------------------------------------------


def simulate_frames(sc, n_frames, verbose=True):
    """Run n_frames of simulation, return list of position snapshots."""
    state_0, state_1 = sc["state_0"], sc["state_1"]
    control, contacts = sc["control"], sc["contacts"]
    pipeline, solver = sc["collision_pipeline"], sc["solver"]
    dt = sc["dt"]

    positions = []
    for fi in range(n_frames):
        for si in range(SIM_SUBSTEPS):
            state_0.clear_forces()
            state_1.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        pos = state_0.particle_q.numpy().copy()
        positions.append(pos)
        if verbose and (fi + 1) % 20 == 0:
            min_z = float(pos[:, 2].min())
            print(f"    Frame {fi + 1}/{n_frames}: min_z={min_z:.4f}")
    # Update sc to reflect swapped states
    sc["state_0"], sc["state_1"] = state_0, state_1
    return positions


# ---------------------------------------------------------------------------
# 1. Run to resting phase
# ---------------------------------------------------------------------------


def phase1_simulate_to_rest():
    """Run 120 frames to reach resting phase, return scenario dict + positions."""
    print("=" * 80)
    print("PHASE 1: Simulating 120 frames (seed 42, All Fixes IPC, ke=kf=1e5)")
    print("=" * 80)

    sc = build_scenario(SEED, iterations=10)
    t0 = time.perf_counter()
    positions = simulate_frames(sc, NUM_FRAMES)
    elapsed = time.perf_counter() - t0
    print(f"  Simulation complete in {elapsed:.1f}s")

    # Quick jitter summary for frames 90-120
    rms_vels = []
    for fi in range(90, len(positions)):
        delta = positions[fi] - positions[fi - 1]
        speed = np.linalg.norm(delta / FRAME_DT, axis=1)
        rms_vels.append(float(np.sqrt(np.mean(speed ** 2))))
    print(f"  Rest-phase (90-120) RMS velocity: median={np.median(rms_vels):.6f} m/s, "
          f"max={np.max(rms_vels):.6f} m/s")
    print(f"  Final min_z: {positions[-1][:, 2].min():.4f}")
    has_nan = np.any(np.isnan(positions[-1]))
    print(f"  NaN check: {has_nan}")

    return sc, positions


# ---------------------------------------------------------------------------
# 2. Detailed 40-iteration convergence analysis at frame 100
# ---------------------------------------------------------------------------


def phase2_detailed_convergence(sc):
    """At frame 100 (already reached), run ONE substep with 40 iterations and track_convergence."""
    print("\n" + "=" * 80)
    print("PHASE 2: Detailed convergence analysis (40 iterations, 1 substep at frame 100)")
    print("=" * 80)

    solver = sc["solver"]
    state_0, state_1 = sc["state_0"], sc["state_1"]
    control, contacts = sc["control"], sc["contacts"]
    pipeline = sc["collision_pipeline"]
    dt = sc["dt"]

    # Save current state for reuse
    pos_snapshot = state_0.particle_q.numpy().copy()
    vel_snapshot = state_0.particle_qd.numpy().copy()

    # Reconfigure solver for 40 iterations with convergence tracking
    old_iters = solver.iterations
    solver.iterations = 40
    solver.reset_convergence_data()
    solver.track_convergence = True

    state_0.clear_forces()
    state_1.clear_forces()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)

    solver.track_convergence = False
    solver.iterations = old_iters

    conv_data = solver.get_convergence_data()
    if not conv_data:
        print("  ERROR: No convergence data recorded!")
        return None

    step_data = conv_data[0]
    residuals = step_data.get("iteration_residuals", [])

    print(f"\n  Per-iteration force residuals ({len(residuals)} iterations):")
    print(f"  {'Iter':>4}  {'RMS':>12}  {'Mean':>12}  {'Max':>12}  {'P50':>12}  {'P95':>12}  {'P99':>12}  {'RMS_disp':>12}")
    print("  " + "-" * 100)
    for i, r in enumerate(residuals):
        print(f"  {i:>4}  {r['rms_force_residual']:>12.6f}  {r['mean_force_residual']:>12.6f}  "
              f"{r['max_force_residual']:>12.4f}  {r['p50_force_residual']:>12.6f}  "
              f"{r['p95_force_residual']:>12.6f}  {r['p99_force_residual']:>12.6f}  "
              f"{r['rms_displacement']:>12.8f}")

    # Summarize convergence rate
    rms_vals = [r["rms_force_residual"] for r in residuals]
    if len(rms_vals) >= 2:
        ratio_first = rms_vals[1] / rms_vals[0] if rms_vals[0] > 0 else float("inf")
        ratio_last = rms_vals[-1] / rms_vals[-2] if rms_vals[-2] > 0 else float("inf")
        total_reduction = rms_vals[-1] / rms_vals[0] if rms_vals[0] > 0 else float("inf")
        print(f"\n  Convergence summary:")
        print(f"    First iteration reduction ratio: {ratio_first:.4f}")
        print(f"    Last iteration reduction ratio:  {ratio_last:.4f}")
        print(f"    Total reduction (iter 0 -> {len(rms_vals)-1}): {total_reduction:.6f}")
        if total_reduction > 0.5:
            print("    ** POOR CONVERGENCE: less than 2x total reduction over 40 iterations **")
        if ratio_last > 0.95:
            print("    ** PLATEAU DETECTED: last iteration barely reduces residual **")

    # Restore state for subsequent phases
    state_1.particle_q.assign(pos_snapshot)
    state_1.particle_qd.assign(vel_snapshot)
    sc["state_0"], sc["state_1"] = state_1, state_0

    return step_data


# ---------------------------------------------------------------------------
# 3. Per-vertex force residual analysis
# ---------------------------------------------------------------------------


def phase3_vertex_residual_analysis(sc):
    """Analyze per-vertex residuals: find worst vertices, classify contact type."""
    print("\n" + "=" * 80)
    print("PHASE 3: Per-vertex force residual analysis (top 20 worst vertices)")
    print("=" * 80)

    solver = sc["solver"]
    model = sc["model"]
    state_0 = sc["state_0"]

    # The force residual norms from the last convergence-tracked step are still in solver
    residual_norms = solver._force_residual_norms.numpy()
    positions = state_0.particle_q.numpy()
    flags = model.particle_flags.numpy()
    masses = model.particle_mass.numpy()

    active_mask = (flags & 1).astype(bool) & (masses > 0)
    n_active = int(active_mask.sum())
    n_total = len(residual_norms)
    print(f"  Total particles: {n_total}, Active: {n_active}")

    # Sort active vertices by residual
    active_indices = np.where(active_mask)[0]
    active_residuals = residual_norms[active_indices]
    sorted_order = np.argsort(active_residuals)[::-1]

    # Get particle forces and hessians
    forces = solver.particle_forces.numpy()
    hessians = solver.particle_hessians.numpy()

    # Get self-contact collision info
    vt_counts = solver.trimesh_collision_detector.vertex_colliding_triangles_count.numpy()
    ee_counts = solver.trimesh_collision_detector.edge_colliding_edges_count.numpy()
    vt_buf_sizes = solver.trimesh_collision_detector.vertex_colliding_triangles_buffer_sizes.numpy()
    ee_buf_sizes = solver.trimesh_collision_detector.edge_colliding_edges_buffer_sizes.numpy()

    # Get soft contacts (body-particle / ground contacts)
    contacts = sc["contacts"]
    soft_contact_count_val = contacts.soft_contact_count.numpy()[0]
    soft_contact_particles = contacts.soft_contact_particle.numpy()[:soft_contact_count_val]
    # Build set of particles in ground contact
    ground_contact_particles = set(int(p) for p in soft_contact_particles)

    print(f"  Ground contacts: {len(ground_contact_particles)} particles in soft contact")
    print(f"  Self-contact vertex-tri collisions detected: {int(np.sum(vt_counts > 0))}")
    print(f"  Self-contact edge-edge collisions detected: {int(np.sum(ee_counts > 0))}")

    print(f"\n  Top 20 highest-residual vertices:")
    print(f"  {'Rank':>4}  {'VIdx':>6}  {'Residual':>12}  {'Pos (x,y,z)':>36}  {'|Force|':>10}  "
          f"{'Contact':>12}  {'VT_cnt':>6}  {'Hdiag':>30}")
    print("  " + "-" * 130)

    top_vertex_indices = []
    for rank, si in enumerate(sorted_order[:20]):
        vidx = active_indices[si]
        res = residual_norms[vidx]
        pos = positions[vidx]
        force = forces[vidx]
        force_mag = float(np.linalg.norm(force))
        hess = hessians[vidx]  # 3x3 matrix
        hdiag = np.diag(hess) if hess.ndim == 2 else hess[:3]

        # Classify contact type
        in_ground = vidx in ground_contact_particles
        vt_count = int(vt_counts[vidx])
        contact_type = []
        if in_ground:
            contact_type.append("GND")
        if vt_count > 0:
            contact_type.append(f"SC({vt_count})")
        if not contact_type:
            contact_type.append("FREE")
        contact_str = "+".join(contact_type)

        top_vertex_indices.append(vidx)

        print(f"  {rank+1:>4}  {vidx:>6}  {res:>12.4f}  ({pos[0]:>10.5f}, {pos[1]:>10.5f}, {pos[2]:>10.5f})  "
              f"{force_mag:>10.4f}  {contact_str:>12}  {vt_count:>6}  "
              f"({hdiag[0]:>8.1f}, {hdiag[1]:>8.1f}, {hdiag[2]:>8.1f})")

    # Statistics on contact types of high-residual vertices
    n_ground = sum(1 for vi in top_vertex_indices if vi in ground_contact_particles)
    n_sc = sum(1 for vi in top_vertex_indices if vt_counts[vi] > 0)
    n_free = sum(1 for vi in top_vertex_indices
                 if vi not in ground_contact_particles and vt_counts[vi] == 0)
    print(f"\n  Top-20 contact breakdown: Ground={n_ground}, Self-contact={n_sc}, Free={n_free}")

    # Distribution of residuals
    print(f"\n  Active residual statistics:")
    print(f"    Mean: {np.mean(active_residuals):.6f}")
    print(f"    Median: {np.median(active_residuals):.6f}")
    print(f"    P90: {np.percentile(active_residuals, 90):.6f}")
    print(f"    P99: {np.percentile(active_residuals, 99):.6f}")
    print(f"    Max: {np.max(active_residuals):.6f}")
    print(f"    Fraction with residual > 1.0: {np.sum(active_residuals > 1.0) / len(active_residuals):.4f}")
    print(f"    Fraction with residual > 10.0: {np.sum(active_residuals > 10.0) / len(active_residuals):.4f}")
    print(f"    Fraction with residual > 100.0: {np.sum(active_residuals > 100.0) / len(active_residuals):.4f}")

    return top_vertex_indices


# ---------------------------------------------------------------------------
# 4. Contact buffer overflow check
# ---------------------------------------------------------------------------


def phase4_contact_buffer_overflow(sc):
    """Check if any contact buffers are full (indicating overflow)."""
    print("\n" + "=" * 80)
    print("PHASE 4: Contact buffer overflow check")
    print("=" * 80)

    solver = sc["solver"]
    detector = solver.trimesh_collision_detector

    # Resize flags
    resize_flags = detector.resize_flags.numpy()
    print(f"  Resize flags: {resize_flags}")
    flag_names = ["vertex-tri resize", "tri-vertex resize", "edge-edge resize", "reserved"]
    for i, (flag, name) in enumerate(zip(resize_flags, flag_names)):
        status = "OVERFLOW DETECTED" if flag > 0 else "ok"
        print(f"    Flag {i} ({name}): {flag} -- {status}")

    # Vertex-triangle buffer usage
    vt_counts = detector.vertex_colliding_triangles_count.numpy()
    vt_buf_sizes = detector.vertex_colliding_triangles_buffer_sizes.numpy()
    vt_at_max = np.sum(vt_counts >= vt_buf_sizes)
    vt_max_count = int(np.max(vt_counts)) if len(vt_counts) > 0 else 0
    vt_mean_count = float(np.mean(vt_counts[vt_counts > 0])) if np.any(vt_counts > 0) else 0.0
    print(f"\n  Vertex-triangle collision buffers (size={VERTEX_BUFFER}):")
    print(f"    Vertices with any VT collision: {int(np.sum(vt_counts > 0))}")
    print(f"    Max collisions for single vertex: {vt_max_count}")
    print(f"    Mean collisions (non-zero only): {vt_mean_count:.1f}")
    print(f"    Vertices at buffer capacity: {int(vt_at_max)}")
    if vt_at_max > 0:
        print(f"    ** WARNING: {int(vt_at_max)} vertices may have missed collisions! **")

    # Edge-edge buffer usage
    ee_counts = detector.edge_colliding_edges_count.numpy()
    ee_buf_sizes = detector.edge_colliding_edges_buffer_sizes.numpy()
    ee_at_max = np.sum(ee_counts >= ee_buf_sizes)
    ee_max_count = int(np.max(ee_counts)) if len(ee_counts) > 0 else 0
    ee_mean_count = float(np.mean(ee_counts[ee_counts > 0])) if np.any(ee_counts > 0) else 0.0
    print(f"\n  Edge-edge collision buffers (size={EDGE_BUFFER}):")
    print(f"    Edges with any EE collision: {int(np.sum(ee_counts > 0))}")
    print(f"    Max collisions for single edge: {ee_max_count}")
    print(f"    Mean collisions (non-zero only): {ee_mean_count:.1f}")
    print(f"    Edges at buffer capacity: {int(ee_at_max)}")
    if ee_at_max > 0:
        print(f"    ** WARNING: {int(ee_at_max)} edges may have missed collisions! **")

    # Histogram of buffer usage
    print(f"\n  VT collision count distribution:")
    for threshold in [0, 1, 5, 10, 20, 40, 64]:
        count = int(np.sum(vt_counts >= threshold))
        print(f"    >= {threshold:>3}: {count:>6} vertices")

    print(f"\n  EE collision count distribution:")
    for threshold in [0, 1, 5, 10, 20, 40, 80]:
        count = int(np.sum(ee_counts >= threshold))
        print(f"    >= {threshold:>3}: {count:>6} edges")


# ---------------------------------------------------------------------------
# 5. Stiffness / Hessian analysis for worst vertices
# ---------------------------------------------------------------------------


def phase5_stiffness_analysis(sc, top_vertices):
    """Analyze Hessian diagonal for worst-residual vertices to identify stiffness imbalance."""
    print("\n" + "=" * 80)
    print("PHASE 5: Stiffness / Hessian analysis for worst vertices")
    print("=" * 80)

    solver = sc["solver"]
    model = sc["model"]
    state_0 = sc["state_0"]

    # The particle_forces and particle_hessians contain the CONTACT contributions
    # (body contact + spring + self-contact) accumulated during the last iteration.
    # The elasticity solve also adds elastic + bending contributions internally.
    # We can read the contact-only forces/hessians since those are what's stored.
    forces = solver.particle_forces.numpy()
    hessians = solver.particle_hessians.numpy()
    positions = state_0.particle_q.numpy()
    masses = model.particle_mass.numpy()
    residual_norms = solver._force_residual_norms.numpy()

    # Compute inertia contribution: m / h^2
    dt = sc["dt"]
    h2 = dt * dt

    print(f"  Substep dt = {dt:.6f}s, h^2 = {h2:.10f}")
    print(f"  Inertia scale (m/h^2) for typical mass:")

    # Get unique masses
    unique_masses = np.unique(masses[masses > 0])
    for m in unique_masses[:5]:
        print(f"    mass={m:.6f} kg -> inertia_stiffness = {m/h2:.2f}")

    print(f"\n  Analysis for top {len(top_vertices)} worst-residual vertices:")
    print(f"  {'VIdx':>6}  {'Mass':>8}  {'m/h2':>10}  {'|Force|':>10}  {'Residual':>10}  "
          f"{'Hdiag_x':>10}  {'Hdiag_y':>10}  {'Hdiag_z':>10}  {'H/inertia':>10}  {'Z pos':>8}")
    print("  " + "-" * 110)

    inertia_ratios = []
    contact_forces_mag = []
    for vidx in top_vertices:
        m = masses[vidx]
        inertia_stiff = m / h2 if h2 > 0 else 0.0
        force = forces[vidx]
        force_mag = float(np.linalg.norm(force))
        hess = hessians[vidx]
        hdiag = np.diag(hess) if hess.ndim == 2 else hess[:3]
        max_hdiag = float(np.max(np.abs(hdiag)))
        ratio = max_hdiag / inertia_stiff if inertia_stiff > 0 else float("inf")
        inertia_ratios.append(ratio)
        contact_forces_mag.append(force_mag)

        print(f"  {vidx:>6}  {m:>8.5f}  {inertia_stiff:>10.1f}  {force_mag:>10.4f}  "
              f"{residual_norms[vidx]:>10.4f}  {hdiag[0]:>10.1f}  {hdiag[1]:>10.1f}  "
              f"{hdiag[2]:>10.1f}  {ratio:>10.2f}  {positions[vidx][2]:>8.4f}")

    print(f"\n  Stiffness ratio (max |Hessian_diag| / inertia) statistics:")
    print(f"    Mean: {np.mean(inertia_ratios):.2f}")
    print(f"    Max:  {np.max(inertia_ratios):.2f}")
    print(f"    Min:  {np.min(inertia_ratios):.2f}")
    if np.max(inertia_ratios) > 100:
        print(f"    ** STIFFNESS DOMINANCE: contact Hessian >> inertia by {np.max(inertia_ratios):.0f}x **")
        print(f"       This means the block solve is ill-conditioned: the contact stiffness")
        print(f"       overwhelms the inertia regularizer, causing the per-vertex Newton step")
        print(f"       to be almost entirely determined by contact forces.")

    # Also analyze globally: what fraction of all active vertices have high stiffness ratio?
    all_hdiags = np.array([np.max(np.abs(np.diag(hessians[i]) if hessians[i].ndim == 2 else hessians[i][:3]))
                           for i in range(len(hessians))])
    all_inertia = masses / h2
    mask = all_inertia > 0
    all_ratios = np.zeros(len(all_hdiags))
    all_ratios[mask] = all_hdiags[mask] / all_inertia[mask]

    active_mask = (model.particle_flags.numpy() & 1).astype(bool) & (masses > 0)
    active_ratios = all_ratios[active_mask]

    print(f"\n  Global Hessian/inertia ratio distribution (all active vertices):")
    for threshold in [0, 1, 10, 100, 1000, 10000]:
        count = int(np.sum(active_ratios >= threshold))
        frac = count / len(active_ratios) if len(active_ratios) > 0 else 0
        print(f"    >= {threshold:>6}: {count:>6} ({frac*100:.1f}%)")


# ---------------------------------------------------------------------------
# 6. Iteration scaling test
# ---------------------------------------------------------------------------


def phase6_iteration_scaling():
    """Test convergence with different iteration counts: 10, 20, 40, 80."""
    print("\n" + "=" * 80)
    print("PHASE 6: Iteration scaling test (10, 20, 40, 80 iterations)")
    print("=" * 80)

    iteration_counts = [10, 20, 40, 80]
    results = []

    for n_iter in iteration_counts:
        print(f"\n  Testing {n_iter} iterations...")
        sc = build_scenario(SEED, iterations=10)  # always use 10 for warmup

        # Simulate to frame 99 (just before analysis frame)
        _ = simulate_frames(sc, ANALYSIS_FRAME, verbose=False)

        # Now run ONE substep with n_iter iterations and convergence tracking
        solver = sc["solver"]
        solver.iterations = n_iter
        solver.reset_convergence_data()
        solver.track_convergence = True

        state_0, state_1 = sc["state_0"], sc["state_1"]
        control, contacts = sc["control"], sc["contacts"]
        pipeline = sc["collision_pipeline"]
        dt = sc["dt"]

        state_0.clear_forces()
        state_1.clear_forces()
        pipeline.collide(state_0, contacts)
        t0 = time.perf_counter()
        solver.step(state_0, state_1, control, contacts, dt)
        elapsed = time.perf_counter() - t0

        solver.track_convergence = False
        conv_data = solver.get_convergence_data()

        if conv_data:
            step_data = conv_data[0]
            residuals = step_data.get("iteration_residuals", [])
            if residuals:
                final_r = residuals[-1]
                first_r = residuals[0]
                result = {
                    "n_iter": n_iter,
                    "first_rms": first_r["rms_force_residual"],
                    "final_rms": final_r["rms_force_residual"],
                    "final_max": final_r["max_force_residual"],
                    "final_p99": final_r["p99_force_residual"],
                    "reduction": final_r["rms_force_residual"] / first_r["rms_force_residual"]
                    if first_r["rms_force_residual"] > 0 else float("inf"),
                    "elapsed": elapsed,
                }
                results.append(result)
                print(f"    RMS residual: {first_r['rms_force_residual']:.6f} -> "
                      f"{final_r['rms_force_residual']:.6f} "
                      f"(reduction: {result['reduction']:.4f}, time: {elapsed:.2f}s)")

    if results:
        print(f"\n  Iteration scaling summary:")
        print(f"  {'Iters':>6}  {'First RMS':>12}  {'Final RMS':>12}  {'Final Max':>12}  "
              f"{'Final P99':>12}  {'Reduction':>10}  {'Time(s)':>8}")
        print("  " + "-" * 80)
        for r in results:
            print(f"  {r['n_iter']:>6}  {r['first_rms']:>12.6f}  {r['final_rms']:>12.6f}  "
                  f"{r['final_max']:>12.4f}  {r['final_p99']:>12.6f}  "
                  f"{r['reduction']:>10.4f}  {r['elapsed']:>8.2f}")

        # Check for plateau
        if len(results) >= 2:
            r10 = results[0]["final_rms"]
            r80 = results[-1]["final_rms"]
            if r10 > 0:
                improvement = r80 / r10
                print(f"\n  10->80 iteration improvement: {improvement:.4f}")
                if improvement > 0.5:
                    print(f"  ** CONVERGENCE BARRIER: 8x more iterations yields < 2x improvement **")
                    print(f"     The solver has a fundamental convergence limitation that more")
                    print(f"     iterations alone cannot overcome.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("VBD Convergence Diagnostic Script")
    print(f"  Seed: {SEED}")
    print(f"  Frames: {NUM_FRAMES}, Analysis at frame {ANALYSIS_FRAME}")
    print(f"  Config: All Fixes IPC, ke=kf={SOFT_CONTACT_KE:.0e}, buffers {VERTEX_BUFFER}/{EDGE_BUFFER}")
    print(f"  Substeps: {SIM_SUBSTEPS}, friction_epsilon: {FRICTION_EPSILON}")
    print(f"  avbd_beta: {AVBD_BETA}, k_start_body_contact: {CONTACT_KE_BODY}")
    print()

    try:
        # Phase 1: Simulate to rest
        sc, positions = phase1_simulate_to_rest()

        # Phase 2: Detailed 40-iteration convergence at frame 100
        step_data = phase2_detailed_convergence(sc)

        # Phase 3: Per-vertex force residual analysis
        top_vertices = phase3_vertex_residual_analysis(sc)

        # Phase 4: Contact buffer overflow
        phase4_contact_buffer_overflow(sc)

        # Phase 5: Stiffness analysis
        if top_vertices:
            phase5_stiffness_analysis(sc, top_vertices)

        # Phase 6: Iteration scaling (needs fresh scenarios)
        phase6_iteration_scaling()

        print("\n" + "=" * 80)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 80)

    finally:
        _restore_friction()


if __name__ == "__main__":
    main()
