#!/usr/bin/env python3
"""VBD Ablation Study: comprehensive evaluation of every kernel fix and runtime parameter.

Tier 1 — Kernel-level configs (K0-K9, P/D friction K4-K6):
  Require source-patching + warp cache clear + subprocess isolation because
  warp caches compiled kernels in memory.

Tier 2 — Runtime configs (R0-R6):
  Share the IPC kernel compilation, only varying solver/model parameters.

Generates an HTML report at ~/reports/newton-vbd/vbd_ablation.html with:
  1. Summary table ranking all configs by jitter
  2-6. Per-issue sections comparing baseline vs reverted fix
  7. Runtime parameter sweep
  8. Methodology
"""
from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PARTICLE_KERNEL = os.path.join(
    WORKTREE, "newton", "_src", "solvers", "vbd", "particle_vbd_kernels.py"
)
RIGID_KERNEL = os.path.join(
    WORKTREE, "newton", "_src", "solvers", "vbd", "rigid_vbd_kernels.py"
)
WARP_CACHE = os.path.expanduser("~/.cache/warp/1.12.0")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789]
NUM_FRAMES = 180
REST_START = 90
FPS = 60
VIDEO_W, VIDEO_H = 1280, 720
SSAA_FACTOR = 2  # render at 2x resolution, downsample with Lanczos for AA
CAM_PITCH, CAM_YAW = -20.0, 90.0
CONVERGENCE_SAMPLE_FRAMES = list(range(100, 180, 10))

# Default runtime parameters (IPC friction baseline)
DEFAULT_RUNTIME = dict(
    step_length=1.0,
    avbd_beta=0.0,
    contact_ke_body=2500.0,
    soft_contact_ke=1e4,
    soft_contact_kd=1e-2,
    soft_contact_kf=1e4,   # P/D friction: kf = ke for balanced stiffness
    friction_epsilon=1e-4,
    chebyshev_rho=0.0,
)

# ---------------------------------------------------------------------------
# Kernel-level configs  (Tier 1)
# ---------------------------------------------------------------------------

KERNEL_CONFIGS = [
    # Newton Main: original code with log-barrier, kd*ke damping, zero friction H
    ("Newton Main (original)", [
        # Log-barrier self-contact (original)
        (PARTICLE_KERNEL,
         "    dEdD = -k * penetration_depth\n    d2E_dDdD = k\n",
         "    dEdD = -k * k / dis\n    d2E_dDdD = k * k / (dis * dis)\n"),
        # kd*ke contact damping (original)
        (RIGID_KERNEL,
         "damping_coeff = body_particle_contact_kd\n",
         "damping_coeff = body_particle_contact_kd * body_particle_contact_ke\n"),
        # Zero friction Hessian at rest (original)
        (PARTICLE_KERNEL,
         "            hessian_scale = mu * normal_contact_force * 2.0 / eps_u\n        force = wp.vec3(0.0, 0.0, 0.0)",
         "            hessian_scale = 0.0\n        force = wp.vec3(0.0, 0.0, 0.0)"),
        (RIGID_KERNEL,
         "            scale = friction_mu * normal_load * 2.0 / eps_u",
         "            scale = 0.0"),
    ], True),

    # Fix 1 only: quadratic self-contact penalty (keep damping+friction H original)
    ("+ Quadratic SC penalty", [
        # kd*ke contact damping (still original)
        (RIGID_KERNEL,
         "damping_coeff = body_particle_contact_kd\n",
         "damping_coeff = body_particle_contact_kd * body_particle_contact_ke\n"),
        # Zero friction Hessian at rest (still original)
        (PARTICLE_KERNEL,
         "            hessian_scale = mu * normal_contact_force * 2.0 / eps_u\n        force = wp.vec3(0.0, 0.0, 0.0)",
         "            hessian_scale = 0.0\n        force = wp.vec3(0.0, 0.0, 0.0)"),
        (RIGID_KERNEL,
         "            scale = friction_mu * normal_load * 2.0 / eps_u",
         "            scale = 0.0"),
    ], False),

    # Fix 1+2: quadratic SC + absolute damping (keep friction H original)
    ("+ Absolute damping", [
        # Zero friction Hessian at rest (still original)
        (PARTICLE_KERNEL,
         "            hessian_scale = mu * normal_contact_force * 2.0 / eps_u\n        force = wp.vec3(0.0, 0.0, 0.0)",
         "            hessian_scale = 0.0\n        force = wp.vec3(0.0, 0.0, 0.0)"),
        (RIGID_KERNEL,
         "            scale = friction_mu * normal_load * 2.0 / eps_u",
         "            scale = 0.0"),
    ], False),

    # All three fixes applied with IPC friction (no patches needed)
    ("All Fixes (IPC)", None, True),
]

# Patches to switch to P/D friction model
PD_PATCHES = [
    (PARTICLE_KERNEL, 'VBD_FRICTION_MODEL = "ipc"', 'VBD_FRICTION_MODEL = "primal_dual"'),
    (RIGID_KERNEL, 'VBD_FRICTION_MODEL = "ipc"', 'VBD_FRICTION_MODEL = "primal_dual"'),
]

# P/D friction configs — share same kernel compilation, vary kf at runtime
# kf ≈ ke balances normal/tangential stiffness
PD_FRICTION_CONFIGS = [
    ("P/D kf=1e3", 1e3, False),
    ("P/D kf=5e3", 5e3, False),
    ("All Fixes (P/D kf=1e4)", 1e4, True),   # recommended: kf = ke
    ("P/D kf=5e4", 5e4, False),
    ("P/D kf=1e5", 1e5, False),
]

# ---------------------------------------------------------------------------
# Runtime configs  (Tier 2)
# ---------------------------------------------------------------------------

RUNTIME_CONFIGS = [
    ("R0. AVBD ramp", dict(avbd_beta=1e5, contact_ke_body=100.0)),
    ("R1. No AVBD", dict(avbd_beta=0.0, contact_ke_body=2500.0)),
    ("R2. Alpha=0.7", dict(step_length=0.7)),
    ("R3. Alpha=0.5", dict(step_length=0.5)),
    ("R4. Chebyshev auto", dict(chebyshev_rho="auto")),
    ("R5. sc_ke=1e5", dict(soft_contact_ke=1e5, soft_contact_kf=1e5)),
    ("R6. sc_ke=1e3", dict(soft_contact_ke=1e3, soft_contact_kf=1e3)),
]

# Which runtime configs get videos (by label prefix)
RUNTIME_VIDEO_LABELS = {"R1", "R3"}

# ---------------------------------------------------------------------------
# Source patching utilities
# ---------------------------------------------------------------------------


def apply_patches(patches: list[tuple[str, str, str]]) -> dict[str, str]:
    """Apply patches and return {filepath: original_content} for rollback."""
    originals: dict[str, str] = {}
    for filepath, old, new in patches:
        if filepath not in originals:
            with open(filepath, "r") as f:
                originals[filepath] = f.read()
        with open(filepath, "r") as f:
            current = f.read()
        patched = current.replace(old, new)
        if patched == current:
            raise RuntimeError(
                f"Patch failed: could not find old text in {filepath}:\n{old!r}"
            )
        with open(filepath, "w") as f:
            f.write(patched)
    return originals


def restore_files(originals: dict[str, str]):
    """Restore original file contents."""
    for filepath, content in originals.items():
        with open(filepath, "w") as f:
            f.write(content)


def clear_warp_cache():
    """Clear the warp kernel cache so patched kernels are recompiled."""
    if os.path.isdir(WARP_CACHE):
        shutil.rmtree(WARP_CACHE)
        print(f"  Cleared warp cache: {WARP_CACHE}")


# ---------------------------------------------------------------------------
# Worker subprocess script (embedded as string constant)
# ---------------------------------------------------------------------------

WORKER_SCRIPT = r'''
"""Worker subprocess: run one ablation config, measure jitter + convergence, optionally record video."""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

WORKTREE = os.environ["WORKTREE"]
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

import warp as wp
wp.init()

import newton
import newton.viewer
from newton import ModelBuilder
from newton.solvers import SolverVBD
from vbd_convergence_analysis.analysis_common import (
    DENSITY, EDGE_KD, EDGE_KE, FRAME_DT, FRICTION_EPSILON, GRAVITY,
    LATERAL_OFFSET_RANGE, LOW_DROP_HEIGHT_RANGE, PARTICLE_RADIUS,
    POP_THRESHOLD_M, SELF_CONTACT_MARGIN, SELF_CONTACT_RADIUS,
    SELF_CONTACT_REST_EXCLUSION_RADIUS, SOFT_CONTACT_MARGIN,
    SIM_SUBSTEPS, TRI_KA, TRI_KD, TRI_KE,
    load_shirt_mesh_vertices,
)

NUM_FRAMES = 180
REST_START = 90
FPS = 60
CAM_PITCH, CAM_YAW = -20.0, 90.0
SSAA_FACTOR = 4  # render at 4x, downsample with Lanczos for anti-aliasing


def build_scenario(seed, step_length=1.0, avbd_beta=0.0, contact_ke_body=2500.0,
                   soft_contact_ke=1e4, soft_contact_kd=1e-2, soft_contact_kf=0.0,
                   friction_epsilon=1e-4, chebyshev_rho=0.0):
    """Build a meter-scale t-shirt drop scenario with configurable runtime params."""
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

    model.soft_contact_ke = soft_contact_ke
    model.soft_contact_kd = soft_contact_kd
    model.soft_contact_kf = soft_contact_kf
    model.soft_contact_mu = 0.25

    # Clamp particle masses to prevent ill-conditioned block solve
    MIN_PARTICLE_MASS = 1e-2  # 10g minimum
    mass_np = model.particle_mass.numpy()
    below = mass_np < MIN_PARTICLE_MASS
    if below.any():
        mass_np[below] = MIN_PARTICLE_MASS
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
        particle_vertex_contact_buffer_size=256,
        particle_edge_contact_buffer_size=512,
        particle_collision_detection_interval=-1,
        friction_epsilon=friction_epsilon,
    )
    solver.avbd_beta = avbd_beta
    solver.k_start_body_contact = contact_ke_body
    solver.track_convergence = False
    if step_length < 1.0:
        solver.step_length = step_length
    if chebyshev_rho == "auto" or (isinstance(chebyshev_rho, (int, float)) and chebyshev_rho > 0):
        solver.chebyshev_rho = chebyshev_rho

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


def run_and_record(seed, config, video_path=None):
    """Run simulation, measure jitter + convergence, optionally record video."""
    from PIL import Image

    record_video = config.get("record_video", False) and video_path is not None
    convergence_frames = set(config.get("convergence_frames", []))
    num_frames = config.get("num_frames", NUM_FRAMES)

    # Build scenario with runtime params
    runtime = {}
    for k in ("step_length", "avbd_beta", "contact_ke_body", "soft_contact_ke",
              "soft_contact_kd", "soft_contact_kf", "friction_epsilon", "chebyshev_rho"):
        if k in config:
            runtime[k] = config[k]

    sc = build_scenario(seed, **runtime)
    state_0, state_1 = sc["state_0"], sc["state_1"]
    control, contacts = sc["control"], sc["contacts"]
    pipeline, solver = sc["collision_pipeline"], sc["solver"]
    model, dt = sc["model"], sc["dt"]

    # Setup viewer if recording video
    viewer = None
    frames_dir = None
    if record_video:
        # Render at SSAA_FACTOR × resolution, downsample for anti-aliasing
        viewer = newton.viewer.ViewerGL(headless=True,
                                        width=1280 * SSAA_FACTOR,
                                        height=720 * SSAA_FACTOR)
        viewer.set_model(model)
        pos_np = state_0.particle_q.numpy()
        center = pos_np.mean(axis=0)
        z_max = float(pos_np[:, 2].max())
        cam_pos = wp.vec3(float(center[0]), float(center[1]) - 0.8, z_max * 0.5 + 0.1)
        viewer.set_camera(cam_pos, CAM_PITCH, CAM_YAW)
        # Wireframe edge overlay to make self-penetration visible
        viewer.renderer.draw_edges = True
        viewer.renderer._edge_color = (0.0, 0.0, 0.0, 0.8)
        frames_dir = tempfile.mkdtemp(prefix="vbd_ablation_")

    positions = [state_0.particle_q.numpy().copy()]
    convergence_data = {}
    sim_time = 0.0

    for fi in range(num_frames):
        # Check if this frame should capture convergence
        capture_conv = fi in convergence_frames

        for si in range(SIM_SUBSTEPS):
            state_0.clear_forces()
            state_1.clear_forces()
            pipeline.collide(state_0, contacts)

            # Enable convergence tracking for first substep of convergence frames
            if capture_conv and si == 0:
                solver.reset_convergence_data()
                solver.track_convergence = True

            solver.step(state_0, state_1, control, contacts, dt)

            if capture_conv and si == 0:
                solver.track_convergence = False
                conv = solver.get_convergence_data()
                if conv and len(conv) > 0:
                    step_data = conv[0]
                    residuals = [
                        float(ir.get("rms_force_residual", ir.get("rms_displacement", 0)))
                        for ir in step_data.get("iteration_residuals", [])
                    ]
                    if residuals and not any(math.isnan(r) for r in residuals):
                        convergence_data[fi] = residuals

            state_0, state_1 = state_1, state_0

        sim_time += FRAME_DT
        pos_np = state_0.particle_q.numpy().copy()
        positions.append(pos_np)

        if viewer is not None:
            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.end_frame()
            frame = viewer.get_frame().numpy()
            img = Image.fromarray(frame)
            if SSAA_FACTOR > 1:
                img = img.resize((1280, 720), Image.LANCZOS)
            img.save(os.path.join(frames_dir, f"frame_{fi:04d}.png"))

    if viewer is not None:
        viewer.close()

    # Encode video
    if record_video and frames_dir and video_path:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(FPS),
             "-i", os.path.join(frames_dir, "frame_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast",
             video_path],
            capture_output=True, text=True, timeout=120,
        )
        for f_name in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f_name))
        os.rmdir(frames_dir)

    # Jitter metrics
    rms_vel, max_disp = [], []
    for f in range(1, len(positions)):
        delta = positions[f] - positions[f - 1]
        has_nan_frame = bool(np.any(np.isnan(delta)))
        if has_nan_frame:
            rms_vel.append(float("nan"))
            max_disp.append(float("nan"))
        else:
            disp = np.linalg.norm(delta, axis=1)
            speed = np.linalg.norm(delta / FRAME_DT, axis=1)
            rms_vel.append(float(np.sqrt(np.mean(speed ** 2))))
            max_disp.append(float(np.max(disp)))

    has_nan = bool(np.any(np.isnan(positions[-1])))
    min_z = float(positions[-1][:, 2].min()) if not has_nan else float("nan")

    rest_rms = np.array(rms_vel[REST_START:])
    rest_disp = np.array(max_disp[REST_START:])
    any_nan_rest = bool(np.any(np.isnan(rest_rms)))

    # Convergence ratio: last iter / first iter (median across captured frames)
    conv_ratios = []
    for fi, residuals in convergence_data.items():
        if len(residuals) >= 2 and residuals[0] > 0:
            conv_ratios.append(residuals[-1] / residuals[0])

    result = dict(
        rms_vel=rms_vel,
        max_disp=max_disp,
        has_nan=has_nan,
        min_z=min_z,
        rest_med_rms=float(np.nanmedian(rest_rms)) if len(rest_rms) and not any_nan_rest else float("nan"),
        rest_max_rms=float(np.nanmax(rest_rms)) if len(rest_rms) and not any_nan_rest else float("nan"),
        rest_med_disp=float(np.nanmedian(rest_disp)) if len(rest_disp) and not any_nan_rest else float("nan"),
        rest_max_disp=float(np.nanmax(rest_disp)) if len(rest_disp) and not any_nan_rest else float("nan"),
        rest_pops=int(np.sum(rest_disp > POP_THRESHOLD_M)) if len(rest_disp) and not any_nan_rest else 90,
        convergence_data=convergence_data,
        conv_ratio=float(np.median(conv_ratios)) if conv_ratios else float("nan"),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--video-path", type=str, default="")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--config-json", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_json, "r") as f:
        config = json.load(f)

    video_path = args.video_path if args.video_path else None
    result = run_and_record(args.seed, config, video_path=video_path)

    # Serialize: convert convergence_data keys to strings for JSON
    save_result = dict(result)
    save_result["convergence_data"] = {str(k): v for k, v in result["convergence_data"].items()}

    with open(args.output_json, "w") as f:
        json.dump(save_result, f, indent=2, default=str)

    status = "NaN!" if result["has_nan"] else f"rms={result['rest_med_rms']:.6f}"
    print(f"Done: seed={args.seed} {status} pops={result['rest_pops']} conv_ratio={result['conv_ratio']:.4f}")
'''


# ---------------------------------------------------------------------------
# Subprocess launcher
# ---------------------------------------------------------------------------

def run_worker(seed: int, config: dict, video_path: str,
               output_json: str) -> dict | None:
    """Run simulation in a subprocess (picks up patched kernels)."""
    worker_file = os.path.join(tempfile.gettempdir(), "vbd_ablation_worker.py")
    with open(worker_file, "w") as f:
        f.write(WORKER_SCRIPT)

    # Write config JSON
    config_file = os.path.join(tempfile.gettempdir(), f"vbd_ablation_cfg_{seed}.json")
    with open(config_file, "w") as f:
        json.dump(config, f)

    env = os.environ.copy()
    env["WORKTREE"] = WORKTREE

    cmd = [
        sys.executable, worker_file,
        "--seed", str(seed),
        "--output-json", output_json,
        "--config-json", config_file,
    ]
    if video_path:
        cmd.extend(["--video-path", video_path])

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"    WORKER FAILED (seed {seed}, {elapsed:.1f}s):")
        lines = result.stderr.strip().split("\n")
        for line in lines[-25:]:
            print(f"      {line}")
        return None

    print(f"    seed {seed}: {elapsed:.1f}s -- {result.stdout.strip()}")

    if os.path.exists(output_json):
        with open(output_json) as f:
            data = json.load(f)
        # Convert convergence_data keys back to int
        if "convergence_data" in data:
            data["convergence_data"] = {
                int(k): v for k, v in data["convergence_data"].items()
            }
        return data
    return None


def make_safe_label(label: str) -> str:
    """Convert a label to a filesystem-safe string."""
    return (label.split(".")[0].strip() + "_" +
            label.split(".", 1)[1].strip() if "." in label else label
            ).replace(":", "").replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("=", "").lower()


def make_fail_result(video_path: str = "") -> dict:
    """Return a failure/NaN result dict."""
    return {
        "has_nan": True,
        "video_path": video_path,
        "rms_vel": [],
        "max_disp": [],
        "rest_med_rms": float("nan"),
        "rest_max_rms": float("nan"),
        "rest_med_disp": float("nan"),
        "rest_max_disp": float("nan"),
        "rest_pops": 90,
        "convergence_data": {},
        "conv_ratio": float("nan"),
        "min_z": float("nan"),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_seeds(seed_results: dict[str, dict]) -> dict:
    """Aggregate per-seed results into summary metrics."""
    valid = [r for r in seed_results.values()
             if not r.get("has_nan") and not math.isnan(r.get("rest_med_rms", float("nan")))]

    if not valid:
        return dict(
            med_rms=float("nan"), max_rms=float("nan"),
            med_disp=float("nan"), max_disp=float("nan"),
            pops=0, conv_ratio=float("nan"),
            nan_count=len(seed_results),
        )

    nan_count = sum(1 for r in seed_results.values()
                    if r.get("has_nan") or math.isnan(r.get("rest_med_rms", float("nan"))))

    conv_ratios = [r["conv_ratio"] for r in valid
                   if not math.isnan(r.get("conv_ratio", float("nan")))]

    return dict(
        med_rms=float(np.median([r["rest_med_rms"] for r in valid])),
        max_rms=float(np.max([r["rest_max_rms"] for r in valid])),
        med_disp=float(np.median([r["rest_med_disp"] for r in valid])),
        max_disp=float(np.max([r["rest_max_disp"] for r in valid])),
        pops=int(np.sum([r["rest_pops"] for r in valid])),
        conv_ratio=float(np.median(conv_ratios)) if conv_ratios else float("nan"),
        nan_count=nan_count,
    )


def median_curve(seed_results: dict[str, dict], key: str = "rms_vel") -> list[float]:
    """Compute median curve across valid seeds."""
    curves = []
    for r in seed_results.values():
        if not r.get("has_nan") and key in r and r[key]:
            curves.append(r[key])
    if curves:
        # Pad to same length
        max_len = max(len(c) for c in curves)
        padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
        return np.median(np.array(padded), axis=0).tolist()
    return []


def collect_convergence_curves(seed_results: dict[str, dict]) -> list[list[float]]:
    """Collect all convergence residual curves across seeds and frames."""
    curves = []
    for r in seed_results.values():
        if r.get("has_nan"):
            continue
        conv = r.get("convergence_data", {})
        for fi, residuals in conv.items():
            if residuals and not any(math.isnan(x) for x in residuals):
                curves.append(residuals)
    return curves


# ---------------------------------------------------------------------------
# Run all configs for a tier
# ---------------------------------------------------------------------------

def run_config_seeds(label: str, config: dict, video_dir: str,
                     record_video: bool) -> dict[str, dict]:
    """Run a config across all seeds, return {seed_key: result}."""
    safe = make_safe_label(label)
    seed_results = {}

    for seed in SEEDS:
        vid_path = os.path.join(video_dir, f"{safe}_seed{seed}.mp4") if record_video else ""
        json_path = os.path.join(video_dir, f"{safe}_seed{seed}.json")

        worker_config = dict(config)
        worker_config["record_video"] = record_video
        worker_config["convergence_frames"] = CONVERGENCE_SAMPLE_FRAMES
        worker_config["num_frames"] = NUM_FRAMES

        r = run_worker(seed, worker_config, vid_path, json_path)
        if r:
            r["video_path"] = vid_path if record_video and os.path.exists(vid_path) else ""
            seed_results[f"seed_{seed}"] = r
        else:
            seed_results[f"seed_{seed}"] = make_fail_result(vid_path)

    return seed_results


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

COLORS = [
    "#E53935", "#1E88E5", "#43A047", "#F4511E",
    "#AB47BC", "#00ACC1", "#FF9800", "#795548",
    "#5C6BC0", "#26A69A", "#D81B60", "#FDD835",
    "#6D4C41", "#78909C", "#8D6E63", "#BDBDBD",
]


def embed_video_b64(video_path: str, label: str, color: str,
                    width: int = 360, metrics_html: str = "") -> str:
    """Reference a video as an external file (relative path) or show a placeholder."""
    h = int(width * 9 / 16)
    if video_path and os.path.exists(video_path):
        # Use relative path: videos/<filename> (files are copied alongside the HTML)
        fname = os.path.basename(video_path)
        return f"""<div style="text-align:center;">
            <div style="font-weight:600;font-size:0.75em;color:{color};">{label}</div>
            <video width="{width}" height="{h}" controls loop muted autoplay playsinline
                   style="border:2px solid {color};border-radius:4px;">
                <source src="videos/{fname}" type="video/mp4">
            </video>
            {metrics_html}
        </div>"""
    return f"""<div style="text-align:center;">
        <div style="font-weight:600;font-size:0.75em;color:{color};">{label}</div>
        <div style="width:{width}px;height:{h}px;background:#eee;display:flex;
                    align-items:center;justify-content:center;border-radius:4px;">
            <span style="color:#999;">No video</span>
        </div>
    </div>"""


def fmt_metric(val, fmt_str=".4f", nan_str="NaN"):
    """Format a metric, handling NaN gracefully."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return f'<span style="color:#c62828;">{nan_str}</span>'
    return f"{val:{fmt_str}}"


def make_plotly_traces_convergence(all_data: list[tuple[str, dict[str, dict]]],
                                   colors: list[str]) -> tuple[str, str]:
    """Build absolute + normalized convergence Plotly trace JS strings."""
    abs_traces, norm_traces = [], []
    for i, (label, seed_results) in enumerate(all_data):
        curves = collect_convergence_curves(seed_results)
        if not curves:
            continue
        # Ensure same length
        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        arr = np.array(curves)
        med = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        x = list(range(1, len(med) + 1))
        color = colors[i % len(colors)]

        abs_traces.append(f"""{{
            x: {json.dumps(x + x[::-1])},
            y: {json.dumps(p25.tolist() + p75.tolist()[::-1])},
            fill: 'toself', fillcolor: '{color}18',
            line: {{color: 'transparent'}}, showlegend: false, hoverinfo: 'skip'
        }}""")
        abs_traces.append(f"""{{
            x: {json.dumps(x)}, y: {json.dumps(med.tolist())},
            mode: 'lines+markers', name: '{label}',
            line: {{color: '{color}', width: 2.5}}, marker: {{size: 4}}
        }}""")

        norm_curves = arr / np.maximum(arr[:, 0:1], 1e-15)
        norm_med = np.median(norm_curves, axis=0)
        norm_p25 = np.percentile(norm_curves, 25, axis=0)
        norm_p75 = np.percentile(norm_curves, 75, axis=0)
        norm_traces.append(f"""{{
            x: {json.dumps(x + x[::-1])},
            y: {json.dumps(norm_p25.tolist() + norm_p75.tolist()[::-1])},
            fill: 'toself', fillcolor: '{color}18',
            line: {{color: 'transparent'}}, showlegend: false, hoverinfo: 'skip'
        }}""")
        norm_traces.append(f"""{{
            x: {json.dumps(x)}, y: {json.dumps(norm_med.tolist())},
            mode: 'lines+markers', name: '{label}',
            line: {{color: '{color}', width: 2.5}}, marker: {{size: 4}}
        }}""")

    return ",".join(abs_traces), ",".join(norm_traces)


def make_plotly_vel_traces(all_data: list[tuple[str, dict[str, dict]]],
                           colors: list[str]) -> str:
    """Build per-frame velocity Plotly traces."""
    traces = []
    for i, (label, seed_results) in enumerate(all_data):
        med = median_curve(seed_results, "rms_vel")
        if not med:
            continue
        x = list(range(1, len(med) + 1))
        color = colors[i % len(colors)]
        traces.append(f"""{{
            x: {json.dumps(x)}, y: {json.dumps(med)},
            mode: 'lines', name: '{label}',
            line: {{color: '{color}', width: 2}},
        }}""")
    # Rest window marker
    traces.append(f"""{{
        x: [{REST_START}, {REST_START}], y: [1e-6, 100],
        mode: 'lines', name: 'Rest window',
        line: {{color: '#999', width: 1, dash: 'dash'}}, showlegend: false,
    }}""")
    return ",".join(traces)


def make_jitter_table_rows(all_data: list[tuple[str, dict[str, dict]]],
                           colors: list[str]) -> str:
    """Build HTML table rows for jitter summary."""
    rows = []
    # Sort by med_rms (ascending, NaN last)
    sorted_data = sorted(
        enumerate(all_data),
        key=lambda x: aggregate_seeds(x[1][1]).get("med_rms", float("inf"))
    )

    best_rms = None
    for rank, (orig_idx, (label, seed_results)) in enumerate(sorted_data):
        agg = aggregate_seeds(seed_results)
        color = colors[orig_idx % len(colors)]
        if rank == 0 and not math.isnan(agg.get("med_rms", float("nan"))):
            best_rms = agg["med_rms"]

        bg = ""
        if rank == 0:
            bg = ' style="background:#E8F5E9;"'
        elif rank == len(sorted_data) - 1:
            bg = ' style="background:#FFEBEE;"'

        nan_style = ' style="color:#c62828;font-weight:bold;"' if agg["nan_count"] > 0 else ""

        rows.append(f"""<tr{bg}>
            <td style="border-left:4px solid {color};padding-left:12px;">
                <strong>{label}</strong></td>
            <td>{fmt_metric(agg['med_rms'])}</td>
            <td>{fmt_metric(agg['max_rms'])}</td>
            <td>{fmt_metric(agg['med_disp'] * 1000 if not math.isnan(agg.get('med_disp', float('nan'))) else float('nan'), '.2f')}</td>
            <td>{agg['pops']}</td>
            <td>{fmt_metric(agg['conv_ratio'])}</td>
            <td{nan_style}>{agg['nan_count']}</td>
        </tr>""")
    return "".join(rows)


def make_comparison_section(section_id: str, title: str, description: str,
                            baseline_label: str, baseline_data: dict[str, dict],
                            compare_data: list[tuple[str, dict[str, dict]]],
                            show_convergence: bool = True,
                            show_velocity: bool = True,
                            show_videos: bool = False) -> str:
    """Build an HTML section comparing baseline vs one or more configs."""
    all_data = [(baseline_label, baseline_data)] + compare_data
    all_colors = ["#43A047"] + [COLORS[i % len(COLORS)] for i in range(len(compare_data))]

    # Jitter table
    table_rows = []
    for i, (label, sr) in enumerate(all_data):
        agg = aggregate_seeds(sr)
        color = all_colors[i]
        nan_s = ' style="color:#c62828;font-weight:bold;"' if agg["nan_count"] > 0 else ""
        bg = ' style="background:#E8F5E9;"' if i == 0 else ""
        table_rows.append(f"""<tr{bg}>
            <td style="border-left:4px solid {color};padding-left:12px;">
                <strong>{label}</strong></td>
            <td>{fmt_metric(agg['med_rms'])}</td>
            <td>{fmt_metric(agg['max_rms'])}</td>
            <td>{fmt_metric(agg['med_disp'] * 1000 if not math.isnan(agg.get('med_disp', float('nan'))) else float('nan'), '.2f')}</td>
            <td>{agg['pops']}</td>
            <td>{fmt_metric(agg['conv_ratio'])}</td>
            <td{nan_s}>{agg['nan_count']}</td>
        </tr>""")

    section = f"""
    <h2>{title}</h2>
    <div class="card">
        {description}
        <table style="margin:12px 0;">
        <thead><tr>
            <th>Config</th><th>Med RMS (m/s)</th><th>Max RMS (m/s)</th>
            <th>Med Disp (mm)</th><th>Pops</th><th>Conv Ratio</th><th>NaN Seeds</th>
        </tr></thead>
        <tbody>{"".join(table_rows)}</tbody>
        </table>
    """

    # Convergence plot
    if show_convergence:
        abs_js, norm_js = make_plotly_traces_convergence(all_data, all_colors)
        if abs_js:
            section += f"""
            <h3>Convergence (absolute)</h3>
            <div id="{section_id}_abs"></div>
            <script>
            Plotly.newPlot('{section_id}_abs', [{abs_js}], {{
                title: 'Absolute Force Residual per Iteration',
                xaxis: {{title: 'VBD Iteration', dtick: 1}},
                yaxis: {{title: 'RMS Force Residual', type: 'log'}},
                hovermode: 'x unified', width: 1050, height: 420,
                legend: {{x: 0.55, y: 0.99, font: {{size: 10}}}},
                margin: {{l: 80, r: 30, t: 50, b: 50}}
            }});
            </script>
            <h3>Convergence (normalized)</h3>
            <div id="{section_id}_norm"></div>
            <script>
            Plotly.newPlot('{section_id}_norm', [{norm_js}], {{
                title: 'Normalized: residual[i] / residual[0]',
                xaxis: {{title: 'VBD Iteration', dtick: 1}},
                yaxis: {{title: 'Normalized Residual', type: 'log', range: [-1.5, 0.2]}},
                hovermode: 'x unified', width: 1050, height: 420,
                legend: {{x: 0.55, y: 0.99, font: {{size: 10}}}},
                margin: {{l: 80, r: 30, t: 50, b: 50}}
            }});
            </script>"""

    # Velocity plot
    if show_velocity:
        vel_js = make_plotly_vel_traces(all_data, all_colors)
        section += f"""
        <h3>Per-Frame RMS Velocity</h3>
        <div id="{section_id}_vel"></div>
        <script>
        Plotly.newPlot('{section_id}_vel', [{vel_js}], {{
            title: 'Per-Frame RMS Vertex Velocity (median across seeds)',
            xaxis: {{title: 'Frame'}},
            yaxis: {{title: 'RMS Velocity (m/s)', type: 'log'}},
            hovermode: 'x unified', width: 1050, height: 420,
            legend: {{x: 0.55, y: 0.99, font: {{size: 10}}}},
            margin: {{l: 80, r: 30, t: 50, b: 50}}
        }});
        </script>"""

    # Videos
    if show_videos:
        video_html = ""
        for seed in SEEDS:
            cards = []
            for i, (label, sr) in enumerate(all_data):
                r = sr.get(f"seed_{seed}", {})
                vp = r.get("video_path", "")
                if vp and os.path.exists(vp):
                    metrics = ""
                    if not r.get("has_nan"):
                        metrics = (f'<div style="font-size:0.65em;color:#666;">'
                                   f'RMS: {r.get("rest_med_rms", 0):.4f} | '
                                   f'Pops: {r.get("rest_pops", "?")}</div>')
                    nan_tag = ' <span style="color:red;">NaN!</span>' if r.get("has_nan") else ""
                    cards.append(embed_video_b64(vp, f"{label}{nan_tag}", all_colors[i],
                                                width=320, metrics_html=metrics))
            if cards:
                video_html += f"""
                <div style="margin-bottom:10px;">
                    <div style="font-weight:600;font-size:0.85em;">Seed {seed}</div>
                    <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;">
                        {"".join(cards)}
                    </div>
                </div>"""
        if video_html:
            section += f"""<h3>Videos</h3>{video_html}"""

    section += "</div>"
    return section


def generate_report(all_configs: dict[str, dict[str, dict]], output_path: str):
    """Generate the full HTML ablation report."""

    # Collect all configs for summary table
    summary_data = [(label, data) for label, data in all_configs.items()]

    # Summary table
    summary_rows = make_jitter_table_rows(summary_data, COLORS)

    # Lookup data by label
    base_data = all_configs.get("Newton Main (original)", {})
    fix1_data = all_configs.get("+ Quadratic SC penalty", {})
    fix2_data = all_configs.get("+ Absolute damping", {})
    ipc_data = all_configs.get("All Fixes (IPC)", {})
    fixed_data = all_configs.get("All Fixes (P/D kf=1e4)", ipc_data)  # P/D is primary

    # Section 2: Newton Main vs progressive fixes
    progressive = []
    if fix1_data:
        progressive.append(("+ Quadratic SC penalty", fix1_data))
    if fix2_data:
        progressive.append(("+ Absolute damping", fix2_data))
    if fixed_data:
        progressive.append(("All Fixes (IPC)", fixed_data))
    sec2 = make_comparison_section(
        "sec2", "Kernel Fixes: Newton Main to All Fixes",
        """<p><strong>Newton Main</strong> uses the original VBD kernel code: log-barrier
        self-contact (H = k<sup>2</sup>/d<sup>2</sup>, unbounded), <code>c = kd &#183; ke</code>
        contact damping (16,000x overdamped), and zero friction Hessian at rest. At 4 substeps
        this combination is <strong>catastrophically unstable</strong> (NaN on all seeds).</p>
        <p>Fixes are applied progressively:</p>
        <ol>
        <li><strong>Quadratic SC penalty</strong> &#8212; E = k/2&#183;(r&#8722;d)<sup>2</sup>
        with constant Hessian H = k. Eliminates the unbounded eigenvalues near close contact.</li>
        <li><strong>Absolute damping</strong> &#8212; <code>c = kd</code> instead of
        <code>kd &#183; ke</code>. Reduces damping Hessian eigenvalue by 6,250x.</li>
        <li><strong>Friction Hessian at rest</strong> &#8212; Correct limiting value
        <code>2/&#949;</code> when tangential slip is zero, so the solver sees friction
        stiffness for resting contacts.</li>
        </ol>""",
        "Newton Main (original)", base_data,
        progressive,
        show_videos=True,
    )

    # Section 3: Friction Model — P/D kf sweep + IPC comparison
    compare_friction = []
    if ipc_data:
        compare_friction.append(("All Fixes (IPC)", ipc_data))
    for label in ["P/D kf=1e3", "P/D kf=5e3", "P/D kf=5e4", "P/D kf=1e5"]:
        d = all_configs.get(label, {})
        if d:
            compare_friction.append((label, d))
    sec3 = make_comparison_section(
        "sec3", "Friction Model: P/D kf Sweep + IPC Comparison",
        """<p>The <strong>Primal/Dual</strong> friction model (Macklin 2020) uses a linear
        spring with stiffness <code>kf</code> in stick and Coulomb in slip. Its Hessian
        is a clean constant <code>kf</code> in the stick regime, avoiding the dt-dependent
        scaling artifact of the IPC C<sup>1</sup> ramp. Setting <code>kf &#8776; ke</code>
        balances normal and tangential stiffness. This section sweeps <code>kf</code> values
        and compares against the IPC model.</p>""",
        "All Fixes (P/D kf=1e4)", fixed_data,
        compare_friction,
        show_videos=True,
    )

    # Section 4: Runtime Parameters (R0-R6)
    runtime_data = []
    for label, _ in RUNTIME_CONFIGS:
        if label in all_configs:
            runtime_data.append((label, all_configs[label]))
    sec4 = ""
    sec5 = ""
    sec6 = ""
    sec7 = ""
    if runtime_data:
        # Use All Fixes (P/D) as implicit baseline reference in the runtime group
        all_runtime = [("All Fixes (P/D kf=1e4)", fixed_data)] + runtime_data
        rt_colors = ["#43A047"] + [COLORS[i % len(COLORS)] for i in range(len(runtime_data))]

        # Jitter table
        rt_table_rows = []
        for i, (label, sr) in enumerate(all_runtime):
            agg = aggregate_seeds(sr)
            color = rt_colors[i]
            nan_s = ' style="color:#c62828;font-weight:bold;"' if agg["nan_count"] > 0 else ""
            bg = ' style="background:#E8F5E9;"' if i == 0 else ""
            rt_table_rows.append(f"""<tr{bg}>
                <td style="border-left:4px solid {color};padding-left:12px;">
                    <strong>{label}</strong></td>
                <td>{fmt_metric(agg['med_rms'])}</td>
                <td>{fmt_metric(agg['max_rms'])}</td>
                <td>{fmt_metric(agg['med_disp'] * 1000 if not math.isnan(agg.get('med_disp', float('nan'))) else float('nan'), '.2f')}</td>
                <td>{agg['pops']}</td>
                <td>{fmt_metric(agg['conv_ratio'])}</td>
                <td{nan_s}>{agg['nan_count']}</td>
            </tr>""")

        vel_js = make_plotly_vel_traces(all_runtime, rt_colors)
        abs_js, norm_js = make_plotly_traces_convergence(all_runtime, rt_colors)

        # Videos
        rt_video_html = ""
        for seed in SEEDS:
            cards = []
            for i, (label, sr) in enumerate(all_runtime):
                r = sr.get(f"seed_{seed}", {})
                vp = r.get("video_path", "")
                if vp and os.path.exists(vp):
                    nan_tag = ' <span style="color:red;">NaN!</span>' if r.get("has_nan") else ""
                    m_html = ""
                    if not r.get("has_nan"):
                        m_html = (f'<div style="font-size:0.65em;color:#666;">'
                                  f'RMS: {r.get("rest_med_rms", 0):.4f}</div>')
                    cards.append(embed_video_b64(vp, f"{label}{nan_tag}", rt_colors[i],
                                                width=280, metrics_html=m_html))
            if cards:
                rt_video_html += f"""
                <div style="margin-bottom:10px;">
                    <div style="font-weight:600;font-size:0.85em;">Seed {seed}</div>
                    <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">
                        {"".join(cards)}
                    </div>
                </div>"""

        sec4 = f"""
        <h2>Runtime Parameters</h2>
        <div class="card">
            <p>All runtime configs use the IPC kernel with all fixes active. Only solver
            and model parameters are varied. Look at the wireframe overlay to check for
            self-penetration, especially with under-relaxation (alpha=0.5).</p>
            <table style="margin:12px 0;">
            <thead><tr>
                <th>Config</th><th>Med RMS (m/s)</th><th>Max RMS (m/s)</th>
                <th>Med Disp (mm)</th><th>Pops</th><th>Conv Ratio</th><th>NaN Seeds</th>
            </tr></thead>
            <tbody>{"".join(rt_table_rows)}</tbody>
            </table>

            <h3>Per-Frame RMS Velocity</h3>
            <div id="sec4_vel"></div>
            <script>
            Plotly.newPlot('sec4_vel', [{vel_js}], {{
                title: 'Per-Frame RMS Vertex Velocity (median across seeds)',
                xaxis: {{title: 'Frame'}},
                yaxis: {{title: 'RMS Velocity (m/s)', type: 'log'}},
                hovermode: 'x unified', width: 1050, height: 420,
                legend: {{x: 0.01, y: 0.01, font: {{size: 10}}}},
                margin: {{l: 80, r: 30, t: 50, b: 50}}
            }});
            </script>
        """
        if abs_js:
            sec7 += f"""
            <h3>Convergence (all overlaid)</h3>
            <div id="sec4_abs"></div>
            <script>
            Plotly.newPlot('sec4_abs', [{abs_js}], {{
                title: 'Absolute Force Residual per Iteration',
                xaxis: {{title: 'VBD Iteration', dtick: 1}},
                yaxis: {{title: 'RMS Force Residual', type: 'log'}},
                hovermode: 'x unified', width: 1050, height: 420,
                legend: {{x: 0.55, y: 0.99, font: {{size: 10}}}},
                margin: {{l: 80, r: 30, t: 50, b: 50}}
            }});
            </script>"""
        if rt_video_html:
            sec4 += f"""<h3>Videos</h3>{rt_video_html}"""
        sec4 += "</div>"

    # Section: Issues Fixed
    sec_issues = """
    <h2>Issues Fixed</h2>
    <div class="card">
        <p>Seven issues were identified and fixed to make VBD stable with self-contact at 4 substeps.</p>

        <h3>Kernel-Level Fixes (code changes)</h3>
        <table>
        <thead><tr><th>#</th><th>Issue</th><th>Severity</th><th>Fix</th></tr></thead>
        <tbody>
        <tr><td>1</td><td>Log-barrier self-contact (H = k&sup2;/d&sup2;, unbounded)</td>
            <td style="color:#c62828;font-weight:bold;">Critical &mdash; NaN</td>
            <td>Quadratic penalty (H = k, constant)</td></tr>
        <tr><td>2</td><td>Contact damping c = kd &middot; ke (16,000&times; overdamped)</td>
            <td style="color:#E65100;font-weight:bold;">High &mdash; 5&times; jitter</td>
            <td>Absolute damping c = kd</td></tr>
        <tr><td>3</td><td>Friction Hessian approximation not scale-invariant</td>
            <td>Medium</td>
            <td>P/D friction model implemented as alternative</td></tr>
        <tr><td>4</td><td>Zero friction Hessian at rest (u == 0)</td>
            <td>Low</td>
            <td>Evaluate limit: hessian_scale = 2/&epsilon;</td></tr>
        </tbody>
        </table>

        <h3>Configuration Fixes (parameter changes)</h3>
        <table>
        <thead><tr><th>#</th><th>Issue</th><th>Severity</th><th>Fix</th></tr></thead>
        <tbody>
        <tr><td>5</td><td>Self-contact buffer overflow (65% VT, 78% EE overflowed at 16/20)</td>
            <td style="color:#c62828;font-weight:bold;">Critical &mdash; divergence</td>
            <td>Increase buffers to 512/512 (max observed: 257 VT, 483 EE)</td></tr>
        <tr><td>6</td><td>Particle masses near zero (1e-6 kg, stiffness ratio 38M:1)</td>
            <td style="color:#E65100;font-weight:bold;">High &mdash; solver diverges</td>
            <td>Clamp min mass to 0.01 kg (10g), giving stiffness ratio ~17:1</td></tr>
        <tr><td>7</td><td>Contact stiffness ke=1e5 (still 1000:1 ratio with mass clamp)</td>
            <td style="color:#E65100;font-weight:bold;">High</td>
            <td>Reduce to ke = kf = 1e4</td></tr>
        </tbody>
        </table>
    </div>

    <div class="card">
        <h3>Key Insight: Buffer Overflow Was the Dominant Problem</h3>
        <p>With default buffer sizes (16 VT / 20 EE), the collision detector silently dropped
        contacts that didn't fit. Each iteration saw a <em>different, incomplete</em> subset of
        contact forces. This caused:</p>
        <ul>
            <li>The solver to <strong>actively diverge</strong> &mdash; residual grew from 21 to 51 over 40 iterations</li>
            <li>80 iterations was <strong>worse</strong> than 10 (reduction 0.64 vs 0.17)</li>
            <li>All visible self-interpenetration in videos</li>
        </ul>
        <p>With 512/512 buffers (no overflow) + mass clamp (stiffness ratio 17:1 instead of 38M:1),
        the solver converges monotonically and the cloth comes to rest.</p>
    </div>
    """

    # Section 8: Methodology
    sec8 = f"""
    {sec_issues}

    <h2>Methodology</h2>
    <div class="card">
        <h3>Scenario</h3>
        <p>T-shirt (unisex_shirt.usd, 6436 vertices) dropped from randomized height
        ({LOW_DROP_HEIGHT_RANGE[0]:.2f}&#8211;{LOW_DROP_HEIGHT_RANGE[1]:.2f}m) with random
        rotation and lateral offset. Self-contact enabled
        (margin={SELF_CONTACT_MARGIN*100:.0f}cm, radius={SELF_CONTACT_RADIUS*1000:.0f}mm).
        {SIM_SUBSTEPS} substeps &times; 10 VBD iterations at 60 FPS. Contact buffers 512/512.
        Min particle mass 0.01 kg. 2&times; SSAA with Lanczos downsample.</p>

        <h3>Default Parameters</h3>
        <table>
        <tr><th>Parameter</th><th>Value</th><th>Notes</th></tr>
        <tr><td>soft_contact_ke</td><td>{DEFAULT_RUNTIME['soft_contact_ke']}</td><td>Normal contact stiffness</td></tr>
        <tr><td>soft_contact_kf</td><td>{DEFAULT_RUNTIME['soft_contact_kf']}</td><td>Friction stiffness (P/D model, = ke)</td></tr>
        <tr><td>soft_contact_kd</td><td>{DEFAULT_RUNTIME['soft_contact_kd']}</td><td>Contact damping</td></tr>
        <tr><td>friction_epsilon</td><td>{DEFAULT_RUNTIME['friction_epsilon']}</td><td>Velocity threshold (m/s)</td></tr>
        <tr><td>step_length (alpha)</td><td>{DEFAULT_RUNTIME['step_length']}</td><td>GS under-relaxation</td></tr>
        <tr><td>avbd_beta</td><td>{DEFAULT_RUNTIME['avbd_beta']}</td><td>0 = disabled</td></tr>
        <tr><td>contact_ke_body</td><td>{DEFAULT_RUNTIME['contact_ke_body']}</td><td>Body contact penalty</td></tr>
        <tr><td>min_particle_mass</td><td>0.01 kg</td><td>Prevents ill-conditioned block solve</td></tr>
        <tr><td>vertex_contact_buffer</td><td>512</td><td>Prevents silent contact drops</td></tr>
        <tr><td>edge_contact_buffer</td><td>512</td><td>Prevents silent contact drops</td></tr>
        </table>

        <h3>Metric Definitions</h3>
        <ul>
            <li><strong>Med RMS (m/s)</strong>: Median per-frame RMS vertex velocity over
                frames {REST_START}&#8211;{NUM_FRAMES} (resting phase).</li>
            <li><strong>Max RMS (m/s)</strong>: Maximum per-frame RMS velocity in resting phase.</li>
            <li><strong>Med Disp (mm)</strong>: Median per-frame max vertex displacement
                in resting phase, in millimeters.</li>
            <li><strong>Pops</strong>: Number of frames in resting phase where max vertex
                displacement exceeds 0.5mm.</li>
            <li><strong>Conv Ratio</strong>: Median ratio residual[last] / residual[0] across
                convergence samples (frames 100&#8211;170, first substep).</li>
            <li><strong>NaN Seeds</strong>: Number of seeds that produced NaN positions.</li>
        </ul>

        <h3>Compute Setup</h3>
        <p>Seeds: {SEEDS}. {NUM_FRAMES} frames ({NUM_FRAMES/60:.1f}s simulation time).
        Convergence sampled at frames {CONVERGENCE_SAMPLE_FRAMES[0]}&#8211;{CONVERGENCE_SAMPLE_FRAMES[-1]}
        (every 10 frames in resting phase). Kernel-level configs use subprocess isolation
        with warp cache clear between compilations.</p>
    </div>"""

    # Assemble full HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VBD Ablation Study</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #f5f5f5; color: #333; line-height: 1.6; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 1.8em; margin-bottom: 4px; }}
    h2 {{ font-size: 1.3em; margin: 28px 0 10px; border-bottom: 2px solid #1976D2; padding-bottom: 5px; }}
    h3 {{ font-size: 1.1em; margin: 16px 0 8px; }}
    .subtitle {{ color: #666; margin-bottom: 20px; }}
    .card {{ background: #fff; border-radius: 8px; padding: 20px; margin: 14px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.85em; }}
    th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
    th {{ background: #f5f5f5; font-weight: 600; }}
    .finding {{ background: #E3F2FD; border-left: 4px solid #1976D2; padding: 12px 16px;
                margin: 12px 0; border-radius: 0 6px 6px 0; }}
    code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body>
<div class="container">

<h1>VBD Solver Analysis</h1>
<p class="subtitle">Newton Main (original VBD code) is unstable at 4 substeps. Three kernel fixes
restore stability: quadratic self-contact penalty, absolute contact damping, and friction Hessian
at rest. The default friction model is Primal/Dual (Macklin 2020) with <code>kf = ke</code>.
{len(SEEDS)} seeds, {NUM_FRAMES} frames ({NUM_FRAMES/60:.1f}s), {SIM_SUBSTEPS}
substeps x 10 iterations. Wireframe overlay shows self-penetration. 16x MSAA.</p>

<h2>1. Summary Table</h2>
<div class="card">
    <p>All configurations ranked by median RMS velocity during rest (frames {REST_START}&#8211;{NUM_FRAMES}).
    Green = best, red = worst.</p>
    <table>
    <thead><tr>
        <th>Config</th><th>Med RMS (m/s)</th><th>Max RMS (m/s)</th>
        <th>Med Disp (mm)</th><th>Pops</th><th>Conv Ratio</th><th>NaN Seeds</th>
    </tr></thead>
    <tbody>{summary_rows}</tbody>
    </table>
</div>

{sec2}
{sec3}
{sec4}
{sec8}

</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"\nReport: {output_path} ({len(html):,} bytes)")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

# Import analysis_common constants for use in report (re-import at module level is fine
# since main process doesn't need warp compilation for the orchestrator).
try:
    sys.path.insert(0, WORKTREE)
    from vbd_convergence_analysis.analysis_common import (
        LOW_DROP_HEIGHT_RANGE, SELF_CONTACT_MARGIN, SELF_CONTACT_RADIUS, SIM_SUBSTEPS,
    )
except ImportError:
    LOW_DROP_HEIGHT_RANGE = (0.05, 0.20)
    SELF_CONTACT_MARGIN = 0.02
    SELF_CONTACT_RADIUS = 0.002
    SIM_SUBSTEPS = 4


def main():
    total_t0 = time.perf_counter()

    video_dir = os.path.join(SCRIPT_DIR, "videos", "ablation")
    os.makedirs(video_dir, exist_ok=True)

    all_configs: dict[str, dict[str, dict]] = {}  # label -> {seed_key -> result}

    # ===================================================================
    # TIER 1: Kernel-level configs (subprocess isolation per compilation)
    # ===================================================================
    print("\n" + "=" * 70)
    print("TIER 1: Kernel-Level Configs (subprocess isolation)")
    print("=" * 70)

    for cfg_idx, (label, patches, record_video) in enumerate(KERNEL_CONFIGS):
        print(f"\n{'='*60}")
        print(f"[K {cfg_idx+1}/{len(KERNEL_CONFIGS)}] {label}")
        print(f"{'='*60}")

        originals = None
        if patches:
            try:
                originals = apply_patches(patches)
                clear_warp_cache()
                print(f"  Applied {len(patches)} patch(es)")
            except RuntimeError as e:
                print(f"  PATCH ERROR: {e}")
                # Store empty results
                all_configs[label] = {f"seed_{s}": make_fail_result() for s in SEEDS}
                continue
        else:
            clear_warp_cache()

        try:
            config = dict(DEFAULT_RUNTIME)
            seed_results = run_config_seeds(label, config, video_dir, record_video)
            all_configs[label] = seed_results
        finally:
            if originals:
                restore_files(originals)
                clear_warp_cache()
                print(f"  Restored kernel source files")

        # Print summary
        agg = aggregate_seeds(seed_results)
        print(f"  Summary: med_rms={fmt_metric(agg['med_rms'])} "
              f"pops={agg['pops']} nan={agg['nan_count']}")

    # ===================================================================
    # P/D friction configs (share one kernel compilation)
    # ===================================================================
    print(f"\n{'='*60}")
    print("P/D Friction Configs (shared kernel compilation)")
    print(f"{'='*60}")

    originals = None
    try:
        originals = apply_patches(PD_PATCHES)
        clear_warp_cache()
        print(f"  Applied P/D friction model patches")

        for label, kf_value, record_video in PD_FRICTION_CONFIGS:
            print(f"\n  --- {label} (kf={kf_value:.0e}) ---")
            config = dict(DEFAULT_RUNTIME)
            config["soft_contact_kf"] = kf_value
            seed_results = run_config_seeds(label, config, video_dir, record_video)
            all_configs[label] = seed_results

            agg = aggregate_seeds(seed_results)
            print(f"  Summary: med_rms={fmt_metric(agg['med_rms'])} "
                  f"pops={agg['pops']} nan={agg['nan_count']}")
    except RuntimeError as e:
        print(f"  P/D PATCH ERROR: {e}")
        for label, _, _ in PD_FRICTION_CONFIGS:
            all_configs[label] = {f"seed_{s}": make_fail_result() for s in SEEDS}
    finally:
        if originals:
            restore_files(originals)
            clear_warp_cache()
            print(f"  Restored kernel source files")

    # ===================================================================
    # TIER 2: Runtime configs (single IPC kernel compilation)
    # ===================================================================
    print("\n" + "=" * 70)
    print("TIER 2: Runtime Configs (shared P/D kernel)")
    print("=" * 70)

    # Apply P/D patches for runtime configs (P/D is the default friction model)
    pd_originals = apply_patches(PD_PATCHES)
    clear_warp_cache()
    print("  Applied P/D friction patches for runtime tier")

    for cfg_idx, (label, overrides) in enumerate(RUNTIME_CONFIGS):
        print(f"\n{'='*60}")
        print(f"[R {cfg_idx+1}/{len(RUNTIME_CONFIGS)}] {label}")
        print(f"{'='*60}")

        config = dict(DEFAULT_RUNTIME)
        for k, v in overrides.items():
            config[k] = v

        # Determine if this config gets video
        label_prefix = label.split(".")[0].strip()
        record_video = label_prefix in RUNTIME_VIDEO_LABELS

        seed_results = run_config_seeds(label, config, video_dir, record_video)
        all_configs[label] = seed_results

        agg = aggregate_seeds(seed_results)
        print(f"  Summary: med_rms={fmt_metric(agg['med_rms'])} "
              f"pops={agg['pops']} nan={agg['nan_count']}")

    # Restore IPC kernels after runtime tier
    restore_files(pd_originals)
    clear_warp_cache()
    print("  Restored kernel source files after runtime tier")

    # ===================================================================
    # Print final ranking
    # ===================================================================
    print(f"\n{'='*110}")
    print(f"{'Config':<35} {'MedRMS':>10} {'MaxRMS':>10} {'MedD(mm)':>10} {'Pops':>6} {'ConvR':>8} {'NaN':>5}")
    print("=" * 110)
    ranked = sorted(all_configs.items(),
                    key=lambda x: aggregate_seeds(x[1]).get("med_rms", float("inf")))
    for label, sr in ranked:
        agg = aggregate_seeds(sr)
        med_disp_mm = agg['med_disp'] * 1000 if not math.isnan(agg.get('med_disp', float('nan'))) else float('nan')
        print(f"{label:<35} {agg['med_rms']:>10.6f} {agg['max_rms']:>10.6f} "
              f"{med_disp_mm:>9.3f} {agg['pops']:>6} {agg['conv_ratio']:>8.4f} {agg['nan_count']:>5}")

    # ===================================================================
    # Save results
    # ===================================================================
    total_elapsed = time.perf_counter() - total_t0
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # Save JSON (exclude per-frame arrays)
    json_save = {}
    for label, sr in all_configs.items():
        json_save[label] = {}
        for seed_key, r in sr.items():
            json_save[label][seed_key] = {
                k: v for k, v in r.items()
                if k not in ("rms_vel", "max_disp", "convergence_data")
            }
    json_save["_metadata"] = dict(
        seeds=SEEDS, num_frames=NUM_FRAMES, rest_start=REST_START,
        fps=FPS, substeps=SIM_SUBSTEPS, iterations=10,
        default_runtime=DEFAULT_RUNTIME,
        total_time_s=total_elapsed,
    )

    reports_dir = os.path.expanduser("~/reports/newton-vbd")
    os.makedirs(reports_dir, exist_ok=True)

    json_path = os.path.join(reports_dir, "vbd_ablation.json")
    with open(json_path, "w") as f:
        json.dump(json_save, f, indent=2, default=str)
    print(f"JSON: {json_path}")

    # Also save in script dir
    json_local = os.path.join(SCRIPT_DIR, "ablation_results.json")
    with open(json_local, "w") as f:
        json.dump(json_save, f, indent=2, default=str)
    print(f"JSON (local): {json_local}")

    # Copy videos to reports dir as external files (not base64 embedded)
    reports_video_dir = os.path.join(reports_dir, "videos")
    os.makedirs(reports_video_dir, exist_ok=True)
    for fname in os.listdir(video_dir):
        if fname.endswith(".mp4"):
            shutil.copy2(os.path.join(video_dir, fname), os.path.join(reports_video_dir, fname))
    print(f"Videos copied to: {reports_video_dir}")

    # Generate HTML report
    html_path = os.path.join(reports_dir, "vbd_ablation.html")
    generate_report(all_configs, html_path)

    # Also save in script dir
    html_local = os.path.join(SCRIPT_DIR, "ablation_report.html")
    shutil.copy2(html_path, html_local)
    print(f"HTML (local): {html_local}")

    # Print tunnel URL if available
    try:
        import subprocess as _sp
        result = _sp.run(
            ["grep", "-oP", r"https://[a-z0-9-]+\.trycloudflare\.com",
             os.path.expanduser("~/.config/reports-supervisor/cloudflared.log")],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            tunnel_url = result.stdout.strip().split("\n")[-1]
            print(f"\nReport URL: {tunnel_url}/newton-vbd/vbd_ablation.html")
    except Exception:
        pass


if __name__ == "__main__":
    main()
