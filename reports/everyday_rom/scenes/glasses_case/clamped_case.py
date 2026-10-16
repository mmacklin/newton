"""Fixed C1 rear clamp, force-driven tip fixture, passive nonlinear shape energy."""

import argparse
import json
import os
import shutil
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from clamped_geometry import (
    EI,
    ROOT_WORLD,
    SHELL_CENTER,
    THICKNESS,
    continuity,
    curvature,
    flap_basis,
    keyframe,
    material_orientation,
    material_segments,
    shell_geometry,
)
from glasses_case import Example as PreviousExample
from opening_coverage import opening_coverage
from peel_case_geometry import Config as BaseConfig
from peel_case_geometry import (
    Simulation,
    box,
    ellipsoid,
    make_basis,
    pose,
    ramp,
    surface_strain_metrics,
    write_kinematic,
)
from pod_path_energy import ModalPathEnergy
from solver_vbd_modal_path import SolverVBDModalPath

import newton


@dataclass
class Config(BaseConfig):
    frames: int = 360
    tip_attachment: bool = True
    close_enabled: bool = True
    path_energy: bool = True
    # Membrane frequency scale EA/(m L), with rectangular strip EA=12 EI/t^2.
    # This is a declared shape-prior scale, not a calibrated material law.
    path_transverse_stiffness: float = 12 * EI / (THICKNESS**2 * 0.018 * material_segments()[0].sum())
    path_restoring_energy: float = 0.03


@wp.func
def clamp_pose(t: float, enabled: bool, ds: wp.array[float], beta: wp.array[float]):
    progress = float(0.0)
    if enabled:
        progress = ramp(t, 0.8, 2.6) * (1.0 - ramp(t, 4.4, 1.1))
    tip = wp.vec3(0.0, 0.020, 0.052)
    for i in range(ds.shape[0]):
        a = progress * beta[i]
        tip += wp.vec3(0.0, -ds[i] * wp.sin(a), ds[i] * wp.cos(a))
    rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), progress * beta[beta.shape[0] - 1])
    return wp.transform(tip, rotation)


@wp.kernel
def drive_tip(
    clock: wp.array[float],
    dt: float,
    body: int,
    enabled: bool,
    ds: wp.array[float],
    beta: wp.array[float],
    p0: wp.array[wp.transform],
    v0: wp.array[wp.spatial_vector],
    p1: wp.array[wp.transform],
    v1: wp.array[wp.spatial_vector],
):
    t = clock[0]
    write_kinematic(body, clamp_pose(t, enabled, ds, beta), clamp_pose(t + dt, enabled, ds, beta), dt, p0, v0, p1, v1)


class Example(Simulation):
    def __init__(self, cfg=None):
        self.begin(None, cfg or Config())
        self.expected_frames = self.cfg.frames
        (p, f), _ = shell_geometry()
        basis = make_basis(
            p, PreviousExample.wall_trial_modes(p), 0.030, [90, 160, 220], label="rear-clamped pouch wall"
        )
        self.shell = self.elastic(p, basis, 0.030, SHELL_CENTER, "fixed_rear_pouch_frame_solved_wall_modes", True)
        self.shell_part = self.mesh("elastic_pouch_wall", p, f, self.shell, (0.022, 0.025, 0.028), True, roughness=0.77)
        for sign in [-1, 1]:
            self.mesh(
                f"oval_end_{sign}",
                *ellipsoid((0.007, 0.020, 0.035), (sign * 0.074, 0, 0), rings=20, sides=40),
                self.shell,
                (0.026, 0.029, 0.032),
                True,
                roughness=0.78,
            )
        p, f, _, basis = flap_basis()
        phi = basis.sample_phi[: len(p)].astype(float)
        inverse = np.linalg.pinv(phi.transpose(0, 2, 1).reshape(-1, basis.mode_count))
        self.path_coefficients = np.stack([inverse @ (keyframe(t)[0] - p).ravel() for t in np.linspace(0, 1, 25)])
        self.path = ModalPathEnergy(
            self.path_coefficients,
            np.linspace(0, 1, 25),
            self.cfg.path_transverse_stiffness,
            self.cfg.path_restoring_energy,
        )
        if self.cfg.path_energy:
            basis.mode_stiffness[:] = 0
            basis.raw_stiffness_matrix[:] = 0
        self.flap = self.elastic(p, basis, 0.018, ROOT_WORLD, "fixed_material_root_solved_whole_flap", True)
        self.flap_part = self.mesh(
            "whole_elastic_root_clamped_flap", p, f, self.flap, (0.030, 0.034, 0.039), True, roughness=0.72
        )
        self.former = self.builder.add_body(
            xform=pose(SHELL_CENTER), is_kinematic=True, label="visible_fixed_rigid_internal_former"
        )
        self.former_part = self.mesh(
            "visible_rigid_internal_former",
            *ellipsoid((0.067, 0.019, 0.034), rings=30, sides=64),
            self.former,
            (0.18, 0.30, 0.32),
            True,
            roughness=0.55,
        )
        ds, beta = material_segments()
        self.ds = wp.array(ds, dtype=float)
        self.beta = wp.array(beta, dtype=float)
        tip = ROOT_WORLD + np.array([0, 0, ds.sum()])
        self.clamp = self.builder.add_body(
            xform=pose(tip), is_kinematic=True, label="visible_prescribed_two_line_tip_clamp"
        )
        self.clamp_parts = []
        jaw_half_width = float(np.abs(basis.sample_points[basis.tip_samples, 0]).max()) + 0.0025
        for sign in [-1, 1]:
            self.clamp_parts.append(
                self.mesh(
                    f"visible_tip_clamp_jaw_{sign}",
                    *box((jaw_half_width, 0.0014, 0.003), (0, sign * 0.0022, -ds[-1] / 2)),
                    self.clamp,
                    (0.32, 0.40, 0.42),
                    True,
                    roughness=0.4,
                    metallic=0.4,
                )
            )
        if self.cfg.tip_attachment:
            for sample in basis.tip_samples:
                point = basis.sample_points[sample]
                self.builder.add_joint_ball(
                    self.clamp,
                    self.flap,
                    parent_xform=pose((point[0], 0, point[2] - ds.sum())),
                    child_xform=pose(point),
                    label="physical_tip_clamp_attachment",
                )
        self.mesh(
            "support_plinth", *box((0.088, 0.029, 0.004), (0, 0, 0.010)), color=(0.22, 0.28, 0.29), collision=True
        )
        # Jaws transfer forces through the declared attachments, not duplicate collision constraints.
        for index in self.clamp_parts:
            shape = self.parts[index]["shape"]
            for part in self.parts:
                if part["shape"] != shape:
                    self.builder.add_shape_collision_filter_pair(shape, part["shape"])
        self.finish()

    def make_solver(self, model, **options):
        if self.cfg.path_energy:
            return SolverVBDModalPath(model, paths={self.flap: self.path}, **options)
        return super().make_solver(model, **options)

    def drive(self):
        wp.launch(
            drive_tip,
            dim=1,
            inputs=[
                self.clock,
                self.dt,
                self.clamp,
                self.cfg.close_enabled,
                self.ds,
                self.beta,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.state_1.body_q,
                self.state_1.body_qd,
            ],
        )

    def measure(self):
        q = self.saved_q[-1]
        by_mesh = {}
        for p in self.parts:
            if p["phi"].shape[1]:
                local = p["rest"] + np.einsum("nmc,m->nc", p["phi"], q[p["q_start"] : p["q_start"] + p["phi"].shape[1]])
                by_mesh[p["name"]] = surface_strain_metrics(p["rest"], local, p["faces"])
        flap = self.parts[self.flap_part]
        local = flap["world"][-1] - ROOT_WORLD
        amplitudes = q[flap["q_start"] : flap["q_start"] + flap["phi"].shape[1]]
        value = self.path.evaluate(amplitudes)
        return dict(
            penetration_m=0.0,
            strain_by_mesh=by_mesh,
            **opening_coverage(flap["world"][-1]),
            **continuity(self.parts[self.shell_part]["world"][-1], flap["world"][-1]),
            **curvature(local),
            **material_orientation(local),
            path_progress=value.progress,
            path_energy_j=value.energy,
            path_coefficient_residual=float(np.linalg.norm(amplitudes - value.closest)),
            **{key: max(v[key] for v in by_mesh.values()) for key in next(iter(by_mesh.values()))},
        )

    def summary(self):
        m = super().summary()
        for key in [
            "max_abs_principal_strain",
            "max_abs_edge_strain",
            "max_abs_area_strain",
            "root_gap_max_m",
            "root_tangent_error_max_degrees",
            "root_normal_error_max_degrees",
            "material_inverted_cells",
            "curvature_max_width_degrees",
            "path_coefficient_residual",
        ]:
            m[key] = max(r[key] for r in self.records)
        m["material_director_min_cosine"] = min(r["material_director_min_cosine"] for r in self.records)
        m["open_curvature_degrees"] = float(
            np.mean([r["curvature_min_width_degrees"] for r in self.records if r["time"] < 0.6])
        )
        m["closed_curvature_degrees"] = (
            float(np.mean([r["curvature_min_width_degrees"] for r in self.records if 3.6 < r["time"] < 4.2]))
            if self.sim_time > 4.2
            else None
        )
        m["root_frames_fixed"] = bool(
            np.all(
                np.asarray(self.saved_poses)[:, [self.shell, self.flap]] == self.saved_poses[0][[self.shell, self.flap]]
            )
        )
        closed = [r for r in self.records if 3.6 < r["time"] < 4.2]
        m["closed_opening_covered_fraction"] = min((r["opening_covered_fraction"] for r in closed), default=None)
        m["closed_opening_min_width_margin_m"] = min((r["opening_min_width_margin_m"] for r in closed), default=None)
        return m

    def test_final(self):
        m = self.summary()
        assert m["finite_state"] and m["graph_enabled"] and m["root_frames_fixed"], m
        for key in ["max_abs_principal_strain", "max_abs_edge_strain", "max_abs_area_strain"]:
            assert m[key] < 0.15, (key, m[key])
        assert m["max_penetration_m"] < 0.001, m
        assert m["root_gap_max_m"] < 1e-6, m
        assert m["root_tangent_error_max_degrees"] < 0.01 and m["root_normal_error_max_degrees"] < 0.01, m
        assert m["material_inverted_cells"] == 0 and m["material_director_min_cosine"] > 0.9, m
        if self.cfg.close_enabled and self.cfg.tip_attachment:
            assert m["closed_opening_covered_fraction"] > 0.9999, m
            assert m["closed_opening_min_width_margin_m"] > -1e-7, m
            assert m["open_curvature_degrees"] < 10 and m["closed_curvature_degrees"] > 140, m
            assert (
                max(r["curvature_peak_six_segment_fraction"] for r in self.records if 3.6 < r["time"] < 4.2) < 0.35
            ), m

    def metadata(self):
        return {
            "slug": "glasses-case",
            "title": "Scalloped root-clamped bending pouch",
            "status": "experimental-unaccepted",
            "camera": {"position": [0.21, -0.36, 0.21], "target": [0, 0, 0.10], "fov": 38},
            "floor_z": 0.006,
            "description": "A full-width flap covers the opening before tapering into concave side scallops and a rounded front tongue, bending under a visible tip clamp.",
            "driven": "Only the visible rigid tip clamp pose. Rear rigid frames are fixed; all modal coordinates are Newton VBD unknowns.",
            "limitations": [
                "Six-mode fixed linear POD span with a passive ordered shape-path prior, not calibrated leather.",
                "Retains the accepted 18 g equal nodal mass model; it is not a uniform-density leather discretization.",
                "The rounded tongue terminates in a 3 mm chord to keep the structured full-thickness mesh nondegenerate.",
                "Parent-authored report-local private SolverVBD override pinned to 8228bf6; not a public solver extension API.",
                "Both pouch and flap rigid frames are fixed at the rear clamp; their elastic modes remain solved.",
                "Tip jaws use four explicit attachment constraints; jaw collision is disabled in every control.",
                "Rigid internal former substitutes for unsupported elastic-elastic contact.",
                "Finite-rotation material-director orientation check proposed in place of the old rest-normal-dot sign gate; strain thresholds unchanged.",
            ],
            "source_files": [
                "clamped_case.py",
                "clamped_geometry.py",
                "opening_coverage.py",
                "glasses_case.py",
                "peel_case_geometry.py",
                "contact_capacity.py",
                "bottle_geometry_local.py",
                "pod_path_energy.py",
                "solver_vbd_modal_path.py",
            ],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--no-tip-attachment", action="store_true")
    parser.add_argument("--no-close", action="store_true")
    parser.add_argument("--linear-energy", action="store_true")
    parser.add_argument("--device", default="cuda:0", help="CUDA device; full scene requires graph capture")
    parser.add_argument("--cache-dir", type=Path, help="Warp cache (default: inside output directory)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    launch_source = args.output / "launch-source"
    launch_source.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(source, launch_source / source.name)
    runtime = Path(newton.__file__).resolve().parents[1]

    def git_info(*command):
        try:
            result = subprocess.run(["git", "-C", str(runtime), *command], capture_output=True, text=True, check=False)
            return result.stdout.strip() if result.returncode == 0 else None
        except FileNotFoundError:
            return None

    head = git_info("rev-parse", "HEAD")
    status = git_info("status", "--porcelain")
    (args.output / "runtime.json").write_text(
        json.dumps({"newton_file": newton.__file__, "head": head, "status": status}, indent=2) + "\n"
    )
    wp.config.kernel_cache_dir = str(args.cache_dir or os.environ.get("WARP_CACHE_PATH") or args.output / "warp-cache")
    wp.init()
    wp.set_device(args.device)
    ex = None
    try:
        ex = Example(
            Config(
                frames=args.frames,
                tip_attachment=not args.no_tip_attachment,
                close_enabled=not args.no_close,
                path_energy=not args.linear_energy,
            )
        )
        for frame in range(args.frames):
            ex.step()
            if (frame + 1) % 60 == 0:
                print(
                    json.dumps(
                        {
                            k: ex.records[-1][k]
                            for k in [
                                "time",
                                "curvature_min_width_degrees",
                                "root_tangent_error_max_degrees",
                                "max_abs_principal_strain",
                                "penetration_m",
                                "path_progress",
                            ]
                        }
                    ),
                    flush=True,
                )
        metrics = ex.summary()
        try:
            ex.test_final()
            metrics["test_passed"] = True
        except AssertionError:
            metrics["test_failure"] = traceback.format_exc()
        ex.export(args.output, ex.metadata(), metrics)
        np.savez_compressed(
            args.output / "path.npz",
            coefficients=ex.path_coefficients,
            progress=ex.path.progress,
            stiffness=ex.path.k_perp,
            restoring=ex.path.k_progress,
        )
        print(json.dumps({k: v for k, v in metrics.items() if k != "records"}, indent=2), flush=True)
        assert metrics["test_passed"], metrics.get("test_failure")
    except BaseException:
        (args.output / "failure.txt").write_text(traceback.format_exc())
        if ex is not None:
            np.savez_compressed(args.output / "failure_trace.npz", **ex.contact_trace.data())
        raise


if __name__ == "__main__":
    main()
