"""NIMBLE-informed finite modal-force finger exercise and passive release.

The supplied target shifts the elastic energy minimum without overwriting state:
E(q,t) = .5 q^T K q - q^T K q_target(t). Once released q_target is zero.
All hand coordinates remain Newton VBD outputs. This is an explicitly actuated,
uncalibrated pose-ROM shell, not muscle dynamics or kinematic pose playback.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import warp as wp
from hand_contact import HandContact, digest, recover
from individual import FINGERS, finger_patches

import newton

STARTS = 0.5 + np.arange(5) * 2.2
DURATION = 12.0
AMPLITUDE = 0.8
RELEASE_DURATION = 1 / 60
MODEL_REVISION = "expressive-hand-20261016"


def target_path(targets, duration=DURATION, amplitude=AMPLITUDE, step_hz=240):
    ts = np.arange(round(duration * step_hz) + 1) / step_hz
    path = np.zeros((len(ts), targets.shape[1]), dtype=np.float32)
    for start, target in zip(STARTS, targets, strict=True):
        a = np.clip((ts - start) / 0.6, 0, 1)
        pulse = a * a * (3 - 2 * a)
        r = np.clip((ts - start - 0.9) / RELEASE_DURATION, 0, 1)
        pulse *= 1 - r * r * (3 - 2 * r)
        pulse[ts >= start + 0.9 + RELEASE_DURATION - 1e-9] = 0
        path += pulse[:, None] * target * amplitude
    return path


@wp.kernel
def apply_target(
    clock: wp.array[int],
    path: wp.array2d[float],
    stiffness: wp.array[float],
    enabled: bool,
    start: int,
    force: wp.array[float],
    saved: wp.array2d[float],
):
    i = wp.tid()
    frame = wp.min(clock[0], path.shape[0] - 1)
    value = float(0)
    if enabled:
        value = stiffness[i] * path[frame, i]
    force[start + i] = value
    if saved:
        saved[frame, i] = value


class Example(HandContact):
    def __init__(
        self,
        inputs,
        targets,
        *,
        actuation_enabled=True,
        audited=True,
        duration=DURATION,
        amplitude=AMPLITUDE,
        substeps=4,
        iterations=20,
    ):
        self.targets_path = Path(targets)
        self.actuation_enabled = actuation_enabled
        self.amplitude = amplitude
        super().__init__(
            travel=0,
            duration=duration,
            modes=48,
            bar_height=0.5,
            audited=audited,
            rom_path=Path(inputs) / "rom-48.npz",
            substeps=substeps,
            iterations=iterations,
        )
        # The old contact probe remains in the model, parked well outside reach.
        # Its topology/collision settings are unchanged, but it is not an actuator.
        self.meshes = self.meshes[:-1]

    def prepare_drive(self, device):
        with np.load(self.targets_path) as a:
            self.target_coordinates = a["q"].copy()
        self.path_cpu = target_path(self.target_coordinates, self.duration, self.amplitude, 1 / self.dt)
        self.path = wp.array(self.path_cpu, dtype=float, device=device)
        self.stiffness = wp.array(self.basis.mode_stiffness, dtype=float, device=device)
        owner = int(self.model.elastic_joint.numpy()[0])
        self.qd_start = int(self.model.joint_qd_start.numpy()[owner]) + 6
        self.force_trace = (
            wp.zeros((self.capacity + 1, self.modes), dtype=float, device=device) if self.audited else None
        )
        self.drive_step()

    def drive_step(self):
        super().drive_step()
        wp.launch(
            apply_target,
            self.modes,
            inputs=[
                self.cursor,
                self.path,
                self.stiffness,
                self.actuation_enabled,
                self.qd_start,
                self.control.joint_f,
                self.force_trace,
            ],
            device=self.model.device,
        )

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)

    def export(self, output):
        result = super().export(output)
        np.savez_compressed(
            output / "actuation.npz",
            forces=self.force_trace.numpy()[: self.capacity],
            q_target=self.path_cpu[: self.capacity],
            times=np.arange(self.capacity) * self.dt,
            stiffness=self.basis.mode_stiffness,
            enabled=self.actuation_enabled,
            qd_start=self.qd_start,
            starts=STARTS,
            releases=STARTS + 0.9 + RELEASE_DURATION,
            release_duration_s=RELEASE_DURATION,
            amplitude=self.amplitude,
            dt=self.dt,
        )
        result.update(
            model_revision=MODEL_REVISION,
            solver_substeps=self.substeps,
            solver_iterations=self.iterations,
            runtime_root=str(Path(newton.__file__).resolve().parents[1]),
            actuation_enabled=self.actuation_enabled,
            modes=48,
            prescription="Generalized force K*q_target, zero after release; no joint_q or mesh overwrite",
            target_sha256=digest(self.targets_path),
            amplitude=self.amplitude,
        )
        result.pop("travel_m")
        result.pop("bar_height_m")
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        meta = json.loads((output / "scene.json").read_text())
        meta.update(
            slug="hand-expressive-finger-actuation",
            private=False,
            poster_frame=round(STARTS[1] * 60 + 45),
            sequence=[
                {
                    "finger": name,
                    "start_s": float(start),
                    "hold_s": [float(start + 0.6), float(start + 0.9)],
                    "release_start_s": float(start + 0.9),
                    "release_s": float(start + 0.9 + RELEASE_DURATION),
                }
                for name, start in zip(FINGERS, STARTS, strict=True)
            ],
            actuation=result["prescription"],
        )
        (output / "scene.json").write_text(json.dumps(meta, indent=2) + "\n")
        shutil.copy2(output / "original_states.npz", output / "solved_states.npz")
        self.result = result
        return result

    def test_final(self):
        if not self.result["geometry_passed"]:
            raise AssertionError({k: v for k, v in self.result["checks"].items() if not v})
        if hasattr(self, "paired_checks") and not all(self.paired_checks):
            raise AssertionError("Finger response, dynamic lag or release failed")


def paired(main, control):
    with np.load(main / "substeps.npz") as a, np.load(control / "substeps.npz") as b:
        qa, qb, ts = a["q"], b["q"], a["times"]
    with np.load(main / "original_states.npz") as a:
        rest, phi = a["rest"], a["vertex_phi"].astype(float)
    with np.load(main / "actuation.npz") as a, np.load(control / "actuation.npz") as b:
        forces = a["forces"]
        qtarget = a["q_target"]
        dt = float(a["dt"]) if "dt" in a else 1 / 240
        np.testing.assert_array_equal(b["forces"], 0)
        np.testing.assert_allclose(forces, a["q_target"] * a["stiffness"], rtol=2e-7, atol=1e-6)
    masks = finger_patches(rest)
    causal = np.empty((len(qa), 5))
    for start in range(0, len(qa), 32):
        delta = recover(np.zeros_like(rest), phi, qa[start : start + 32] - qb[start : start + 32])
        for i, mask in enumerate(masks):
            causal[start : start + 32, i] = np.linalg.norm(delta[:, mask], axis=-1).max(axis=1)
    rows = []
    for i, (name, start) in enumerate(zip(FINGERS, STARTS, strict=True)):
        hold = (ts >= start + 0.7) & (ts < start + 0.9)
        release = int(round((start + 0.9 + RELEASE_DURATION) / dt))
        recovered = (ts >= start + 1.8) & (ts < start + 2.1)
        active_response = float(causal[hold, i].mean())
        final = float(causal[recovered, i].max())
        release_motion = recover(np.zeros_like(rest), phi, qa[release : release + round(0.05 / dt)] - qa[release - 1])
        release_travel = float(np.linalg.norm(release_motion[:, masks[i]], axis=-1).max())
        unforced = bool(np.all(forces[release : release + round(0.1 / dt)] == 0))
        ramp = (ts >= start + 0.1) & (ts < start + 0.5)
        steady = (qa[hold] - qb[hold]).mean(axis=0)
        # Compare with the quasi-static trajectory scaled from the measured hold.
        # This removes static wrist compliance from the dynamic-lag diagnostic.
        factor = np.linalg.norm(qtarget[ramp], axis=1) / np.linalg.norm(qtarget[hold], axis=1).mean()
        lag = recover(np.zeros_like(rest), phi, qa[ramp] - qb[ramp] - factor[:, None] * steady)
        max_lag = float(np.linalg.norm(lag[:, masks[i]], axis=-1).max())
        rows.append(
            {
                "finger": name,
                "hold_distal_response_m": active_response,
                "recovered_response_m": final,
                "recovery_fraction": 1 - final / active_response,
                "first_50ms_release_motion_m": release_travel,
                "released_force_zero": unforced,
                "ramp_dynamic_lag_m": max_lag,
                "checks": {
                    "visible_response": active_response >= 0.009,
                    "recovery": final <= 0.001 and final <= 0.1 * active_response,
                    "release_dynamics": release_travel >= 0.003,
                    "release_unforced": unforced,
                    "dynamic_lag": max_lag >= 1e-5,
                },
            }
        )
    np.savez_compressed(main / "paired_response.npz", times=ts, distal_causal=causal)
    return rows


def benchmark(args):
    rows = []
    for repeat in range(4):
        e = Example(
            args.inputs,
            args.targets,
            audited=False,
            amplitude=args.amplitude,
            substeps=args.substeps,
            iterations=args.iterations,
        )
        assert e.graph is not None and e.force_trace is None
        wp.synchronize_device(e.model.device)
        t = time.perf_counter()
        for _ in range(e.frames):
            e.step()
        wp.synchronize_device(e.model.device)
        elapsed = time.perf_counter() - t
        row = {"repeat": repeat, "warmup": repeat == 0, "frames": e.frames, "mean_ms": elapsed / e.frames * 1000}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {
        "method": "CUDA graph full trajectories, one discarded warmup, no metrics/render/readbacks in loop",
        "rows": rows,
        "median_ms": float(np.median([r["mean_ms"] for r in rows[1:]])),
        "modes": 48,
        "iterations": args.iterations,
        "substeps": args.substeps,
        "model_revision": MODEL_REVISION,
        "runtime_root": str(Path(newton.__file__).resolve().parents[1]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("smoke", "validate", "benchmark", "qualify"))
    assets = Path(__file__).with_name("assets")
    p.add_argument("--inputs", type=Path, default=assets)
    p.add_argument("--targets", type=Path, default=assets / "targets.npz")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--amplitude", type=float, default=AMPLITUDE)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--common-manifest", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.action == "qualify":
        if args.common_manifest is None or (args.substeps, args.iterations) != (2, 12):
            raise ValueError("Optimized qualification requires the common runtime and 2x12")
        runtime = Path(newton.__file__).resolve().parents[1]
        files = json.loads(args.common_manifest.read_text())
        for name, expected in files.items():
            if digest(runtime / name) != expected:
                raise AssertionError(f"Common runtime mismatch: {name}")
        (args.output / "runtime-identity.json").write_text(
            json.dumps(
                {"root": str(runtime), "manifest_sha256": digest(args.common_manifest), "files_checked": len(files)},
                indent=2,
            )
            + "\n"
        )
    if args.action == "benchmark":
        (args.output / "benchmark.json").write_text(json.dumps(benchmark(args), indent=2) + "\n")
        return
    results = []
    duration = 2.5 if args.action == "smoke" else DURATION
    for enabled, name in ((True, "actuated"), (False, "no-actuation")):
        e = Example(
            args.inputs,
            args.targets,
            actuation_enabled=enabled,
            duration=duration,
            amplitude=args.amplitude,
            substeps=args.substeps,
            iterations=args.iterations,
        )
        e.run()
        result = e.export(args.output / name)
        try:
            e.test_final()
            result["test_final_passed"] = True
        except AssertionError as error:
            result["test_final_passed"] = False
            result["test_final_failure"] = str(error)
        results.append(result)
        print(
            json.dumps({"arm": name, "checks": result["checks"], "strain": result["max_principal_strain"]}), flush=True
        )
    if args.action in ("validate", "qualify"):
        rows = paired(args.output / "actuated", args.output / "no-actuation")
        e.paired_checks = [all(row["checks"].values()) for row in rows]
        try:
            e.test_final()
            paired_final = True
        except AssertionError:
            paired_final = False
        result = {
            "test_passed": all(r["geometry_passed"] for r in results) and all(all(r["checks"].values()) for r in rows),
            "negative_control_passed": results[1]["geometry_passed"],
            "paired_test_final_passed": paired_final,
            "arms": results,
            "fingers": rows,
            "model_revision": MODEL_REVISION,
            "solver_substeps": args.substeps,
            "solver_iterations": args.iterations,
        }
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"test_passed": result["test_passed"], "fingers": rows}), flush=True)
        if not result["test_passed"] or not paired_final:
            raise AssertionError("Hand validation failed; see results.json")
        if args.action == "qualify" and result["test_passed"] and paired_final:
            timing = benchmark(args)
            timing["optimized_sim_frame_ms"] = timing["median_ms"]
            timing["physical_results_sha256"] = digest(args.output / "results.json")
            (args.output / "benchmark.json").write_text(json.dumps(timing, indent=2) + "\n")
    else:
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
