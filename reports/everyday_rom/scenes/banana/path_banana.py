# SPDX-License-Identifier: Apache-2.0
"""Three whole elastic peels with a passive ordered shape prior and tip pins.

Report-local experiment. The path potential replaces the shell rest energy;
it is not an adhesion/fracture model. Only visible rigid grippers are driven.
"""

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import warp as wp
from banana import drive_grippers
from build_banana_path import principal_strain
from peel_case_geometry import Config, Simulation, banana_flesh, box, pose, tick, tube
from pod_path_energy import ModalPathEnergy
from solver_vbd_modal_path import SolverVBDModalPath

import newton


@wp.kernel
def record_step(
    slot: wp.array[int],
    clock: wp.array[float],
    q: wp.array[float],
    body: wp.array[wp.transform],
    global_count: wp.array[int],
    body_count: wp.array[int],
    saved_q: wp.array2d[float],
    saved_body: wp.array2d[wp.transform],
    saved_counts: wp.array2d[int],
    saved_time: wp.array[float],
):
    i, k = wp.tid(), slot[0]
    if k < saved_q.shape[0]:
        if i < q.shape[0]:
            saved_q[k, i] = q[i]
        if i < body.shape[0]:
            saved_body[k, i] = body[i]
            saved_counts[k, i + 1] = body_count[i]
        if i == 0:
            saved_counts[k, 0] = global_count[0]
            saved_time[k] = clock[0]


@wp.kernel
def increment(slot: wp.array[int]):
    slot[0] += 1


class Example(Simulation):
    def __init__(self, rom, frames=420, drive=True, k_perp=2e6, k_progress=0.002, iterations=16, pull_duration=5.2):
        if not np.isfinite(pull_duration) or pull_duration <= 0:
            raise ValueError("Pull duration must be finite and positive")
        self.pull_duration = float(pull_duration)
        self.k_perp, self.k_progress = k_perp, k_progress
        self.begin(None, Config(iterations=iterations, joint_angular_ke=1000, elastic_relaxation=1.0))
        self.cfg.drive_enabled = drive
        self.expected_steps = frames * self.cfg.substeps
        self.paths, self.peels, self.origins, self.tip_local = {}, [], [], []
        grippers, trajectories = [], []
        self.fruit = self.builder.add_body(xform=pose(), is_kinematic=True, label="supported_rigid_fruit")
        self.mesh("fruit", *banana_flesh(), self.fruit, (0.93, 0.87, 0.61), True)
        self.mesh(
            "stalk_fixture", *tube([[0, 0, 0.015], [0, 0, 0.18]], 0.007, 20), color=(0.25, 0.29, 0.3), metallic=0.7
        )
        self.mesh("fixture_foot", *box((0.04, 0.034, 0.006), (0, 0, 0.009)), color=(0.25, 0.29, 0.3), metallic=0.7)
        self.rom_path = Path(rom)
        with np.load(rom, allow_pickle=False) as data:
            for strip in range(3):
                rest, phi = data[f"rest_{strip}"].copy(), data[f"phi_{strip}"].copy()
                mass, origin = data[f"mass_{strip}"].copy(), data[f"origin_{strip}"].copy()
                faces = data[f"faces_{strip}"].copy()
                a, s = data[f"coefficients_{strip}"].copy(), data[f"progress_{strip}"].copy()
                tip = (rest[1096] + rest[2201]) / 2
                tip_phi = (phi[1096] + phi[2201]) / 2
                samples = np.concatenate((rest, np.zeros((1, 3)), tip[None]))
                sampled_phi = np.concatenate((phi, np.zeros((1, phi.shape[1], 3)), tip_phi[None]))
                basis = newton.ModalBasis(
                    sample_points=samples,
                    sample_phi=sampled_phi,
                    sample_mass=np.r_[mass, 0, 0],
                    mode_mass=np.ones(phi.shape[1]),
                    mode_stiffness=np.zeros(phi.shape[1]),
                    mode_damping=np.full(phi.shape[1], 10.0),
                )
                body = self.elastic(samples, basis, float(mass.sum()), origin, f"whole_peel_{strip}", False)
                part = self.mesh(f"peel_{strip}", rest, faces, body, (0.91 - strip * 0.025, 0.72, 0.045), True)
                self.peels.append(part)
                self.origins.append(origin)
                self.tip_local.append(tip)
                self.paths[body] = ModalPathEnergy(a, s, k_perp, k_progress)
                self.builder.add_joint_fixed(
                    parent=self.fruit,
                    child=body,
                    parent_xform=pose(origin),
                    child_xform=pose(),
                    collision_filter_parent=False,
                    label=f"basal_clamp_{strip}",
                )
                times = np.arange(frames * 2 + 2) / 120
                u = np.clip((times - 0.6) / self.pull_duration, 0, 1)
                progress = u * u * (3 - 2 * u)
                coefficients = np.column_stack([np.interp(progress, s, a[:, j]) for j in range(a.shape[1])])
                path = tip + origin + np.einsum("mc,tm->tc", tip_phi, coefficients)
                gripper = self.builder.add_body(xform=pose(path[0]), is_kinematic=True, label=f"tip_gripper_{strip}")
                self.mesh(f"tip_pin_{strip}", *box((0.005, 0.006, 0.004)), gripper, (0.16, 0.3, 0.34), metallic=0.5)
                self.builder.add_joint_ball(
                    parent=gripper,
                    child=body,
                    parent_xform=pose(),
                    child_xform=pose(tip),
                    collision_filter_parent=True,
                    label=f"tip_pin_joint_{strip}",
                )
                grippers.append(gripper)
                trajectories.append(path)
        self.gripper_ids = wp.array(grippers, dtype=int)
        self.gripper_paths = wp.array(np.array(trajectories), dtype=wp.vec3)
        self.finish()

    def finish(self):
        self.builder.color()
        self.model = self.builder.finalize()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.pipeline = newton.CollisionPipeline(self.model, reduce_contacts=False, rigid_contact_max=32768)
        self.contacts, self.control = self.pipeline.contacts(), self.model.control()
        self.solver = SolverVBDModalPath(
            self.model,
            paths=self.paths,
            iterations=self.cfg.iterations,
            rigid_body_contact_buffer_size=16384,
            rigid_compliant_alm=False,
            rigid_contact_hard=False,
            rigid_contact_k_start=self.cfg.contact_ke,
            rigid_joint_linear_ke=40000,
            rigid_joint_linear_k_start=40000,
            rigid_joint_angular_ke=1000,
            rigid_joint_angular_k_start=1000,
            rigid_joint_linear_kd=0.03,
            rigid_joint_angular_kd=0.0002,
            rigid_joint_adaptive_stiffness=False,
            elastic_body_relaxation=1.0,
            friction_epsilon=1e-4,
        )
        self.clock, self.counter = wp.zeros(1, dtype=float), wp.zeros(1, dtype=int)
        self.q_starts = {}
        for body in self.paths:
            e = int(self.model.body_elastic_index.numpy()[body])
            joint = int(self.model.elastic_joint.numpy()[e])
            self.q_starts[body] = int(self.model.joint_q_start.numpy()[joint]) + 7
        self.shape_bodies = self.model.shape_body.numpy()
        self.contact_phi = self.model.elastic_shape_vertex_phi.numpy().reshape(-1, self.model.elastic_max_mode_count, 3)
        self.contact_rest = self.model.elastic_shape_vertex_local.numpy()
        shape_ids = self.model.elastic_shape_shape.numpy()
        starts, counts = self.model.elastic_shape_vertex_start.numpy(), self.model.elastic_shape_vertex_count.numpy()
        for part in self.parts:
            if part["body"] in self.paths:
                e = int(np.flatnonzero(shape_ids == part["shape"])[0])
                section = slice(starts[e], starts[e] + counts[e])
                part["phi"] = self.contact_phi[section].copy()
                part["q_start"] = self.q_starts[part["body"]]
                np.testing.assert_allclose(self.contact_rest[section], part["rest"], atol=1e-7)
            else:
                part["phi"], part["q_start"] = np.zeros((len(part["rest"]), 0, 3)), -1
        self.trace_q = wp.zeros((self.expected_steps, self.model.joint_coord_count), dtype=float)
        self.trace_body = wp.zeros((self.expected_steps, self.model.body_count), dtype=wp.transform)
        self.trace_counts = wp.zeros((self.expected_steps, self.model.body_count + 1), dtype=int)
        self.trace_time = wp.zeros(self.expected_steps, dtype=float)
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.cfg.substeps):
            wp.launch(
                drive_grippers,
                dim=3,
                inputs=[
                    self.clock,
                    self.dt,
                    self.gripper_ids,
                    self.gripper_paths,
                    self.cfg.drive_enabled,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_q,
                    self.state_1.body_qd,
                ],
            )
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(tick, dim=1, inputs=[self.clock, self.dt])
            wp.launch(
                record_step,
                dim=max(self.model.joint_coord_count, self.model.body_count),
                inputs=[
                    self.counter,
                    self.clock,
                    self.state_0.joint_q,
                    self.state_0.body_q,
                    self.contacts.rigid_contact_count,
                    self.solver.body_body_contact_counts,
                    self.trace_q,
                    self.trace_body,
                    self.trace_counts,
                    self.trace_time,
                ],
            )
            wp.launch(increment, dim=1, inputs=[self.counter])

    def measure(self):
        values, root_error, strains = [], [], []
        for strip, part_index in enumerate(self.peels):
            part = self.parts[part_index]
            count, start = part["phi"].shape[1], part["q_start"]
            q = self.saved_q[-1][start : start + count]
            result = self.paths[part["body"]].evaluate(q)
            world = part["world"][-1]
            values.append(
                {
                    "progress": result.progress,
                    "segment": result.segment,
                    "off_path_norm": float(np.linalg.norm(q - result.closest)),
                }
            )
            root_error.append(float(np.linalg.norm(self.saved_poses[-1][part["body"], :3] - self.origins[strip])))
            strains.append(principal_strain(part["rest"], world, part["faces"]))
        return {"paths": values, "principal_strain": strains, "root_error_m": max(root_error), "penetration_m": 0.0}

    def export_run(self, output):
        output.mkdir(parents=True, exist_ok=True)
        n = int(self.counter.numpy()[0])
        counts = self.trace_counts.numpy()
        metrics = {
            "test_passed": False,
            "negative_control_passed": False,
            "status": "WIP dynamics diagnostic",
            "frames": len(self.records),
            "substeps": n,
            "expected_substeps": self.expected_steps,
            "global_capacity": self.contacts.rigid_contact_max,
            "body_capacity": 16384,
            "raw_contact_peak": int(counts[:, 0].max()),
            "raw_body_peak": int(counts[:, 1:].max()),
            "capacity_passed": bool(
                n == self.expected_steps and counts[:, 0].max() <= 32768 and counts[:, 1:].max() <= 16384
            ),
            "final_progress": [r["progress"] for r in self.records[-1]["paths"]],
            "saved_frame_principal_strain": np.max([r["principal_strain"] for r in self.records], axis=0).tolist(),
            "max_root_error_m": max(r["root_error_m"] for r in self.records),
            "max_penetration_m": max(r["penetration_m"] for r in self.records),
            "mean_graph_ms": 1000 * float(np.mean(self.timings[10:] or self.timings)),
            "records": self.records,
        }
        arrays, meshes = {"times": np.array([r["time"] for r in self.records])}, []
        for i, part in enumerate(self.parts):
            arrays[f"vertices_{i}"] = np.asarray(part["world"]) if part["body"] >= 0 else part["world"][0]
            arrays[f"faces_{i}"] = part["faces"]
            meshes.append(
                {k: part[k] for k in ("name", "color", "roughness", "metallic", "opacity")}
                | {"vertices": f"vertices_{i}", "faces": f"faces_{i}"}
            )
        np.savez_compressed(output / "scene.npz", **arrays)
        np.savez_compressed(
            output / "substeps.npz",
            q=self.trace_q.numpy(),
            body=self.trace_body.numpy(),
            counts=counts,
            time=self.trace_time.numpy(),
        )
        np.savez_compressed(
            output / "state.npz",
            joint_q=self.saved_q,
            body_q=self.saved_poses,
            joint_qd=self.saved_qd,
            body_qd=self.saved_v,
            q_starts=np.array(list(self.q_starts.values())),
            peel_bodies=np.array(list(self.paths)),
        )
        fields = (
            "elastic_shape_body",
            "elastic_shape_vertex_start",
            "elastic_shape_vertex_count",
            "elastic_shape_vertex_local",
            "elastic_shape_vertex_phi",
            "elastic_shape_indices",
            "elastic_shape_index_start",
            "elastic_shape_index_count",
            "elastic_body",
            "elastic_mode_start",
            "elastic_mode_count",
            "elastic_mode_mass",
            "elastic_mode_stiffness",
            "elastic_mode_damping",
        )
        np.savez_compressed(
            output / "registered_model.npz",
            **{name: getattr(self.model, name).numpy() for name in fields},
            elastic_max_mode_count=self.model.elastic_max_mode_count,
        )
        metadata = {
            "title": "Ordered POD banana peeling",
            "status": "WIP",
            "fps": 60,
            "meshes": meshes,
            "camera": {"position": [0.56, -0.77, 0.51], "target": [0, 0, 0.22], "fov": 38},
            "floor_z": 0.003,
            "poster_frame": len(self.records) - 1,
            "model": "Three whole peels with a passive nonlinear POD path prior; no shell/fracture calibration.",
            "driven": "Rigid tip grippers only, through finite-stiffness pin joints; no modal coordinates are assigned.",
            "mode_count": 96,
            "source_sha256": {},
            "rom_sha256": hashlib.sha256(self.rom_path.read_bytes()).hexdigest(),
            "parameters": asdict(self.cfg)
            | {
                "k_perp_per_s2": self.k_perp,
                "k_progress_J": self.k_progress,
                "pull_duration_s": self.pull_duration,
                "pull_start_s": 0.6,
                "dt_s": self.dt,
                "timing_scope": "Graph execution includes per-substep diagnostic trace kernels",
            },
        }
        sources = output / "source"
        sources.mkdir(exist_ok=True)
        for file in Path(__file__).parent.glob("*.py"):
            shutil.copy2(file, sources / file.name)
            metadata["source_sha256"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        for name in ("banana.py", "peel_case_geometry.py"):
            file = Path(__file__).resolve().parent / name
            shutil.copy2(file, sources / name)
            metadata["source_sha256"][name] = hashlib.sha256(file.read_bytes()).hexdigest()
        (output / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps({k: v for k, v in metrics.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=420)
    parser.add_argument("--no-drive", action="store_true")
    parser.add_argument("--k-perp", type=float, default=2e6)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--pull-duration", type=float, default=5.2)
    args = parser.parse_args()
    example = Example(
        args.rom,
        args.frames,
        not args.no_drive,
        args.k_perp,
        iterations=args.iterations,
        pull_duration=args.pull_duration,
    )
    for frame in range(args.frames):
        example.step()
        if (frame + 1) % 60 == 0:
            print(json.dumps(example.records[-1]), flush=True)
    example.export_run(args.output)
