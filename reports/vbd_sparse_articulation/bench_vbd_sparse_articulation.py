# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark synthetic rigid VBD articulation cases.

This script intentionally uses Newton's public API. It is meant to compare the
current VBD rigid body-local solve against an experimental sparse articulation
mode once that mode is added to :class:`newton.solvers.SolverVBD`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton


@wp.kernel
def perturb_body_transforms(
    body_q: wp.array(dtype=wp.transform),
    body_inv_mass: wp.array(dtype=float),
    amplitude: float,
):
    body_id = wp.tid()
    if body_inv_mass[body_id] == 0.0:
        return

    q = body_q[body_id]
    pos = wp.transform_get_translation(q)
    rot = wp.transform_get_rotation(q)
    phase = float(body_id + 1)

    offset = wp.vec3(
        amplitude * wp.sin(1.7 * phase),
        0.5 * amplitude * wp.cos(2.3 * phase),
        0.25 * amplitude * wp.sin(0.7 * phase),
    )
    axis = wp.normalize(wp.vec3(0.3, 0.5, 0.8))
    angle = 0.5 * amplitude * wp.sin(1.1 * phase)
    dq = wp.quat_from_axis_angle(axis, angle)

    body_q[body_id] = wp.transform(pos + offset, wp.normalize(wp.mul(dq, rot)))


@dataclass
class CaseResult:
    scenario: str
    device: str
    body_count: int
    mode: str
    available: bool
    mean_step_us: float | None
    p50_step_us: float | None
    p95_step_us: float | None
    joint_residual_l2: float | None
    joint_residual_max: float | None
    contact_penetration_max: float | None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "device": self.device,
            "body_count": self.body_count,
            "mode": self.mode,
            "available": self.available,
            "mean_step_us": self.mean_step_us,
            "p50_step_us": self.p50_step_us,
            "p95_step_us": self.p95_step_us,
            "joint_residual_l2": self.joint_residual_l2,
            "joint_residual_max": self.joint_residual_max,
            "contact_penetration_max": self.contact_penetration_max,
            "error": self.error,
        }


def _quat_identity() -> wp.quat:
    return wp.quat_identity()


def _box_cfg(collide: bool) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0
    cfg.has_shape_collision = collide
    cfg.ke = 1.0e5
    cfg.kd = 1.0e1
    cfg.mu = 0.5
    return cfg


def _body_inertia(scale: float = 1.0) -> wp.mat33:
    return wp.mat33(
        scale,
        0.0,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        0.0,
        scale,
    )


def _add_box_body(builder: newton.ModelBuilder, pos: np.ndarray, label: str, collide: bool) -> int:
    body = builder.add_link(
        xform=wp.transform(p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), q=_quat_identity()),
        mass=1.0,
        inertia=_body_inertia(0.1),
        label=label,
    )
    builder.add_shape_box(body, hx=0.08, hy=0.08, hz=0.08, cfg=_box_cfg(collide))
    return body


def build_single(body_count: int, collide: bool, device: str) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    if collide:
        builder.add_ground_plane()
    for i in range(body_count):
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(float(i) * 0.3, 0.0, 1.0), q=_quat_identity()),
            mass=1.0,
            inertia=_body_inertia(0.1),
            label=f"free_{i}",
        )
        builder.add_shape_box(body, hx=0.08, hy=0.08, hz=0.08, cfg=_box_cfg(collide))
    builder.color()
    return builder.finalize(device=device)


def build_chain(body_count: int, joint_kind: str, collide: bool, device: str) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    if collide:
        builder.add_ground_plane()

    spacing = 0.3
    bodies = []
    for i in range(body_count):
        bodies.append(_add_box_body(builder, np.array([spacing * i, 0.0, 1.0]), f"chain_{i}", collide))

    joints = []
    joints.append(
        builder.add_joint_fixed(
            parent=-1,
            child=bodies[0],
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=_quat_identity()),
            child_xform=wp.transform(),
            label="root_fixed",
        )
    )

    for i in range(1, body_count):
        parent = bodies[i - 1]
        child = bodies[i]
        parent_xform = wp.transform(p=wp.vec3(spacing * 0.5, 0.0, 0.0), q=_quat_identity())
        child_xform = wp.transform(p=wp.vec3(-spacing * 0.5, 0.0, 0.0), q=_quat_identity())
        if joint_kind == "fixed":
            joint = builder.add_joint_fixed(parent, child, parent_xform, child_xform, label=f"fixed_{i}")
        elif joint_kind == "revolute":
            joint = builder.add_joint_revolute(
                parent,
                child,
                parent_xform=parent_xform,
                child_xform=child_xform,
                axis=wp.vec3(0.0, 0.0, 1.0),
                limit_lower=-0.25,
                limit_upper=0.25,
                limit_ke=1.0e4,
                limit_kd=1.0e-2,
                label=f"hinge_{i}",
            )
        else:
            raise ValueError(f"unsupported joint_kind: {joint_kind}")
        joints.append(joint)

    builder.add_articulation(joints, label=f"chain_{joint_kind}")
    builder.color()
    return builder.finalize(device=device)


def build_loop_fixed(body_count: int, collide: bool, device: str) -> newton.Model:
    if body_count < 4:
        raise ValueError("loop_fixed requires at least 4 bodies")

    builder = newton.ModelBuilder(gravity=0.0)
    if collide:
        builder.add_ground_plane()

    radius = 0.05 * body_count
    center = np.array([0.0, 0.0, 1.0])
    positions = []
    for i in range(body_count):
        theta = 2.0 * math.pi * i / body_count
        positions.append(center + np.array([radius * math.cos(theta), radius * math.sin(theta), 0.0]))

    bodies = [_add_box_body(builder, p, f"loop_{i}", collide) for i, p in enumerate(positions)]

    joints = []
    joints.append(
        builder.add_joint_fixed(
            parent=-1,
            child=bodies[0],
            parent_xform=wp.transform(
                p=wp.vec3(float(positions[0][0]), float(positions[0][1]), float(positions[0][2])),
                q=_quat_identity(),
            ),
            child_xform=wp.transform(),
            label="loop_root_fixed",
        )
    )

    for i in range(1, body_count):
        parent_pos = positions[i - 1]
        child_pos = positions[i]
        mid = 0.5 * (parent_pos + child_pos)
        joints.append(
            builder.add_joint_fixed(
                parent=bodies[i - 1],
                child=bodies[i],
                parent_xform=wp.transform(
                    p=wp.vec3(*(float(x) for x in (mid - parent_pos))),
                    q=_quat_identity(),
                ),
                child_xform=wp.transform(
                    p=wp.vec3(*(float(x) for x in (mid - child_pos))),
                    q=_quat_identity(),
                ),
                label=f"loop_tree_fixed_{i}",
            )
        )

    parent_pos = positions[-1]
    child_pos = positions[0]
    mid = 0.5 * (parent_pos + child_pos)
    joints.append(
        builder.add_joint_fixed(
            parent=bodies[-1],
            child=bodies[0],
            parent_xform=wp.transform(p=wp.vec3(*(float(x) for x in (mid - parent_pos))), q=_quat_identity()),
            child_xform=wp.transform(p=wp.vec3(*(float(x) for x in (mid - child_pos))), q=_quat_identity()),
            label="loop_closure_fixed",
        )
    )

    builder.add_articulation(joints, label="loop_tree")
    builder.color()
    return builder.finalize(device=device)


def build_contact_stack(body_count: int, device: str) -> newton.Model:
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.add_ground_plane()
    for i in range(body_count):
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.11 + 0.18 * float(i)), q=_quat_identity()),
            mass=1.0,
            inertia=_body_inertia(0.1),
            label=f"stack_{i}",
        )
        builder.add_shape_box(body, hx=0.08, hy=0.08, hz=0.08, cfg=_box_cfg(True))
    builder.color()
    return builder.finalize(device=device)


def build_model(scenario: str, body_count: int, device: str) -> newton.Model:
    if scenario == "single":
        return build_single(body_count, collide=False, device=device)
    if scenario == "chain_fixed":
        return build_chain(body_count, "fixed", collide=False, device=device)
    if scenario == "chain_revolute":
        return build_chain(body_count, "revolute", collide=False, device=device)
    if scenario == "loop_fixed":
        return build_loop_fixed(body_count, collide=False, device=device)
    if scenario == "contact_stack":
        return build_contact_stack(body_count, device=device)
    raise ValueError(f"unknown scenario: {scenario}")


def _array_np(array) -> np.ndarray:
    return np.asarray(array.to("cpu").numpy())


def _transform_parts(transforms) -> tuple[np.ndarray, np.ndarray]:
    values = _array_np(transforms)
    if values.dtype.fields:
        names = values.dtype.names or ()
        if "p" in names and "q" in names:
            return np.asarray(values["p"], dtype=np.float64), np.asarray(values["q"], dtype=np.float64)
    flat = np.asarray(values, dtype=np.float64).reshape(values.shape[0], -1)
    if flat.shape[1] < 7:
        raise RuntimeError(f"expected transform arrays to contain at least 7 scalars, got shape {flat.shape}")
    return flat[:, 0:3], flat[:, 3:7]


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_inv(q: np.ndarray) -> np.ndarray:
    denom = float(np.dot(q, q))
    if denom == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64) / denom


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    return _quat_mul(_quat_mul(q, qv), _quat_inv(q))[0:3]


def _quat_angle(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        return 0.0
    q = q / norm
    if q[3] < 0.0:
        q = -q
    return 2.0 * math.atan2(float(np.linalg.norm(q[0:3])), max(-1.0, min(1.0, float(q[3]))))


def _transform_point(pos: np.ndarray, quat: np.ndarray, local: np.ndarray) -> np.ndarray:
    return pos + _quat_rotate(quat, local)


def compute_joint_residual(model: newton.Model, state: newton.State) -> tuple[float, float]:
    if model.joint_count == 0 or model.body_count == 0:
        return 0.0, 0.0

    body_pos, body_quat = _transform_parts(state.body_q)
    joint_parent = _array_np(model.joint_parent).astype(np.int64)
    joint_child = _array_np(model.joint_child).astype(np.int64)
    joint_type = _array_np(model.joint_type).astype(np.int64)
    joint_enabled = _array_np(model.joint_enabled).astype(bool)
    joint_xp_pos, joint_xp_quat = _transform_parts(model.joint_X_p)
    joint_xc_pos, joint_xc_quat = _transform_parts(model.joint_X_c)

    residuals = []
    for joint_index in range(model.joint_count):
        if not joint_enabled[joint_index]:
            continue

        jt = int(joint_type[joint_index])
        if jt == int(newton.JointType.FREE):
            continue

        child = int(joint_child[joint_index])
        parent = int(joint_parent[joint_index])

        child_anchor = _transform_point(body_pos[child], body_quat[child], joint_xc_pos[joint_index])
        if parent >= 0:
            parent_anchor = _transform_point(body_pos[parent], body_quat[parent], joint_xp_pos[joint_index])
            parent_quat = _quat_mul(body_quat[parent], joint_xp_quat[joint_index])
        else:
            parent_anchor = joint_xp_pos[joint_index]
            parent_quat = joint_xp_quat[joint_index]

        child_quat = _quat_mul(body_quat[child], joint_xc_quat[joint_index])
        linear = float(np.linalg.norm(parent_anchor - child_anchor))
        residuals.append(linear)

        if jt == int(newton.JointType.FIXED):
            q_err = _quat_mul(_quat_inv(parent_quat), child_quat)
            residuals.append(_quat_angle(q_err))
        elif jt == int(newton.JointType.REVOLUTE):
            parent_axis = _quat_rotate(parent_quat, np.array([0.0, 0.0, 1.0]))
            child_axis = _quat_rotate(child_quat, np.array([0.0, 0.0, 1.0]))
            residuals.append(float(np.linalg.norm(np.cross(parent_axis, child_axis))))

    if not residuals:
        return 0.0, 0.0
    residual_array = np.asarray(residuals, dtype=np.float64)
    return float(np.linalg.norm(residual_array)), float(np.max(np.abs(residual_array)))


def compute_contact_penetration(contacts) -> float:
    if contacts is None:
        return 0.0
    try:
        count = int(_array_np(contacts.rigid_contact_count).reshape(-1)[0])
    except Exception:
        return 0.0
    if count <= 0:
        return 0.0
    try:
        p0 = _array_np(contacts.rigid_contact_point0)[:count]
        p1 = _array_np(contacts.rigid_contact_point1)[:count]
        n = _array_np(contacts.rigid_contact_normal)[:count]
        gap = np.einsum("ij,ij->i", p0 - p1, n)
        return float(max(0.0, -float(np.min(gap))))
    except Exception:
        return 0.0


def make_solver(model: newton.Model, mode: str, iterations: int):
    kwargs = {"iterations": iterations}
    if mode != "local":
        signature = inspect.signature(newton.solvers.SolverVBD)
        if "rigid_articulation_solve" not in signature.parameters:
            raise RuntimeError("SolverVBD has no rigid_articulation_solve option yet")
        kwargs["rigid_articulation_solve"] = mode
    return newton.solvers.SolverVBD(model, **kwargs)


def run_case(args: argparse.Namespace, scenario: str, body_count: int, mode: str) -> CaseResult:
    wp_device = wp.get_device(args.device)
    with wp.ScopedDevice(wp_device):
        def create_runtime():
            model = build_model(scenario, body_count, args.device)
            state_0 = model.state()
            state_1 = model.state()
            control = model.control()
            contacts = model.contacts()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
            if args.perturb > 0.0 and scenario != "contact_stack":
                wp.launch(
                    kernel=perturb_body_transforms,
                    dim=model.body_count,
                    inputs=[state_0.body_q, model.body_inv_mass, args.perturb],
                    device=wp_device,
                )
            solver = make_solver(model, mode, args.iterations)
            return model, state_0, state_1, control, contacts, solver

        try:
            model, state_0, state_1, control, contacts, solver = create_runtime()
        except Exception as exc:
            return CaseResult(
                scenario=scenario,
                device=args.device,
                body_count=body_count,
                mode=mode,
                available=False,
                mean_step_us=None,
                p50_step_us=None,
                p95_step_us=None,
                joint_residual_l2=None,
                joint_residual_max=None,
                contact_penetration_max=None,
                error=str(exc),
            )

        def step_once():
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, args.dt)

        for _ in range(args.warmup):
            step_once()
            state_0, state_1 = state_1, state_0
        wp.synchronize_device()

        graph = None
        if args.cuda_graph and wp_device.is_cuda:
            with wp.ScopedCapture(device=wp_device) as capture:
                step_once()
            graph = capture.graph
            wp.synchronize_device()

        times_us = []
        for _ in range(args.steps):
            start = time.perf_counter()
            if graph is not None:
                wp.capture_launch(graph)
            else:
                step_once()
            wp.synchronize_device()
            end = time.perf_counter()
            times_us.append((end - start) * 1.0e6)
            state_0, state_1 = state_1, state_0

        accuracy_model, accuracy_state_0, accuracy_state_1, accuracy_control, accuracy_contacts, accuracy_solver = (
            create_runtime()
        )
        accuracy_state_0.clear_forces()
        accuracy_model.collide(accuracy_state_0, accuracy_contacts)
        accuracy_solver.step(accuracy_state_0, accuracy_state_1, accuracy_control, accuracy_contacts, args.dt)
        wp.synchronize_device()

        joint_l2, joint_max = compute_joint_residual(accuracy_model, accuracy_state_1)
        contact_max = compute_contact_penetration(accuracy_contacts)

        return CaseResult(
            scenario=scenario,
            device=args.device,
            body_count=body_count,
            mode=mode,
            available=True,
            mean_step_us=float(statistics.fmean(times_us)),
            p50_step_us=float(statistics.median(times_us)),
            p95_step_us=float(np.percentile(np.asarray(times_us), 95.0)),
            joint_residual_l2=joint_l2,
            joint_residual_max=joint_max,
            contact_penetration_max=contact_max,
        )


def add_comparisons(results: list[dict]) -> list[dict]:
    by_key = {}
    for row in results:
        key = (row["scenario"], row["device"], row["body_count"])
        by_key.setdefault(key, {})[row["mode"]] = row

    for modes in by_key.values():
        baseline = modes.get("local")
        if not baseline or not baseline.get("available"):
            continue
        base_time = baseline.get("mean_step_us")
        base_resid = baseline.get("joint_residual_l2")
        for row in modes.values():
            row["cost_ratio_vs_local"] = None
            row["residual_reduction_vs_local"] = None
            if row.get("available") and base_time and row.get("mean_step_us"):
                row["cost_ratio_vs_local"] = row["mean_step_us"] / base_time
            if row.get("available") and row.get("joint_residual_l2") is not None and base_resid is not None:
                denom = max(float(row["joint_residual_l2"]), 1.0e-12)
                row["residual_reduction_vs_local"] = float(base_resid) / denom
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scenarios", nargs="+", default=["single", "chain_fixed", "chain_revolute", "loop_fixed"])
    parser.add_argument("--body-counts", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--modes", nargs="+", default=["local", "block_sparse_joints"])
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--perturb", type=float, default=2.0e-2)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--jsonl", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = []
    for scenario in args.scenarios:
        for body_count in args.body_counts:
            for mode in args.modes:
                result = run_case(args, scenario, body_count, mode)
                results.append(result.to_dict())

    results = add_comparisons(results)
    payload = {"results": results}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("a", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
