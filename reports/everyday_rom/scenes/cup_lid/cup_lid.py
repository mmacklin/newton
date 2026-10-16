# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Audit and export the supported cup/lid/straw without changing its dynamics."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import warp as wp
from baseline import FRAMES, PIN, STAGE, digest, provenance, supported, write_json

import newton.solvers
from newton.examples.basic._reduced_elastic_cap import cup_mesh, ring_mesh

METRIC_NAMES = ("initial_residual_norm", "solve_residual_norm", "applied_residual_norm", "update_norm", "update_max")
STATE_NAMES = ("body_q", "body_qd", "joint_q", "joint_qd")


@wp.kernel
def record_collision(
    cursor: wp.array[int],
    clock: wp.array[float],
    raw: wp.array[int],
    times: wp.array[float],
    counts: wp.array[int],
    calls: wp.array[int],
):
    row = cursor[0]
    if row < counts.shape[0]:
        times[row] = clock[0]
        counts[row] = raw[0]
        calls[row] = calls[row] + 1


@wp.kernel
def record_solve(
    cursor: wp.array[int],
    calls: wp.array[int],
    raw: wp.array[int],
    overflow: wp.array[int],
    bodies: wp.array2d[int],
    overflows: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    poses: wp.array2d[wp.transform],
    velocities: wp.array2d[wp.spatial_vector],
    qs: wp.array2d[float],
    qds: wp.array2d[float],
    initial: wp.array[float],
    solve: wp.array[float],
    applied: wp.array[float],
    update: wp.array[float],
    maximum: wp.array[float],
    metrics: wp.array2d[float],
):
    row = cursor[0]
    if row < calls.shape[0]:
        calls[row] = calls[row] + 1
        overflows[row] = overflow[0]
        for i in range(raw.shape[0]):
            bodies[row, i] = raw[i]
            poses[row, i] = body_q[i]
            velocities[row, i] = body_qd[i]
        for i in range(joint_q.shape[0]):
            qs[row, i] = joint_q[i]
        for i in range(joint_qd.shape[0]):
            qds[row, i] = joint_qd[i]
        metrics[row, 0] = initial[0]
        metrics[row, 1] = solve[0]
        metrics[row, 2] = applied[0]
        metrics[row, 3] = update[0]
        metrics[row, 4] = maximum[0]
    cursor[0] = row + 1


def check_trace(data, expected, global_capacity, body_capacity, dt):
    assert int(data["cursor"]) == expected, "Missing or extra substeps"
    assert expected > 0
    for name in (
        "times",
        "global_raw",
        "collisions",
        "solves",
        "body_overflow",
        "body_raw",
        "solver_metrics",
        *STATE_NAMES,
    ):
        assert len(data[name]) == expected, f"Incomplete {name}"
        assert np.isfinite(data[name]).all(), f"Nonfinite {name}"
    assert np.all(data["collisions"] == 1), "Missing or repeated collision"
    assert np.all(data["solves"] == 1), "Missing or repeated solve"
    assert np.all((data["global_raw"] >= 0) & (data["global_raw"] <= global_capacity)), "Global contact overflow"
    assert np.all((data["body_raw"] >= 0) & (data["body_raw"] <= body_capacity)), "Body contact overflow"
    assert np.all(data["body_overflow"] == 0), "Solver contact overflow"
    np.testing.assert_allclose(data["times"], np.arange(expected) * dt, atol=2e-4, rtol=0)
    return {
        "passed": True,
        "observed_substeps": expected,
        "expected_substeps": expected,
        "global_capacity": int(global_capacity),
        "body_capacity": int(body_capacity),
        "global_raw_peak": int(data["global_raw"].max()),
        "body_raw_peaks": data["body_raw"].max(axis=0).tolist(),
        "overflow": False,
        "includes_settling": True,
        "scope": "Every actual collision before solve and every completed solve, including approach, insertion, hold, stirring, withdrawal, release; graph capture does not advance simulation.",
        "body_count_semantics": "Raw pre-clamp VBD body-list counters. Kinematic lid has elastic DOFs and is eligible; rigid kinematic straw is ineligible. Valid only after global-capacity check.",
    }


class SubstepTrace:
    def __init__(self, example):
        self.example = e = example
        n = FRAMES * e.cfg.substeps
        d = e.model.device
        self.cursor = wp.zeros(1, dtype=int, device=d)
        self.times = wp.zeros(n, dtype=float, device=d)
        self.global_raw = wp.full(n, -1, dtype=int, device=d)
        self.body_raw = wp.full((n, e.model.body_count), -1, dtype=int, device=d)
        self.collisions = wp.zeros(n, dtype=int, device=d)
        self.solves = wp.zeros(n, dtype=int, device=d)
        self.body_overflow = wp.zeros(n, dtype=int, device=d)
        self.body_q = wp.zeros((n, e.model.body_count), dtype=wp.transform, device=d)
        self.body_qd = wp.zeros((n, e.model.body_count), dtype=wp.spatial_vector, device=d)
        self.joint_q = wp.zeros((n, e.state_0.joint_q.shape[0]), dtype=float, device=d)
        self.joint_qd = wp.zeros((n, e.state_0.joint_qd.shape[0]), dtype=float, device=d)
        self.solver_metrics = wp.zeros((n, 5), dtype=float, device=d)
        collide, solve = e.pipeline.collide, e.solver.step

        def traced_collide(state, contacts):
            collide(state, contacts)
            wp.launch(
                record_collision,
                dim=1,
                inputs=[
                    self.cursor,
                    e.clock,
                    contacts.rigid_contact_count,
                    self.times,
                    self.global_raw,
                    self.collisions,
                ],
                device=d,
            )

        def traced_solve(state_in, state_out, control, contacts, dt):
            solve(state_in, state_out, control, contacts, dt)
            wp.launch(
                record_solve,
                dim=1,
                inputs=[
                    self.cursor,
                    self.solves,
                    e.solver.body_body_contact_counts,
                    e.solver.body_body_contact_overflow_max,
                    self.body_raw,
                    self.body_overflow,
                    state_out.body_q,
                    state_out.body_qd,
                    state_out.joint_q,
                    state_out.joint_qd,
                    self.body_q,
                    self.body_qd,
                    self.joint_q,
                    self.joint_qd,
                    *[getattr(e.solver, "elastic_body_block_" + name) for name in METRIC_NAMES],
                    self.solver_metrics,
                ],
                device=d,
            )

        e.pipeline.collide, e.solver.step = traced_collide, traced_solve
        with wp.ScopedCapture() as capture:
            e.simulate()
        e.graph = capture.graph
        assert int(self.cursor.numpy()[0]) == 0
        assert float(e.clock.numpy()[0]) == 0.0

    def data(self):
        return {
            name: getattr(self, name).numpy()
            for name in (
                "cursor",
                "times",
                "global_raw",
                "body_raw",
                "collisions",
                "solves",
                "body_overflow",
                *STATE_NAMES,
                "solver_metrics",
            )
        }


def world_points(local, poses):
    rotations = np.stack([np.asarray(wp.quat_to_matrix(wp.quat(*p[3:])), dtype=float).reshape(3, 3) for p in poses])
    return (
        np.einsum("fij,fvj->fvi", rotations, np.broadcast_to(local, (len(poses), *local.shape[-2:])))
        + poses[:, None, :3]
    ).astype(np.float32)


def strain_operator(rest, faces):
    edges = rest[faces[:, 1:]] - rest[faces[:, :1]]
    first = np.linalg.norm(edges[:, 0], axis=1)
    tangent = edges[:, 0] / first[:, None]
    parallel = np.sum(tangent * edges[:, 1], axis=1)
    perpendicular = np.linalg.norm(edges[:, 1] - parallel[:, None] * tangent, axis=1)
    assert np.all(first > 0) and np.all(perpendicular > 0)
    inv = np.zeros((len(faces), 2, 2))
    inv[:, 0, 0] = 1 / first
    inv[:, 0, 1] = -parallel / (first * perpendicular)
    inv[:, 1, 1] = 1 / perpendicular
    return inv, first * perpendicular / 2


def surface_strain(points, faces, inverse):
    ds = (points[faces[:, 1:]] - points[faces[:, :1]]).transpose(0, 2, 1)
    gradient = ds @ inverse
    gram = gradient.transpose(0, 2, 1) @ gradient
    stretches = np.sqrt(np.maximum(np.linalg.eigvalsh(gram), 0))
    return np.max(np.abs(stretches - 1), axis=1)


def physical_metrics(e, trace):
    n = len(trace["times"])
    mid_inverse, mid_area = strain_operator(e.mid, e.mid_faces)
    skin_inverse, _ = strain_operator(e.vertices, e.faces)
    result = {
        key: np.zeros(n)
        for key in (
            "max_deformation_m",
            "penetration_m",
            "max_rim_displacement_m",
            "max_edge_length_change",
            "max_midsurface_principal_stretch_deviation",
            "max_skin_principal_stretch_deviation",
            "midsurface_area_fraction_above_10_percent",
        )
    }
    result["tip_drop_m"] = np.zeros((n, 4))
    result["times"] = trace["times"].astype(float) + e.sim_dt
    result["contacts"] = trace["global_raw"]
    q = trace["joint_q"][:, e.q_start : e.q_start + 4]
    for i in range(n):
        disp = np.einsum("vmc,m->vc", e.vertex_phi, q[i])
        vertices = e.vertices + disp
        world = world_points(vertices, trace["body_q"][i : i + 1, e.cap])[0]
        tri = world[e.faces]
        samples = np.concatenate(
            (
                world,
                tri.mean(axis=1),
                (tri[:, 0] + tri[:, 1]) / 2,
                (tri[:, 1] + tri[:, 2]) / 2,
                (tri[:, 2] + tri[:, 0]) / 2,
            )
        )
        sdf = supported.cylinder_sdf(samples, trace["body_q"][i, e.straw], e.cfg.straw_radius, e.cfg.straw_half_height)
        edge = vertices[e.edges[:, 1]] - vertices[e.edges[:, 0]]
        strain = surface_strain(vertices[: len(e.mid)], e.mid_faces, mid_inverse)
        result["max_deformation_m"][i] = np.linalg.norm(disp, axis=1).max()
        result["penetration_m"][i] = max(0.0, -float(sdf.min()))
        result["max_rim_displacement_m"][i] = np.linalg.norm(disp[: len(e.mid)][e.sector < 0], axis=1).max()
        result["max_edge_length_change"][i] = np.max(np.abs(np.linalg.norm(edge, axis=1) / e.rest_edge_lengths - 1))
        result["max_midsurface_principal_stretch_deviation"][i] = strain.max()
        result["max_skin_principal_stretch_deviation"][i] = surface_strain(vertices, e.faces, skin_inverse).max()
        result["midsurface_area_fraction_above_10_percent"][i] = mid_area[strain > 0.1].sum() / mid_area.sum()
        result["tip_drop_m"][i] = -disp[e.tip_indices, 2]
    return result


def check_physics(metrics, contact_enabled):
    assert all(np.isfinite(v).all() for v in metrics.values())
    time_s = metrics["times"]
    hold, stir, release = (time_s > 1.8) & (time_s < 2.25), (time_s > 2.8) & (time_s < 6.4), time_s > 9.1
    stopped = (time_s > 7.0) & (time_s < 7.25)
    result = {key: float(np.max(value)) for key, value in metrics.items() if key.startswith("max_")}
    result["max_penetration_m"] = float(metrics["penetration_m"].max())
    result["midsurface_area_fraction_above_10_percent_max"] = float(
        metrics["midsurface_area_fraction_above_10_percent"].max()
    )
    result["hold_tip_peak_to_peak_m"] = float(np.ptp(metrics["tip_drop_m"][hold], axis=0).max())
    result["release_max_deformation_m"] = float(metrics["max_deformation_m"][release].max())
    result["stir_tip_range_m"] = np.ptp(metrics["tip_drop_m"][stir], axis=0).tolist()
    result["hold_contact_dropouts"] = int(np.sum(metrics["contacts"][hold] == 0))
    result["stir_contact_dropouts"] = int(np.sum(metrics["contacts"][stir] == 0))
    result["post_stir_hold_contact_dropouts"] = int(np.sum(metrics["contacts"][stopped] == 0))
    result["post_stir_hold_tip_peak_to_peak_m"] = float(np.ptp(metrics["tip_drop_m"][stopped], axis=0).max())
    result["stir_max_tip_step_m"] = float(np.abs(np.diff(metrics["tip_drop_m"][stir], axis=0)).max())
    result["stir_max_tip_second_difference_m"] = float(np.abs(np.diff(metrics["tip_drop_m"][stir], n=2, axis=0)).max())
    assert result["max_rim_displacement_m"] < 1e-6
    if contact_enabled:
        assert 0.003 < result["max_deformation_m"] < 0.020, result
        assert result["max_penetration_m"] < 0.0006, result
        assert result["max_edge_length_change"] < 0.15, result
        assert result["hold_tip_peak_to_peak_m"] < 0.0002, result
        assert result["release_max_deformation_m"] < 0.0002, result
        assert min(result["stir_tip_range_m"]) > 0.001, result
        assert (
            result["hold_contact_dropouts"]
            == result["stir_contact_dropouts"]
            == result["post_stir_hold_contact_dropouts"]
            == 0
        ), result
        assert result["post_stir_hold_tip_peak_to_peak_m"] < 0.0002, result
        assert result["stir_max_tip_step_m"] < 0.0004, result
        assert result["stir_max_tip_second_difference_m"] < 0.0001, result
    else:
        assert result["max_deformation_m"] < 1e-5, result
        assert np.all(metrics["contacts"] == 0)
        assert result["max_penetration_m"] > 0.002, "Negative control did not cross the lid"
    result["passed"] = True
    result["acceptance_scope"] = (
        "Supported example contact, bounded edge change and recovery thresholds; not a low-principal-strain acceptance."
    )
    result["principal_strain_screen"] = {
        "threshold": 0.15,
        "midsurface_passed": result["max_midsurface_principal_stretch_deviation"] < 0.15,
        "complete_skin_passed": result["max_skin_principal_stretch_deviation"] < 0.15,
        "claim": "Independent 15-percent principal-stretch screen, reported even when the supported contact tests pass.",
    }
    result["measured_substeps"] = len(time_s)
    return result


def cylinder_mesh(radius, half_height, segments=128):
    """Same side/end-cap tessellation as the original cup RTX exporter."""
    angle = np.arange(segments) * (2 * np.pi / segments)
    rings = [
        np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.full(segments, z)))
        for z in (-half_height, half_height)
    ]
    points = np.concatenate([*rings, *rings, [[0, 0, -half_height], [0, 0, half_height]]]).astype(np.float32)
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend(
            [
                (i, j, j + segments),
                (i, j + segments, i + segments),
                (4 * segments, j + 2 * segments, i + 2 * segments),
                (4 * segments + 1, i + 3 * segments, j + 3 * segments),
            ]
        )
    return points, np.asarray(faces, dtype=np.int32)


def export_scene(e, trace, initial, output):
    times = np.arange(FRAMES + 1, dtype=float) / 60
    saved = {
        key: np.concatenate((initial[key][None], trace[key][e.cfg.substeps - 1 :: e.cfg.substeps]))
        for key in STATE_NAMES
    }
    saved["times"] = times
    saved["modal_q"] = saved["joint_q"][:, e.q_start : e.q_start + 4]
    local = e.vertices[None] + np.einsum("vmc,fm->fvc", e.vertex_phi, saved["modal_q"])
    straw, straw_faces = cylinder_mesh(e.cfg.straw_radius, e.cfg.straw_half_height)
    cup, cup_faces = cup_mesh(height=e.cfg.lid_z)
    rim, rim_faces = ring_mesh(0.0445, 0.047, e.cfg.lid_z + 0.001, 0.0014)
    scene = {
        "times": times,
        "vertices_0": world_points(local, saved["body_q"][:, e.cap]),
        "faces_0": e.faces,
        "vertices_1": world_points(straw, saved["body_q"][:, e.straw]),
        "faces_1": straw_faces,
        "vertices_2": cup,
        "faces_2": cup_faces,
        "vertices_3": rim,
        "faces_3": rim_faces,
    }
    np.savez_compressed(output / "scene.npz", **scene)
    np.savez_compressed(output / "trajectory.npz", **saved)
    np.savez_compressed(
        output / "basis.npz",
        rest_vertices=e.vertices,
        faces=e.faces,
        phi=e.vertex_phi,
        mode_mass=e.basis.mode_mass,
        mode_stiffness=e.basis.mode_stiffness,
        mode_damping=e.basis.mode_damping,
        sample_mass=e.basis.sample_mass,
        mid=e.mid,
        mid_faces=e.mid_faces,
        sector=e.sector,
        depth=e.depth,
        snapshots=e.snapshots,
    )
    np.savez_compressed(output / "surface_map.npz", mesh_0_vertex_ids=np.arange(len(e.vertices), dtype=np.int32))
    meshes = []
    for i, (name, color, roughness, opacity) in enumerate(
        (
            ("contact-driven-four-mode-lid", [0.72, 0.76, 0.79], 0.3, 1.0),
            ("prescribed-solid-straw-proxy", [0.62, 0.018, 0.009], 0.27, 1.0),
            ("fixed-drinking-cup-display", [0.65, 0.73, 0.78], 0.28, 0.065),
            ("fixed-lid-rim", [0.72, 0.76, 0.79], 0.3, 1.0),
        )
    ):
        meshes.append(
            {
                "name": name,
                "vertices": f"vertices_{i}",
                "faces": f"faces_{i}",
                "color": color,
                "roughness": roughness,
                "metallic": 0.0,
                "opacity": opacity,
            }
        )
    meta = {
        "slug": "cup-lid",
        "title": "Drinking cup, slit lid and straw",
        "status": "contact validated; strain-limited ROM",
        "fps": 60,
        "meshes": meshes,
        "poster_frame": 240,
        "camera": {"position": [0.23, -0.36, 0.30], "target": [0, 0, 0.15], "fov": 42},
        "camera_detail": {"position": [0.028, -0.040, 0.164], "target": [0, 0, 0.1215], "fov": 38},
        "floor_z": 0.0,
        "mode_count": 4,
        "description": "A prescribed straw inserts at 10 degrees, stirs twice, withdraws and releases a contact-driven four-mode slit lid.",
        "model": "The original supported cup/straw scene: one four-mode POD lid with fixed kinematic frame, 4 substeps, 12 VBD iterations.",
        "driven": "Straw rigid pose/velocity only. Lid floating frame fixed; all four modal coordinates are solved by contact, never prescribed.",
        "limitations": [
            "Cup and rim are fixed, visual-only supports with no cup collision, motion or deformation.",
            "Straw collision and display are a solid capped cylinder proxy, not a hollow tube; no liquid or fluid forces.",
            "Four linear POD modes from geometric bending exemplars, not shell-FEM snapshots; no tearing, plasticity, self-contact, buckling or arbitrary folding.",
            "Retain the approved 10-degree tilt and 0.5 mm orbit; the original 25-degree trial overextended the basis and snagged.",
            "Material values are assumed, not calibrated. Thin lid top and bottom share the same midsurface displacement field.",
            "Surface strain is geometric principal stretch, not stress; thin side-wall shear is not a calibrated shell strain.",
            "The approved trajectory fails a 15-percent principal-stretch screen: approximately 18.1 percent midsurface and 47.9 percent complete-skin peak deviation, despite only 9.44 percent edge-length change. Do not present as inextensible or low-strain.",
            "Penetration is sampled at collision vertices, edge midpoints and triangle centroids, not an exhaustive continuous intersection bound.",
        ],
        "parameters": {
            **asdict(e.cfg),
            "frames": FRAMES,
            "gravity_m_s2": [0, 0, -9.81],
            "elastic_body_relaxation": 0.6,
            "rigid_compliant_alm": False,
            "rigid_contact_hard": False,
            "reduce_contacts": False,
            "global_contact_capacity": e.contacts.rigid_contact_max,
            "body_contact_capacity": e.solver.body_body_contact_buffer_pre_alloc,
        },
        "source_files": ["source/baseline.py", "source/cup_lid.py", "source/test_cup_lid.py"],
        "reproduce": "uv run python reports/everyday_rom/scenes/cup_lid/cup_lid.py --output output/everyday-rom/cup-lid",
        "surface_recovery": "rest_vertices + einsum('vmc,fm->fvc', phi, modal_q), then saved body rotation/translation; no amplification",
        "simulation_state": "trajectory.npz",
        "original_basis": "basis.npz",
        "render_vertex_map": "surface_map.npz",
        "timeline_s": {
            "insert": [0.3, 1.5],
            "hold": [1.5, 2.3],
            "stir": [2.5, 6.5],
            "withdraw": [7.3, 8.5],
            "release": [8.5, 9.5],
        },
        "render_notes": "Use all four original meshes. Only mesh 0 is elastic. Preserve normal then strain sequence without changing geometry. Overview includes the entire straw; detail camera targets the slit.",
        "runtime_commit": PIN,
        "original_report": "https://reports.mmacklin.com/cup-straw-rom/index.html",
    }
    write_json(output / "scene.json", meta)
    return saved


def run_audit(output, contact_enabled):
    output.mkdir(parents=True, exist_ok=True)
    info = provenance()
    e = supported.Example(None, config=supported.Example.Config(contact_enabled=contact_enabled))
    # The only solver option changed is read-only residual instrumentation.
    e.solver = newton.solvers.SolverVBD(
        e.model,
        iterations=e.cfg.iterations,
        rigid_body_contact_buffer_size=1024,
        rigid_contact_k_start=e.cfg.contact_ke,
        rigid_compliant_alm=False,
        rigid_contact_hard=False,
        friction_epsilon=1e-4,
        elastic_body_relaxation=0.6,
        elastic_solver_metrics=True,
    )
    initial = {key: getattr(e.state_0, key).numpy() for key in STATE_NAMES}
    trace = SubstepTrace(e)
    e.metrics_enabled = True
    start = time.perf_counter()
    for _ in range(FRAMES):
        e.step()
    wp.synchronize()
    audit_seconds = time.perf_counter() - start
    data = trace.data()
    data["cursor"] = np.asarray(int(data["cursor"][0]))
    np.testing.assert_allclose(
        data["body_q"][:, e.cap], np.broadcast_to(initial["body_q"][e.cap], (len(data["times"]), 7)), atol=1e-7
    )
    np.testing.assert_allclose(data["body_qd"][:, e.cap], 0, atol=1e-7)
    np.testing.assert_allclose(data["joint_q"][:, e.q_start - 7 : e.q_start], data["body_q"][:, e.cap], atol=1e-7)
    np.savez_compressed(output / "substep-audit.npz", **data)
    capacity = check_trace(
        data,
        FRAMES * e.cfg.substeps,
        e.contacts.rigid_contact_max,
        e.solver.body_body_contact_buffer_pre_alloc,
        e.sim_dt,
    )
    write_json(output / "contact_capacity.json", capacity)
    measured = physical_metrics(e, data)
    np.savez_compressed(output / "substep_metrics.npz", **measured)
    physical = check_physics(measured, contact_enabled)
    e.test_final()
    summary = e.summary()
    write_json(output / "frame_metrics.json", e.records)
    info.update(
        {
            "test_passed": True,
            "contact_enabled": contact_enabled,
            "config": asdict(e.cfg),
            "supported_example_tests_passed": True,
            "physical_all_substeps": physical,
            "max_displacement_m": physical["max_deformation_m"],
            "final_displacement_m": float(measured["max_deformation_m"][-1]),
            "max_penetration_m": physical["max_penetration_m"],
            "max_principal_strain": physical["max_midsurface_principal_stretch_deviation"],
            "principal_strain_screen_passed": physical["principal_strain_screen"]["midsurface_passed"],
            "contact_capacity": capacity,
            "frame_summary": summary,
            "mode_count": 4,
            "frames": FRAMES,
            "saved_frames": FRAMES + 1,
            "audit_wall_s": audit_seconds,
            "audit_timing_scope": "Not a performance baseline: includes every-substep trace/residual GPU work, every-frame CPU validation, readbacks and Python dispatch; excludes later all-substep geometry metrics.",
            "test_scope": "Every actual substep and supported Example.test_final; no contact-disabled modal forcing.",
        }
    )
    residual = data["solver_metrics"]
    ratio = residual[:, 1] / np.maximum(residual[:, 0], 1e-12)
    info["solver_metrics"] = {
        "columns": METRIC_NAMES,
        "maxima": residual.max(axis=0).tolist(),
        "max_solve_residual_ratio": float(ratio.max()),
        "scope": "Final VBD iteration block residual for every actual substep",
    }
    assert ratio.max() < 0.05
    saved = export_scene(e, data, initial, output)
    write_json(output / "metrics.json", info)
    print(json.dumps({"contact_enabled": contact_enabled, "physical": physical, "capacity": capacity}), flush=True)
    return info, saved, data


def bundle(output):
    wp.set_device("cuda:0")
    positive, saved, trace = run_audit(output, True)
    negative, _, _ = run_audit(output / "negative-control", False)
    positive["negative_control_passed"] = negative["test_passed"]
    positive["negative_control_max_deformation_m"] = negative["physical_all_substeps"]["max_deformation_m"]
    baseline = json.loads((output / "baseline/performance.json").read_text())
    assert baseline["passed"]
    differences = {}
    for trial in range(len(baseline["trials"])):
        clean = np.load(output / f"baseline/graph-final-{trial}.npz")
        for key in STATE_NAMES:
            differences[f"trial_{trial}_{key}"] = float(np.max(np.abs(clean[key] - saved[key][-1])))
            np.testing.assert_allclose(clean[key], saved[key][-1], atol=2e-6, rtol=2e-5)
    profiles = np.load(output / "baseline/profile-states.npz")
    for phase in baseline["kernel_profile"]:
        index = (phase["start_frame"] + phase["frames"]) * positive["config"]["substeps"] - 1
        for key in STATE_NAMES:
            clean = profiles[f"{phase['phase']}_{key}"]
            differences[f"{phase['phase']}_{key}"] = float(np.max(np.abs(clean - trace[key][index])))
            np.testing.assert_allclose(clean, trace[key][index], atol=2e-6, rtol=2e-5)
    positive["clean_vs_audit_state_check_passed"] = True
    positive["clean_vs_audit_max_abs"] = differences
    positive["simulation_wall_ms_per_frame"] = baseline["median_graph_wall_ms_per_frame"]
    positive["simulation_cuda_ms_per_frame"] = baseline["median_graph_cuda_ms_per_frame"]
    positive["sim_frame_ms"] = baseline["median_graph_cuda_ms_per_frame"]
    positive["timing_scope"] = baseline["timing_scope"]
    write_json(output / "metrics.json", positive)
    negative["negative_control_passed"] = True
    write_json(output / "negative-control/metrics.json", negative)
    sources = output / "source"
    sources.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, sources / path.name)
    shutil.copy2(Path(__file__).with_name("README.md"), sources / "README.md")
    for path in (Path(supported.__file__), Path(supported.__file__).with_name("_reduced_elastic_cap.py")):
        shutil.copy2(path, sources / path.name)
    manifest = {
        "status": "staged-contact-tested-strain-warning-not-rendered",
        "gallery_slug": "cup-lid",
        "replaces_gallery_slug": "mustard-bottle",
        "runtime_commit": PIN,
        "worktree": str(Path(newton.__file__).resolve().parents[1]),
        "branch": "codex/cup-lid-refinement-20261016",
        "stage": str(output),
        "parent_owns": ["common report builder", "evaluator integration", "publishing"],
        "lorentz_owns": ["normal then strain rendering"],
        "schema_handoff": "RENDER_HANDOFF.md",
        "original_report_preserved": "https://reports.mmacklin.com/cup-straw-rom/index.html",
        "runtime_modified": False,
        "hidden_modal_animation": False,
        "contact_capacity_passed_both_runs": True,
        "actual_substeps_per_run": FRAMES * positive["config"]["substeps"],
        "physical_results": positive["physical_all_substeps"],
        "principal_strain_screen_passed": positive["principal_strain_screen_passed"],
        "integration_warning": "Supported contact tests pass; independent 15-percent principal-strain screen fails. Preserve this limitation in the normal/strain presentation; no kinetics were retuned.",
        "baseline": "baseline/performance.json",
        "physical_metrics": "metrics.json",
        "clean_graph_ms_per_frame": baseline["median_graph_cuda_ms_per_frame"],
        "clean_vs_audit_state_check_passed": True,
        "render_scene_files": ["scene.json", "scene.npz"],
        "source_files": sorted(str(p.relative_to(output)) for p in sources.iterdir()),
    }
    manifest["sha256"] = {
        str(p.relative_to(output)): digest(p)
        for p in output.rglob("*")
        if p.is_file() and "warp-cache" not in p.parts and p.name != "integration_manifest.json"
    }
    write_json(output / "integration_manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STAGE)
    args = parser.parse_args()
    bundle(args.output)
