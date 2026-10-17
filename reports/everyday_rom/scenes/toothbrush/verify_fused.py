# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent CPU audit of fused exports; never edit inputs or run simulation."""

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
from brush_glove_geometry import (
    Config,
    brush_handle,
    coupled_mass_audit,
    sampled_mass_properties,
    surface_mass,
    transform_points,
)
from capacity_audit import validate as validate_capacity
from field_coverage import coverage_summary
from fused_toothbrush import HostArray, virtual_handle
from test_toothbrush_scrub import PHYSICS_KEYS, scrub_strokes
from toothbrush import Example as OriginalExample

HERE = Path(__file__).resolve().parent
STATE_KEYS = ("body_q", "body_qd", "joint_q", "joint_qd", "joint_enabled")


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_motion_limits(motion):
    assert motion["max_surface_step_m"] < 0.001, f"modal surface step: {motion['max_surface_step_m']}"
    assert motion["max_surface_second_difference_m"] < 0.001, (
        f"modal second difference: {motion['max_surface_second_difference_m']}"
    )


def rest_metric(points, faces):
    triangles = np.asarray(points, dtype=float)[faces]
    a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    length = np.linalg.norm(a, axis=1)
    area2 = np.linalg.norm(np.cross(a, b), axis=1)
    scale = length * np.linalg.norm(b, axis=1)
    valid = (scale > 0) & (area2 > 1e-10 * scale)
    if not valid.all():
        raise ValueError(f"{np.count_nonzero(~valid)} degenerate reference triangles")
    projection = np.einsum("ij,ij->i", a, b) / length
    height = area2 / length
    inverse = np.zeros((len(faces), 2, 2))
    inverse[:, 0, 0] = 1 / length
    inverse[:, 0, 1] = -projection / (length * height)
    inverse[:, 1, 1] = 1 / height
    return inverse, area2 / 2


def stretches(points, faces, inverse):
    triangles = np.asarray(points, dtype=float)[faces]
    edges = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1)
    deformation = edges @ inverse
    metric = np.swapaxes(deformation, -1, -2) @ deformation
    return np.sqrt(np.maximum(np.linalg.eigvalsh(metric), 0))


def check_strain_limits(first_frame, material_rest):
    assert first_frame < 0.10, f"first-export strain: {first_frame}"
    assert material_rest < 0.10, f"material-rest strain: {material_rest}"


def modal_motion(phi, q, equilibrium):
    assert all(np.isfinite(a).all() for a in (phi, q, equilibrium)), "Non-finite modal input"
    displacement = np.einsum("vmc,fm->fvc", phi, q)
    relative = displacement - np.einsum("vmc,m->vc", phi, equilibrium)
    return {
        "displacement": displacement,
        "relative": relative,
        "max_surface_step_m": float(np.linalg.norm(np.diff(displacement, axis=0), axis=2).max()),
        "max_surface_second_difference_m": float(np.linalg.norm(np.diff(displacement, n=2, axis=0), axis=2).max()),
    }


def check_residuals(raw):
    initial, solved = raw["initial_residual_norm"], raw["solve_residual_norm"]
    assert initial.shape == solved.shape == (450, 1), "Need 450 residual observations for one elastic body"
    assert np.isfinite(initial).all() and np.isfinite(solved).all(), "Non-finite residual"
    assert np.all(initial >= 0) and np.all(solved >= 0), "Negative residual norm"
    ratios = solved / np.maximum(initial, 1e-12)
    assert float(ratios.max()) < 0.05, f"Original modal residual gate: {ratios.max()}"
    return ratios[:, 0]


def check_saved_capacity(raw):
    expected = (60 + 450) * 4
    assert int(raw["expected_substeps"]) == expected
    assert int(raw["global_capacity"]) == int(raw["body_capacity"]) == 16384
    assert float(raw["dt"]) == 1 / 240
    data = {
        key: value
        for key, value in raw.items()
        if key not in ("global_capacity", "body_capacity", "expected_substeps", "dt")
    }
    return validate_capacity(data, expected, 16384, 16384, 1 / 240)


def check_pair_states(positive, control):
    for key in ("times", "joint_enabled"):
        np.testing.assert_array_equal(positive[key], control[key], err_msg=key)
    for key in ("body_q", "body_qd"):
        np.testing.assert_array_equal(positive[key][:, 1:], control[key][:, 1:], err_msg=f"kinematic {key}")


def verify_static(scene, basis, model, contact_enabled):
    reference = json.loads((HERE / "fused_reference.json").read_text())
    assert scene["rigid_fusion"] and len(scene["meshes"]) == 30
    expected_config = {
        "substeps": 4,
        "iterations": 16,
        "contact_enabled": contact_enabled,
        "contact_ke": 3000.0,
        "contact_kd": 0.0,
        "friction": 0.2,
        "contact_gap": 0.00015,
        "elastic_relaxation": 0.4,
    }
    assert scene["parameters"] == expected_config, "Physical configuration changed"
    for name, expected in reference["model"].items():
        if name == "shape_contact_pairs" and not contact_enabled:
            continue
        array = model[name]
        assert list(array.shape) == expected["shape"] and str(array.dtype) == expected["dtype"], name
        assert hashlib.sha256(array.tobytes()).hexdigest() == expected["sha256"], name
    for name in (
        "body_mass",
        "body_com",
        "body_inertia",
        "body_flags",
        "joint_parent",
        "joint_child",
        "joint_type",
        "joint_X_p",
        "joint_X_c",
        "shape_material_ke",
        "shape_material_kd",
        "shape_material_mu",
    ):
        np.testing.assert_array_equal(basis[f"model_{name}"], model[name])
    frozen = load(HERE / "frozen/basis.npz")
    for i, mesh in enumerate(scene["meshes"]):
        expected = reference["coarse_meshes"][str(i)]
        assert mesh["name"] == expected["name"] and mesh["body_index"] == expected["body"]
        for key, label in (("rest_vertices", "points"), ("faces", "faces")):
            value = basis[f"{key}_{i}"]
            assert list(value.shape) == expected[label]["shape"] and str(value.dtype) == expected[label]["dtype"]
            assert hashlib.sha256(value.tobytes()).hexdigest() == expected[label]["sha256"]
        if mesh["q_start"] is not None:
            assert mesh["q_start"] == 7 and mesh["body_index"] == 0
            for key in ("phi", "source_phi", "sample_mass", *PHYSICS_KEYS):
                np.testing.assert_array_equal(basis[f"{key}_{i}"], frozen[f"{key}_{i}"])
    np.testing.assert_array_equal(basis["solver_elastic_implicit_mass_coupling"], [True])


def recover(scene, basis, state, settled):
    """Rebuild observables from saved states, not the exported summary."""
    times = state["times"]
    np.testing.assert_allclose(times, np.arange(1, 451) / 60, rtol=0, atol=1e-12)
    for key, shape in (
        ("body_q", (450, 5, 7)),
        ("body_qd", (450, 5, 6)),
        ("joint_q", (450, 75)),
        ("joint_qd", (450, 70)),
        ("joint_enabled", (450, 6)),
    ):
        assert state[key].shape == shape and np.isfinite(state[key]).all(), key
    handle_mesh = brush_handle(split_head=True)[0]
    _, handle_com, _ = sampled_mass_properties(handle_mesh[0], surface_mass(*handle_mesh, 0.018))
    root = -np.asarray(scene["diagnostic_handle_local_translation"], np.float32)
    fused_com = basis["model_body_com"][0]
    poses, speeds = zip(
        *(
            virtual_handle(p, v, root, handle_com, fused_com)
            for p, v in zip(state["body_q"][:, 0], state["body_qd"][:, 0], strict=True)
        ),
        strict=True,
    )
    poses, speeds = np.asarray(poses), np.asarray(speeds)
    initial, _ = virtual_handle(settled["body_q"][0], settled["body_qd"][0], root, handle_com, fused_com)
    result = {
        "handle_position_m": poses[:, :3],
        "handle_velocity": speeds,
        "handle_com_m": np.stack([transform_points(handle_com[None], pose)[0] for pose in poses]),
        "max_deformation_m": np.zeros(450),
        "max_total_displacement_m": np.zeros(450),
        "plate_modal_displacement_m": np.zeros(450),
        "floor_penetration_m": np.zeros(450),
        "penetration_m": np.zeros(450),
    }
    result["root_anchor_error_m"] = np.array(
        [
            np.linalg.norm(transform_points(root[None], pose)[0] - actual[:3])
            for pose, actual in zip(poses, state["body_q"][:, 0], strict=True)
        ]
    )
    result["root_angular_error_rad"] = np.array(
        [
            2 * np.arccos(np.clip(abs(float(np.dot(pose[3:], actual[3:]))), 0, 1))
            for pose, actual in zip(poses, state["body_q"][:, 0], strict=True)
        ]
    )
    tips_x, tips_y, worlds = [], [], {}
    max_step = max_second = max_first_strain = max_rest_strain = 0.0
    for i, mesh in enumerate(scene["meshes"]):
        rest, faces, body = basis[f"rest_vertices_{i}"], basis[f"faces_{i}"], mesh["body_index"]
        local = np.broadcast_to(rest, (450, *rest.shape))
        if mesh["q_start"] is not None:
            phi, start = basis[f"phi_{i}"], mesh["q_start"]
            np.testing.assert_allclose(
                basis[f"mode_stiffness_{i}"] * basis[f"analytic_equilibrium_q_{i}"],
                basis[f"mode_coupling_linear_{i}"] @ np.array([0, 0, -9.81]),
                rtol=1e-6,
                atol=1e-9,
            )
            motion = modal_motion(phi, state["joint_q"][:, start : start + 40], basis[f"equilibrium_q_{i}"])
            max_step = max(max_step, motion["max_surface_step_m"])
            max_second = max(max_second, motion["max_surface_second_difference_m"])
            local = rest + motion["displacement"]
            deformation = np.linalg.norm(motion["relative"], axis=2).max(axis=1)
            result["max_deformation_m"] = np.maximum(result["max_deformation_m"], deformation)
            result["max_total_displacement_m"] = np.maximum(
                result["max_total_displacement_m"], np.linalg.norm(motion["displacement"], axis=2).max(axis=1)
            )
            if mesh["name"].startswith("tuft_"):
                tip = rest[:, 2] > rest[:, 2].max() - 1e-6
                tips_x.append(motion["relative"][:, tip, 0].mean(axis=1))
                tips_y.append(motion["relative"][:, tip, 1].mean(axis=1))
            else:
                result["plate_modal_displacement_m"] = np.maximum(result["plate_modal_displacement_m"], deformation)
        world = (
            np.stack(
                [transform_points(points, pose[body]) for points, pose in zip(local, state["body_q"], strict=True)]
            )
            if body >= 0
            else rest
        )
        worlds[mesh["vertices"]] = world
        frames = world if world.ndim == 3 else world[None]
        inverse, _ = rest_metric(frames[0], faces)
        max_first_strain = max(
            max_first_strain, *(float(np.abs(stretches(points, faces, inverse) - 1).max()) for points in frames)
        )
        if mesh["q_start"] is not None:
            inverse, _ = rest_metric(rest, faces)
            max_rest_strain = max(
                max_rest_strain, *(float(np.abs(stretches(points, faces, inverse) - 1).max()) for points in local)
            )
        if i == 0 or mesh["q_start"] is not None:
            result["floor_penetration_m"] = np.maximum(result["floor_penetration_m"], -world[:, :, 2].min(axis=1))
        if mesh["q_start"] is not None:
            for frame in range(450):
                points = world[frame]
                triangle = points[faces]
                samples = np.concatenate(
                    [
                        points,
                        triangle.mean(axis=1),
                        (triangle[:, 0] + triangle[:, 1]) / 2,
                        (triangle[:, 1] + triangle[:, 2]) / 2,
                        (triangle[:, 2] + triangle[:, 0]) / 2,
                    ]
                )
                tool = state["body_q"][frame, 1]
                rotation = np.asarray(wp.quat_to_matrix(wp.quat(*tool[3:])), dtype=float).reshape(3, 3)
                tool_local = (samples - tool[:3]) @ rotation
                tool_local[:, 1] -= np.clip(tool_local[:, 1], -0.035, 0.035)
                result["penetration_m"][frame] = max(
                    result["penetration_m"][frame], float((0.0025 - np.linalg.norm(tool_local, axis=1)).max())
                )
    result["tip_x_displacement_m"], result["tip_y_displacement_m"] = np.asarray(tips_x).T, np.asarray(tips_y).T
    assert result["tip_x_displacement_m"].shape == (450, 20)
    result["tuft_tip_deformation_m"] = np.hypot(result["tip_x_displacement_m"], result["tip_y_displacement_m"])
    return (
        result,
        worlds,
        {
            "max_surface_step_m": max_step,
            "max_surface_second_difference_m": max_second,
            "initial_handle_z_m": float(initial[2]),
            "max_first_export_strain": max_first_strain,
            "max_material_rest_strain": max_rest_strain,
        },
    )


def physical_summary(metrics, recovered, state, raw_contacts, field, residuals, model, capacity, motion):
    """Supply independently recovered quantities to the unchanged physical gates."""
    records = [dict(record) for record in metrics["records"]]
    assert len(records) == 450
    np.testing.assert_allclose([r["time"] for r in records], state["times"], rtol=0, atol=1e-10)
    offsets = raw_contacts["offsets"]
    assert offsets.shape == (451,) and offsets[0] == 0 and np.all(np.diff(offsets) >= 0)
    assert offsets[-1] == len(raw_contacts["shape0"]) == len(raw_contacts["shape1"])
    assert field["counts"].shape == (2040, 20)
    assert int(field["settling_substeps"]) == 240 and int(field["simulation_substeps"]) == 1800
    assert np.all(field["counts"] >= 0)
    ratios = check_residuals(residuals)
    for frame, record in enumerate(records):
        for key, values in recovered.items():
            np.testing.assert_allclose(record[key], values[frame], rtol=0, atol=2e-8, err_msg=f"frame {frame} {key}")
            record[key] = values[frame].tolist()
        a, b = (raw_contacts[name][offsets[frame] : offsets[frame + 1]] for name in ("shape0", "shape1"))
        tool, head = (a == 5) | (b == 5), (a == 4) | (b == 4)
        for key, value in (
            ("contacts", len(a)),
            ("tool_contacts", int(np.count_nonzero(tool & head))),
            ("other_tool_contacts", int(np.count_nonzero(tool & ~head))),
            ("floor_contacts", int(np.count_nonzero((a == 7) | (b == 7)))),
        ):
            assert record[key] == value, key
        expected_field = field["counts"][240 + 4 * frame : 244 + 4 * frame].sum(axis=0)
        np.testing.assert_array_equal(record["tuft_contacts_all_substeps"], expected_field)
        np.testing.assert_allclose(record["modal_solve_residual_ratio"], ratios[frame], rtol=0, atol=1e-12)
        assert record["grasp_enabled"] == bool(state["joint_enabled"][frame, 5])
    held = [r for r in records if 2.5 < r["time"] < 4.2]
    falling = [r for r in records if 5.22 < r["time"] < 5.32]
    landed = [r for r in records if r["time"] > 6.8]
    scrubbing = [r for r in records if 2.5 < r["time"] < 4.5]
    tangent = np.array([r["tip_y_displacement_m"] for r in scrubbing])
    mass = coupled_mass_audit(
        float(model["body_mass"][0]),
        model["body_com"][0].astype(float),
        model["body_inertia"][0].astype(float),
        model["elastic_mode_mass"].astype(float),
        model["elastic_mode_coupling_linear"].astype(float),
        model["elastic_mode_coupling_angular"].astype(float),
    )
    mass["actual_implicit_mass_coupling"] = [True]
    summary = dict(metrics)
    summary.update(
        records=records,
        mass_audit=mass,
        contact_buffer_overflow=not capacity["passed"],
        max_modal_solve_residual_ratio=float(ratios.max()),
        max_root_anchor_error_m=float(recovered["root_anchor_error_m"].max()),
        max_root_angular_error_rad=float(recovered["root_angular_error_rad"].max()),
        max_penetration_m=max(
            float(recovered["floor_penetration_m"].max()),
            float(recovered["penetration_m"].max()) if metrics["config"]["contact_enabled"] else 0,
        ),
        max_deformation_m=float(recovered["max_deformation_m"].max()),
        contact_window_deformation_m=max(r["max_deformation_m"] for r in held),
        tool_contact_frames=sum(r["tool_contacts"] > 0 for r in held),
        handle_lift_m=float(recovered["handle_position_m"][:, 2].max()) - motion["initial_handle_z_m"],
        freefall_vertical_acceleration_m_s2=float(np.mean(np.diff([r["handle_velocity"][2] for r in falling])) * 60),
        freefall_drop_m=falling[0]["handle_com_m"][2] - falling[-1]["handle_com_m"][2],
        final_handle_speed_m_s=float(np.linalg.norm(recovered["handle_velocity"][-1, :3])),
        final_handle_angular_speed_rad_s=float(np.linalg.norm(recovered["handle_velocity"][-1, 3:])),
        landed_position_range_m=float(np.ptp([r["handle_com_m"] for r in landed], axis=0).max()),
        release_max_deformation_m=max(r["max_deformation_m"] for r in landed),
        scrub_tuft_reversals=int(np.count_nonzero((tangent.max(axis=0) > 0.00025) & (tangent.min(axis=0) < -0.00025))),
        scrub_tangential_peak_to_peak_m=float(np.ptp(tangent, axis=0).max()),
        scrub_contact_frames=sum(r["tool_contacts"] > 0 for r in scrubbing),
        field_coverage=coverage_summary(records),
    )
    return summary


def verify_arm(directory, output):
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scene, metrics = (json.loads((directory / name).read_text()) for name in ("scene.json", "metrics.json"))
    basis, state, model = (load(directory / name) for name in ("basis.npz", "state.npz", "model.npz"))
    assert metrics["graph_enabled"] and metrics["config"] == scene["parameters"]
    assert scene["contact_tool"] == {"shape": "capsule_y", "radius_m": 0.0025, "half_length_m": 0.035}
    np.testing.assert_array_equal(metrics["gravity_m_s2"], [0, 0, -9.81])
    assert metrics["root_joints"] == [] and metrics["handle_body_index"] == metrics["head_body_index"] == 0
    assert metrics["grasp_joint_index"] == 5 and metrics["mode_count"] == 40
    assert {name: sha(directory / "source" / name) for name in scene["source_files"]} == metrics["source_sha256"]
    verify_static(scene, basis, model, metrics["config"]["contact_enabled"])
    recovered, worlds, motion = recover(scene, basis, state, load(directory / "settled.npz"))
    np.savez_compressed(output / "recovered.npz", **recovered)
    arrays = load(directory / "scene.npz")
    np.testing.assert_array_equal(arrays["times"], state["times"])
    error = 0.0
    for i, mesh in enumerate(scene["meshes"]):
        np.testing.assert_array_equal(arrays[mesh["faces"]], basis[f"faces_{i}"])
        assert np.isfinite(arrays[mesh["vertices"]]).all()
        error = max(error, float(np.abs(worlds[mesh["vertices"]] - arrays[mesh["vertices"]]).max()))
    capacity = check_saved_capacity(load(directory / "contact_capacity.npz"))
    summary = physical_summary(
        metrics,
        recovered,
        state,
        load(directory / "frame_contacts.npz"),
        load(directory / "field_contacts.npz"),
        load(directory / "residuals.npz"),
        model,
        capacity,
        motion,
    )
    result = dict(motion, export_recovery_max_error_m=error, capacity=capacity, recovered_summary=summary)
    (output / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
    assert error < 1e-7, f"Export reconstruction {error}"
    check_motion_limits(motion)
    check_strain_limits(motion["max_first_export_strain"], motion["max_material_rest_strain"])
    view = SimpleNamespace(
        records=summary["records"],
        cfg=Config(**metrics["config"]),
        graph=True,
        summary=lambda: summary,
        model=SimpleNamespace(elastic_body_count=1, body_flags=HostArray(model["body_flags"])),
        elastic=[None],
        expected_dynamic_body_count=1,
        capacity_audit=SimpleNamespace(
            check=lambda expected: check_saved_capacity(load(directory / "contact_capacity.npz"))
        ),
    )
    OriginalExample.test_final(view)
    times, positions = state["times"], recovered["handle_position_m"]
    strokes = scrub_strokes(times, positions)
    assert np.ptp(positions[times < 0.35, :2], axis=0).max() < 0.0002
    assert np.linalg.norm(positions[times < 1.5, :2] - positions[0, :2], axis=1).max() < 0.0005
    assert positions[(times > 1.4) & (times < 1.5), 2].min() - positions[0, 2] > 0.10
    enabled = state["joint_enabled"][:, 5]
    assert not np.any(enabled[times > 5.22])
    assert np.all(enabled[(times > 0.37) & (times < 5.19)])
    result.update(
        passed=True,
        actual_handle_strokes_m=strokes,
        runtime_test_flags_retained={key: metrics.get(key) for key in ("test_passed", "negative_control_passed")},
    )
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_pair(positive, control, output):
    output = Path(output)
    reports = [verify_arm(path, output / name) for path, name in ((positive, "positive"), (control, "control"))]
    states = [load(Path(path) / "state.npz") for path in (positive, control)]
    check_pair_states(*states)
    a, b = (row["recovered_summary"] for row in reports)
    ratio = b["contact_window_deformation_m"] / a["contact_window_deformation_m"]
    assert ratio < 0.1
    assert not b["contact_window_deformation_m"] > 0.001, "Control must fail the original positive gate"
    model_a, model_b = (load(Path(path) / "model.npz") for path in (positive, control))
    pairs_a, pairs_b = (set(map(tuple, model["shape_contact_pairs"])) for model in (model_a, model_b))
    assert pairs_a - pairs_b == {(4, 5)} and not pairs_b - pairs_a
    basis = load(Path(positive) / "basis.npz")
    scene = json.loads((Path(positive) / "scene.json").read_text())
    final = peak = held_control = 0.0
    held = (states[0]["times"] > 2.5) & (states[0]["times"] < 4.2)
    for i, mesh in enumerate(scene["meshes"]):
        if mesh["q_start"] is None:
            continue
        phi, start = basis[f"phi_{i}"], mesh["q_start"]
        q, control_q = (state["joint_q"][:, start : start + 40] for state in states)
        difference = np.einsum("vmc,fm->fvc", phi, q - control_q)
        drift = np.einsum("vmc,fm->fvc", phi, control_q - basis[f"equilibrium_q_{i}"])
        final = max(final, float(np.linalg.norm(difference[-1], axis=1).max()))
        peak = max(peak, float(np.linalg.norm(difference, axis=2).max()))
        held_control = max(held_control, float(np.linalg.norm(drift[held], axis=2).max()))
    result = {
        "contact_window_control_ratio": ratio,
        "final_matched_modal_difference_m": final,
        "peak_matched_modal_difference_m": peak,
        "held_control_drift_from_positive_equilibrium_m": held_control,
        "private_trajectory_parity_required": False,
    }
    (output / "pair-measurements.json").write_text(json.dumps(result, indent=2) + "\n")
    assert final < 0.0002 and peak > 0.001 and held_control < 0.0001, result
    result["passed"] = True
    (output / "pair-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for source in (args.positive, args.control):
        if source is not None:
            assert not args.output.resolve().is_relative_to(source.resolve()), "Audit output must not modify its inputs"
    assert not args.output.exists(), "Use a fresh audit directory"
    args.output.mkdir(parents=True)
    try:
        result = (
            verify_pair(args.positive, args.control, args.output)
            if args.control
            else verify_arm(args.positive, args.output)
        )
        print(json.dumps({key: value for key, value in result.items() if key != "recovered_summary"}, indent=2))
    except Exception:
        (args.output / "failure.json").write_text(
            json.dumps({"passed": False, "traceback": traceback.format_exc()}, indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    main()
