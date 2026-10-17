# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Jointed dynamic toothbrush: sweep the full bristle field, recover and drop."""

from dataclasses import replace

import numpy as np
import warp as wp
from brush_glove_geometry import (
    CapturedExample,
    Config,
    advance_clock,
    brush_handle,
    brush_head_rom,
    coupled_mass_audit,
    ellipsoid,
    join_meshes,
    record_contact_peak,
    run_cli,
    sampled_mass_properties,
    shape_config,
    smooth,
    surface_mass,
    transform_points,
)
from field_coverage import FieldAudit, coverage_summary, require_field_coverage

import newton


@wp.func
def nominal_handle(t: float, initial_z: float):
    scrub = (
        0.006 * smooth(t, 2.55, 0.45)
        - 0.012 * smooth(t, 3.15, 0.60)
        + 0.012 * smooth(t, 3.85, 0.60)
        - 0.006 * smooth(t, 4.50, 0.30)
    )
    return wp.vec3(
        0.040 * smooth(t, 1.5, 0.8) - 0.030 * smooth(t, 4.5, 0.6),
        0.025 * smooth(t, 1.5, 0.8) + scrub,
        initial_z + 0.12 * smooth(t, 0.5, 0.9),
    )


@wp.func
def mechanism_position(t: float, initial_z: float, part: int):
    nominal = nominal_handle(t, initial_z)
    if part == 0:
        # A narrow capsule enters each inter-column lane, clearing the tips
        # during crossover rather than dragging them across the whole head.
        return wp.vec3(
            0.043 - 0.006 * smooth(t, 3.28, 0.06) - 0.030 * smooth(t, 4.5, 0.6),
            0.036,
            initial_z
            + 0.1583
            - 0.0125 * smooth(t, 2.3, 0.3)
            + 0.005 * smooth(t, 3.0, 0.28)
            - 0.005 * smooth(t, 3.34, 0.28)
            + 0.0125 * smooth(t, 4.5, 0.6),
        )
    carrier = nominal + wp.vec3(0.0, -0.105, 0.010 + 0.050 * smooth(t, 5.2, 0.5))
    if part == 1:
        return carrier
    opening = 0.010 + 0.012 * (1.0 - smooth(t, 0.10, 0.20) + smooth(t, 5.2, 0.2))
    side = float(2 * part - 5)
    return carrier + wp.vec3(side * opening, 0.0, 0.0)


@wp.kernel
def drive_mechanism(
    clock: wp.array[float],
    dt: float,
    initial_z: float,
    bodies: wp.array[int],
    grasp_joint: int,
    enabled: wp.array[wp.bool],
    pose0: wp.array[wp.transform],
    vel0: wp.array[wp.spatial_vector],
    pose1: wp.array[wp.transform],
    vel1: wp.array[wp.spatial_vector],
):
    part = wp.tid()
    t = clock[0]
    p = mechanism_position(t, initial_z, part)
    nxt = mechanism_position(t + dt, initial_z, part)
    body = bodies[part]
    pose = wp.transform(p, wp.quat_identity())
    velocity = wp.spatial_vector((nxt - p) / dt, wp.vec3(0.0))
    pose0[body] = pose
    pose1[body] = pose
    vel0[body] = velocity
    vel1[body] = velocity
    if part == 0:
        enabled[grasp_joint] = t >= 0.35 and t < 5.2


class Example(CapturedExample):
    """Scrub, unload and release a dynamic handle with a physical 40-mode head."""

    slug = "toothbrush"
    dynamic_assembly = True
    default_iterations = 24
    expected_dynamic_body_count = 2

    def __init__(self, viewer=None, cfg: Config | None = None):
        cfg = replace(cfg or Config(iterations=self.default_iterations), friction=0.20)
        self.begin(viewer, cfg, kind=0)
        handle, grip, ridges, plate, plastic_head, root = brush_handle(split_head=True)
        self.initial_z = -float(handle[0][:, 2].min()) + 0.0003
        mass, com, inertia = sampled_mass_properties(handle[0], surface_mass(*handle, 0.018))
        self.handle_com = com
        self.handle = self.builder.add_body(
            xform=wp.transform(wp.vec3(0, 0, self.initial_z), wp.quat_identity()),
            mass=mass,
            com=wp.vec3(*com),
            inertia=wp.mat33(*inertia.ravel()),
            lock_inertia=True,
            label="dynamic_rigid_handle",
        )
        for name, mesh, color, roughness in [
            ("ivory_handle", handle, (0.86, 0.92, 0.91), 0.28),
            ("teal_grip", grip, (0.025, 0.40, 0.40), 0.55),
            ("grip_ribs", ridges, (0.06, 0.60, 0.54), 0.5),
        ]:
            self.add_visual(name, mesh, color, self.handle, roughness=roughness)
        floor_cfg = shape_config(self.cfg)
        floor_cfg.mu = 0.5
        floor_cfg.ke = 300000.0
        # Cover a full impact substep in contact generation without inflating
        # the collision surface or changing its zero rest margin.
        floor_cfg.gap = 0.008
        self.handle_collision = self.builder.add_shape_mesh(
            self.handle, mesh=newton.Mesh(handle[0], handle[1].ravel()), cfg=floor_cfg, label="handle_contact"
        )
        p, f, basis, self.head_parts = brush_head_rom(plastic_head, plate, root)
        self.head_body = self.add_elastic(
            "physical_elastic_head",
            (p, f),
            basis,
            root + np.array([0, 0, self.initial_z]),
            (0.86, 0.92, 0.91),
            dynamic=True,
        )
        item = self.elastic[0]
        item["root"] = root
        template = self.meshes.pop()
        # Render material regions of the one collision mesh without resampling
        # its modes or creating additional floating frames.
        for part in self.head_parts:
            start, end = part["start"], part["start"] + part["count"]
            self.meshes.append(
                {
                    **template,
                    "name": part["name"],
                    "p": p[start:end],
                    "f": part["mesh"][1],
                    "color": part["color"],
                    "elastic_vertex_start": start,
                    "elastic_vertex_end": end,
                }
            )
        self.root_joints = [
            self.builder.add_joint_fixed(
                self.handle,
                self.head_body,
                parent_xform=wp.transform(wp.vec3(*root), wp.quat_identity()),
                child_xform=wp.transform_identity(),
                label="fixed_neck_to_zero_phi_plastic_head",
            )
        ]
        self.add_sphere_actuator(0.0025, (0.95, 0.94, 0.88), half_length=0.035)
        if not self.cfg.contact_enabled:
            for item in self.elastic:
                self.builder.add_shape_collision_filter_pair(item["shape"], self.tool_shape)
        self.floor_shape = self.builder.add_ground_plane(cfg=floor_cfg)
        self.add_visual(
            "rigid_floor",
            (
                np.array([[-0.30, -0.30, 0], [0.32, -0.30, 0], [0.32, 0.23, 0], [-0.30, 0.23, 0]], dtype=np.float32),
                np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            ),
            (0.36, 0.40, 0.43),
            roughness=0.72,
        )
        self.gripper = []
        for part in range(1, 4):
            body = self.builder.add_body(
                xform=wp.transform(mechanism_position(0.0, self.initial_z, part), wp.quat_identity()),
                is_kinematic=True,
                label=f"visible_gripper_{part}",
            )
            self.gripper.append(body)
            if part == 1:
                mesh = join_meshes(
                    [
                        ellipsoid((0.035, 0.012, 0.006), (0, 0, 0.035)),
                        ellipsoid((0.006, 0.007, 0.030), (0, 0, 0.062)),
                    ]
                )
                color = (0.16, 0.19, 0.22)
            else:
                mesh = join_meshes(
                    [
                        ellipsoid((0.003, 0.013, 0.019), (0, 0, 0.016)),
                        ellipsoid((0.004, 0.013, 0.005)),
                    ]
                )
                color = (0.78, 0.29, 0.12)
            self.add_visual(f"gripper_part_{part}", mesh, color, body, roughness=0.35, metallic=0.3)
        self.grasp_joint = self.builder.add_joint_fixed(
            self.gripper[0],
            self.handle,
            child_xform=wp.transform(wp.vec3(0, -0.105, 0.010), wp.quat_identity()),
            enabled=False,
            label="visible_jaw_ideal_grasp_released_at_5_2s",
        )
        self.mechanism_bodies = wp.array([self.actuator, *self.gripper], dtype=int)
        self.finish()
        self.initial_handle_pose = self.state_0.body_q.numpy()[self.handle].copy()

    def mass_audit(self):
        item = self.elastic[0]
        body = self.head_body
        expected_mass, expected_com, expected_inertia = sampled_mass_properties(item["p"], item["source_sample_mass"])
        actual_mass = float(self.model.body_mass.numpy()[body])
        actual_com = self.model.body_com.numpy()[body]
        actual_inertia = self.model.body_inertia.numpy()[body]
        np.testing.assert_array_equal(actual_inertia, expected_inertia.astype(np.float32))
        np.testing.assert_array_equal(actual_com, expected_com.astype(np.float32))
        assert actual_mass == float(np.float32(expected_mass))
        result = coupled_mass_audit(
            actual_mass,
            actual_com.astype(float),
            actual_inertia.astype(float),
            self.model.elastic_mode_mass.numpy().astype(float),
            self.model.elastic_mode_coupling_linear.numpy().astype(float),
            self.model.elastic_mode_coupling_angular.numpy().astype(float),
        )
        result.update(
            head_mass_kg=actual_mass,
            plastic_plate_mass_kg=0.002,
            handle_mass_kg=float(self.model.body_mass.numpy()[self.handle]),
            head_com_m=actual_com.tolist(),
            physical_principal_inertia_kg_m2=np.linalg.eigvalsh(expected_inertia).tolist(),
            exported_principal_inertia_kg_m2=np.linalg.eigvalsh(actual_inertia.astype(float)).tolist(),
            inertia_regularization_applied=False,
            inertia_matches_samples_exact_float32=True,
            actual_implicit_mass_coupling=self.solver.elastic_implicit_mass_coupling.numpy().tolist(),
        )
        return result

    def solver_options(self):
        return {
            "rigid_joint_linear_ke": 1e5,
            "rigid_joint_angular_ke": 1e3,
            "rigid_joint_linear_k_start": 1e5,
            "rigid_joint_angular_k_start": 1e3,
            "rigid_joint_adaptive_stiffness": False,
        }

    def simulate(self):
        for _ in range(self.cfg.substeps):
            wp.launch(
                drive_mechanism,
                4,
                inputs=[
                    self.clock,
                    self.dt,
                    self.initial_z,
                    self.mechanism_bodies,
                    self.grasp_joint,
                    self.model.joint_enabled,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_q,
                    self.state_1.body_qd,
                ],
            )
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.capacity_audit.after_collide()
            self.field_audit.observe()
            wp.launch(record_contact_peak, 1, inputs=[self.contacts.rigid_contact_count, self.contact_peak])
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.capacity_audit.after_solve()
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_clock, 1, inputs=[self.clock, self.dt])

    def setup_audits(self):
        self.field_audit = FieldAudit(self)

    def update_metrics(self):
        super().update_metrics()
        frame = self.records[-1]
        frame["tuft_contacts_all_substeps"] = self.field_audit.last_frame().tolist()
        frame["tuft_tip_deformation_m"] = np.hypot(
            frame["tip_x_displacement_m"], frame["tip_y_displacement_m"]
        ).tolist()

    def object_metrics(self, frame, displacements):
        poses = self.state_0.body_q.numpy()
        velocity = self.state_0.body_qd.numpy()[self.handle]
        root_error, angle_error = 0.0, 0.0
        min_z = float(transform_points(self.meshes[0]["p"], poses[self.handle])[:, 2].min())
        for item, disp in zip(self.elastic, displacements, strict=True):
            parent = transform_points(item["root"][None], poses[self.handle])[0]
            root_error = max(root_error, float(np.linalg.norm(parent - poses[item["body"], :3])))
            dot = abs(float(np.dot(poses[self.handle, 3:], poses[item["body"], 3:])))
            angle_error = max(angle_error, float(2 * np.arccos(np.clip(dot, 0, 1))))
            surface = transform_points(item["p"] + disp + item["equilibrium_disp"], poses[item["body"]])
            min_z = min(min_z, float(surface[:, 2].min()))
        shape0 = self.contacts.rigid_contact_shape0.numpy()[: frame["contacts"]]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[: frame["contacts"]]
        tool_pair = (shape0 == self.tool_shape) | (shape1 == self.tool_shape)
        elastic_shapes = [item["shape"] for item in self.elastic]
        bristle_pair = np.isin(shape0, elastic_shapes) | np.isin(shape1, elastic_shapes)
        tip_x = [float(displacements[0][part["tip_indices"], 0].mean()) for part in self.head_parts[2:]]
        tip_y = [float(displacements[0][part["tip_indices"], 1].mean()) for part in self.head_parts[2:]]
        plate_count = self.head_parts[2]["start"]
        frame.update(
            handle_position_m=poses[self.handle, :3].tolist(),
            handle_velocity=velocity.tolist(),
            handle_com_m=transform_points(self.handle_com[None], poses[self.handle])[0].tolist(),
            plate_modal_displacement_m=float(np.linalg.norm(displacements[0][:plate_count], axis=1).max()),
            root_anchor_error_m=root_error,
            root_angular_error_rad=angle_error,
            floor_penetration_m=max(0.0, -min_z),
            tool_contacts=int(np.count_nonzero(tool_pair & bristle_pair)),
            other_tool_contacts=int(np.count_nonzero(tool_pair & ~bristle_pair)),
            floor_contacts=int(np.count_nonzero((shape0 == self.floor_shape) | (shape1 == self.floor_shape))),
            grasp_enabled=bool(self.model.joint_enabled.numpy()[self.grasp_joint]),
            tip_x_displacement_m=tip_x,
            tip_y_displacement_m=tip_y,
        )

    def summary(self):
        result = super().summary()
        held = [r for r in self.records if 2.5 < r["time"] < 4.2]
        falling = [r for r in self.records if 5.22 < r["time"] < 5.32]
        landed = [r for r in self.records if r["time"] > 6.8]
        scrubbing = [r for r in self.records if 2.5 < r["time"] < 4.5]
        tangential = np.asarray([r["tip_y_displacement_m"] for r in scrubbing])
        result.update(
            dynamic_assembly=True,
            single_physical_head=True,
            head_body_index=self.head_body,
            mass_audit=self.mass_audit(),
            floor_contact_generation_gap_m=0.008,
            floor_contact_rest_margin_m=0.0,
            floor_contact_stiffness_n_m=300000.0,
            floor_friction=0.5,
            handle_body_index=self.handle,
            root_joints=self.root_joints,
            grasp_joint_index=self.grasp_joint,
            root_parent_local_m=[item["root"].tolist() for item in self.elastic],
            max_root_anchor_error_m=max(r["root_anchor_error_m"] for r in self.records),
            max_root_angular_error_rad=max(r["root_angular_error_rad"] for r in self.records),
            max_floor_penetration_m=max(r["floor_penetration_m"] for r in self.records),
            contact_window_deformation_m=max((r["max_deformation_m"] for r in held), default=0),
            tool_contact_frames=sum(r["tool_contacts"] > 0 for r in held),
            handle_lift_m=max(r["handle_position_m"][2] for r in self.records) - self.initial_handle_pose[2].item(),
            freefall_vertical_acceleration_m_s2=float(np.mean(np.diff([r["handle_velocity"][2] for r in falling])) * 60)
            if len(falling) > 1
            else None,
            freefall_drop_m=falling[0]["handle_com_m"][2] - falling[-1]["handle_com_m"][2] if falling else 0,
            final_handle_speed_m_s=float(np.linalg.norm(self.records[-1]["handle_velocity"][:3])),
            final_handle_angular_speed_rad_s=float(np.linalg.norm(self.records[-1]["handle_velocity"][3:])),
            landed_position_range_m=float(np.ptp([r["handle_com_m"] for r in landed], axis=0).max())
            if landed
            else None,
            displacement_reference="Modal change from floor-settled initial state. Driven versus matched no-tool-contact trajectory isolates tool bending from gravity/inertia/landing loads.",
            scrub_tuft_reversals=int(
                np.count_nonzero((tangential.max(axis=0) > 0.00025) & (tangential.min(axis=0) < -0.00025))
            )
            if len(tangential)
            else 0,
            scrub_tangential_peak_to_peak_m=float(np.ptp(tangential, axis=0).max()) if len(tangential) else 0.0,
            scrub_contact_frames=sum(r["tool_contacts"] > 0 for r in scrubbing),
            stationary_surface=False,
        )
        result["field_coverage"] = coverage_summary(self.records)
        result["max_tool_overlap_m"] = result["max_penetration_m"]
        result["max_penetration_m"] = max(
            result["max_tool_overlap_m"] if self.cfg.contact_enabled else 0.0,
            result["max_floor_penetration_m"],
        )
        return result

    def test_final(self):
        s = {k: v for k, v in self.summary().items() if k != "records"}
        assert len(self.records) >= 420, s
        assert self.cfg.substeps == 4 and self.graph is not None, s
        assert not s["contact_buffer_overflow"], s
        self.capacity_audit.check((60 + len(self.records)) * self.cfg.substeps)
        assert s["max_modal_solve_residual_ratio"] < 0.05, s
        assert self.model.elastic_body_count == 1 and len(self.elastic) == 1, s
        assert (
            np.count_nonzero(self.model.body_flags.numpy() == int(newton.BodyFlags.DYNAMIC))
            == self.expected_dynamic_body_count
        ), s
        assert s["mass_audit"]["dimension"] == 46, s
        assert s["mass_audit"]["runtime_spd_passed"], s
        assert s["mass_audit"]["actual_implicit_mass_coupling"] == [True], s
        assert s["mass_audit"]["normalized_condition_number"] < 10, s
        assert max(r["plate_modal_displacement_m"] for r in self.records) == 0, s
        if self.cfg.contact_enabled:
            assert s["contact_window_deformation_m"] > 0.001, s
        assert s["max_root_anchor_error_m"] < 0.0005, s
        assert s["max_root_angular_error_rad"] < 0.02, s
        assert s["max_penetration_m"] < 0.0008, s
        assert s["max_deformation_m"] < 0.006, s
        assert 0.10 < s["handle_lift_m"] < 0.14, s
        assert -10.2 < s["freefall_vertical_acceleration_m_s2"] < -9.4, s
        assert s["freefall_drop_m"] > 0.04, s
        assert s["final_handle_speed_m_s"] < 0.02, s
        assert s["final_handle_angular_speed_rad_s"] < 0.5, s
        assert s["landed_position_range_m"] < 0.0005, s
        assert s["release_max_deformation_m"] < 0.0002, s
        assert all(not r["grasp_enabled"] for r in self.records if r["time"] > 5.22), s
        if self.cfg.contact_enabled:
            assert s["contact_window_deformation_m"] > 0.001, s
            assert s["tool_contact_frames"] > 35, s
            tips = np.asarray([r["tip_x_displacement_m"] for r in self.records if 2.5 < r["time"] < 4.2])
            assert np.count_nonzero(np.max(np.abs(tips), axis=0) > 0.0005) >= 5, s
            assert s["scrub_tuft_reversals"] >= 5, s
            assert s["scrub_tangential_peak_to_peak_m"] > 0.001, s
            assert s["scrub_contact_frames"] > 100, s
            require_field_coverage(s["field_coverage"])
        else:
            assert s["contact_window_deformation_m"] < 0.0001, s
            assert s["tool_contact_frames"] == 0, s

    def validate_negative_control(self, negative):
        assert negative.summary()["contact_window_deformation_m"] < 0.1 * self.summary()["contact_window_deformation_m"]

    def scene_metadata(self):
        return {
            "title": "Toothbrush: contact-driven bristle scrubbing",
            "camera": {
                "position": [0.000, 0.036, 0.215],
                "target": [0.042, 0.036, 0.140],
                "fov": 32,
                "track_meshes": ["teal_tuft_plate"],
                "reference_frame": 189,
            },
            "floor_z": -0.0001,
            "dynamic_assembly": True,
            "single_physical_head": True,
            "description": "The lifted brush scrubs back and forth while the enamel-colored capsule sweeps smoothly across all four transverse columns of the bristle field. Contact bends successive tufts; withdrawal lets them recover before the handle is released.",
            "model": "One physical elastic head with 20 independent two-mode tuft fields, attached by one fixed neck joint to an 18 g dynamic handle. A 2 g zero-Phi plastic head plate supplies physical floating-frame inertia; 140 visible nylon strands retain the original modal mass, stiffness and damping.",
            "driven": "The visible jaw carriage and sweeping capsule are kinematic. The grasp joint enables at 0.35 s and disables at 5.2 s; both object frames and all 40 modal coordinates are solved by Newton VBD with implicit mass coupling enabled. The close-up camera tracks the saved head position without changing geometry.",
            "limitations": [
                "Visible jaws indicate an ideal switchable fixed-joint grasp, not a frictional grasp simulation; jaw geometry has no collision.",
                "Grouped strands share one beam field per tuft; individual-filament splay and bristle-bristle contact are not resolved.",
                "Linear small-deflection beam ROM with root strain clamped in each dynamic frame; no inextensibility or nonlinear filament model.",
                "Assumed nylon density 1150 kg/m^3 and modulus 1.3 GPa. Plastic mass is explicitly assumed 18 g handle plus 2 g head plate; a surface-area-lumped mass distribution supplies consistent mass, COM and inertia, not a calibrated solid-volume density model.",
                "Head plate and tuft roots have zero modal displacement; plate bending is omitted. No inertia regularization or runtime cutoff modification is used.",
                "Rounded tool is an analytic capsule proxy, not anatomical tooth geometry. No cleaning, toothpaste or fluid model.",
                "Separate handle and joined head/tuft collision meshes; decorative handle grip ribs do not add collision geometry. Floor material friction 0.5, bristle and tool material friction 0.2. Floor contact-generation gap 8 mm covers impact travel per substep; rest margin remains zero, with no geometry inflation. Tool gap is 0.15 mm.",
            ],
        }


if __name__ == "__main__":
    run_cli(Example)
