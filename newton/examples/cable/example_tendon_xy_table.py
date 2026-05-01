# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Tendon XY Table
#
# Cross-base cable-driven XY table topology inspired by the MathWorks
# sm_pulleys_xytable_cross example.  This reproduces its Pulley Angles
# mode: a single tendon routes from one side of the table through seven
# pulleys to the other side, the platform is constrained to X/Y sliders,
# and only the two base pulleys are driven.
#
# Command: python -m newton.examples tendon_xy_table
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.examples.cable.cable import (
    assert_tendon_total_length,
    get_tendon_cable_lines,
)

DRIVE_TIMES = np.array(
    [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
        4.0,
        4.25,
        4.5,
        4.75,
        5.0,
        5.25,
        5.5,
        5.75,
        6.0,
        6.25,
        6.5,
        6.75,
        7.0,
        7.25,
        7.5,
        7.75,
        8.0,
        8.25,
        8.5,
        8.75,
        9.0,
        9.25,
        9.5,
        9.75,
        10.0,
        10.25,
        10.5,
        10.75,
        11.0,
        11.25,
        11.5,
        11.75,
        12.0,
        12.25,
        12.5,
        12.75,
        13.0,
    ],
    dtype=np.float32,
)

# Radian pulley-angle trajectories sampled from the MathWorks
# sm_pulleys_xytable_cross_driveqs.mat reference data.
P2_TARGET = np.array(
    [
        0.000097,
        0.910122,
        1.556299,
        1.608036,
        1.045439,
        0.083773,
        -0.909973,
        -1.556147,
        -1.607844,
        -1.045251,
        -0.083577,
        -0.909924,
        -1.556097,
        -1.607833,
        -1.045246,
        -0.083579,
        0.910162,
        1.556332,
        1.60803,
        1.045447,
        0.083771,
        0.000097,
        0.000097,
        0.000097,
        0.000097,
        0.000097,
        0.000097,
        0.000097,
        0.000097,
        0.000097,
        0.000097,
        -0.638078,
        -0.482919,
        0.493326,
        1.917883,
        3.246474,
        3.971646,
        3.816451,
        2.840144,
        1.415642,
        0.086918,
        -1.181705,
        -2.629063,
        -3.708756,
        -4.008412,
        -3.413551,
        -2.151414,
        -0.70407,
        0.375623,
        0.675263,
        0.080614,
        0.000068,
        0.000095,
    ],
    dtype=np.float32,
)

P6_TARGET = np.array(
    [
        0.000078,
        0.910103,
        1.55628,
        1.608017,
        1.045419,
        0.083753,
        -0.909992,
        -1.556167,
        -1.607864,
        -1.04527,
        -0.083596,
        0.910099,
        1.556272,
        1.608008,
        1.045421,
        0.083754,
        -0.909987,
        -1.556157,
        -1.607855,
        -1.045272,
        -0.083596,
        1.820041,
        3.112294,
        3.215737,
        2.090842,
        0.167397,
        -1.820092,
        -3.112264,
        -3.215705,
        -2.090595,
        -0.167278,
        1.181904,
        2.629273,
        3.709114,
        4.008552,
        3.413783,
        2.151595,
        0.704216,
        -0.375523,
        -0.675088,
        -0.080442,
        0.638361,
        0.483019,
        -0.493256,
        -1.917711,
        -3.246249,
        -3.971442,
        -3.816208,
        -2.839924,
        -1.415467,
        -0.086746,
        0.000108,
        0.00008,
    ],
    dtype=np.float32,
)

DRIVE_HOLD_TIME = 8.0


def _interp_target(t: float, values: np.ndarray) -> tuple[float, float]:
    t = float(np.clip(t, DRIVE_TIMES[0], DRIVE_TIMES[-1]))
    value = float(np.interp(t, DRIVE_TIMES, values))

    idx = int(np.searchsorted(DRIVE_TIMES, t, side="right") - 1)
    idx = max(0, min(idx, len(DRIVE_TIMES) - 2))
    dt = float(DRIVE_TIMES[idx + 1] - DRIVE_TIMES[idx])
    velocity = float((values[idx + 1] - values[idx]) / dt) if dt > 0.0 else 0.0
    return value, velocity


def _desired_table_xy(t: float) -> tuple[float, float]:
    period = 2.5
    amp = 0.05
    local = t % period
    s = np.sin(2.0 * np.pi * local / period)

    if t < 2.5:
        return amp * s, 0.0
    if t < 5.0:
        return 0.0, amp * s
    if t < 7.5:
        return amp * s, amp * s
    if t < 10.0:
        return amp * (np.sin(2.0 * np.pi * local / period + 1.5 * np.pi) + 1.0), amp * s
    if t < 12.5:
        return amp * (np.sin(2.0 * np.pi * local / period + 0.5 * np.pi) - 1.0), amp * s
    return 0.0, 0.0


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=0.0)

        self.rad_drive = 0.030
        self.rad_axis = 0.030
        self.rad_ctr = 0.015
        self.drive_target_scale = 1.0

        visual_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            collision_group=0,
        )
        table_cfg = newton.ModelBuilder.ShapeConfig(
            density=2700.0,
            has_shape_collision=False,
            collision_group=0,
        )
        pulley_cfg = newton.ModelBuilder.ShapeConfig(
            density=7000.0,
            has_shape_collision=False,
            collision_group=0,
        )

        base = builder.add_body(xform=wp.transform(), mass=0.0, is_kinematic=True, label="base")
        builder.add_shape_box(
            base,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.012)),
            hx=0.22,
            hy=0.055,
            hz=0.005,
            cfg=visual_cfg,
            color=(0.46, 0.51, 0.58),
        )
        builder.add_shape_box(
            base,
            xform=wp.transform(p=wp.vec3(0.0, 0.195, -0.014)),
            hx=0.20,
            hy=0.010,
            hz=0.004,
            cfg=visual_cfg,
            color=(0.36, 0.39, 0.44),
        )
        builder.add_shape_box(
            base,
            xform=wp.transform(p=wp.vec3(0.0, -0.225, -0.014)),
            hx=0.08,
            hy=0.010,
            hz=0.004,
            cfg=visual_cfg,
            color=(0.36, 0.39, 0.44),
        )

        slider = builder.add_link(xform=wp.transform(), label="x_slider")
        builder.add_shape_box(
            slider,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.006)),
            hx=0.075,
            hy=0.050,
            hz=0.005,
            cfg=table_cfg,
            color=(0.42, 0.64, 0.46),
        )

        table = builder.add_link(xform=wp.transform(), label="y_table")
        builder.add_shape_box(
            table,
            xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.003)),
            hx=0.0125,
            hy=0.225,
            hz=0.005,
            cfg=table_cfg,
            color=(0.77, 0.72, 0.60),
        )

        for px, py in [(-0.045, -0.045), (-0.045, 0.045), (0.045, 0.045), (0.045, -0.045)]:
            builder.add_shape_cylinder(
                slider,
                xform=wp.transform(p=wp.vec3(px, py, 0.0)),
                radius=self.rad_ctr,
                half_height=0.005,
                cfg=pulley_cfg,
                color=(0.34, 0.52, 0.38),
            )

        builder.add_shape_cylinder(
            table,
            xform=wp.transform(p=wp.vec3(0.0, 0.195, 0.0)),
            radius=self.rad_axis,
            half_height=0.005,
            cfg=pulley_cfg,
            color=(0.62, 0.57, 0.47),
        )
        for px, py in [(-0.030, -0.225), (0.030, -0.225)]:
            builder.add_shape_sphere(
                table,
                xform=wp.transform(p=wp.vec3(px, py, 0.0)),
                radius=0.007,
                cfg=visual_cfg,
                color=(0.12, 0.12, 0.12),
            )

        p2_pos = wp.vec3(-0.190, 0.0, 0.0)
        p6_pos = wp.vec3(0.190, 0.0, 0.0)
        p2 = builder.add_link(xform=wp.transform(p=p2_pos), label="drive_p2")
        p6 = builder.add_link(xform=wp.transform(p=p6_pos), label="drive_p6")
        for body in [p2, p6]:
            builder.add_shape_cylinder(
                body,
                radius=self.rad_drive,
                half_height=0.006,
                cfg=pulley_cfg,
                color=(0.22, 0.28, 0.35),
            )

        j_slider = builder.add_joint_prismatic(
            parent=-1,
            child=slider,
            axis=Axis.X,
            limit_lower=-0.075,
            limit_upper=0.075,
            limit_ke=1.0e4,
            limit_kd=100.0,
            friction=0.0,
            label="slider_x",
        )
        j_table = builder.add_joint_prismatic(
            parent=slider,
            child=table,
            axis=Axis.Y,
            limit_lower=-0.075,
            limit_upper=0.075,
            limit_ke=1.0e4,
            limit_kd=100.0,
            friction=0.0,
            label="table_y",
        )

        drive_args = {
            "axis": Axis.Z,
            "target_ke": 12.0,
            "target_kd": 1.2,
            "armature": 0.0,
            "effort_limit": 50.0,
            "velocity_limit": 200.0,
            "friction": 0.0,
            "actuator_mode": newton.JointTargetMode.POSITION_VELOCITY,
        }
        j_p2 = builder.add_joint_revolute(
            parent=-1,
            child=p2,
            parent_xform=wp.transform(p=p2_pos),
            child_xform=wp.transform(),
            label="drive_p2_z",
            **drive_args,
        )
        j_p6 = builder.add_joint_revolute(
            parent=-1,
            child=p6,
            parent_xform=wp.transform(p=p6_pos),
            child_xform=wp.transform(),
            label="drive_p6_z",
            **drive_args,
        )

        builder.add_articulation([j_slider, j_table])
        builder.add_articulation([j_p2])
        builder.add_articulation([j_p6])

        axis = (0.0, 0.0, 1.0)
        compliance = 3.0e-5
        damping = 0.02
        drive_mu = 10.0

        builder.add_tendon()
        for body, link_type, radius, orientation, mu, offset in [
            (table, TendonLinkType.ATTACHMENT, 0.0, 1, 0.0, (-0.030, -0.225, 0.0)),
            (slider, TendonLinkType.ROLLING, self.rad_ctr, 1, 0.0, (-0.045, -0.045, 0.0)),
            (p2, TendonLinkType.ROLLING, self.rad_drive, -1, drive_mu, (0.0, 0.0, 0.0)),
            (slider, TendonLinkType.ROLLING, self.rad_ctr, 1, 0.0, (-0.045, 0.045, 0.0)),
            (table, TendonLinkType.ROLLING, self.rad_axis, -1, 0.0, (0.0, 0.195, 0.0)),
            (slider, TendonLinkType.ROLLING, self.rad_ctr, 1, 0.0, (0.045, 0.045, 0.0)),
            (p6, TendonLinkType.ROLLING, self.rad_drive, -1, drive_mu, (0.0, 0.0, 0.0)),
            (slider, TendonLinkType.ROLLING, self.rad_ctr, 1, 0.0, (0.045, -0.045, 0.0)),
            (table, TendonLinkType.ATTACHMENT, 0.0, 1, 0.0, (0.030, -0.225, 0.0)),
        ]:
            builder.add_tendon_link(
                body=body,
                link_type=int(link_type),
                radius=radius,
                orientation=orientation,
                mu=mu,
                offset=offset,
                axis=axis,
                compliance=compliance,
                damping=damping,
                rest_length=-1.0,
            )

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=12,
            joint_linear_relaxation=0.8,
        )
        self._apply_cable_pretension(0.99995)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        dof_starts = self.model.joint_qd_start.numpy()
        self.slider_dof_start = int(dof_starts[j_slider])
        self.table_dof_start = int(dof_starts[j_table])
        self.p2_dof_start = int(dof_starts[j_p2])
        self.p6_dof_start = int(dof_starts[j_p6])
        self.slider_idx = slider
        self.table_idx = table
        self._table_xy_history = []
        self._slider_x_history = []

        self.reference_path = self._make_reference_path()

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(0.0, -0.72, 0.72), pitch=-58.0, yaw=90.0)
            if hasattr(self.viewer, "renderer"):
                self.viewer.renderer.show_wireframe_overlay = True

    def _make_reference_path(self):
        ts = np.linspace(0.0, 12.5, 180)
        pts = np.array([(*_desired_table_xy(float(t)), 0.002) for t in ts], dtype=np.float32)
        starts = wp.array(pts[:-1], dtype=wp.vec3)
        ends = wp.array(pts[1:], dtype=wp.vec3)
        return starts, ends

    def _apply_cable_pretension(self, scale: float):
        tendon_total = self.solver.tendon_total_cable.numpy() * scale
        segment_rest = self.solver.tendon_seg_rest_length.numpy() * scale
        self.solver.tendon_total_cable = wp.array(
            tendon_total,
            dtype=wp.float32,
            device=self.model.device,
        )
        self.solver.tendon_seg_rest_length = wp.array(
            segment_rest,
            dtype=wp.float32,
            device=self.model.device,
        )

    def _set_drive_targets(self, t: float):
        active_t = min(t, DRIVE_HOLD_TIME)
        p2, p2d = _interp_target(active_t, P2_TARGET)
        p6, p6d = _interp_target(active_t, P6_TARGET)
        p2 *= self.drive_target_scale
        p2d *= self.drive_target_scale
        p6 *= self.drive_target_scale
        p6d *= self.drive_target_scale
        if t >= DRIVE_HOLD_TIME:
            p2d = 0.0
            p6d = 0.0

        self.control.joint_target_pos[self.p2_dof_start : self.p2_dof_start + 1].fill_(p2)
        self.control.joint_target_vel[self.p2_dof_start : self.p2_dof_start + 1].fill_(p2d)
        self.control.joint_target_pos[self.p6_dof_start : self.p6_dof_start + 1].fill_(p6)
        self.control.joint_target_vel[self.p6_dof_start : self.p6_dof_start + 1].fill_(p6d)

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            self._set_drive_targets(t)

            self.state_0.clear_forces()
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self._record_motion_sample()

    def _record_motion_sample(self):
        body_q = self.state_0.body_q.numpy()
        self._table_xy_history.append(np.array(body_q[self.table_idx][:2], dtype=np.float64))
        self._slider_x_history.append(float(body_q[self.slider_idx][0]))

    def test_post_step(self):
        assert_tendon_total_length(self)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in body positions"

    def test_final(self):
        assert_tendon_total_length(self)
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite values in body positions"
        if not self._table_xy_history:
            self._record_motion_sample()

        table_xy = np.array(self._table_xy_history)
        slider_x = np.array(self._slider_x_history)
        assert np.isfinite(table_xy).all() and np.isfinite(slider_x).all(), "Non-finite values in table trajectory"

        joint_ke = self.model.joint_target_ke.numpy()
        assert joint_ke[self.slider_dof_start] == 0.0 and joint_ke[self.table_dof_start] == 0.0, (
            "XY axes must stay passive; motion should come from cable friction, not direct joint targets"
        )
        assert joint_ke[self.p2_dof_start] > 0.0 and joint_ke[self.p6_dof_start] > 0.0, (
            "Drive pulley joints should be the only actuated joints"
        )
        link_mu = self.model.tendon_link_mu.numpy()
        assert link_mu[2] > 0.0 and link_mu[6] > 0.0, "Driven pulleys must use frictional rolling contact"

        max_abs_table = float(np.max(np.abs(table_xy)))
        max_abs_slider = float(np.max(np.abs(slider_x)))
        assert max_abs_table < 0.09 and max_abs_slider < 0.09, (
            f"Table exceeded expected travel: max_abs_table={max_abs_table:.5f}, max_abs_slider={max_abs_slider:.5f}"
        )

        total_motion = float(np.max(np.linalg.norm(table_xy, axis=1)))
        assert total_motion > 0.006, f"Driven table should move from origin: max_motion={total_motion:.5f}"

        if len(table_xy) >= 120:
            max_step = float(np.max(np.linalg.norm(np.diff(table_xy, axis=0), axis=1)))
            assert max_step < 0.02, f"Table trajectory jumped unexpectedly: max_frame_delta={max_step:.5f}"

        if len(table_xy) >= 300:
            max_y = float(np.max(table_xy[:, 1]))
            assert max_y > 0.012, f"Expected early upward travel from P2/P6 drive schedule, max_y={max_y:.5f}"

        if len(table_xy) >= 420:
            max_x = float(np.max(table_xy[:, 0]))
            min_x = float(np.min(table_xy[:, 0]))
            assert max_x > 0.025 and min_x < -0.025, (
                f"Expected bidirectional X travel before the midpoint: min_x={min_x:.5f}, max_x={max_x:.5f}"
            )

        if len(table_xy) >= 720:
            tail_step = float(np.max(np.linalg.norm(np.diff(table_xy[-120:], axis=0), axis=1)))
            assert tail_step < 0.01, f"Held pulley targets should settle smoothly: tail_step={tail_step:.5f}"

    def render(self):
        if self.viewer is not None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
            self.viewer.log_lines("cable", starts, ends, colors=(0.95, 0.55, 0.12), width=0.004)
            ref_starts, ref_ends = self.reference_path
            self.viewer.log_lines("reference_path", ref_starts, ref_ends, colors=(0.25, 0.55, 1.0), width=0.0015)
            self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
