# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable-driven four-bar linkage (XPBD).

A standing four-bar linkage (two ground pivots, two uprights, a top coupler bar) is driven by
a massless cable. A **force-controlled** linear actuator (a prismatic slider, commanded by a
ramped force) tensions a cable that runs from the coupler, through a **frictional pinhole
groove**, to the slider. The cable lifts the linkage against gravity until the driven ground
joint reaches its joint limit.

This exercises the routed-cable feature together with a closed kinematic loop, a joint limit,
pinhole (groove) friction, and the XPBD joint-reaction probe:

- The open chain (left upright -> coupler -> right upright) is an articulation (a tree); the
  fourth revolute (right upright back to its world pivot) closes the loop and is added
  *outside* the articulation, since XPBD solves every joint as an independent constraint.
- The driven ground joint carries a hard positional **joint limit**.
- The groove is a ``PINHOLE`` cable link with friction ``mu`` (capstan slip through the bend);
  per-segment cable tension is read as ``(d - rest) / compliance``.
- **Reaction-dependent joint friction** is applied at an interior linkage joint: a Coulomb-like
  friction torque whose magnitude scales with that joint's reaction load, read from the solver's
  :attr:`~newton.solvers.SolverXPBD.joint_reaction_f` probe and applied via :attr:`Control.joint_f`.
"""

import numpy as np
import warp as wp

import newton
from newton import Axis, TendonLinkType
from newton.examples.cable.cable import get_tendon_cable_lines

# Standing four-bar geometry (planar X-Z plane; revolutes spin about +Y).
GROUND_A = np.array([0.0, 0.0, 0.0])  # driven ground pivot (left)
GROUND_D = np.array([0.5, 0.0, 0.0])  # passive ground pivot (right)
UPRIGHT_LEN = 0.40
# Diagonal parallelogram (not near-vertical): the coupler then has vertical travel, so a vertical
# cable pull has good leverage to lift it.
START_ANGLE = np.deg2rad(50.0)  # upright angle from +X; parallelogram four-bar
B_TIP = GROUND_A + UPRIGHT_LEN * np.array([np.cos(START_ANGLE), 0.0, np.sin(START_ANGLE)])
C_TIP = GROUND_D + UPRIGHT_LEN * np.array([np.cos(START_ANGLE), 0.0, np.sin(START_ANGLE)])
COUPLER_MID = 0.5 * (B_TIP + C_TIP)

# Cable route, matching the reference sketch: coupler -> frictional pinhole groove -> fixed rolling
# redirect pulley -> force-controlled actuator. Anchored relative to the coupler: up to the groove,
# across to the pulley, then down to the slider -- ~90 deg bends at both the groove and the pulley.
GROOVE_CENTER = COUPLER_MID + np.array([0.0, 0.0, 0.40])  # arc center, above the coupler (cable goes up)
GROOVE_MU = 0.2
GROOVE_PINHOLES = 5  # the groove is modeled as a row of pinholes along an arc (capstan friction accumulates)
GROOVE_RADIUS = 0.06
PULLEY_CENTER = GROOVE_CENTER + np.array([0.60, 0.0, 0.0])  # across to the right at groove height
PULLEY_RADIUS = 0.06
PULLEY_MU = 0.2
SLIDER_START = PULLEY_CENTER + np.array([0.0, 0.0, -0.55])  # below the pulley, prismatic Z (cable goes down)
CABLE_COMPLIANCE = 1.0e-4


def _groove_pinhole_points():
    """Points along the groove arc (in order from the coupler side to the pulley side).

    The cable comes up from the coupler and leaves to the right toward the pulley, so the arc
    turns it ~90 deg; threading several pinholes along it accumulates the capstan friction.
    """
    a0, a1 = np.deg2rad(200.0), np.deg2rad(70.0)
    return [
        GROOVE_CENTER + GROOVE_RADIUS * np.array([np.cos(a), 0.0, np.sin(a)])
        for a in np.linspace(a0, a1, GROOVE_PINHOLES)
    ]


# Joint limit on the driven ground joint: gravity sags the linkage to LIMIT_HI; the actuator
# lifts it (angle goes negative) until it reaches LIMIT_LO.
LIMIT_LO = np.deg2rad(-10.0)
LIMIT_HI = np.deg2rad(30.0)

# Reaction-dependent friction at the crank->coupler joint: friction torque = JOINT_MU * |reaction|.
JOINT_MU = 0.15

ACTUATOR_PEAK = 30.0  # N
ACTUATOR_RATE = 10.0  # N/s


def build_cable_fourbar():
    """Assemble the standing four-bar, the pinhole groove, the slider actuator, and the cable."""
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

    def add_bar(p0, p1, mass=0.20):
        """A rigid bar between world pivots p0 and p1: body at the midpoint, drawn as a capsule.

        Inertia is set explicitly to a transverse rod (I = m*L^2/12). The visual capsule has zero
        density, so otherwise the body inertia would floor near zero and the linkage would explode.
        """
        center = 0.5 * (p0 + p1)
        length = float(np.linalg.norm(p1 - p0))
        moment = mass * length * length / 12.0
        body = builder.add_link(
            xform=wp.transform(p=wp.vec3(*center), q=wp.quat_identity()),
            mass=mass,
            inertia=wp.mat33(moment, 0.0, 0.0, 0.0, moment, 0.0, 0.0, 0.0, moment),
            lock_inertia=True,
        )
        direction = wp.vec3(*((p1 - p0) / length))
        builder.add_shape_capsule(
            body,
            xform=wp.transform(p=wp.vec3(), q=wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), direction)),
            radius=0.014,
            half_height=0.5 * length - 0.014,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
        )
        return body, center

    left, left_c = add_bar(GROUND_A, B_TIP)
    coupler, coupler_c = add_bar(B_TIP, C_TIP)
    right, right_c = add_bar(GROUND_D, C_TIP)

    def pin(parent, parent_c, child, child_c, pivot, limit_lower=None, limit_upper=None):
        """Revolute joint about +Y connecting two bodies at a shared world pivot.

        Each anchor is the pivot expressed in that body's local frame. The bodies start unrotated,
        so that is ``pivot - center`` (or the pivot itself for the world, -1).
        """
        parent_anchor = pivot if parent == -1 else pivot - parent_c
        return builder.add_joint_revolute(
            parent=parent,
            child=child,
            axis=Axis.Y,
            parent_xform=wp.transform(p=wp.vec3(*parent_anchor), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(*(pivot - child_c)), q=wp.quat_identity()),
            limit_lower=limit_lower,
            limit_upper=limit_upper,
        )

    # The open chain world -> left -> coupler -> right forms the articulation (a tree); the
    # driven ground joint jA carries the joint limit.
    jA = pin(-1, None, left, left_c, GROUND_A, limit_lower=LIMIT_LO, limit_upper=LIMIT_HI)
    jB = pin(left, left_c, coupler, coupler_c, B_TIP)
    jC = pin(coupler, coupler_c, right, right_c, C_TIP)
    builder.add_articulation([jA, jB, jC])
    # The fourth revolute closes the loop (right back to the world), added OUTSIDE the articulation.
    pin(-1, None, right, right_c, GROUND_D)

    # Markers, groove, and slider share an isolated collision group so they never collide with
    # the moving bars.
    decor = newton.ModelBuilder.ShapeConfig(density=0.0, collision_group=7)

    ground = builder.add_body(xform=wp.transform(), mass=0.0, is_kinematic=True)
    builder.add_shape_sphere(ground, xform=wp.transform(p=wp.vec3(*GROUND_A)), radius=0.02, cfg=decor)
    builder.add_shape_sphere(ground, xform=wp.transform(p=wp.vec3(*GROUND_D)), radius=0.02, cfg=decor)

    # Frictional groove: a row of pinholes along an arc (each a small eyelet the cable threads).
    groove = builder.add_body(xform=wp.transform(), mass=0.0, is_kinematic=True)
    groove_points = _groove_pinhole_points()
    for p in groove_points:
        builder.add_shape_sphere(groove, xform=wp.transform(p=wp.vec3(*p)), radius=0.012, cfg=decor)

    # Fixed rolling redirect pulley; cylinder axis is +Y so the cable wraps the rim in the X-Z plane.
    pulley = builder.add_body(xform=wp.transform(p=wp.vec3(*PULLEY_CENTER)), mass=0.0, is_kinematic=True)
    builder.add_shape_cylinder(
        pulley,
        xform=wp.transform(q=wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 1.0, 0.0))),
        radius=PULLEY_RADIUS,
        half_height=0.01,
        cfg=decor,
    )

    # Force-controlled linear actuator: a slider on a prismatic X joint, commanded by joint_f.
    slider = builder.add_link(xform=wp.transform(p=wp.vec3(*SLIDER_START)), mass=0.1)
    builder.add_shape_sphere(slider, radius=0.025, cfg=decor)
    j_slider = builder.add_joint_prismatic(
        parent=-1,
        child=slider,
        axis=Axis.Z,
        parent_xform=wp.transform(p=wp.vec3(*SLIDER_START)),
        child_xform=wp.transform(),
    )
    builder.add_articulation([j_slider])

    # Cable: coupler -> frictional pinhole groove -> rolling redirect pulley -> slider.
    # Both the groove (pinhole) and the pulley (rolling) apply capstan friction; cable plane normal +Y.
    cable_plane = (0.0, 1.0, 0.0)
    builder.add_tendon()
    builder.add_tendon_link(
        body=coupler, link_type=int(TendonLinkType.ATTACHMENT), offset=(0.0, 0.0, 0.0), axis=cable_plane
    )
    for p in groove_points:
        builder.add_tendon_link(
            body=groove,
            link_type=int(TendonLinkType.PINHOLE),
            mu=GROOVE_MU,
            offset=(float(p[0]), float(p[1]), float(p[2])),
            axis=cable_plane,
            compliance=CABLE_COMPLIANCE,
            damping=0.0,
            rest_length=-1.0,
        )
    builder.add_tendon_link(
        body=pulley,
        link_type=int(TendonLinkType.ROLLING),
        radius=PULLEY_RADIUS,
        orientation=1,
        mu=PULLEY_MU,
        offset=(0.0, 0.0, 0.0),
        axis=cable_plane,
        compliance=CABLE_COMPLIANCE,
        damping=0.0,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=slider,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=cable_plane,
        compliance=CABLE_COMPLIANCE,
        damping=0.0,
        rest_length=-1.0,
    )

    model = builder.finalize()
    bar_centers = {left: left_c, coupler: coupler_c, right: right_c}
    loop_pivots = {"B": (left, coupler, B_TIP), "C": (coupler, right, C_TIP), "D": (right, None, GROUND_D)}
    return model, left, coupler, right, slider, jA, jB, j_slider, loop_pivots, bar_centers


def _world_point(q, bidx, local):
    tf = wp.transform(wp.vec3(*q[bidx][:3]), wp.quat(*q[bidx][3:]))
    return np.array(wp.transform_point(tf, wp.vec3(*local)))


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        (
            self.model,
            self.left,
            self.coupler,
            self.right,
            self.slider,
            self.jA,
            self.jB,
            self.j_slider,
            self._pivots,
            self._centers,
        ) = build_cable_fourbar()
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=6, joint_linear_relaxation=0.6)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        qd_start = self.model.joint_qd_start.numpy()
        self.slider_dof = int(qd_start[self.j_slider])
        self.jB_dof = int(qd_start[self.jB])
        self._joint_f = np.zeros(self.model.joint_dof_count, dtype=np.float32)

        self.left_x0 = float(self.state_0.body_q.numpy()[self.left][0])
        self.max_tension = 0.0
        self.max_left_travel = 0.0
        self.max_pivot_gap = 0.0
        self.max_friction_torque = 0.0

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.5, -2.8, 0.55), pitch=-3.0, yaw=90.0)
        self.capture()

    def capture(self):
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _segment_tensions(self):
        # Geometric cable tension per segment [N]: max(stretch, 0) / compliance.
        att_l = self.solver.tendon_seg_attachment_l.numpy()
        att_r = self.solver.tendon_seg_attachment_r.numpy()
        rest = self.solver.tendon_seg_rest_length.numpy()
        compliance = self.model.tendon_seg_compliance.numpy()
        length = np.linalg.norm(att_r - att_l, axis=1)
        return np.maximum(length - rest, 0.0) / np.maximum(compliance, 1e-8)

    def _actuator_force(self):
        # Triangle ramp 0 -> PEAK -> 0; -Z pulls the slider down (away from the pulley), tensioning
        # the cable and lifting the linkage against gravity.
        t = self.sim_time
        up = ACTUATOR_PEAK / ACTUATOR_RATE
        return -(ACTUATOR_RATE * t if t <= up else max(ACTUATOR_PEAK - ACTUATOR_RATE * (t - up), 0.0))

    def step(self):
        # Apply last frame's commanded forces (actuator + reaction friction) for this frame's substeps.
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

        # Read the crank->coupler joint reaction (Miles's probe) and its velocity, then command a
        # Coulomb-like friction torque whose magnitude scales with the reaction load. One-frame lag.
        newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)
        jB_vel = float(self.state_0.joint_qd.numpy()[self.jB_dof])
        reaction = self.solver.joint_reaction_f.numpy()[self.jB]
        reaction_load = float(np.linalg.norm(reaction[:3]))  # linear force part [N]
        friction_torque = -JOINT_MU * reaction_load * np.tanh(jB_vel / 0.05)
        self.max_friction_torque = max(self.max_friction_torque, abs(friction_torque))

        self._joint_f[self.slider_dof] = self._actuator_force()
        self._joint_f[self.jB_dof] = friction_torque
        self.control.joint_f.assign(self._joint_f)

        tension = self._segment_tensions()
        self.viewer.log_scalar("Cable tension: coupler side [N]", float(tension[0]))
        self.viewer.log_scalar("Cable tension: slider side [N]", float(tension[-1]))
        self.viewer.log_scalar("Crank-coupler reaction load [N]", reaction_load)
        self.viewer.log_scalar("Crank-coupler friction torque [N*m]", float(friction_torque))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        starts, ends = get_tendon_cable_lines(self.solver, self.model, self.state_0)
        if len(starts) > 0:
            self.viewer.log_lines(
                "cable",
                wp.array(starts, dtype=wp.vec3),
                wp.array(ends, dtype=wp.vec3),
                colors=(0.95, 0.2, 0.2),
                width=0.004,
            )
        self.viewer.end_frame()

    def _pivot_gap(self, q):
        # Largest world-space separation across the loop joints (the loop must stay closed).
        gap = 0.0
        for ba, bb, pivot in self._pivots.values():
            pa = _world_point(q, ba, pivot - self._centers[ba])
            pb = _world_point(q, bb, pivot - self._centers[bb]) if bb is not None else pivot
            gap = max(gap, float(np.linalg.norm(pa - pb)))
        return gap

    def test_post_step(self):
        q = self.state_0.body_q.numpy()
        assert np.isfinite(q).all(), "non-finite body state"
        gap = self._pivot_gap(q)
        self.max_pivot_gap = max(self.max_pivot_gap, gap)
        assert gap < 0.02, f"four-bar loop separated: pivot gap {gap * 1e3:.1f} mm (linkage exploded)"
        self.max_tension = max(self.max_tension, float(self._segment_tensions().max()))
        self.max_left_travel = max(self.max_left_travel, abs(float(q[self.left][0]) - self.left_x0))

    def test_final(self):
        assert self.max_tension > 0.0, "cable never tensioned"
        assert np.isfinite(self.max_tension), "non-finite tension"
        assert self.max_left_travel > 0.02, f"actuator did not move the linkage: {self.max_left_travel:.4f} m"
        assert self.max_pivot_gap < 0.02, f"loop did not stay closed: {self.max_pivot_gap * 1e3:.1f} mm"
        assert self.max_friction_torque > 0.0, "reaction-dependent joint friction never engaged"
        assert np.isfinite(self.solver.joint_reaction_f.numpy()).all(), "non-finite joint reaction"


if __name__ == "__main__":
    import newton.examples

    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
