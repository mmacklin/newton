# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.tests.unittest_utils import sanitize_identifier


def build_asymmetric_atwood(mass_left=1.0, mass_right=3.0, mu=0.0,
                            pulley_radius=0.15):
    """Atwood machine with unequal masses and configurable capstan friction.

    Kinematic pulley at z=3.5, weights at z=2.0.  Returns (model, left_idx,
    right_idx) where left is the lighter weight.
    """
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

    pulley = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 3.5), q=wp.quat_identity()),
        mass=0.0,
        is_kinematic=True,
    )
    builder.add_shape_cylinder(pulley, radius=pulley_radius, half_height=0.04)

    Dof = newton.ModelBuilder.JointDofConfig
    planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
    planar_ang = [Dof(axis=Axis.Y)]

    left = builder.add_link(
        xform=wp.transform(p=wp.vec3(-0.5, 0.0, 2.0), q=wp.quat_identity()),
        mass=mass_left,
    )
    builder.add_shape_box(left, hx=0.06, hy=0.06, hz=0.06)
    j1 = builder.add_joint_d6(
        parent=-1, child=left,
        linear_axes=planar_lin, angular_axes=planar_ang,
        parent_xform=wp.transform(p=wp.vec3(-0.5, 0.0, 2.0)),
        child_xform=wp.transform(),
    )
    builder.add_articulation([j1])

    right = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.5, 0.0, 2.0), q=wp.quat_identity()),
        mass=mass_right,
    )
    builder.add_shape_box(right, hx=0.06, hy=0.06, hz=0.06)
    j2 = builder.add_joint_d6(
        parent=-1, child=right,
        linear_axes=planar_lin, angular_axes=planar_ang,
        parent_xform=wp.transform(p=wp.vec3(0.5, 0.0, 2.0)),
        child_xform=wp.transform(),
    )
    builder.add_articulation([j2])

    axis = (0.0, 1.0, 0.0)
    builder.add_tendon()
    builder.add_tendon_link(
        body=left,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
    )
    builder.add_tendon_link(
        body=pulley,
        link_type=int(TendonLinkType.ROLLING),
        radius=pulley_radius,
        orientation=1,
        mu=mu,
        offset=(0.0, 0.0, 0.0),
        axis=axis,
        compliance=1.0e-6,
        damping=0.1,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
        compliance=1.0e-6,
        damping=0.1,
        rest_length=-1.0,
    )

    builder.add_ground_plane()
    return builder.finalize(), left, right


def build_dynamic_atwood(mass_left=1.0, mass_right=3.0, mu=0.0,
                         pulley_radius=0.15, pulley_mass=5.0):
    """Atwood machine with a dynamic pulley on a hinge joint.

    Returns (model, left_idx, right_idx, pulley_idx).
    """
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

    pulley_pos = wp.vec3(0.0, 0.0, 3.5)
    pulley = builder.add_body(
        xform=wp.transform(p=pulley_pos, q=wp.quat_identity()),
        mass=pulley_mass,
    )
    q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
    builder.add_shape_cylinder(
        pulley, xform=wp.transform(q=q_cyl),
        radius=pulley_radius, half_height=0.04,
    )
    Dof = newton.ModelBuilder.JointDofConfig
    j_pulley = builder.add_joint_d6(
        parent=-1, child=pulley,
        linear_axes=[],
        angular_axes=[Dof(axis=Axis.Y)],
        parent_xform=wp.transform(p=pulley_pos),
        child_xform=wp.transform(),
    )
    builder.add_articulation([j_pulley])

    planar_lin = [Dof(axis=Axis.X), Dof(axis=Axis.Z)]
    planar_ang = [Dof(axis=Axis.Y)]

    left = builder.add_link(
        xform=wp.transform(p=wp.vec3(-0.4, 0.0, 2.0), q=wp.quat_identity()),
        mass=mass_left,
    )
    builder.add_shape_box(left, hx=0.06, hy=0.06, hz=0.06)
    j1 = builder.add_joint_d6(
        parent=-1, child=left,
        linear_axes=planar_lin, angular_axes=planar_ang,
        parent_xform=wp.transform(p=wp.vec3(-0.4, 0.0, 2.0)),
        child_xform=wp.transform(),
    )
    builder.add_articulation([j1])

    right = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.4, 0.0, 2.0), q=wp.quat_identity()),
        mass=mass_right,
    )
    builder.add_shape_box(right, hx=0.06, hy=0.06, hz=0.06)
    j2 = builder.add_joint_d6(
        parent=-1, child=right,
        linear_axes=planar_lin, angular_axes=planar_ang,
        parent_xform=wp.transform(p=wp.vec3(0.4, 0.0, 2.0)),
        child_xform=wp.transform(),
    )
    builder.add_articulation([j2])

    axis = (0.0, 1.0, 0.0)
    builder.add_tendon()
    builder.add_tendon_link(
        body=left,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
    )
    builder.add_tendon_link(
        body=pulley,
        link_type=int(TendonLinkType.ROLLING),
        radius=pulley_radius,
        orientation=1,
        mu=mu,
        offset=(0.0, 0.0, 0.0),
        axis=axis,
        compliance=1.0e-6,
        damping=0.1,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
        compliance=1.0e-6,
        damping=0.1,
        rest_length=-1.0,
    )

    builder.add_ground_plane()
    return builder.finalize(), left, right, pulley


def run_atwood(model, num_frames=120, substeps=16, fps=60):
    dt = 1.0 / fps / substeps
    solver = newton.solvers.SolverXPBD(
        model, iterations=8, joint_linear_relaxation=0.8,
    )
    s0 = model.state()
    s1 = model.state()
    control = model.control()
    contacts = model.contacts()

    for _ in range(num_frames):
        for _ in range(substeps):
            s0.clear_forces()
            model.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0

    return s0


class TestTendonCapstan(unittest.TestCase):
    pass


def add_test(cls, name, devices, test_fn):
    for device in devices:
        test_name = f"test_{sanitize_identifier(name)}_{sanitize_identifier(device)}"
        setattr(cls, test_name, lambda self, d=device, fn=test_fn: fn(self, d))


def test_frictionless_atwood(test, device):
    """mu=0: standard Atwood — heavy descends, light ascends."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx = build_asymmetric_atwood(
            mass_left=1.0, mass_right=3.0, mu=0.0,
        )
        state = run_atwood(model, num_frames=60)
        bq = state.body_q.numpy()
        test.assertTrue(np.isfinite(bq).all(), "Non-finite body positions")

        left_z = float(bq[left_idx][2])
        right_z = float(bq[right_idx][2])

        test.assertGreater(left_z, 2.0,
                           f"Light weight should ascend: z={left_z:.3f}")
        test.assertLess(right_z, 2.0,
                        f"Heavy weight should descend: z={right_z:.3f}")

        displacement = abs(left_z - 2.0)
        test.assertGreater(displacement, 0.1,
                           f"Significant motion expected: Δ={displacement:.3f}")


def test_high_friction_locked(test, device):
    """mu=10: cable locked by friction — neither weight moves significantly."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx = build_asymmetric_atwood(
            mass_left=1.0, mass_right=3.0, mu=10.0,
        )
        state = run_atwood(model, num_frames=60)
        bq = state.body_q.numpy()
        test.assertTrue(np.isfinite(bq).all(), "Non-finite body positions")

        left_z = float(bq[left_idx][2])
        right_z = float(bq[right_idx][2])

        left_drift = abs(left_z - 2.0)
        right_drift = abs(right_z - 2.0)

        test.assertLess(left_drift, 0.3,
                        f"Light weight should stay near start: Δ={left_drift:.3f}")
        test.assertLess(right_drift, 0.3,
                        f"Heavy weight should stay near start: Δ={right_drift:.3f}")


def test_critical_friction(test, device):
    """mu near critical value: motion is much slower than frictionless."""
    with wp.ScopedDevice(device):
        # mu_crit = ln(3)/pi ≈ 0.35 for 3:1 mass ratio
        # Use slightly below critical so there's still some motion
        mu_sub = 0.30
        model_sub, left_sub, right_sub = build_asymmetric_atwood(
            mass_left=1.0, mass_right=3.0, mu=mu_sub,
        )
        state_sub = run_atwood(model_sub, num_frames=60)
        bq_sub = state_sub.body_q.numpy()

        model_free, left_free, right_free = build_asymmetric_atwood(
            mass_left=1.0, mass_right=3.0, mu=0.0,
        )
        state_free = run_atwood(model_free, num_frames=60)
        bq_free = state_free.body_q.numpy()

        test.assertTrue(np.isfinite(bq_sub).all(), "Non-finite (subcritical)")
        test.assertTrue(np.isfinite(bq_free).all(), "Non-finite (frictionless)")

        disp_sub = abs(float(bq_sub[right_sub][2]) - 2.0)
        disp_free = abs(float(bq_free[right_free][2]) - 2.0)

        test.assertGreater(disp_free, 0.1,
                           f"Frictionless should move: Δ={disp_free:.3f}")
        test.assertLess(disp_sub, disp_free,
                        f"Subcritical friction should slow motion: "
                        f"Δ_sub={disp_sub:.3f} >= Δ_free={disp_free:.3f}")


def test_friction_monotonic(test, device):
    """Higher friction → less displacement (monotonicity)."""
    with wp.ScopedDevice(device):
        mus = [0.0, 0.15, 0.30, 1.0]
        displacements = []
        for mu in mus:
            model, _, right_idx = build_asymmetric_atwood(
                mass_left=1.0, mass_right=3.0, mu=mu,
            )
            state = run_atwood(model, num_frames=60)
            bq = state.body_q.numpy()
            test.assertTrue(np.isfinite(bq).all(), f"Non-finite at mu={mu}")
            disp = abs(float(bq[right_idx][2]) - 2.0)
            displacements.append(disp)

        for i in range(len(mus) - 1):
            test.assertGreaterEqual(
                displacements[i] + 0.01,  # small tolerance
                displacements[i + 1],
                f"Monotonicity violated: mu={mus[i]} disp={displacements[i]:.3f} "
                f"vs mu={mus[i+1]} disp={displacements[i+1]:.3f}",
            )


def test_frictionless_dynamic_no_rotation(test, device):
    """mu=0 on a dynamic pulley: cable slides freely, pulley does not rotate."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx, pulley_idx = build_dynamic_atwood(
            mass_left=1.0, mass_right=3.0, mu=0.0, pulley_mass=5.0,
        )
        state = run_atwood(model, num_frames=60)
        bq = state.body_q.numpy()
        test.assertTrue(np.isfinite(bq).all(), "Non-finite body positions")

        # extract pulley Y-axis rotation from quaternion
        qy = float(bq[pulley_idx][4])
        qw = float(bq[pulley_idx][6])
        rot_y_rad = abs(2.0 * np.arctan2(qy, qw))

        test.assertLess(rot_y_rad, 0.05,
                        f"Frictionless pulley should not rotate: "
                        f"|rot_y|={np.degrees(rot_y_rad):.1f}°")

        # Atwood motion should still work (light ascends, heavy descends)
        left_z = float(bq[left_idx][2])
        right_z = float(bq[right_idx][2])
        test.assertGreater(left_z, 2.0,
                           f"Light weight should ascend: z={left_z:.3f}")
        test.assertLess(right_z, 2.0,
                        f"Heavy weight should descend: z={right_z:.3f}")


def test_stick_dx_equals_dtheta_r(test, device):
    """mu=∞ on dynamic pulley: cable displacement matches rim displacement."""
    with wp.ScopedDevice(device):
        pulley_radius = 0.15
        model, left_idx, right_idx, pulley_idx = build_dynamic_atwood(
            mass_left=1.0, mass_right=3.0, mu=10.0,
            pulley_radius=pulley_radius, pulley_mass=5.0,
        )
        state_init = model.state()
        bq0 = state_init.body_q.numpy()
        left_z0 = float(bq0[left_idx][2])
        right_z0 = float(bq0[right_idx][2])

        state = run_atwood(model, num_frames=60)
        bq = state.body_q.numpy()
        test.assertTrue(np.isfinite(bq).all(), "Non-finite body positions")

        left_z = float(bq[left_idx][2])
        right_z = float(bq[right_idx][2])
        dx_left = abs(left_z - left_z0)
        dx_right = abs(right_z - right_z0)

        # extract pulley rotation
        qy = float(bq[pulley_idx][4])
        qw = float(bq[pulley_idx][6])
        dtheta = abs(2.0 * np.arctan2(qy, qw))
        rim_disp = dtheta * pulley_radius

        test.assertGreater(dtheta, 0.1,
                           f"Pulley should rotate: dtheta={np.degrees(dtheta):.1f}°")
        test.assertAlmostEqual(dx_left, rim_disp, delta=0.15,
                               msg=f"Left displacement should match rim: "
                               f"dx_left={dx_left:.3f}, dtheta*R={rim_disp:.3f}")
        test.assertAlmostEqual(dx_right, rim_disp, delta=0.15,
                               msg=f"Right displacement should match rim: "
                               f"dx_right={dx_right:.3f}, dtheta*R={rim_disp:.3f}")


def test_small_inertia_equivalence(test, device):
    """Small-inertia pulley: frictionless and high-friction behave similarly."""
    with wp.ScopedDevice(device):
        model_free, _, right_free, _ = build_dynamic_atwood(
            mass_left=1.0, mass_right=3.0, mu=0.0,
            pulley_mass=0.1,
        )
        state_free = run_atwood(model_free, num_frames=60)
        bq_free = state_free.body_q.numpy()
        disp_free = abs(float(bq_free[right_free][2]) - 2.0)

        model_stick, _, right_stick, _ = build_dynamic_atwood(
            mass_left=1.0, mass_right=3.0, mu=10.0,
            pulley_mass=0.1,
        )
        state_stick = run_atwood(model_stick, num_frames=60)
        bq_stick = state_stick.body_q.numpy()
        disp_stick = abs(float(bq_stick[right_stick][2]) - 2.0)

        test.assertTrue(np.isfinite(bq_free).all(), "Non-finite (frictionless)")
        test.assertTrue(np.isfinite(bq_stick).all(), "Non-finite (stick)")

        test.assertGreater(disp_free, 0.1,
                           f"Frictionless should move: Δ={disp_free:.3f}")
        test.assertGreater(disp_stick, 0.1,
                           f"Stick should also move: Δ={disp_stick:.3f}")
        ratio = disp_stick / disp_free if disp_free > 0.01 else 0.0
        test.assertGreater(ratio, 0.5,
                           f"Small-inertia pulley: stick and frictionless should "
                           f"be similar: ratio={ratio:.2f}")


devices = ["cpu"]
if wp.is_cuda_available():
    devices.append("cuda:0")

add_test(TestTendonCapstan, "frictionless_atwood", devices, test_frictionless_atwood)
add_test(TestTendonCapstan, "high_friction_locked", devices, test_high_friction_locked)
add_test(TestTendonCapstan, "critical_friction", devices, test_critical_friction)
add_test(TestTendonCapstan, "friction_monotonic", devices, test_friction_monotonic)
add_test(TestTendonCapstan, "frictionless_dynamic_no_rotation", devices, test_frictionless_dynamic_no_rotation)
add_test(TestTendonCapstan, "stick_dx_equals_dtheta_r", devices, test_stick_dx_equals_dtheta_r)
add_test(TestTendonCapstan, "small_inertia_equivalence", devices, test_small_inertia_equivalence)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
