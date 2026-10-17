# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Mutation-sensitive CPU tests for the independent fused trajectory verifier."""

import unittest
from types import SimpleNamespace

import numpy as np
from brush_glove_geometry import Config
from fused_toothbrush import HostArray
from toothbrush import Example as OriginalExample
from verify_fused import (
    check_motion_limits,
    check_pair_states,
    check_residuals,
    check_saved_capacity,
    check_strain_limits,
    modal_motion,
    rest_metric,
    stretches,
)


class TrajectoryTests(unittest.TestCase):
    def test_unchanged_physical_assertions_reject_mutated_summaries(self):
        """Exercise original physical/control predicates, not rewritten thresholds."""
        records = [
            {
                "time": (i + 1) / 60,
                "plate_modal_displacement_m": 0,
                "grasp_enabled": (i + 1) / 60 < 5.2,
                "tip_x_displacement_m": [0.001] * 20,
            }
            for i in range(450)
        ]
        summary = {
            "records": records,
            "contact_buffer_overflow": False,
            "max_modal_solve_residual_ratio": 0.01,
            "mass_audit": {
                "dimension": 46,
                "runtime_spd_passed": True,
                "actual_implicit_mass_coupling": [True],
                "normalized_condition_number": 2,
            },
            "contact_window_deformation_m": 0.002,
            "max_root_anchor_error_m": 0,
            "max_root_angular_error_rad": 0,
            "max_penetration_m": 0.0001,
            "max_deformation_m": 0.003,
            "handle_lift_m": 0.12,
            "freefall_vertical_acceleration_m_s2": -9.81,
            "freefall_drop_m": 0.05,
            "final_handle_speed_m_s": 0.001,
            "final_handle_angular_speed_rad_s": 0.01,
            "landed_position_range_m": 0.0001,
            "release_max_deformation_m": 0.00001,
            "tool_contact_frames": 100,
            "scrub_tuft_reversals": 20,
            "scrub_tangential_peak_to_peak_m": 0.003,
            "scrub_contact_frames": 110,
            "field_coverage": {
                "covered_tufts": 20,
                "covered_tufts_per_transverse_column": [5] * 4,
                "covered_tufts_per_longitudinal_row": [4] * 5,
            },
        }
        view = SimpleNamespace(
            records=records,
            cfg=Config(),
            graph=True,
            summary=lambda: summary,
            model=SimpleNamespace(elastic_body_count=1, body_flags=HostArray(np.array([1]))),
            elastic=[None],
            expected_dynamic_body_count=1,
            capacity_audit=SimpleNamespace(check=lambda count: None),
        )
        OriginalExample.test_final(view)
        for key, bad in (
            ("contact_window_deformation_m", 0),
            ("max_penetration_m", 0.00081),
            ("max_modal_solve_residual_ratio", 0.051),
            ("freefall_vertical_acceleration_m_s2", 0),
            ("release_max_deformation_m", 0.00021),
            ("scrub_contact_frames", 99),
        ):
            before = summary[key]
            summary[key] = bad
            with self.subTest(mutation=key), self.assertRaises(AssertionError):
                OriginalExample.test_final(view)
            summary[key] = before

    def test_original_strain_limits_reject_deformation(self):
        """Retain first-export and material-rest ten-percent strain gates."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        faces = np.array([[0, 1, 2]])
        inverse, _ = rest_metric(points, faces)
        np.testing.assert_allclose(stretches(points, faces, inverse), 1)
        bad = points.copy()
        bad[1, 0] = 1.101
        strain = float(np.abs(stretches(bad, faces, inverse) - 1).max())
        for a, b in ((strain, 0), (0, strain)):
            with self.assertRaises(AssertionError):
                check_strain_limits(a, b)

    def test_modal_recovery_and_original_one_mm_limits(self):
        """Reject a step and a second difference independently at the original bound."""
        phi = np.zeros((2, 40, 3), np.float32)
        phi[0, 0, 0] = 1
        q = np.zeros((450, 40), np.float32)
        good = modal_motion(phi, q, np.zeros(40, np.float32))
        check_motion_limits(good)
        q[225:, 0] = 0.00101
        with self.assertRaisesRegex(AssertionError, "surface step"):
            check_motion_limits(modal_motion(phi, q, np.zeros(40, np.float32)))
        q[:] = 0
        q[225, 0] = 0.0006
        spike = modal_motion(phi, q, np.zeros(40, np.float32))
        self.assertLess(spike["max_surface_step_m"], 0.001)
        with self.assertRaisesRegex(AssertionError, "second difference"):
            check_motion_limits(spike)

    def test_nonfinite_motion_rejected(self):
        """NaN coordinates cannot disappear through a maximum reduction."""
        phi = np.zeros((1, 40, 3), np.float32)
        q = np.zeros((450, 40), np.float32)
        q[12, 0] = np.nan
        with self.assertRaisesRegex(AssertionError, "finite"):
            modal_motion(phi, q, np.zeros(40, np.float32))

    def test_raw_residual_gate_rejects_bad_ratio_and_missing_frames(self):
        """Do not trust a green summary if the captured solve residual fails."""
        values = {
            "initial_residual_norm": np.ones((450, 1), np.float32),
            "solve_residual_norm": np.full((450, 1), 0.01, np.float32),
        }
        self.assertLess(float(check_residuals(values).max()), 0.05)
        values["solve_residual_norm"][35, 0] = 0.05001
        with self.assertRaisesRegex(AssertionError, "residual"):
            check_residuals(values)
        values["solve_residual_norm"] = np.zeros((449, 1), np.float32)
        with self.assertRaisesRegex(AssertionError, "450"):
            check_residuals(values)

    def test_capacity_coverage_and_overflow_mutations(self):
        """Require every settling/trajectory substep and unclamped capacity guards."""
        n = 2040
        times, t = [], np.float32(0)
        for i in range(n):
            if i <= 240 and i % 4 == 0:
                t = np.float32(0)
            times.append(t)
            t = np.float32(t + np.float32(1 / 240))
        good = {
            "collision_count": np.array(n),
            "solve_count": np.array(n),
            "raw_global": np.zeros(n, np.int32),
            "solved_raw_global": np.zeros(n, np.int32),
            "raw_per_body": np.zeros((n, 5), np.int32),
            "body_overflow_max": np.zeros(n, np.int32),
            "time_s": np.array(times),
            "collision_pair": np.arange(1, n + 1),
            "collision_seen": np.ones(n, np.int32),
            "solve_seen": np.ones(n, np.int32),
            "global_capacity": np.array(16384),
            "body_capacity": np.array(16384),
            "expected_substeps": np.array(n),
            "dt": np.array(1 / 240),
        }
        self.assertTrue(check_saved_capacity(good)["passed"])
        for key, index, value in (
            ("solve_seen", 77, 0),
            ("raw_per_body", (77, 0), 16385),
            ("body_overflow_max", 77, 1),
        ):
            bad = {name: array.copy() for name, array in good.items()}
            bad[key][index] = value
            with self.subTest(mutation=key), self.assertRaises(AssertionError):
                check_saved_capacity(bad)

    def test_matched_control_schedule_and_kinematics(self):
        """Reject changed kinematic states, release schedule, or time samples."""
        state = {
            "times": np.arange(1, 451) / 60,
            "body_q": np.zeros((450, 5, 7), np.float32),
            "body_qd": np.zeros((450, 5, 6), np.float32),
            "joint_enabled": np.zeros((450, 6), bool),
        }
        control = {key: array.copy() for key, array in state.items()}
        check_pair_states(state, control)
        for key, index in (("times", 20), ("body_q", (20, 1, 0)), ("body_qd", (20, 2, 0)), ("joint_enabled", (20, 5))):
            bad = {name: array.copy() for name, array in control.items()}
            bad[key][index] = 1
            with self.subTest(mutation=key), self.assertRaises(AssertionError):
                check_pair_states(state, bad)


if __name__ == "__main__":
    unittest.main()
