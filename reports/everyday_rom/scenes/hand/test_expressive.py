"""Actuation, strain derivatives and unchanged-discretization regressions."""

import ast
import unittest
from pathlib import Path

import numpy as np
import warp as wp
from expressive import AMPLITUDE, RELEASE_DURATION, STARTS, apply_target, target_path
from fit_bounded_targets import StrainChart
from hand_contact import finite_strain

ASSETS = Path(__file__).with_name("assets")


class ActuationTests(unittest.TestCase):
    def test_schedule_is_individual_and_releases_exactly(self):
        targets = np.eye(5)
        path = target_path(targets)
        self.assertLessEqual(int(np.count_nonzero(path, axis=1).max()), 1)
        for i, start in enumerate(STARTS):
            hold = int(round((start + 0.7) * 240))
            self.assertAlmostEqual(float(path[hold, i]), AMPLITUDE, places=7)
            release = int(round((start + 0.9 + RELEASE_DURATION) * 240))
            np.testing.assert_array_equal(path[release : release + 240], 0)
        np.testing.assert_array_equal(path[:120], 0)

    def test_two_substep_schedule_matches_four_substep_samples(self):
        targets = np.random.default_rng(4).normal(size=(5, 48))
        np.testing.assert_array_equal(target_path(targets, step_hz=120), target_path(targets, step_hz=240)[::2])

    def test_force_matches_shifted_elastic_energy_and_negative_control(self):
        for enabled in (True, False):
            path = np.array([[0.2, -0.3, 0.4]], dtype=np.float32)
            stiffness = np.array([2.0, 3.0, 4.0], dtype=np.float32)
            force = wp.zeros(12, dtype=float, device="cpu")
            saved = wp.zeros((1, 3), dtype=float, device="cpu")
            wp.launch(
                apply_target,
                3,
                inputs=[
                    wp.zeros(1, dtype=int, device="cpu"),
                    wp.array(path, dtype=float, device="cpu"),
                    wp.array(stiffness, dtype=float, device="cpu"),
                    enabled,
                    6,
                    force,
                    saved,
                ],
                device="cpu",
            )
            expected = path[0] * stiffness if enabled else np.zeros(3)
            np.testing.assert_allclose(force.numpy()[6:9], expected)
            np.testing.assert_array_equal(force.numpy()[:6], 0)
            np.testing.assert_array_equal(force.numpy()[9:], 0)
            np.testing.assert_array_equal(saved.numpy()[0], force.numpy()[6:9])
            q = np.array([0.1, 0.5, -0.2])
            gradient = stiffness * q - expected
            eps = 1e-6

            def energy(x, stiffness=stiffness, expected=expected):
                return 0.5 * np.dot(stiffness * x, x) - np.dot(expected, x)

            numeric = np.array([(energy(q + eps * d) - energy(q - eps * d)) / (2 * eps) for d in np.eye(3)])
            np.testing.assert_allclose(gradient, numeric, atol=1e-9)

    def test_force_kernel_never_assigns_state(self):
        tree = ast.parse(Path(__file__).with_name("expressive.py").read_text())
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_target")
        names = {a.arg for a in kernel.args.args}
        self.assertNotIn("joint_q", names)
        self.assertNotIn("body_q", names)
        self.assertNotIn("vertices", names)


class StrainTests(unittest.TestCase):
    def test_chart_and_derivative(self):
        rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        phi = np.random.default_rng(7).normal(size=(3, 4, 3)) * 0.1
        q = np.array([0.2, -0.3, 0.4, 0.1])
        chart = StrainChart(rest, faces, phi)
        e, d = chart.evaluate(q, np.array([0]))
        physical = finite_strain(rest, rest + np.einsum("nmc,m->nc", phi, q), faces)
        np.testing.assert_allclose(e, np.sort(physical, axis=-1)[:, ::-1], atol=1e-14)
        for i in range(4):
            delta = np.eye(4)[i] * 1e-6
            numerical = (chart.evaluate(q + delta, [0])[0] - chart.evaluate(q - delta, [0])[0]) / 2e-6
            np.testing.assert_allclose(d[:, :, i], numerical, atol=2e-10)

    def test_rigid_rotation_has_zero_strain(self):
        rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        r = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(finite_strain(rest, rest @ r, np.array([[0, 1, 2]])), 0, atol=1e-14)


class DerivedBasisTests(unittest.TestCase):
    def test_physical_discretization_and_mass_unchanged(self):
        with np.load(ASSETS / "rom-48.npz", allow_pickle=False) as a:
            self.assertEqual(a["rest"].shape, (4994, 3))
            self.assertEqual(a["faces"].shape, (9984, 3))
            self.assertAlmostEqual(float(a["body_mass"]), 0.35, places=6)
            for key in a.files:
                self.assertTrue(np.isfinite(a[key]).all(), key)
            self.assertEqual(a["sample_phi"].shape, (5991, 48, 3))
            gram = np.einsum(
                "n,nmc,nkc->mk",
                a["sample_mass"].astype(float),
                a["sample_phi"].astype(float),
                a["sample_phi"].astype(float),
            )
            np.testing.assert_allclose(gram, np.eye(48), atol=2e-7)


if __name__ == "__main__":
    unittest.main()
