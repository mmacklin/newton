"""Frozen six-substep withdrawal tests and complete saved-force validation."""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from audit_equilibrium import audit_saved
from expressive import AMPLITUDE, RELEASE_DURATION, STARTS, target_path


class WithdrawalTests(unittest.TestCase):
    def test_six_substep_force_schedule_and_exact_zero(self):
        """Check the frozen release50 pulse and exact zero after withdrawal."""
        self.assertEqual(RELEASE_DURATION, 3 / 60)
        self.assertEqual(AMPLITUDE, 1)
        path = target_path(np.eye(5), step_hz=120)
        self.assertTrue(np.isfinite(path).all())
        self.assertLessEqual(int(np.count_nonzero(path, axis=1).max()), 1)
        for i, start in enumerate(STARTS):
            first = int(round((start + 0.9) * 120))
            expected = np.array([1, 25 / 27, 20 / 27, 1 / 2, 7 / 27, 2 / 27, 0], dtype=np.float32)
            np.testing.assert_array_equal(path[first : first + 7, i], expected)
            end = int(round(STARTS[i + 1] * 120)) if i + 1 < 5 else len(path)
            np.testing.assert_array_equal(path[first + 6 : end], 0)

    def test_rise_and_hold_unchanged(self):
        """Preserve the approved rise and hold without modifying modal state."""
        path = target_path(np.eye(5), step_hz=120)
        for i, start in enumerate(STARTS):
            t = np.arange(73) / 120
            expected = np.clip(t / 0.6, 0, 1)
            expected = (expected * expected * (3 - 2 * expected)).astype(np.float32)
            begin = int(round(start * 120))
            np.testing.assert_allclose(path[begin : begin + 73, i], expected, atol=1e-7, rtol=0)
            np.testing.assert_array_equal(path[begin + 72 : begin + 109, i], 1)


class ProvenanceTests(unittest.TestCase):
    def test_accepted_derived_assets_and_source_hashes(self):
        """Require the reviewed derived-only assets and exact bundled sources."""
        root = Path(__file__).parent
        expected = {
            "rom-48.npz": "526d8f3d1f9ffc2e2979429036250b7bfaaca6f81923f9c850b3e45d4944daea",
            "targets.npz": "7eafaf8c51a4992d137d38fb874fea24004e54ec0c04ba15c89d79828f641610",
        }
        for name, checksum in expected.items():
            self.assertEqual(hashlib.sha256((root / "assets" / name).read_bytes()).hexdigest(), checksum)
        for name, record in json.loads((root / "sources.json").read_text()).items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), record["bundled_sha256"], name)
        with np.load(root / "assets/targets.npz", allow_pickle=False) as targets:
            self.assertEqual(targets["q"].shape, (5, 48))
            self.assertTrue(np.isfinite(targets["q"]).all())


class SavedForceTests(unittest.TestCase):
    def test_saved_force_audit_and_corruption_negatives(self):
        """Reject double force, nonzero control, residual release force and NaNs."""
        target = target_path(np.random.default_rng(5).normal(size=(5, 48)), step_hz=120)[:-1]
        stiffness = np.linspace(1, 2, 48).astype(np.float32)
        for corruption in (None, "double", "control", "release", "nan"):
            with self.subTest(corruption=corruption), TemporaryDirectory() as temporary:
                root = Path(temporary)
                for arm, enabled in (("actuated", True), ("no-actuation", False)):
                    destination = root / arm
                    destination.mkdir()
                    q = target.copy()
                    forces = q * stiffness if enabled else np.zeros_like(q)
                    if enabled and corruption == "double":
                        forces *= 2
                    if not enabled and corruption == "control":
                        forces[100, 0] = 1
                    if enabled and corruption == "release":
                        # A consistent force/target pair still fails the released-zero check.
                        q[200, 0] = 1
                        forces[200, 0] = stiffness[0]
                    if enabled and corruption == "nan":
                        forces[100, 0] = np.nan
                    np.savez(
                        destination / "actuation.npz",
                        forces=forces,
                        q_target=q,
                        times=np.arange(1440) * (1 / 120),
                        stiffness=stiffness,
                        dt=1 / 120,
                        amplitude=1,
                        starts=STARTS,
                        enabled=enabled,
                        release_duration_s=RELEASE_DURATION,
                    )
                if corruption is None:
                    self.assertTrue(audit_saved(root, STARTS, RELEASE_DURATION)["passed"])
                else:
                    with self.assertRaises(AssertionError):
                        audit_saved(root, STARTS, RELEASE_DURATION)


if __name__ == "__main__":
    unittest.main()
