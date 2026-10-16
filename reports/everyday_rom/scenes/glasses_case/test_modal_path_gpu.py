# SPDX-License-Identifier: Apache-2.0
"""Check the active-color block injection against the independent CPU energy."""

import os
import unittest

import numpy as np
import warp as wp
from pod_path_energy import ModalPathEnergy
from solver_vbd_modal_path import assemble_modal_paths


class PathAssemblyTests(unittest.TestCase):
    def setUp(self):
        wp.init()
        self.device = os.environ.get("MODAL_PATH_TEST_DEVICE", "cpu")
        self.paths = (
            ModalPathEnergy([[0, 0], [1, 0], [1, 1]], [0, 0.5, 1], 100),
            ModalPathEnergy([[0, 0, 0], [0.2, 0.8, -0.4]], [0, 1], 73, 0.8),
        )
        coefficients = np.zeros((2, 3, 3))
        coefficients[0, :, :2] = self.paths[0].coefficients
        coefficients[1, :2] = self.paths[1].coefficients
        progress = np.array([[0, 0.5, 1], [0, 1, 0]], dtype=np.float64)
        self.q_numpy = np.zeros(12, dtype=np.float32)
        self.q_numpy[2:4] = [0.3, 0.12]
        self.q_numpy[7:10] = [0.1, 0.4, -0.1]

        def array(values, dtype):
            return wp.array(values, dtype=dtype, device=self.device)

        self.gradient = array(np.full((2, 12), 0.25), wp.float32)
        self.matrix = array(np.full((2, 12, 12), 0.125), wp.float32)
        self.q = array(self.q_numpy, wp.float32)
        self.selected_progress = wp.zeros(2, dtype=wp.float64, device=self.device)
        self.energy = wp.zeros(2, dtype=wp.float64, device=self.device)
        self.inputs = [
            array([0, -1, 1], wp.int32),
            array([0, 1], wp.int32),
            array([2, 7], wp.int32),
            array([2, 3], wp.int32),
            array([3, 2], wp.int32),
            array(coefficients, wp.float64),
            array(progress, wp.float64),
            array([100, 73], wp.float64),
            array([0, 0.8], wp.float64),
            self.q,
            self.gradient,
            self.matrix,
            self.selected_progress,
            self.energy,
        ]

    def launch(self, body_ids):
        wp.launch(assemble_modal_paths, dim=body_ids.size, inputs=[body_ids, *self.inputs], device=self.device)

    def check(self, active):
        expected_g = np.full((2, 12), 0.25, dtype=np.float32)
        expected_h = np.full((2, 12, 12), 0.125, dtype=np.float32)
        for binding in active:
            start = (2, 7)[binding]
            count = (2, 3)[binding]
            value = self.paths[binding].evaluate(self.q_numpy[start : start + count])
            expected_g[binding, 6 : 6 + count] += value.gradient.astype(np.float32)
            for i in range(count):
                for j in range(i, count):
                    expected_h[binding, 6 + i, 6 + j] += np.float32(value.hessian[i, j])
            self.assertAlmostEqual(self.selected_progress.numpy()[binding], value.progress, places=12)
            self.assertAlmostEqual(self.energy.numpy()[binding], value.energy, places=12)
        np.testing.assert_allclose(self.gradient.numpy(), expected_g, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(self.matrix.numpy(), expected_h, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(self.q.numpy(), self.q_numpy)

    def test_mixed_dimensions_match_cpu_energy_and_preserve_state(self):
        ids = wp.array([0, 1, 2], dtype=wp.int32, device=self.device)
        self.launch(ids)
        self.check((0, 1))

    def test_only_active_color_body_is_assembled(self):
        ids = wp.array([1, 2], dtype=wp.int32, device=self.device)
        self.launch(ids)
        self.check((1,))
        self.assertEqual(self.selected_progress.numpy()[0], 0)
        self.assertEqual(self.energy.numpy()[0], 0)

    @unittest.skipUnless(os.environ.get("MODAL_PATH_TEST_DEVICE", "cpu").startswith("cuda"), "CUDA graph test")
    def test_graph_replay(self):
        ids = wp.array([0, 1, 2], dtype=wp.int32, device=self.device)
        wp.load_module(module="solver_vbd_modal_path", device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            self.launch(ids)
        wp.capture_launch(capture.graph)
        self.check((0, 1))


if __name__ == "__main__":
    unittest.main()
