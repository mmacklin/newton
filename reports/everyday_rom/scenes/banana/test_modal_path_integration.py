# SPDX-License-Identifier: Apache-2.0
"""Full Newton timestep checks, independent of the path assembly implementation."""

import os
import unittest

import numpy as np
import warp as wp
from pod_path_energy import ModalPathEnergy
from solver_vbd_modal_path import SolverVBDModalPath

import newton
from newton.solvers import SolverVBD


class PathIntegrationTests(unittest.TestCase):
    def build(self, coupled=True, restoring=0.0):
        device = os.environ.get("MODAL_PATH_TEST_DEVICE", "cpu")
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        body = builder.add_body_elastic(
            mass=1.0,
            inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            mode_count=2,
            mode_mass=[1.0, 1.0],
            mode_stiffness=[0.0, 0.0],
            mode_damping=[0.0, 0.0],
            mode_q=[0.3, 0.12],
            mode_qd=[0.2, -0.1],
            is_kinematic=True,
        )
        builder.color()
        model = builder.finalize(device=device)
        path = ModalPathEnergy([[0, 0], [1, 0]], [0, 1], 100.0, restoring)
        solver = (
            SolverVBDModalPath(model, paths={body: path}, iterations=8, rigid_compliant_alm=False)
            if coupled
            else SolverVBD(model, iterations=8, rigid_compliant_alm=False)
        )
        return model, solver, path

    def test_implicit_response_and_missing_coupling_negative_control(self):
        dt = 0.1
        expected = np.array([0.32, 0.11 / 2])
        for coupled in (True, False):
            model, solver, _ = self.build(coupled)
            start, end = model.state(), model.state()
            before = start.joint_q.numpy().copy()
            solver.step(start, end, model.control(), None, dt)
            np.testing.assert_array_equal(start.joint_q.numpy(), before)
            q = end.joint_q.numpy()[7:9]
            if coupled:
                np.testing.assert_allclose(q, expected, atol=2e-6)
                np.testing.assert_allclose(end.joint_qd.numpy()[6:8], (expected - before[7:9]) / dt, atol=2e-5)
            else:
                np.testing.assert_allclose(q, [0.32, 0.11], atol=2e-6)
                self.assertGreater(np.linalg.norm(q - expected), 0.05)

    def test_restoring_energy_dissipates_without_external_work(self):
        model, solver, path = self.build(restoring=4.0)
        a, b = model.state(), model.state()
        control = model.control()
        energies = []
        for _ in range(40):
            q, v = a.joint_q.numpy()[7:9], a.joint_qd.numpy()[6:8]
            energies.append(0.5 * np.dot(v, v) + path.evaluate(q).energy)
            solver.step(a, b, control, None, 0.01)
            a, b = b, a
        self.assertLess(max(np.diff(energies)), 2e-6)
        self.assertLess(energies[-1], 0.9 * energies[0])


if __name__ == "__main__":
    unittest.main()
