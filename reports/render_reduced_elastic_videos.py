# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import warp as wp

import newton
import newton.viewer
from newton.examples.basic.example_basic_reduced_elastic_beam import Example as BeamExample
from newton.examples.basic.example_basic_reduced_elastic_beam_vibration import Example as BeamVibrationExample
from newton.examples.basic.example_basic_reduced_elastic_fourbar import Example as FourbarExample

WIDTH = 960
HEIGHT = 540
FPS = 60


def _write_video(path: Path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def _capture(
    example, viewer, frame_count: int, screenshot_path: Path | None = None, screenshot_frame: int | None = None
):
    frames = []
    if screenshot_frame is None:
        screenshot_frame = frame_count // 2
    for i in range(frame_count):
        example.step()
        example.render()
        frame = viewer.get_frame().numpy()
        frames.append(frame)
        if screenshot_path is not None and i == screenshot_frame:
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(screenshot_path, frame)

    stacked = np.asarray(frames)
    if int(stacked.max()) == int(stacked.min()):
        raise RuntimeError("captured frames are blank")
    return frames


class RevoluteEndpointFixture:
    def __init__(self, viewer):
        self.viewer = viewer
        self.fps = FPS
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.length = 1.0
        self.z = 0.18
        self.mode_q0 = np.array([0.65, -0.32, 0.22, -0.14, 0.08], dtype=np.float32)

        torsion_basis = newton.ModalGeneratorBeam(
            length=self.length,
            half_width_y=0.085,
            half_width_z=0.05,
            mode_specs=[
                {
                    "type": newton.ModalGeneratorBeam.Mode.TORSION,
                    "boundary": newton.ModalGeneratorBeam.Boundary.FIXED_FREE,
                    "order": 1,
                },
                {
                    "type": newton.ModalGeneratorBeam.Mode.TORSION,
                    "boundary": newton.ModalGeneratorBeam.Boundary.FIXED_FREE,
                    "order": 2,
                },
                {
                    "type": newton.ModalGeneratorBeam.Mode.TORSION,
                    "boundary": newton.ModalGeneratorBeam.Boundary.FIXED_FREE,
                    "order": 3,
                },
                {
                    "type": newton.ModalGeneratorBeam.Mode.TORSION,
                    "boundary": newton.ModalGeneratorBeam.Boundary.FIXED_FREE,
                    "order": 4,
                },
                {
                    "type": newton.ModalGeneratorBeam.Mode.TORSION,
                    "boundary": newton.ModalGeneratorBeam.Boundary.FIXED_FREE,
                    "order": 5,
                },
            ],
            sample_count=49,
            density=500.0,
            shear_modulus=4.0e4,
            label="torsion_fixture_basis",
        ).build()

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_ground_plane()

        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        self.body = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(0.5 * self.length, 0.0, self.z), wp.quat_identity()),
            mass=1.0,
            inertia=inertia,
            mode_q=self.mode_q0,
            modal_basis=torsion_basis,
            label="elastic_torsion_fixture",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        builder.add_shape_box(self.body, hx=self.length / 2.0, hy=0.085, hz=0.05, cfg=shape_cfg)
        self.joint = builder.add_joint_revolute(
            parent=-1,
            child=self.body,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, self.z), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.5 * self.length, 0.0, 0.0), wp.quat_identity()),
            label="world_revolute_to_elastic_endpoint",
        )
        builder.color()

        self.model = builder.finalize()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.elastic_joint = int(self.model.elastic_joint.numpy()[0])
        self.elastic_q_start = int(self.model.joint_q_start.numpy()[self.elastic_joint])
        self.elastic_qd_start = int(self.model.joint_qd_start.numpy()[self.elastic_joint])
        self._joint_f = self.control.joint_f.numpy()

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=16,
            rigid_joint_linear_k_start=1.0e5,
            rigid_joint_angular_k_start=1.0e3,
            rigid_joint_linear_ke=2.0e6,
            rigid_joint_angular_ke=1.0e5,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.52, -1.35, 0.72), -24.0, 84.0)

    def _mode_value(self) -> float:
        return float(self.state_0.joint_q.numpy()[self.elastic_q_start + 7])

    def step(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            self.state_0.clear_forces()
            self._joint_f[:] = 0.0
            self._joint_f[self.elastic_qd_start + 6] = 0.8 * math.sin(4.0 * t)
            self._joint_f[self.elastic_qd_start + 7] = 0.45 * math.cos(3.1 * t)
            self._joint_f[self.elastic_qd_start + 8] = 0.25 * math.sin(5.2 * t + 0.4)
            self._joint_f[self.elastic_qd_start + 9] = 0.15 * math.cos(6.1 * t)
            self._joint_f[self.elastic_qd_start + 10] = 0.08 * math.sin(7.0 * t)
            self.control.joint_f.assign(self._joint_f)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


def main():
    root = Path(__file__).resolve().parent
    assets = root / "assets"

    viewer = newton.viewer.ViewerGL(width=WIDTH, height=HEIGHT, headless=True)
    fourbar = FourbarExample(viewer, None)
    frames = _capture(
        fourbar,
        viewer,
        frame_count=180,
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_fourbar.jpg"),
    )
    _write_video(assets / "reduced_elastic_fourbar.mp4", frames)

    viewer.clear_model()
    fixture = RevoluteEndpointFixture(viewer)
    frames = _capture(fixture, viewer, frame_count=150)
    _write_video(assets / "elastic_revolute_endpoint_fixture.mp4", frames)

    viewer.clear_model()
    beam = BeamExample(viewer, None)
    frames = _capture(
        beam,
        viewer,
        frame_count=150,
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_beam.jpg"),
    )
    _write_video(assets / "reduced_elastic_cantilever_beam.mp4", frames)

    viewer.clear_model()
    vibration = BeamVibrationExample(viewer, None)
    frames = _capture(
        vibration,
        viewer,
        frame_count=180,
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_beam_vibration.jpg"),
        screenshot_frame=8,
    )
    _write_video(assets / "reduced_elastic_cantilever_vibration.mp4", frames)
    viewer.close()

    print(f"Wrote {assets / 'reduced_elastic_fourbar.mp4'}")
    print(f"Wrote {assets / 'elastic_revolute_endpoint_fixture.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_cantilever_beam.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_cantilever_vibration.mp4'}")


if __name__ == "__main__":
    main()
