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
from newton.examples.basic.example_basic_reduced_elastic_fourbar import Example as FourbarExample

WIDTH = 960
HEIGHT = 540
FPS = 60


def _write_video(path: Path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def _capture(example, viewer, frame_count: int, screenshot_path: Path | None = None):
    frames = []
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
        self.rest_anchor = -0.5
        self.z = 0.14
        self.mode_q0 = 0.12

        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_ground_plane()

        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        self.body = builder.add_body_elastic(
            xform=wp.transform(wp.vec3(abs(self.rest_anchor) + self.mode_q0, 0.0, self.z), wp.quat_identity()),
            mass=1.0,
            inertia=inertia,
            mode_count=1,
            mode_mass=[0.04],
            mode_stiffness=[12.0],
            mode_damping=[0.25],
            mode_q=[self.mode_q0],
            mode_shape_fn=self._mode_shape,
            label="elastic_endpoint_fixture",
        )

        shape_cfg = newton.ModelBuilder.ShapeConfig()
        shape_cfg.density = 0.0
        builder.add_shape_box(self.body, hx=self.length / 2.0, hy=0.035, hz=0.025, cfg=shape_cfg)
        self.joint = builder.add_joint_revolute(
            parent=-1,
            child=self.body,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, self.z), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(self.rest_anchor, 0.0, 0.0), wp.quat_identity()),
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
        self.viewer.set_camera(wp.vec3(0.58, -1.15, 0.65), -28.0, 90.0)

    def _mode_shape(self, x: np.ndarray) -> np.ndarray:
        return np.array([[x[0] / abs(self.rest_anchor), 0.0, 0.0]], dtype=np.float32)

    def _mode_value(self) -> float:
        return float(self.state_0.joint_q.numpy()[self.elastic_q_start + 7])

    def step(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            self.state_0.clear_forces()
            self._joint_f[:] = 0.0
            self._joint_f[self.elastic_qd_start + 6] = 1.4 * math.sin(5.0 * t)
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
    viewer.close()

    print(f"Wrote {assets / 'reduced_elastic_fourbar.mp4'}")
    print(f"Wrote {assets / 'elastic_revolute_endpoint_fixture.mp4'}")


if __name__ == "__main__":
    main()
