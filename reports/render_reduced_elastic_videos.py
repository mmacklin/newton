# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import newton
import newton.viewer
from newton.examples.basic.example_basic_reduced_elastic_beam import Example as BeamExample
from newton.examples.basic.example_basic_reduced_elastic_beam_vibration import Example as BeamVibrationExample
from newton.examples.basic.example_basic_reduced_elastic_fourbar import Example as FourbarExample
from newton.examples.basic.example_basic_reduced_elastic_torsion import Example as TorsionExample

WIDTH = 960
HEIGHT = 540
FPS = 60
VIDEO_ASSETS = (
    "reduced_elastic_fourbar.mp4",
    "elastic_revolute_endpoint_fixture.mp4",
    "reduced_elastic_cantilever_beam.mp4",
    "reduced_elastic_cantilever_vibration.mp4",
)


def _write_video(path: Path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def _update_report_video_cache_busters(report_path: Path, cache_key: str):
    text = report_path.read_text()
    for video_name in VIDEO_ASSETS:
        text = re.sub(
            rf'(data-video-src|src)="assets/{re.escape(video_name)}(?:\?[^"]*)?"',
            rf'\1="assets/{video_name}?datetime={cache_key}"',
            text,
        )
    report_path.write_text(text)


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


def main():
    root = Path(__file__).resolve().parent
    assets = root / "assets"
    cache_key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

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
    fixture = TorsionExample(viewer, None)
    frames = _capture(
        fixture,
        viewer,
        frame_count=150,
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_torsion.jpg"),
    )
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
    _update_report_video_cache_busters(root / "reduced_elastic_links_implementation.html", cache_key)

    print(f"Wrote {assets / 'reduced_elastic_fourbar.mp4'}")
    print(f"Wrote {assets / 'elastic_revolute_endpoint_fixture.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_cantilever_beam.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_cantilever_vibration.mp4'}")
    print(f"Updated report video cache key: {cache_key}")


if __name__ == "__main__":
    main()
