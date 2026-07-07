# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the reduced-elastic floating-frame inertial coupling example videos.

Captures each coupling example headlessly through ViewerGL, writes an MP4 plus a
mid-run screenshot into reports/assets, and (given --report) refreshes the
report's video cache-busting query strings and the mode-count table.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import newton
import newton.viewer
from newton.examples.basic.example_basic_reduced_elastic_angular_frame_coupling import Example as AngularFrameCoupling
from newton.examples.basic.example_basic_reduced_elastic_base_excitation import Example as BaseExcitation
from newton.examples.basic.example_basic_reduced_elastic_base_rotation import Example as BaseRotation
from newton.examples.basic.example_basic_reduced_elastic_centrifugal import Example as Centrifugal
from newton.examples.basic.example_basic_reduced_elastic_coriolis import Example as Coriolis
from newton.examples.basic.example_basic_reduced_elastic_frame_coupling import Example as FrameCoupling
from newton.examples.basic.example_basic_reduced_elastic_gravity_coupling import Example as GravityCoupling

WIDTH = 960
HEIGHT = 540
FPS = 60

MODE_COUNTS_START = "<!-- MODE_COUNTS_START -->"
MODE_COUNTS_END = "<!-- MODE_COUNTS_END -->"

EXAMPLES = (
    ("Gravity and Translational Coupling", GravityCoupling, "reduced_elastic_gravity_coupling", 180, 120),
    ("Base Excitation", BaseExcitation, "reduced_elastic_base_excitation", 240, 150),
    ("Base Rotation (Euler)", BaseRotation, "reduced_elastic_base_rotation", 240, 150),
    ("Centrifugal", Centrifugal, "reduced_elastic_centrifugal", 240, 200),
    ("Coriolis", Coriolis, "reduced_elastic_coriolis", 300, 200),
    ("Frame Coupling (linear recoil)", FrameCoupling, "reduced_elastic_frame_coupling", 180, 60),
    ("Angular Frame Coupling", AngularFrameCoupling, "reduced_elastic_angular_frame_coupling", 240, 120),
)


def _write_video(path: Path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def _elastic_mode_counts(model) -> list[int]:
    elastic_body_count = int(getattr(model, "elastic_body_count", 0))
    if elastic_body_count == 0 or getattr(model, "elastic_mode_count", None) is None:
        return []
    return [int(count) for count in model.elastic_mode_count.numpy()[:elastic_body_count]]


def _capture(example, viewer, frame_count, screenshot_path, screenshot_frame):
    frames = []
    for i in range(frame_count):
        example.step()
        example.render()
        frames.append(viewer.get_frame().numpy())
        if i == screenshot_frame:
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(screenshot_path, frames[-1])
    stacked = np.asarray(frames)
    if int(stacked.max()) == int(stacked.min()):
        raise RuntimeError("captured frames are blank")
    return frames


def _update_cache_busters(report_path: Path, video_names, cache_key):
    text = report_path.read_text()
    for video_name in video_names:
        text = re.sub(
            rf'(data-video-src|src)="assets/{re.escape(video_name)}\.mp4(?:\?[^"]*)?"',
            rf'\1="assets/{video_name}.mp4?datetime={cache_key}"',
            text,
        )
    report_path.write_text(text)


def _update_mode_counts(report_path: Path, rows):
    text = report_path.read_text()
    table_rows = []
    for label, counts in rows:
        total = sum(counts)
        breakdown = " + ".join(str(c) for c in counts) if len(counts) > 1 else (str(counts[0]) if counts else "0")
        table_rows.append(f"        <tr><td>{escape(label)}</td><td>{total}</td><td>{escape(breakdown)}</td></tr>")
    table_html = "\n".join(table_rows)
    pattern = re.compile(rf"{re.escape(MODE_COUNTS_START)}.*?{re.escape(MODE_COUNTS_END)}", re.S)
    if not pattern.search(text):
        print("report has no mode count markers; skipping mode-count table update")
        return
    report_path.write_text(pattern.sub(f"{MODE_COUNTS_START}\n{table_html}\n{MODE_COUNTS_END}", text))


def parse_args():
    parser = argparse.ArgumentParser(description="Render reduced elastic inertial coupling example videos.")
    parser.add_argument("--assets", type=Path, default=None)
    parser.add_argument("--screenshots", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main():
    root = Path(__file__).resolve().parent
    args = parse_args()
    assets = args.assets if args.assets is not None else root / "assets"
    screenshots = args.screenshots if args.screenshots is not None else assets / "screenshots"
    cache_key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    viewer = newton.viewer.ViewerGL(width=WIDTH, height=HEIGHT, headless=True)
    mode_rows = []
    for label, example_cls, video_name, frame_count, screenshot_frame in EXAMPLES:
        if viewer.model is not None:
            viewer.clear_model()
        example = example_cls(viewer, None)
        frames = _capture(
            example,
            viewer,
            frame_count,
            screenshots / f"{video_name}.jpg",
            screenshot_frame,
        )
        _write_video(assets / f"{video_name}.mp4", frames)
        counts = _elastic_mode_counts(example.model)
        mode_rows.append((label, counts))
        print(f"Wrote {assets / f'{video_name}.mp4'}  ({sum(counts)} modes)")
    viewer.close()

    report_path = args.report if args.report is not None else root / "reduced_elastic_inertial_coupling.html"
    if report_path.exists():
        _update_mode_counts(report_path, mode_rows)
        _update_cache_busters(report_path, [name for _, _, name, _, _ in EXAMPLES], cache_key)
        print(f"Updated report cache key: {cache_key}")
    else:
        print(f"Skipped report update because {report_path} does not exist")


if __name__ == "__main__":
    main()
