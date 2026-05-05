# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import newton
import newton.viewer
from newton.examples.basic.example_basic_reduced_elastic_beam import Example as BeamExample
from newton.examples.basic.example_basic_reduced_elastic_beam_vibration import Example as BeamVibrationExample
from newton.examples.basic.example_basic_reduced_elastic_bellcrank import Example as BellcrankExample
from newton.examples.basic.example_basic_reduced_elastic_cantilever_weight import Example as CantileverWeightExample
from newton.examples.basic.example_basic_reduced_elastic_crank_slider import Example as CrankSliderExample
from newton.examples.basic.example_basic_reduced_elastic_dipper import Example as DipperExample
from newton.examples.basic.example_basic_reduced_elastic_fourbar import Example as FourbarExample
from newton.examples.basic.example_basic_reduced_elastic_matrix_rom import Example as MatrixROMExample
from newton.examples.basic.example_basic_reduced_elastic_prismatic import Example as PrismaticExample
from newton.examples.basic.example_basic_reduced_elastic_torsion import Example as TorsionExample
from newton.examples.basic.example_basic_reduced_elastic_vertical_weight import Example as VerticalWeightExample
from newton.examples.basic.example_basic_reduced_elastic_watt import Example as WattExample

WIDTH = 960
HEIGHT = 540
FPS = 60
VIDEO_ASSETS = (
    "reduced_elastic_fourbar.mp4",
    "elastic_revolute_endpoint_fixture.mp4",
    "reduced_elastic_prismatic_compression.mp4",
    "reduced_elastic_crank_slider.mp4",
    "reduced_elastic_watt_linkage.mp4",
    "reduced_elastic_bellcrank.mp4",
    "reduced_elastic_matrix_rom.mp4",
    "reduced_elastic_dipper_arm.mp4",
    "reduced_elastic_cantilever_beam.mp4",
    "reduced_elastic_cantilever_vibration.mp4",
    "reduced_elastic_cantilever_weight.mp4",
    "reduced_elastic_vertical_weight.mp4",
)
MODE_COUNTS_START = "<!-- MODE_COUNTS_START -->"
MODE_COUNTS_END = "<!-- MODE_COUNTS_END -->"


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


def _update_report_mode_counts(report_path: Path, rows: list[tuple[str, list[int]]]):
    text = report_path.read_text()
    table_rows = []
    for label, counts in rows:
        total = sum(counts)
        if len(counts) > 1:
            breakdown = " + ".join(str(count) for count in counts)
        elif counts:
            breakdown = str(counts[0])
        else:
            breakdown = "0"
        table_rows.append(f"        <tr><td>{escape(label)}</td><td>{total}</td><td>{escape(breakdown)}</td></tr>")
    table_html = "\n".join(table_rows)
    pattern = re.compile(rf"{re.escape(MODE_COUNTS_START)}.*?{re.escape(MODE_COUNTS_END)}", re.S)
    replacement = f"{MODE_COUNTS_START}\n{table_html}\n{MODE_COUNTS_END}"
    if not pattern.search(text):
        raise RuntimeError("report is missing mode count markers")
    report_path.write_text(pattern.sub(replacement, text))


def _elastic_mode_counts(example) -> list[int]:
    model = example.model
    elastic_body_count = int(getattr(model, "elastic_body_count", 0))
    if elastic_body_count == 0 or getattr(model, "elastic_mode_count", None) is None:
        return []
    return [int(count) for count in model.elastic_mode_count.numpy()[:elastic_body_count]]


def _prepare_example(label: str, example, mode_rows: list[tuple[str, list[int]]]):
    example.viewer.show_elastic_strain = bool(getattr(example, "show_elastic_strain", False))
    if not example.viewer.show_elastic_strain:
        example.viewer.elastic_strain_color_max = None
    counts = _elastic_mode_counts(example)
    total = sum(counts)
    unit = "mode" if total == 1 else "modes"
    if len(counts) > 1:
        print(f"{label}: {total} {unit} ({' + '.join(str(count) for count in counts)} per elastic body)")
    else:
        print(f"{label}: {total} {unit}")
    mode_rows.append((label, counts))
    return example


def _render_example(
    viewer,
    mode_rows: list[tuple[str, list[int]]],
    label: str,
    example_cls,
    frame_count: int,
    video_path: Path,
    screenshot_path: Path,
    screenshot_frame: int | None = None,
):
    if viewer.model is not None:
        viewer.clear_model()

    example = _prepare_example(label, example_cls(viewer, None), mode_rows)
    frames = _capture(
        example,
        viewer,
        frame_count=frame_count,
        screenshot_path=screenshot_path,
        screenshot_frame=screenshot_frame,
    )
    _write_video(video_path, frames)


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
    mode_rows: list[tuple[str, list[int]]] = []

    viewer = newton.viewer.ViewerGL(width=WIDTH, height=HEIGHT, headless=True)
    _render_example(
        viewer,
        mode_rows,
        "Elastic 4-Bar Linkage",
        FourbarExample,
        frame_count=180,
        video_path=assets / "reduced_elastic_fourbar.mp4",
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_fourbar.jpg"),
    )

    _render_example(
        viewer,
        mode_rows,
        "Revolute Torsion Fixture",
        TorsionExample,
        frame_count=150,
        video_path=assets / "elastic_revolute_endpoint_fixture.mp4",
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_torsion.jpg"),
        screenshot_frame=15,
    )

    _render_example(
        viewer,
        mode_rows,
        "Prismatic Compression Fixture",
        PrismaticExample,
        frame_count=180,
        video_path=assets / "reduced_elastic_prismatic_compression.mp4",
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_prismatic.jpg"),
        screenshot_frame=120,
    )

    _render_example(
        viewer,
        mode_rows,
        "Elastic Slider-Crank",
        CrankSliderExample,
        frame_count=180,
        video_path=assets / "reduced_elastic_crank_slider.mp4",
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_crank_slider.jpg"),
        screenshot_frame=95,
    )

    _render_example(
        viewer,
        mode_rows,
        "Elastic Watt Linkage",
        WattExample,
        frame_count=300,
        video_path=assets / "reduced_elastic_watt_linkage.mp4",
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_watt.jpg"),
        screenshot_frame=180,
    )

    _render_example(
        viewer,
        mode_rows,
        "Elastic Bell-Crank Transfer",
        BellcrankExample,
        frame_count=300,
        video_path=assets / "reduced_elastic_bellcrank.mp4",
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_bellcrank.jpg"),
        screenshot_frame=180,
    )

    _render_example(
        viewer,
        mode_rows,
        "Matrix ROM Handling Beam",
        MatrixROMExample,
        frame_count=300,
        video_path=assets / "reduced_elastic_matrix_rom.mp4",
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_matrix_rom.jpg"),
        screenshot_frame=150,
    )

    _render_example(
        viewer,
        mode_rows,
        "Flexible Dipper Arm",
        DipperExample,
        frame_count=240,
        video_path=assets / "reduced_elastic_dipper_arm.mp4",
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_dipper.jpg"),
        screenshot_frame=150,
    )

    _render_example(
        viewer,
        mode_rows,
        "Cantilever Tip Load",
        BeamExample,
        frame_count=150,
        video_path=assets / "reduced_elastic_cantilever_beam.mp4",
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_beam.jpg"),
    )

    _render_example(
        viewer,
        mode_rows,
        "Cantilever Vibration",
        BeamVibrationExample,
        frame_count=180,
        video_path=assets / "reduced_elastic_cantilever_vibration.mp4",
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_beam_vibration.jpg"),
        screenshot_frame=8,
    )

    _render_example(
        viewer,
        mode_rows,
        "Cantilever With Rigid Weight",
        CantileverWeightExample,
        frame_count=180,
        video_path=assets / "reduced_elastic_cantilever_weight.mp4",
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_cantilever_weight.jpg"),
        screenshot_frame=60,
    )

    _render_example(
        viewer,
        mode_rows,
        "Vertical Suspended Weight",
        VerticalWeightExample,
        frame_count=180,
        video_path=assets / "reduced_elastic_vertical_weight.mp4",
        screenshot_path=Path("docs/images/examples/example_basic_reduced_elastic_vertical_weight.jpg"),
        screenshot_frame=60,
    )
    viewer.close()
    report_path = root / "reduced_elastic_links_implementation.html"
    _update_report_mode_counts(report_path, mode_rows)
    _update_report_video_cache_busters(report_path, cache_key)

    print(f"Wrote {assets / 'reduced_elastic_fourbar.mp4'}")
    print(f"Wrote {assets / 'elastic_revolute_endpoint_fixture.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_prismatic_compression.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_crank_slider.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_watt_linkage.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_bellcrank.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_matrix_rom.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_dipper_arm.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_cantilever_beam.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_cantilever_vibration.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_cantilever_weight.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_vertical_weight.mp4'}")
    print(f"Updated report video cache key: {cache_key}")


if __name__ == "__main__":
    main()
