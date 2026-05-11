# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
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
from newton.examples.basic.example_basic_reduced_elastic_chair_stick_slip import Example as ChairStickSlipExample
from newton.examples.basic.example_basic_reduced_elastic_crank_slider import Example as CrankSliderExample
from newton.examples.basic.example_basic_reduced_elastic_dipper import Example as DipperExample
from newton.examples.basic.example_basic_reduced_elastic_fourbar import Example as FourbarExample
from newton.examples.basic.example_basic_reduced_elastic_gripper_contact import Example as GripperContactExample
from newton.examples.basic.example_basic_reduced_elastic_matrix_rom import Example as MatrixROMExample
from newton.examples.basic.example_basic_reduced_elastic_prismatic import Example as PrismaticExample
from newton.examples.basic.example_basic_reduced_elastic_scraper_contact import Example as ScraperContactExample
from newton.examples.basic.example_basic_reduced_elastic_torsion import Example as TorsionExample
from newton.examples.basic.example_basic_reduced_elastic_vertical_weight import Example as VerticalWeightExample
from newton.examples.basic.example_basic_reduced_elastic_wall_contact import Example as WallContactExample
from newton.examples.basic.example_basic_reduced_elastic_watt import Example as WattExample
from newton.examples.robot.example_robot_ur10_elastic_panel import Example as UR10PanelExample

WIDTH = 960
HEIGHT = 540
FPS = 60
UR10_PANEL_WIDTH = 1920
UR10_PANEL_HEIGHT = 1080
UR10_PANEL_FRAME_COUNT = 240
VIDEO_ASSETS = (
    "reduced_elastic_fourbar.mp4",
    "elastic_revolute_endpoint_fixture.mp4",
    "reduced_elastic_prismatic_compression.mp4",
    "reduced_elastic_crank_slider.mp4",
    "reduced_elastic_watt_linkage.mp4",
    "reduced_elastic_bellcrank.mp4",
    "reduced_elastic_matrix_rom.mp4",
    "reduced_elastic_ur10_panel.mp4",
    "reduced_elastic_dipper_arm.mp4",
    "reduced_elastic_wall_contact.mp4",
    "reduced_elastic_gripper_contact.mp4",
    "reduced_elastic_scraper_contact.mp4",
    "reduced_elastic_chair_stick_slip.mp4",
    "reduced_elastic_chair_mode_exercise.mp4",
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
    width: int | None = None,
    height: int | None = None,
):
    render_viewer = viewer
    owns_viewer = width is not None or height is not None
    if owns_viewer:
        render_viewer = newton.viewer.ViewerGL(width=width or WIDTH, height=height or HEIGHT, headless=True)

    if render_viewer.model is not None:
        render_viewer.clear_model()

    example = _prepare_example(label, example_cls(render_viewer, None), mode_rows)
    frames = _capture(
        example,
        render_viewer,
        frame_count=frame_count,
        screenshot_path=screenshot_path,
        screenshot_frame=screenshot_frame,
    )
    _write_video(video_path, frames)
    if owns_viewer:
        render_viewer.close()


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


def _render_chair_mode_exercise(
    viewer,
    mode_rows: list[tuple[str, list[int]]],
    video_path: Path,
):
    if viewer.model is not None:
        viewer.clear_model()

    example = _prepare_example("Plastic Chair Mode Exercise", ChairStickSlipExample(viewer, None), mode_rows)
    viewer.show_elastic_strain = True
    viewer.elastic_strain_color_max = 0.07

    frame_count = 180
    mode_count = int(example.model.elastic_mode_count.numpy()[0])
    q_start = example.elastic_q_start + 7
    qd_start = int(example.model.joint_qd_start.numpy()[int(example.model.elastic_joint.numpy()[0])]) + 6
    amplitudes = np.linspace(0.075, 0.035, mode_count, dtype=np.float32)
    frames = []

    for i in range(frame_count):
        t = i / max(frame_count - 1, 1)
        active = min(int(t * mode_count), mode_count - 1)
        local_t = t * mode_count - active
        envelope = math.sin(math.pi * local_t)

        q = example.state_0.joint_q.numpy()
        qd = example.state_0.joint_qd.numpy()
        q[q_start : q_start + mode_count] = 0.0
        qd[qd_start : qd_start + mode_count] = 0.0
        q[q_start + active] = float(amplitudes[active] * envelope)

        # Add a low-amplitude blended pass near the end so the full basis can be
        # inspected as a coupled deformation rather than only one mode at a time.
        if t > 0.78:
            blend = (t - 0.78) / 0.22
            for mode in range(mode_count):
                phase = 2.0 * math.pi * (2.0 * blend + 0.13 * mode)
                q[q_start + mode] += float(0.020 * math.sin(phase) / (1.0 + 0.12 * mode))

        example.state_0.joint_q.assign(q)
        example.state_0.joint_qd.assign(qd)
        example.render()
        frames.append(viewer.get_frame().numpy())

    stacked = np.asarray(frames)
    if int(stacked.max()) == int(stacked.min()):
        raise RuntimeError("captured chair mode exercise frames are blank")
    _write_video(video_path, frames)


def parse_args():
    parser = argparse.ArgumentParser(description="Render reduced elastic example videos for a local report.")
    parser.add_argument(
        "--assets",
        type=Path,
        default=None,
        help="Directory for generated videos. Defaults to reports/assets next to this script.",
    )
    parser.add_argument(
        "--screenshots",
        type=Path,
        default=None,
        help="Directory for generated screenshots. Defaults to <assets>/screenshots.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional existing HTML report whose video cache keys and mode counts should be updated.",
    )
    return parser.parse_args()


def main():
    root = Path(__file__).resolve().parent
    args = parse_args()
    assets = args.assets if args.assets is not None else root / "assets"
    screenshots = args.screenshots if args.screenshots is not None else assets / "screenshots"
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
        screenshot_path=screenshots / "example_basic_reduced_elastic_fourbar.jpg",
    )

    _render_example(
        viewer,
        mode_rows,
        "Revolute Torsion Fixture",
        TorsionExample,
        frame_count=150,
        video_path=assets / "elastic_revolute_endpoint_fixture.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_torsion.jpg",
        screenshot_frame=15,
    )

    _render_example(
        viewer,
        mode_rows,
        "Prismatic Compression Fixture",
        PrismaticExample,
        frame_count=180,
        video_path=assets / "reduced_elastic_prismatic_compression.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_prismatic.jpg",
        screenshot_frame=120,
    )

    _render_example(
        viewer,
        mode_rows,
        "Elastic Slider-Crank",
        CrankSliderExample,
        frame_count=180,
        video_path=assets / "reduced_elastic_crank_slider.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_crank_slider.jpg",
        screenshot_frame=95,
    )

    _render_example(
        viewer,
        mode_rows,
        "Elastic Watt Linkage",
        WattExample,
        frame_count=300,
        video_path=assets / "reduced_elastic_watt_linkage.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_watt.jpg",
        screenshot_frame=180,
    )

    _render_example(
        viewer,
        mode_rows,
        "Elastic Bell-Crank Transfer",
        BellcrankExample,
        frame_count=300,
        video_path=assets / "reduced_elastic_bellcrank.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_bellcrank.jpg",
        screenshot_frame=180,
    )

    _render_example(
        viewer,
        mode_rows,
        "Matrix ROM Handling Beam",
        MatrixROMExample,
        frame_count=300,
        video_path=assets / "reduced_elastic_matrix_rom.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_matrix_rom.jpg",
        screenshot_frame=150,
    )

    _render_example(
        viewer,
        mode_rows,
        "UR10 Elastic Car Panel",
        UR10PanelExample,
        frame_count=UR10_PANEL_FRAME_COUNT,
        video_path=assets / "reduced_elastic_ur10_panel.mp4",
        screenshot_path=screenshots / "example_robot_ur10_elastic_panel.jpg",
        screenshot_frame=UR10_PANEL_FRAME_COUNT // 2,
        width=UR10_PANEL_WIDTH,
        height=UR10_PANEL_HEIGHT,
    )

    _render_example(
        viewer,
        mode_rows,
        "Flexible Dipper Arm",
        DipperExample,
        frame_count=240,
        video_path=assets / "reduced_elastic_dipper_arm.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_dipper.jpg",
        screenshot_frame=150,
    )

    _render_example(
        viewer,
        mode_rows,
        "Wall Pad Contact",
        WallContactExample,
        frame_count=120,
        video_path=assets / "reduced_elastic_wall_contact.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_wall_contact.jpg",
        screenshot_frame=80,
    )

    _render_example(
        viewer,
        mode_rows,
        "Two-Gripper Contact",
        GripperContactExample,
        frame_count=150,
        video_path=assets / "reduced_elastic_gripper_contact.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_gripper_contact.jpg",
        screenshot_frame=110,
    )

    _render_example(
        viewer,
        mode_rows,
        "Scraper Contact",
        ScraperContactExample,
        frame_count=150,
        video_path=assets / "reduced_elastic_scraper_contact.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_scraper_contact.jpg",
        screenshot_frame=100,
    )

    _render_example(
        viewer,
        mode_rows,
        "Plastic Chair Stick-Slip",
        ChairStickSlipExample,
        frame_count=360,
        video_path=assets / "reduced_elastic_chair_stick_slip.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_chair_stick_slip.jpg",
        screenshot_frame=230,
    )

    _render_chair_mode_exercise(
        viewer,
        mode_rows,
        video_path=assets / "reduced_elastic_chair_mode_exercise.mp4",
    )

    _render_example(
        viewer,
        mode_rows,
        "Cantilever Tip Load",
        BeamExample,
        frame_count=150,
        video_path=assets / "reduced_elastic_cantilever_beam.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_beam.jpg",
    )

    _render_example(
        viewer,
        mode_rows,
        "Cantilever Vibration",
        BeamVibrationExample,
        frame_count=180,
        video_path=assets / "reduced_elastic_cantilever_vibration.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_beam_vibration.jpg",
        screenshot_frame=8,
    )

    _render_example(
        viewer,
        mode_rows,
        "Cantilever With Rigid Weight",
        CantileverWeightExample,
        frame_count=180,
        video_path=assets / "reduced_elastic_cantilever_weight.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_cantilever_weight.jpg",
        screenshot_frame=60,
    )

    _render_example(
        viewer,
        mode_rows,
        "Vertical Suspended Weight",
        VerticalWeightExample,
        frame_count=180,
        video_path=assets / "reduced_elastic_vertical_weight.mp4",
        screenshot_path=screenshots / "example_basic_reduced_elastic_vertical_weight.jpg",
        screenshot_frame=60,
    )
    viewer.close()
    report_path = args.report if args.report is not None else root / "reduced_elastic_links_implementation.html"
    if report_path.exists():
        _update_report_mode_counts(report_path, mode_rows)
        _update_report_video_cache_busters(report_path, cache_key)
        print(f"Updated report video cache key: {cache_key}")
    else:
        print(f"Skipped report update because {report_path} does not exist")

    print(f"Wrote {assets / 'reduced_elastic_fourbar.mp4'}")
    print(f"Wrote {assets / 'elastic_revolute_endpoint_fixture.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_prismatic_compression.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_crank_slider.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_watt_linkage.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_bellcrank.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_matrix_rom.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_ur10_panel.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_dipper_arm.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_wall_contact.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_gripper_contact.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_scraper_contact.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_chair_stick_slip.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_chair_mode_exercise.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_cantilever_beam.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_cantilever_vibration.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_cantilever_weight.mp4'}")
    print(f"Wrote {assets / 'reduced_elastic_vertical_weight.mp4'}")


if __name__ == "__main__":
    main()
