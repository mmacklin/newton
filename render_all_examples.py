"""Render all tendon examples headless to MP4."""

import os
import sys
import importlib

os.environ["DISPLAY"] = ":99"

import imageio
import numpy as np
import warp as wp

from newton.viewer import ViewerGL

EXAMPLES = [
    ("tendon_pulley", "newton.examples.cable.example_tendon_pulley"),
    ("tendon_rolling_pulley", "newton.examples.cable.example_tendon_rolling_pulley"),
    ("tendon_compound_pulley", "newton.examples.cable.example_tendon_compound_pulley"),
    ("tendon_cable_machine", "newton.examples.cable.example_tendon_cable_machine"),
    ("tendon_3d_routing", "newton.examples.cable.example_tendon_3d_routing"),
]

NUM_FRAMES = 240
FPS = 60
REPORT_DIR = os.path.expanduser("~/reports/cable-sim-research")
os.makedirs(REPORT_DIR, exist_ok=True)


class FakeArgs:
    headless = True
    record = False
    num_frames = NUM_FRAMES
    episode_frames = None


def render_example(name, module_path, viewer):
    print(f"\n{'='*60}")
    print(f"Rendering: {name}")
    print(f"{'='*60}")

    mod = importlib.import_module(module_path)
    example = mod.Example(viewer, FakeArgs())

    mp4_path = os.path.join(REPORT_DIR, f"{name}.mp4")
    writer = imageio.get_writer(
        mp4_path, fps=FPS, codec="libx264",
        output_params=["-crf", "20", "-pix_fmt", "yuv420p"],
    )

    frame_buf = None
    for frame in range(NUM_FRAMES):
        example.step()
        example.render()

        frame_buf = viewer.get_frame(target_image=frame_buf)
        frame_np = frame_buf.numpy()
        writer.append_data(frame_np)

        if frame % 60 == 0:
            print(f"  frame {frame}/{NUM_FRAMES}")

    writer.close()
    print(f"  Saved: {mp4_path}")
    return mp4_path


def main():
    viewer = ViewerGL(width=960, height=720, headless=True)

    paths = {}
    for name, module_path in EXAMPLES:
        try:
            mp4_path = render_example(name, module_path, viewer)
            paths[name] = mp4_path
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Done. Videos:")
    for name, path in paths.items():
        url = f"https://companion.mmacklin.com/cable-sim-research/{os.path.basename(path)}"
        print(f"  {name}: {url}")


if __name__ == "__main__":
    main()
