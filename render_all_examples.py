"""Render all tendon examples headless to MP4."""

import importlib
import os
import traceback

os.environ["DISPLAY"] = ":99"

import imageio
import numpy as np

from newton.viewer import ViewerGL

EXAMPLES = [
    ("tendon_pulley", "newton.examples.cable.example_tendon_pulley", 240),
    ("tendon_rolling_pulley", "newton.examples.cable.example_tendon_rolling_pulley", 180),
    ("tendon_compound_pulley", "newton.examples.cable.example_tendon_compound_pulley", 220),
    ("tendon_cable_machine", "newton.examples.cable.example_tendon_cable_machine", 100),
    ("tendon_3d_routing", "newton.examples.cable.example_tendon_3d_routing", 140),
    ("tendon_xy_table", "newton.examples.cable.example_tendon_xy_table", 480),
]

NUM_FRAMES = 240
FPS = 60
REPORT_DIR = os.path.expanduser("~/reports/cable-sim-research")
os.makedirs(REPORT_DIR, exist_ok=True)


class FakeArgs:
    def __init__(self, num_frames):
        self.headless = True
        self.record = False
        self.num_frames = num_frames
        self.episode_frames = None


def render_example(name, module_path, num_frames, viewer):
    print(f"\n{'=' * 60}")
    print(f"Rendering: {name}")
    print(f"{'=' * 60}")

    mod = importlib.import_module(module_path)
    example = mod.Example(viewer, FakeArgs(num_frames))

    mp4_path = os.path.join(REPORT_DIR, f"{name}.mp4")
    writer = imageio.get_writer(
        mp4_path,
        fps=FPS,
        codec="libx264",
        output_params=["-crf", "20", "-pix_fmt", "yuv420p"],
    )

    frame_buf = None
    last_frame_np = None
    for frame in range(num_frames):
        example.step()

        if hasattr(example, "state_0") and not np.isfinite(example.state_0.body_q.numpy()).all():
            print(f"  frame {frame}: non-finite body state; freezing remaining diagnostic frames")
            if last_frame_np is None:
                break
            for freeze_frame in range(frame, num_frames):
                writer.append_data(last_frame_np)
                if freeze_frame % 60 == 0:
                    print(f"  frame {freeze_frame}/{num_frames}")
            break

        example.render()

        frame_buf = viewer.get_frame(target_image=frame_buf)
        frame_np = frame_buf.numpy()
        writer.append_data(frame_np)
        last_frame_np = np.array(frame_np, copy=True)

        if frame % 60 == 0:
            print(f"  frame {frame}/{num_frames}")

    writer.close()
    print(f"  Saved: {mp4_path}")
    return mp4_path


def main():
    viewer = ViewerGL(width=960, height=720, headless=True)

    paths = {}
    for name, module_path, num_frames in EXAMPLES:
        try:
            mp4_path = render_example(name, module_path, num_frames, viewer)
            paths[name] = mp4_path
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("Done. Videos:")
    for name, path in paths.items():
        url = f"https://reports.mmacklin.com/cable-sim-research/{os.path.basename(path)}"
        print(f"  {name}: {url}")


if __name__ == "__main__":
    main()
