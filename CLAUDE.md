@AGENTS.md

## Headless Video Recording

Newton's `ViewerGL` supports headless rendering on this machine (KasmVNC on `:99`). Use this to capture screenshots or MP4 videos of simulations without a visible window.

### Setup

```bash
uv sync --extra examples           # installs pyglet for ViewerGL
uv add --dev "imageio[ffmpeg]"     # installs imageio + bundled ffmpeg for MP4 encoding
```

### Rendering Pipeline

```python
import os, imageio, warp as wp, newton
from newton.viewer import ViewerGL

os.environ["DISPLAY"] = ":99"  # required — KasmVNC X server

# one viewer instance, reuse across scenes (creating multiple causes GL context conflicts)
viewer = ViewerGL(width=960, height=720, headless=True)
viewer.set_model(model)  # call before set_camera — set_model recreates camera with model's up_axis
viewer.set_camera(pos=wp.vec3(0, 1.5, 4), pitch=-5.0, yaw=-90.0)

# per-frame render loop
viewer.begin_frame(sim_time)
viewer.log_state(state)                              # renders body shapes
viewer.log_lines("cable", starts, ends,              # optional: debug lines
                 colors=(1.0, 0.3, 0.1), width=0.008)
viewer.end_frame()

# capture frame as numpy (H, W, 3) uint8
frame_buf = viewer.get_frame(target_image=frame_buf)  # reuse buffer across frames
frame_np = frame_buf.numpy()
```

### Writing MP4

```python
writer = imageio.get_writer(
    "output.mp4", fps=60, codec="libx264",
    output_params=["-crf", "20", "-pix_fmt", "yuv420p"],
)
# in your sim loop:
writer.append_data(frame_np)
# after loop:
writer.close()
```

### Key Gotchas

- **Single ViewerGL instance**: creating a second `ViewerGL` triggers `GL_INVALID_OPERATION` (0x502) due to conflicting GL contexts. Reuse one viewer and call `set_model()` to switch scenes.
- **`set_model` before `set_camera`**: `set_model()` recreates the camera with the model's `up_axis`. Camera settings applied before `set_model()` are lost.
- **`DISPLAY=:99`**: must be set before importing pyglet. Without it, pyglet cannot create an OpenGL context.
- **CUDA/GL interop warning**: "Could not register GL buffer" is expected on this machine — it falls back to CPU-side copy. Frames still render correctly.
- **No MSAA**: "Could not get MSAA config" is normal in headless mode — renders without anti-aliasing.
