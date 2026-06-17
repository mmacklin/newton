# VBD Sparse Articulation Report Reproduction

These scripts reproduce the measurements and ViewerGL media summarized in:

```text
https://reports.mmacklin.com/vbd-sparse-articulation/
```

Run commands from the repository root.

## Synthetic CPU Timing And Residuals

```bash
uv run reports/vbd_sparse_articulation/bench_vbd_sparse_articulation.py \
  --device cpu \
  --scenarios single chain_fixed chain_revolute loop_fixed \
  --body-counts 4 32 64 \
  --modes local block_sparse_joints \
  --steps 10 \
  --warmup 3 \
  --iterations 1 \
  --json reports/vbd_sparse_articulation/bench_cpu_sparse.json

uv run reports/vbd_sparse_articulation/bench_vbd_sparse_articulation.py \
  --device cpu \
  --scenarios chain_fixed chain_revolute loop_fixed \
  --body-counts 128 \
  --modes local block_sparse_joints \
  --steps 10 \
  --warmup 3 \
  --iterations 1 \
  --json reports/vbd_sparse_articulation/bench_cpu_sparse_128.json
```

## Residual Versus Internal VBD Iterations

```bash
uv run reports/vbd_sparse_articulation/sweep_vbd_sparse_iterations.py \
  --device cpu \
  --scenarios chain_fixed chain_revolute loop_fixed \
  --body-counts 4 32 64 \
  --iterations 1 2 4 8 \
  --modes local block_sparse_joints \
  --json reports/vbd_sparse_articulation/iteration_sweep_cpu_sparse.json \
  --markdown reports/vbd_sparse_articulation/iteration_sweep_cpu_sparse.md
```

## Diagonal Contact Check

```bash
uv run reports/vbd_sparse_articulation/bench_vbd_sparse_articulation.py \
  --device cpu \
  --scenarios contact_stack \
  --body-counts 4 16 \
  --modes local block_sparse_joints \
  --steps 10 \
  --warmup 3 \
  --iterations 2 \
  --json reports/vbd_sparse_articulation/bench_cpu_sparse_contacts.json
```

## Humanoid Contact Smoke

```bash
uv run reports/vbd_sparse_articulation/bench_vbd_humanoid_contact.py \
  --robots h1 g1 \
  --steps 90 \
  --iterations 3 \
  --contact-mode ground \
  --joint-stiffness fixed_high \
  --output reports/vbd_sparse_articulation/humanoid_contact_fixed_high_cpu.json
```

ViewerGL videos require a working GL context. On the CPU-only host used for the
report, this was run under Xvfb/software GL:

```bash
env -u __GLX_VENDOR_LIBRARY_NAME LIBGL_ALWAYS_SOFTWARE=1 PYOPENGL_PLATFORM=glx \
  xvfb-run -a -s "-screen 0 960x540x24" \
  uv run --with 'imageio[ffmpeg]' --with pillow \
  reports/vbd_sparse_articulation/render_humanoid_contact_videos.py \
  --robots h1 g1 \
  --modes local block_sparse_joints \
  --joint-stiffness fixed_high \
  --steps 90 \
  --iterations 3 \
  --output-dir reports/vbd_sparse_articulation/videos
```

## XY Table Selected Validation

The final report keeps only the selected XY table validation, not the diagnostic
parameter sweeps:

```bash
env -u __GLX_VENDOR_LIBRARY_NAME LIBGL_ALWAYS_SOFTWARE=1 PYOPENGL_PLATFORM=glx \
  xvfb-run -a -s "-screen 0 960x540x24" \
  uv run --with 'imageio[ffmpeg]' --with pillow \
  reports/vbd_sparse_articulation/render_cable_cross_slide_video.py \
  --output-dir reports/vbd_sparse_articulation/videos
```

The script defaults match the reported selected configuration:
`block_sparse_joints`, `rigid_articulation_relaxation=0.65`,
`sim_substeps=10`, `sim_iterations=5`, and `cable_bend_stiffness=1e-5`.

## Focused Regression Tests

```bash
uv run --extra dev -m newton.tests -k test_vbd_sparse_articulation
```
