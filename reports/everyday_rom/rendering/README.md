# Exact-State Rendering And Strain Replay

Source-only tools for rendering saved `scene.json` / `scene.npz` surfaces with
ovrtx and replaying the **same trajectory** first normally, then with principal
surface strain. No simulation modules are imported or executed. This directory
contains no videos, HTML, saved trajectories, or trained models.

`source_manifest.json` identifies the origin workspace, its base revision, each
copied source snapshot's SHA256, the packaged hashes, and every adaptation.
The origin snapshots were uncommitted: their file hashes, not the base revision
alone, identify the implementation. `surface_strain.py` and `strain_adapters.py`
are byte-identical to those snapshots. Rendering, strain math, source geometry,
physics gates, camera behavior, and readback tolerances are retained.

## Setup

Run from the checkout root:

```bash
uv sync --extra examples --extra rtx
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
uv run --no-sync python -m reports.everyday_rom.rendering.verify_sources
```

CPU preparation needs NumPy and Pillow. CPU USD tests also need `pxr`, supplied
by the importers extra on supported platforms. Install system `ffmpeg`,
`ffprobe`, and the DejaVu Sans font (`DejaVuSans.ttf`, for example from
`fonts-dejavu-core` on Debian/Ubuntu). The original renderer used ovrtx
`0.3.0.312915`; the explicit render commands below select that version without
changing the project's dependency files. An RTX-capable NVIDIA GPU and compatible
driver are required for rendering, but not for preparation, help, or CPU tests.
No optional renderer is imported by the CPU entry points or `--help`.

## Input Contract

The renderer is scene-generic: it does not choose an object, solver, or runtime.
Use a **new owned copy**, never a sealed input directory, as its output directory.

- `scene.json`: positive `fps`, `camera.position`, `camera.target`, optional
  `camera.fov` (default 38 degrees), and `meshes`. Each mesh has a unique `name`
  and `vertices` / `faces` array keys; normal materials may supply `color`,
  `roughness`, `metallic`, and `opacity`.
- `scene.npz`: finite `times` of shape `(F,)`, triangle indices `(T,3)`, and
  vertices `(N,3)` or `(F,N,3)`. At least two frames and one moving mesh are
  required. Optional per-frame RGB vertex colors have shape `(F,N,3)`.
- `metrics.json`: original, unmodified physical results. A full video requires
  `test_passed` and `negative_control_passed` to be exactly true. Missing or
  failed checks are not repaired here. Diagnostic `--frame N` previews do not
  constitute validation; the retained `fixture_only` exemption is for synthetic
  tests only, never for converting failed physical evidence into an accepted video.
- Strain additionally requires material-rest vertices and matching topology,
  plus the saved basis/state files used by its adapter, or an explicit reference
  contract. Frame zero is **not** a material-rest substitute.

Tracking cameras use `camera.track_meshes` and `camera.reference_frame`; both
halves retain the same camera motion. Frames are not dropped, interpolated,
resimulated, or deformation-amplified. Renderer point/color readback must match
the saved arrays within `2e-7`; independent saved-state surface recovery in
strain preparation uses `2e-6` metres.

## Normal Then Strain

For a generic normal render, `render_scene.py` combines fresh-directory staging
and rendering. It accepts any scene matching the contract, without an object
adapter. This command is GPU work; acquire the shared lock as shown:

```bash
flock -w 1800 "${GPU_LOCK:-/tmp/newton-everyday-rom-gpu.lock}" \
  timeout 8m uv run --no-sync --with 'ovrtx==0.3.0.312915' \
  python reports/everyday_rom/rendering/render_scene.py generated/my-scene \
  --output render-work/my-scene/normal
```

The module form `python -m reports.everyday_rom.rendering.render_scene` also
works. It retains defaults 1280x960 and 96 samples, and accepts `--width`,
`--height`, `--samples`, and diagnostic `--frame`. All original full-video
physics gates still apply. `--help` is CPU-only.

Example using a completed cup/lid export, with its documented fixed 0-50% display
scale. Choose the object and fixed scale documented for your own source; the
renderer has no default strain scale. All output directories below must be new
and outside `SOURCE`. `stage-normal` verifies recognized incoming seals and
copies all source files and attribution byte-for-byte, so use a focused export
directory rather than an entire experiment tree.

```bash
SOURCE="$PWD/generated/cup-lid"
WORK="$PWD/render-work/cup-lid"
mkdir -p "$WORK"

uv run --no-sync python -m reports.everyday_rom.rendering.strain_publication \
  stage-normal "$SOURCE" "$WORK/normal"

# One GPU job per lock acquisition; execution timeout starts after the lock.
# Use the same GPU_LOCK as other local GPU workers.
render_saved() {
  flock -w 1800 "${GPU_LOCK:-/tmp/newton-everyday-rom-gpu.lock}" \
    timeout 8m uv run --no-sync --with 'ovrtx==0.3.0.312915' \
    python -m reports.everyday_rom.rendering.render_strain_replay "$@"
}
render_saved "$WORK/normal" --normal

uv run --no-sync python -m reports.everyday_rom.rendering.strain_replay \
  prepare "$WORK/normal" "$WORK/strain" --object cup-lid --scale-max 0.50
uv run --no-sync python -m reports.everyday_rom.rendering.strain_replay \
  verify "$WORK/strain"
uv run --no-sync python -m reports.everyday_rom.rendering.strain_overlay \
  "$WORK/strain" "$WORK/strain/overlay-audit.json" --anchor top-left
render_saved "$WORK/strain" --legend-anchor top-left

uv run --no-sync python -m reports.everyday_rom.rendering.strain_video \
  plan "$WORK/strain" "$WORK/strain" "$WORK/replay-plan"
uv run --no-sync python -m reports.everyday_rom.rendering.strain_video \
  concat "$WORK/replay-plan"
uv run --no-sync python -m reports.everyday_rom.rendering.strain_publication \
  package "$WORK/replay-plan" "$WORK/object"
uv run --no-sync python -m reports.everyday_rom.rendering.strain_publication \
  verify "$WORK/object" --full-decode
```

The primary result is `WORK/object/replay/rtx.mp4`. It has exactly 2F frames:
NORMAL, then STRAIN. `plan` verifies both existing render audits, source hashes,
camera, scale, frame times and complete decoded clips without writing video.
`concat` uses stream copy if compatible, otherwise lossless H.264 reencoding;
every decoded RGB24 frame of both halves must remain identical. A final file is
promoted from `rtx.partial.mp4` only after these checks pass. `package` and
`verify` are local artifact helpers, not publication or network commands.

Default render settings remain 1280x960, 96 samples, path tracing with denoising.
For another resolution, pass identical `--width` and `--height` to both overlay
audit and renderer. If the overlay intersects an object, select another
`--anchor` / `--legend-anchor` pair and audit again; do not change the saved
camera or geometry to conceal overlap. Each frame is checked, excluding only
floor background. For a diagnostic still, append `--frame 0` to a render command.
That is also GPU work and should use its own lock acquisition.

To reuse a normal clip instead, prepare from the existing normal directory that
contains its original `rtx.json`, `scene.usdc`, and `rtx.mp4`; do not render into
that sealed directory. The planner verifies its hashes and decoded appearance.
Historical headphone-only `--accepted-headphones` reuse is retained and checks
the original exact 0-3% local-state strain; it is not a general bypass.

## References And Relocation

Built-in adapters cover `clog`, `glasses-case`, `toothbrush`, `drink-bottle`,
`hand`, `straw`, `spatula`, `banana`, `headphones`, and `cup-lid`. Common aliases
include `case`, `brush`, `drinkbottle`, and `cup`. They consume the saved schemas
without importing scene or solver code. Use `--references references.json` if
your scene exporter uses another layout:

```json
{
  "schema_version": 1,
  "object": "cup-lid",
  "surfaces": [{
    "mesh": "elastic_lid",
    "rest": {"file": "reference.npz", "key": "rest"},
    "faces": {"file": "reference.npz", "key": "faces"},
    "provenance": "Undeformed runtime skin in original vertex order"
  }],
  "uncolored": {"cup": "Rigid support"}
}
```

Reference paths are relative to that JSON. Every mesh must be classified exactly
once. Optional `local` and `poses` references provide independent recovery:
`local` is `(F,N,3)`; `poses` is `(F,7)` in `[px,py,pz,qx,qy,qz,qw]` order.
Omitting them is recorded as no independent recovery check, not a successful one.

New preparation and video plans record resolved input paths as provenance, not
hardcoded workspace dependencies. Generate new plans at their final working
location; do not rewrite their source paths or hashes after moving them.
Packaged object verification is self-contained. For a relocated prepared strain
bundle whose original inputs are unavailable, use the explicit mode:

```bash
uv run --no-sync python -m reports.everyday_rom.rendering.strain_replay \
  verify "$WORK/strain" --offline-bundle
uv run --no-sync python -m reports.everyday_rom.rendering.strain_replay \
  combine-scene "$WORK/strain" --offline-bundle
```

This retains all output hashes, implementation snapshots, geometry, field and
color checks, but cannot recheck original external inputs. Renderer
`--offline-bundle` has the same semantics and records the reduced verification
scope. The optional `strain/replay/scene.{json,npz}` has both exact geometry
halves in one scene; the two-clip route above preserves the normal video's
original appearance. Neither route changes physical acceptance.

## Interpretation And Tests

The signed principal engineering strains are `sqrt(eigvalsh(F.T @ F)) - 1`.
Colors are the rest-area weighted vertex average of the largest absolute
principal strain, on a fixed scale shared by every colored part and frame.
Values above the scale are clipped **only in color**; raw signed fields remain.
Rigid rotations do not create strain, and coincident disconnected components
are not welded. This is surface kinematics, not bending curvature, calibrated
stress, or material validity. Cup/lid includes full-skin and separate midsurface
fields; its inherited independent 15% strain-screen failure remains a warning,
not hidden by the 50% display scale or its passing original contact gates.

Saved native videos do not establish optimized-runtime trajectory equivalence.
These tools make no new speed claim or blanket physical-pass claim. Existing
failed metrics are preserved, and rendering cannot validate a different run.

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m unittest discover \
  -s reports/everyday_rom/rendering -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m \
  reports.everyday_rom.rendering.render_strain_replay --help
```

Tests cover strain analytics, all ten adapter contracts, exact geometry/time
replay, CPU USD bindings, stale-source rejection, seal copying, overlay geometry,
decoded concat identity, relocated module/script entry points, and unchanged
physics gates. Temporary synthetic media are CPU-generated and deleted.
USD and ffmpeg integration tests explicitly skip if their dependencies are
unavailable. This source packaging was CPU-checked; no GPU render, new trajectory,
publication, or solver qualification is implied.
