# Everyday Reduced-Elastic Objects

Ten research examples from the [everyday ROM report](https://reports.mmacklin.com/everyday-rom/index.html),
with procedural geometry, required derived assets, checks, and exact-state rendering
tools. These are uncalibrated demonstrations, not material-validated digital twins.
This branch adds report-local examples; it does not change the Newton solver.

## Setup

```bash
git clone --branch codex/everyday-rom https://github.com/mmacklin/newton.git
cd newton
uv sync --extra examples --extra rtx
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1
```

The runtime matches reduced-links commit `8228bf6f690fff4dbb359d5de6447b09cb77c1e7`.
The original runs used Warp 1.17.0 and ovrtx 0.3.0.312915 on a 1g.24gb MIG
partition of an RTX PRO 6000 Blackwell Server Edition. CUDA is needed for the
graph-based simulation commands. RTX rendering requires a supported NVIDIA GPU;
install system `ffmpeg` and `ffprobe` for video encoding and verification.
For ViewerGL on Linux, use `export __GLX_VENDOR_LIBRARY_NAME=nvidia`.

## Scenes

Each directory has its own README with runnable commands, parameters, and scope.
Run those commands from the checkout root. Each source snapshot carries its own
helpers to preserve the geometry and physics used by the published scene.

| Scene | Source / Assets |
| --- | --- |
| Rounded clog, compression and recoil | [clog](scenes/clog/README.md) |
| Scalloped glasses case, bending flap | [glasses_case](scenes/glasses_case/README.md) |
| Toothbrush, full bristle-field contact | [toothbrush](scenes/toothbrush/README.md) |
| Squeeze drink bottle | [drink_bottle](scenes/drink_bottle/README.md) |
| Cup lid, angled insertion and stirring | [cup_lid](scenes/cup_lid/README.md) |
| NIMBLE-derived hand, individual finger forces | [hand](scenes/hand/README.md) |
| Bent drinking straw | [straw](scenes/straw/README.md) |
| Flexible spatula and bowl | [spatula](scenes/spatula/README.md) |
| Banana, three continuous peeling strips | [banana](scenes/banana/README.md) |
| Headphones, flexible band and rod cable | [headphones](scenes/headphones/README.md) |

The hand contains an attributed derived ROM, not original NIMBLE model files,
poses or textures. See its [attribution](scenes/hand/ATTRIBUTION.md).
The banana and case use a local passive nonlinear modal-path energy extension;
it is research scaffolding, not a supported core solver API. All other scenes use
the existing reduced-link runtime. Ongoing optimization prototypes and unvalidated
larger-motion trials are intentionally excluded.

## Validation And Performance

```bash
uv run python reports/everyday_rom/check_bundle.py
uv run python reports/everyday_rom/run_tests.py
```

Unit tests are run per directory in separate processes to isolate helper-module
names. Saved-trajectory checks explicitly skip when their generated data is absent;
they are not evidence of a fresh simulation passing. Each scene README describes
how to generate and validate those data. Preserve failed runs and use fresh output
directories. Do not weaken the physical checks to obtain a faster measurement.

The selected gallery's videos and optimized measurements are source/configuration
bound; the portable example commands in this checkout can differ. As of 2026-10-17,
all **10/10 selected scenes** meet the scoped 4x real-time timing target. Timings
are simulation-only CUDA graph measurements, excluding rendering, recording and
diagnostic readback. These are audited gallery results, not benchmark claims for
this checkout.

### Current Gallery And Portable Runtime

The selected gallery as of 2026-10-17 uses separately qualified runtime overlays.
This checkout retains the baseline Newton runtime and the commands documented in
each scene. The table is a comparison, **not** a set of newly qualified portable
presets. Changing an iteration count alone does not reproduce a selected solver,
physical result, saved trajectory or timing.

| Scene | Portable example command | Selected gallery |
| --- | --- | --- |
| Clog | 4 substeps x 24 iterations, relaxation 0.6 | Same geometry/budget; canonical-FP32 native overlay |
| Glasses case | 4x16, relaxation 0.7 | 2x12, relaxation 0.7; same C1/scalloped geometry |
| Toothbrush | Original 4x24/0.3; qualified generic fused 4x16/0.4 | Fused 4x16/0.4 with private persistent soft-dual optimization |
| Drink bottle | 4x16 | 2x12; same geometry/materials |
| Cup lid | 4x12 | Same source/configuration; selected combined native overlay |
| Hand | 48 modes, 2x12, 50 ms release | Same derived assets/motion/configuration; separate optimized runtime |
| Straw | 4x24, relaxation 0.25 | 4x8, relaxation 0.75; same rest geometry |
| Spatula | Original motion, 4x16, relaxation 0.6 | 25 mm attached lift, coupled17 4x12/0.9; composed native overlay, pair counters OFF |
| Banana | 4x16, relaxation 1 | 2x6, relaxation 1; identical ordered ROM |
| Headphones | 4x20, relaxation 0.8 | 4x2 with selected solver/normal-conditioning overlay; same 12 rods |

In particular, this checkout does **not** implement or qualify the selected
spatula's 25 mm lift from 2.9 to 3.2 seconds before release, nor does it contain
the selected native overlays. The gallery's audited counter-OFF measurement is
[4.12058212631579 ms](https://reports.mmacklin.com/everyday-rom/spatula/published-measurement.json),
below the 4x target of 4.166667 ms, with matching 390-frame normal then 390-frame
strain media. Native pair-counter outputs are disabled during measurement;
counter values are unavailable, not reported as zero, and no cost is subtracted.
Historical numerical/history failures and the contact-free gate passing at exactly
two unloaded frames remain disclosed. No all-tests-pass or full-history equivalence
is implied. Use the report's current source/configuration and failure disclosures
rather than attaching its measurements to these portable commands.

`assets.json` and scene source manifests preserve the original content hashes.
When a live gallery path has been replaced, a hash-verified archive URL identifies
the packaged historical snapshot; it is not relabeled as the new selected source.
`original_source_url` retains an earlier URL and `archived_source_url` supplements
an original machine path. The banana ROM now lives under `banana/provenance/`;
the toothbrush frozen basis is an explicitly historical finite-neck reference.
Neither asset has been regenerated or replaced. Media and raw trajectories remain
external generated evidence, not additional Git assets.

## Rendering And Reports

See [rendering](rendering/README.md) for ovrtx rendering of saved Newton surfaces
and a normal-then-strain replay of exactly the same trajectory. Principal surface
strain is a kinematic diagnostic, not calibrated stress. In particular, the cup
lid exceeds the independent small-strain screen; its README retains that warning.

`build_report.py REPORT_ROOT` rebuilds the original gallery from scene artifacts
under `REPORT_ROOT/<scene-slug>/`, including scene-specific MP4 download names.
It does not run simulations or invent missing validation. Optional detailed
performance and historical evidence pages are not generated by this script.
Use existing artifacts from the public report or generate fresh ones using each
scene's instructions. Do not attach old audits or timings to new trajectories.

Rendered HTML, videos, saved trajectories, raw third-party inputs and benchmark
histories are deliberately excluded from Git. Publish generated reports under
`~/reports/<slug>/` (or another static host); the current report remains at
https://reports.mmacklin.com/everyday-rom/index.html.
