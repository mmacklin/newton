# Ordered-POD Banana

Portable packaging of the published whole-peel banana. Run `path_banana.py`,
not the old linear-shell experiment. `assets/ordered_rom.npz` is the exact
published derived ROM: SHA256
`5c7766fe2b931f3cbc06914378484933a451c832dbd7fa7296c248bb25c19f77`.
It contains three 32-mode peels, their rest meshes, masses, ordered coefficients,
and progress knots. No training trajectories are needed to run it.

## Commands

Run from the repository root with its example dependencies installed
(`uv sync --extra examples --extra dev`). Full simulation requires CUDA because
the authoritative example captures a CUDA graph. The published run uses 600
frames, four substeps, 16 iterations, and an 8-second pull:

```bash
uv run python reports/everyday_rom/scenes/banana/path_banana.py reports/everyday_rom/scenes/banana/assets/ordered_rom.npz outputs/banana --frames 600 --iterations 16 --pull-duration 8
uv run python reports/everyday_rom/scenes/banana/path_banana.py reports/everyday_rom/scenes/banana/assets/ordered_rom.npz outputs/banana-no-drive --frames 600 --iterations 16 --pull-duration 8 --no-drive
uv run python reports/everyday_rom/scenes/banana/path_banana.py --help
CUDA_VISIBLE_DEVICES= MODAL_PATH_TEST_DEVICE=cpu uv run python -m unittest discover -s reports/everyday_rom/scenes/banana -p 'test_*.py' -v
```

Use fresh output directories. Helpers resolve beside the script, with no
host-specific imports or asset paths. `build_banana_path.py` retains the
optional offline fitting utility and override tests; regeneration requires
explicit `--sources STRIP0_DIR STRIP1_DIR STRIP2_DIR` containing training data,
which is intentionally not distributed here. The supplied ROM should be used
for reproduction rather than retraining a different basis.

## Research-Only Extension

`solver_vbd_modal_path.py` and `pod_path_energy.py` preserve the published
source AST, allowing only formatting and the repository hook's Warp array
argument-annotation syntax update. Tests normalize only that call/subscript
form; kernel bodies, dtypes, dimensions, and runtime constructors stay strict.
Original byte hashes and the verified original AST hashes remain in
`sources.json`; packaged byte hashes track the formatted copies.
The local `SolverVBDModalPath` subclasses public
`SolverVBD` but overrides its **private** `_launch_elastic_body_solve` hook,
tested against Newton `8228bf6f690fff4dbb359d5de6447b09cb77c1e7` and Warp 1.17.0.
This is not a supported Newton extension API. Keep it scene-local; it is not a
patch to, or proposed merge into, `newton/`. Future runtime revisions may break
this private contract; run the assembly and integration tests first.

All three peels retain their physical frame/modal unknowns, the original
passive nonlinear ordered-path energy, damping, contacts, and pin constraints.
Only the visible rigid tip grippers are driven. The fruit is rigid and fixed.
This is a bounded shape-prior experiment, not calibrated shell mechanics,
adhesion, fracture, or a guarantee outside the trained span. No optimized
runtime or fast preset is included.

The copied runtime exports diagnostic trajectories; its original WIP/status
fields and absence of a built-in paired physical-acceptance verdict are retained.
The published paired acceptance evidence, report/video tooling, and saved
trajectories are not bundled. CPU tests cover POD fitting, energy derivatives,
geometry, retention constraints, assembly, and solver coupling; they do not
repeat the published full physical validation. The CUDA graph unit test skips
in CPU mode and can be exercised with `MODAL_PATH_TEST_DEVICE=cuda:0`.

`banana.py` contains only the unchanged gripper/progression helpers needed by
this scene and its tests. `retained_geometry.py` preserves the pure retention
helpers without importing historical full-order experiment drivers.
`sources.json` records every original origin/hash and all portability changes.
No videos, HTML, saved simulation trajectories, historical experiments, or raw
NIMBLE assets are included.
