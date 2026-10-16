# Full-Opening Glasses Case

Portable packaging of the latest published full-width, fixed-C1 flap revision.
The rear sewn edge is 146 mm wide, expanding to 150 mm over the opening;
scalloped shoulders start at 74% of material length. The rounded tongue ends
in a nondegenerate 3 mm chord. This is not the earlier narrowed flap.

All required model geometry and bases are procedural in the local modules.
No mesh downloads, persisted trajectories, or raw model assets are required.
The original 37 x 67 full-thickness grid, six flap modes, three wall modes,
18 g equal-nodal flap mass, four tip attachments, and solver settings remain.

## Commands

Run from the repository root with its example dependencies installed
(`uv sync --extra examples --extra dev`). Full simulation requires CUDA graph
capture. The original four-substep, 16-iteration, 360-frame setup is retained:

```bash
uv run python reports/everyday_rom/scenes/glasses_case/clamped_case.py --output outputs/glasses-case/candidate
uv run python reports/everyday_rom/scenes/glasses_case/clamped_case.py --output outputs/glasses-case/no-tip --no-tip-attachment
uv run python reports/everyday_rom/scenes/glasses_case/clamped_case.py --output outputs/glasses-case/no-close --no-close
uv run python reports/everyday_rom/scenes/glasses_case/audit_revision.py outputs/glasses-case
uv run python reports/everyday_rom/scenes/glasses_case/clamped_case.py --help
CUDA_VISIBLE_DEVICES= MODAL_PATH_TEST_DEVICE=cpu uv run python -m unittest discover -s reports/everyday_rom/scenes/glasses_case -p 'test_*.py' -v
```

Use fresh output directories. `--device` selects the CUDA device, and
`--cache-dir` optionally sets the Warp kernel cache; otherwise it uses
`WARP_CACHE_PATH` or an output-local cache. Runtime Git information is recorded
when available, without requiring a particular machine path, clean worktree,
or unchanged Git HEAD after packaging commits. No physics parameters changed.

## Research-Only Extension

`solver_vbd_modal_path.py` and `pod_path_energy.py` preserve the published
source AST, allowing only formatting and the repository hook's Warp array
argument-annotation syntax update. Tests normalize only that call/subscript
form; kernel bodies, dtypes, dimensions, and runtime constructors stay strict.
Original byte hashes and the verified original AST hashes remain in
`sources.json`; packaged byte hashes track the formatted copies.
The local `SolverVBDModalPath` subclasses public
`SolverVBD` but overrides the **private** `_launch_elastic_body_solve` hook,
tested against Newton `8228bf6f690fff4dbb359d5de6447b09cb77c1e7` and Warp 1.17.0.
It is research-only, not a supported Newton extension API. Keep it local;
do not merge it into `newton/`. Run the retained assembly/integration tests
before using another runtime revision.

The passive ordered-POD prior is not calibrated leather. Retaining equal nodal
mass while increasing area changes the implied areal density. Rear rigid
frames are fixed; modal coordinates remain solved. Only the visible rigid tip
clamp is driven. Four explicit attachment constraints transfer force; jaw
collision is disabled consistently. A rigid internal former substitutes for
unsupported elastic-elastic contact. No optimized runtime or fast variant is
included. Surface strain diagnostics are not calibrated material stress.

Full-mouth projected coverage is tested in `test_opening_coverage.py` and every
closed-hold frame in the saved-state audit. It is **not** a watertight seal or
contact-pressure certificate: inherited endcaps leave approximately 6.13 mm
of side clearance below the closed flap.

Retained CPU tests cover coverage counterexamples, full-width geometry, fixed
root position/tangent, mass normalization, strain, contact capacity, nonlinear
energy, and solver coupling. Two historical-evidence tests are retained but
skip unless `CASE_HISTORY_ROOT` names an external `c1-revision` directory with
the original failed trial and sibling `accepted/` trajectory. These historical
artifacts are intentionally not bundled. The CUDA graph test also skips in CPU
mode; no full physical or GPU validation is implied by CPU packaging tests.

`glasses_case.py` retains only the unchanged authored wall-field helper; the
current runnable scene is `clamped_case.py`. `banana.py` is only the common
progression helper needed by retained geometry tests, not another scene.
`sources.json` records original origins/hashes and portability changes. No
videos, HTML, trajectories, historical experiment drivers, or raw NIMBLE assets
are included.
