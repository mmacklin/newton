# Published Straw

Run from the repository root at Newton pin `8228bf6f690fff4dbb359d5de6447b09cb77c1e7`:

```sh
PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/straw/straw.py --device cuda:0 --frames 390 --substeps 4 --iterations 24 --validate --output output/straw
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run --extra examples python -m unittest discover -s reports/everyday_rom/scenes/straw -p 'test_*.py' -v
```

Published hollow bent straw with four modes, floating-frame projected basis, original root support/release, material, mass/inertia and motion. All meshes/operators are procedural. The shared CPU suite imports its local `spatula.py` companion; that is not a new scene selection. Default output is `output/straw`.

`sources.json` maps files to this scene's published snapshot, distinct from the older helper retained by spatula. `portability.patch` changes only output, reproduction, test-temporary and local-validator paths. Public Newton imports are used; capacity/modal diagnostics depend on SolverVBD implementation arrays. No experimental solver, coarse geometry or fast setting is included.

CPU tests and `--help` are recorded in `validation.json`. Original full positive/control, support/release, recovery, strain and export gates remain unchanged. No fresh GPU trajectory or RTX rendering was performed. Archived `--diagnose-straw`/`--compare-boosts` utilities still require excluded historical checkpoints; they are not standalone reproduction commands. The local schema validator needs only NumPy; optional rendering needs `ovrtx`, USD and video tooling. No saved trajectory/media is included.
