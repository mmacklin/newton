# Published Drink Bottle

Run from the repository root at Newton pin `8228bf6f690fff4dbb359d5de6447b09cb77c1e7`:

```sh
PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/drink_bottle/bottles.py --slug drink-bottle --device cuda:0 --frames 360 --substeps 4 --iterations 16 --validate --output output/drink-bottle
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run --extra examples python -m unittest discover -s reports/everyday_rom/scenes/drink_bottle -p 'test_*.py' -v
```

Published floating bottle with four modes, original finite-mass support/drop, materials and compliant ALM. Geometry, basis and full surface map are generated procedurally. No native optimization or fast configuration is included. The shared source also contains the mustard spec, needed by its original geometry tests. Default output is `output/drink-bottle`.

`sources.json` records every published origin and hash; `portability.patch` contains only output/reproduction/test-data paths. Public Newton imports are used; contact and modal diagnostics depend on SolverVBD implementation arrays. CUDA is required for the original graph execution.

CPU tests and `--help` are recorded in `validation.json`; no fresh GPU physics or rendering is certified. Original physical/control/strain/export checks remain intact. The two evidence-only tests in `TestBottleExports` skip when generated/prior data is absent. Their optional `BOTTLE_TEST_ROOT` and `BOTTLE_PRIOR_ROOT` default to `output`; historical clamped/rejected trajectories are intentionally not bundled. No real CPU failure is skipped. No saved trajectories, media or raw private assets are included.
