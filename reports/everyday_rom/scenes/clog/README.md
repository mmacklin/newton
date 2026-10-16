# Published Clog

Run from the repository root at Newton pin `8228bf6f690fff4dbb359d5de6447b09cb77c1e7`:

```sh
PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/clog/clog.py --device cuda:0 --frames 360 --substeps 4 --iterations 24 --with-control --output output/clog
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run --extra examples python -m unittest discover -s reports/everyday_rom/scenes/clog -p 'test_*.py' -v
```

Published distal-toe clog, 12 modes, original material/contact/drive and relaxation 0.6. Geometry, basis, mass and inertia are procedural; no external asset is required. Default output is `output/clog`, relative to the working directory. `cap.py` is retained because the published shared helper snapshots it and the shared CPU tests import it; it is not a new selected scene.

`sources.json` maps every copied file to its published origin and both hashes. `portability.patch` contains only path/command edits. No persistent solver or performance configuration is included. Public Newton imports are used; the contact audit and modal diagnostics depend on SolverVBD implementation arrays, including `body_body_contact_counts`, `body_body_contact_overflow_max` and modal block metrics.

CPU tests and `--help` are recorded in `validation.json`; no fresh GPU trajectory is certified. Original positive/control, prepress, strain and surface-recovery assertions are unchanged and can fail; CPU success is not a waiver or a physical pass. The local renderer is retained for its NumPy schema validator; optional RTX rendering needs separately installed `ovrtx`, USD and video tooling and was not tested. No saved trajectory or media is packaged.
