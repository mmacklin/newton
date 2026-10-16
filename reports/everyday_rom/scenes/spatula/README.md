# Published Spatula

Run from the repository root at Newton pin `8228bf6f690fff4dbb359d5de6447b09cb77c1e7`:

```sh
PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/spatula/spatula.py --device cuda:0 --frames 390 --substeps 4 --iterations 16 --validate --output output/spatula
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run --extra examples python -m unittest discover -s reports/everyday_rom/scenes/spatula -p 'test_*.py' -v
```

Published five-mode flexible blade, finite-mass handle, rigid bowl, original grasp/release and material/contact settings. Geometry, basis and full surface deformation are procedural. The published shared tests import their local `straw.py` companion; no straw variant is promoted. Default output is `output/spatula`.

`sources.json` binds the scene's own published source revision, not the different straw helper. `portability.patch` changes only output/reproduction/test-data/local-validator paths. Public Newton imports are used; graph-safe contact audits and modal diagnostics read SolverVBD implementation arrays. No experimental native code, floor-gap change, fast setting or model coarsening is included.

CPU tests and `--help` are recorded in `validation.json`. Original positive/control, grasp/release, penetration, strain, recovery and export assertions remain unchanged; no fresh full GPU run or rendering was done. The retained local schema validator needs NumPy, while optional RTX rendering needs `ovrtx`, USD and video tooling. No trajectories, videos, HTML or private data are included.
