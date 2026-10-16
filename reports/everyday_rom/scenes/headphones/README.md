# Published Headphones

Run from the repository root at Newton pin `8228bf6f690fff4dbb359d5de6447b09cb77c1e7`:

```sh
PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/headphones/headphones.py --duration 15 --output output/headphones
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run --extra examples python -m unittest discover -s reports/everyday_rom/scenes/headphones -p 'test_*.py' -v
```

The normal command runs the full 900-frame positive and no-contact control at the original 4 substeps and 20 iterations on CUDA. It preserves the original two-mode headband, cups, jacks, 12-segment cable, joints, mass/inertia and motions. It does not include cable coarsening or any native optimization. All assets are procedural; default output is `output/headphones`.

`sources.json` records published source provenance and `portability.patch` contains only output/reproduction path edits. Public Newton imports are used, but contact audits/modal diagnostics read SolverVBD implementation arrays. The scene requires CUDA for its original graph workflow; its CLI has no device override. `--quick` intentionally omits control qualification and is not a validated replacement command.

CPU geometry/mass/topology tests and `--help` are recorded in `validation.json`. No fresh full GPU run or rendering was performed. Original predrive, contact, recovery, drop, surface-reconstruction and negative-control gates are retained, not weakened. No archived trajectory, video, HTML or private asset is included.
