# Published Toothbrush

Run from the repository root at Newton pin `8228bf6f690fff4dbb359d5de6447b09cb77c1e7`:

```sh
PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/toothbrush/toothbrush.py --device cuda:0 --frames 450 --substeps 4 --iterations 24 --negative-control --output output/toothbrush
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/toothbrush/refine_render.py --physical output/toothbrush --output output/toothbrush-smooth
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run --extra examples python -m unittest discover -s reports/everyday_rom/scenes/toothbrush -p 'test_*.py' -v
```

This is the published 20-tuft, 40-mode full-field scrub, with original geometry, mass/inertia, operators, material, 0.15 mm tool gap and relaxation 0.3. It is not a fast configuration or sparse/fused native experiment. All simulation and surface-deformation inputs are procedural. Default output is `output/toothbrush`.

`frozen/basis.npz` (472,130 bytes) and `frozen/scene.json` are exact copies of the published **frozen** reference for the original-physics CPU test, not the current simulation/render export. The basis was not trimmed or recompressed and is never loaded by the simulator. Its source URL is `https://reports.mmacklin.com/everyday-rom/toothbrush/frozen/basis.npz`. No frozen trajectory/metrics history is included.

The second command reproduces the latest smooth plastic representation from the **fresh** 450-frame simulator output: two Loop subdivision levels, 14,080 handle triangles and 8,960 plastic-head triangles. `subdivision.py` contains the published functions unchanged; the NumPy quaternion operation order and original 5e-8 m all-frame embedding check are retained. Tuft deformation arrays stay byte-identical and the plastic head's projected modal field must remain exactly zero. Physics inputs, collision samples, mass, inertia, M/K/C and state are not edited. `presentation.json` preserves the latest published camera/poster configuration.

The output must be absent or empty, separate from the input directory, and must not overlap source files. Already refined input is rejected. Fresh input must report passing original positive/control gates and match local simulation source hashes; unlike the old saved-run adapter, this path does not demand archived equilibrium coordinates. The input basis/state hashes are recorded, and the full fresh physical basis is retained under the new render output's `physics/`. No extra binary reference or saved movie state is needed. Synthetic 450-frame CPU tests check mapping, both triangle counts, unchanged tuft arrays/inputs and rejection controls; they are not full physical qualification.

`sources.json` binds published origins/hashes and the unchanged subdivision/quaternion function ASTs. `portability.patch` records output/command/fixture changes and the local fresh-export refinement adapter. Public Newton imports are used; graph-safe contact capacity and field audits access SolverVBD implementation contact arrays. The local source entry point replaces only the old machine-bound provenance launcher, not simulation arithmetic.

CPU tests and `--help` are recorded in `validation.json`. Original full positive/control, penetration, coverage, mass/operator and recovery assertions remain unchanged. Fresh GPU physics and RTX output were not tested. Archived trajectory-dependent field-coverage tests and the generic reproduction framework are not included; the copied simulator retains its actual coverage gates. No saved state, videos, HTML or private data is packaged.
