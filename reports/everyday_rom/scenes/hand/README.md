# Pose-ROM Hand

This is the published 48-mode NIMBLE-derived hand. Run from the checkout root:

```bash
uv run python reports/everyday_rom/scenes/hand/expressive.py validate --substeps 2 --iterations 12 --output output/everyday-rom/hand
uv run python reports/everyday_rom/scenes/hand/expressive.py benchmark --substeps 2 --iterations 12 --output output/everyday-rom/hand-timing
uv run python -m unittest discover -s reports/everyday_rom/scenes/hand -p 'test_*.py' -v
```

Use a fresh output directory. `validate` runs actuation and no-actuation arms,
with every-substep geometry checks and per-finger release/recovery checks.
`benchmark` uses CUDA graphs with auditing, export and rendering excluded.
The report video used four substeps and twenty iterations; select those explicitly
to reproduce its settings. The 2x12 settings passed the published optimized audit.

The wrist support is kinematic. Finite modal forces `K*q_target(t)` curl each
finger independently, then vanish for passive recovery. Neither modal state nor
surface vertices are prescribed. This is an assumed 10 MPa, 1 mm shell with
0.35 kg mass, not a calibrated anatomical, muscle or self-contact model.

`assets/rom-48.npz` contains the derived mass-normalized basis and surface;
`assets/targets.npz` contains five derived modal-force targets. Raw NIMBLE models,
training poses, textures and checkpoints are excluded. See [ATTRIBUTION.md](ATTRIBUTION.md).
The offline fitting helper requires SciPy and separately authorized native input;
neither is needed to run the supplied ROM. `individual.py` retains earlier contact
helper code, but `expressive.py` is the supported entry point for this snapshot.

Published report: https://reports.mmacklin.com/everyday-rom/index.html#hand
