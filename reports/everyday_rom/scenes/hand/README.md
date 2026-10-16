# Pose-ROM Hand

This is the conditioned 48-mode NIMBLE-derived hand, revision
`larger-motion-hand-release50-20261016`. Run from the checkout root:

```bash
uv run python reports/everyday_rom/scenes/hand/expressive.py validate --substeps 2 --iterations 12 --output output/everyday-rom/hand
uv run python reports/everyday_rom/scenes/hand/expressive.py benchmark --substeps 2 --iterations 12 --output output/everyday-rom/hand-timing
uv run python -m unittest discover -s reports/everyday_rom/scenes/hand -p 'test_*.py' -v
```

Use a fresh output directory. `validate` runs actuation and no-actuation arms,
with every-substep geometry, equilibrium, saved-force and per-finger
release/recovery checks. Defaults are two substeps and twelve iterations,
720 frames per arm and 1,440 checked post-solve states per arm.
`benchmark` uses CUDA graphs with auditing, export and rendering excluded.
The accepted release50 trajectory also used 2x12; its timing belongs to its
separately fingerprinted optimized runtime, not automatically to this checkout.

The wrist support is kinematic. Finite modal forces `K*q_target(t)` curl each
finger independently at amplitude 1: 0.6 s rise, 0.3 s hold, 50 ms smooth
withdrawal (six 120 Hz substeps), then exact zero for passive recovery.
Neither modal state nor
surface vertices are prescribed. This is an assumed 10 MPa, 1 mm shell with
0.35 kg mass, not a calibrated anatomical, muscle or self-contact model.

The accepted trajectory reaches 24-29 mm maximum distal vertex travel, distinct
from its 14-17 mm distal centroid travel. Its 14.993235% peak principal strain
leaves only 0.006765 percentage points below the unchanged 15% ceiling; this is
not evidence of broad robustness. Edge, area, orientation, intersections,
wrist, contact and recovery gates remain mandatory. Validation failures exit
nonzero and preserve the output; do not increase tolerances to obtain a pass.

`assets/rom-48.npz` contains the derived mass-normalized basis and surface:
five bounded whole-surface curl targets plus 43 complementary POD directions,
diagonalized under the unchanged material model. It preserves the original
mass, wrist attachment, surface topology and physical sampling. See
[PROVENANCE.md](PROVENANCE.md) for immutable input hashes.
`assets/targets.npz` contains five derived modal-force targets. Raw NIMBLE models,
training poses, textures and checkpoints are excluded. See [ATTRIBUTION.md](ATTRIBUTION.md).
The offline fitting helper requires SciPy and separately authorized native input;
neither is needed to run the supplied ROM. `individual.py` retains earlier contact
helper code, but `expressive.py` is the supported entry point for this snapshot.

Published report: https://reports.mmacklin.com/everyday-rom/index.html#hand
