# Cup, Elastic Lid And Straw

The procedural asset and interactive `Example` are already part of this branch:

```bash
uv run -m newton.examples basic_reduced_elastic_cup_straw --motion stir
```

For the gallery's every-substep contact audit, saved surfaces and negative control,
run from the checkout root, in this order:

```bash
uv run python reports/everyday_rom/scenes/cup_lid/baseline.py --output output/everyday-rom/cup-lid/baseline
uv run python reports/everyday_rom/scenes/cup_lid/cup_lid.py --output output/everyday-rom/cup-lid
uv run python -m unittest discover -s reports/everyday_rom/scenes/cup_lid -p 'test_*.py' -v
```

Use a fresh output directory. The supported cup asset is procedural; no additional
binary mesh is needed. The four-mode lid is contact-driven by the prescribed straw,
using four substeps and twelve iterations over 570 steps plus the initial state.
The scripts require the unchanged `newton/` runtime from the branch's base commit;
report-only commits are allowed.

This is a high-strain illustrative ROM, not validated linear material response.
The complete skin reaches 47.95% principal engineering strain (midsurface 18.13%):
contact/recovery tests pass, but the separate 15% strain screen does not. Use a
50% complete-skin strain scale, not a rescaled or clamped geometry.

Report: https://reports.mmacklin.com/everyday-rom/index.html#cup-lid
