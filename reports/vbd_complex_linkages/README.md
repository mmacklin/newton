# VBD complex linkage comparison

This report harness compares the experimental VBD sparse-direct articulation
solve against VBD local updates and Kamino on closed-loop mechanisms.

The attached NVIDIA linkage demo is kept outside the repository at:

```text
/home/horde/external-assets/kamino_linkage_demos_20260625
```

The harness imports its model-construction function, places every linkage joint
inside one closed-loop Newton articulation, and runs matched controls and model
parameters through each solver.

Initial robot-foot smoke test:

```bash
uv run --extra examples python reports/vbd_complex_linkages/bench_complex_linkages.py \
  --scenario robot-foot --frames 20 --output /tmp/vbd-complex-linkages-smoke.json
```
