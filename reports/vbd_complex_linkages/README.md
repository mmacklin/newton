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

Trained DR Legs policy comparison:

```bash
uv run --extra torch-cu12 python reports/vbd_complex_linkages/bench_dr_legs_policy.py \
  --cases kamino local_i8 local_i32 sparse_i8 sparse_no_armature_i8
```

Render one policy rollout through the same harness:

```bash
uv run --extra torch-cu12 python reports/vbd_complex_linkages/render_dr_legs_policy.py \
  --case sparse_i8
```

The VBD policy rows use an isotropic child-body inertia approximation for the
policy's joint armature. The no-armature case records the unsupported baseline;
the approximation and its limitations are defined in the report.
