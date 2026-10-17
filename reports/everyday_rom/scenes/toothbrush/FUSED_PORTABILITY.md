# Fused Toothbrush Portability

## Scope

`fused_toothbrush.py` is an additive public-API construction adapter. The original
`toothbrush.py` finite-neck example remains available. Both use the same motion,
integration loop, generic `newton.solvers.SolverVBD`, contact materials, physical
assertions, and matched no-tool-contact control. The only structural assertion
change is two dynamic bodies for the original versus one for the fused model.
The fused defaults are four substeps, 16 iterations, relaxation 0.4.

CPU construction/algebra checks and fresh 450-frame positive/control/repeat CUDA
arms pass. Independent exported-trajectory audits preserve the original gates.
This is not a reproduction of the published private 4.246491822727274 ms timing.
No private dispatcher, native runtime patch, benchmark or new rendering is included.

## Physical Qualification

On 2026-10-17, one bounded CUDA job collected three fresh arms on an RTX PRO 6000
Blackwell MIG 1g.24gb device after CPU/AOT prewarm. All three passed the original
live assertions and independent CPU audits, with no numerical or gate changes.
Each arm includes 60 settling frames and 450 exported frames at 60 Hz.

| Independent measurement | Positive | Control | Repeat | Original bound |
|---|---:|---:|---:|---:|
| Modal surface step, mm | 0.980926910 | 0.001109187 | 0.980927492 | <1 |
| Modal second difference, mm | 0.951729482 | 0.002055548 | 0.951730006 | <1 |
| Maximum solve residual ratio | 7.06528e-7 | 9.04454e-6 | 4.25859e-6 | <0.05 |
| Collision and solve observations | 2040 each | 2040 each | 2040 each | 2040 each |
| Unclamped global/body peak | 2706/2706 | 2706/2706 | 2706/2706 | <=16384 |

Geometry reconstruction error is zero in every arm. Both positive arms cover all
20 tufts and pass the original overlap, mass/SPD, deformation, lift, freefall,
landing, release, capacity, residual and strain gates. The control has no
tool/head contact and passes the original control gates. Positive/control
kinematics and enable schedules match exactly; the deformation ratio is
`3.726611100557633e-5`, below the original 0.1 bound.

The repeat is **not bitwise identical** to the first positive. Its largest
translation-component difference is 0.256483 mm at 7.5 s; its largest quaternion
component difference is 0.0326777. Both trajectories independently pass. Repeat
parity was diagnostic, not a substituted or weakened physical gate. No cause is
inferred, and private trajectory parity was not measured. Three passing arms do
not establish deterministic replay or unlimited robustness.

The separate evidence stage is `brush-fused-portable-physical`, with 257 sealed
artifacts and manifest SHA-256
`f1e902408cee02e5ed4a135bc50f4fffc13c46e58d9ae34286440a3a7ba23124`.
`fused_sources.json` binds its handoff, result and tested source hashes. Raw
states and videos are intentionally absent from this source bundle. Source
metrics retain their original false acceptance flags; the independent audit
reports establish the physical pass without rewriting those records.

## Commands

Run from the repository root with the normal example dependencies installed:

```bash
PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/toothbrush/fused_toothbrush.py --check-construction
CUDA_VISIBLE_DEVICES= PYTHONPATH=. uv run --extra examples python -m unittest discover -s reports/everyday_rom/scenes/toothbrush -p 'test_*.py' -v
PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/toothbrush/fused_toothbrush.py --help
```

The construction command explicitly selects CPU and never settles, steps,
captures, or renders. Construction requires the reduced-elastic public APIs on
this example branch; an arbitrary released Newton version is not promised to
provide them. The branch's normal dependencies include Warp and NumPy; no new
dependency is introduced.

The simulation entry point runs the original live physical assertions and a
matched negative control. The qualification collector below additionally saves
the raw diagnostics needed for the independent full exported-trajectory audit:

```bash
PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/toothbrush/fused_toothbrush.py --device cuda:0 --frames 450 --substeps 4 --iterations 16 --negative-control --output output/toothbrush-fused
CUDA_VISIBLE_DEVICES= PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/toothbrush/refine_render.py --physical output/toothbrush-fused --output output/toothbrush-fused-smooth
```

For full physical qualification, `qualify_fused.py` adds raw residual, contact-pair
and settled-state export and a fresh positive/control/repeat sequence. It does
not benchmark or render. GPU users of a shared machine must acquire its GPU lock
and prewarm before collecting. Independent verification is CPU-only and may run
after releasing that lock:

```bash
PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/toothbrush/qualify_fused.py --collect-only --output output/fused-qualification
CUDA_VISIBLE_DEVICES= PYTHONPATH=. uv run --extra examples python reports/everyday_rom/scenes/toothbrush/qualify_fused.py --verify-only --output output/fused-qualification
```

Collection alone never reports full acceptance. Every original live physical
assertion runs unchanged (apart from the approved dynamic-body count); failure
stops later arms. `verify_fused.py` independently recovers geometry, virtual
handle COM motion, contact coverage, raw residual ratios and capacity histories.
It keeps both original 1 mm modal motion limits and both 10% strain limits.
Synthetic mutations must fail before a fresh trajectory is attempted. Inputs
are never rewritten by the verifier; failed arrays and source flags are retained.

The optional refinement command retains the existing 450-frame, single-application,
5e-8 m embedding bound. It requires passing positive/control flags and local
producer hashes. It does not load gallery states or simulate. The existing
finite-neck exported-state verifier remains unchanged. The separate fused
verifier accounts for its one-body/no-neck topology, without weakening gates.
The qualification collector deliberately leaves source acceptance flags false;
its output is not automatically accepted by the refinement command. Review and
bind the independent result before any separate rendering workflow; do not edit
the source flags to bypass that check. No new rendering was part of qualification.

## Preserved Construction

- One dynamic fused head frame, four kinematic tool/jaw bodies; no finite neck.
- Original 40 modal masses, stiffnesses, damping values and all inertial coupling
  tensors; 20 independent two-mode tufts and 140 strands.
- Original finalized float32 handle/head masses, centers and inertias combined
  in float64 with the parallel-axis theorem, then exported float32 unchanged.
  No mass, inertia, stiffness or solver regularization is introduced.
- Rigid handle/grip/rib vertices and grasp seeded with explicit zero Phi/Psi,
  zero-mass samples in the head frame. Existing head samples remain unchanged.
- All 30 authored mesh parts, collision samples/materials, and the exact detailed
  render embedding. Handle: 7,042 vertices/14,080 triangles. Plastic head: 4,482
  vertices/8,960 triangles. All 20 tufts remain 518 vertices/1,008 triangles each.
- Original handle-origin/COM diagnostics recovered by a host-only virtual view;
  no native state, force, contact or modal coordinate is rewritten for metrics.

`fused_reference.json` contains only array hashes/dtypes/shapes derived from the
independently verified 142-file approved physical-repeat seal. It contains no
trajectory, binary mesh, generated media or raw third-party model. The origin
manifest and input hashes are retained. All procedural geometry is local.

## Portability Boundaries

The accepted gallery run used a research-only rigid-attachment routing flag in
its collision pipeline. This public runtime has no equivalent per-shape option:
all shapes on an elastic body use elastic vertex contact generation, including
zero-Phi handle attachments. The same material geometry, frame and Jacobians
do not prove the same generated contact set or trajectory. This is the main
blocker to claiming exact execution parity, despite the public path's fresh
physical pass. No private flag is silently copied.

The private optimized dispatcher is absent. Generic solver timing is unknown
and must be measured separately, with diagnostics disabled, only after physical
qualification. Neither CPU algebra nor exact mesh/model arrays imply the
published frame time or a full-scene speedup.

## Revalidation and Limits

1. The documented physical pass applies to the tested source and configuration,
   not arbitrary Newton releases or changed settings. Requalify changes with the
   collector and independent verifier; never reuse a prior green summary.
2. Preserve fresh full positive/control trajectories and repeat with all physical,
   capacity, coverage, residual, motion and negative-control gates. A repeat's
   bitwise difference is diagnostic; private trajectory bitwise parity is not
   required for this distinct public collision path.
3. Retain failures. Do not weaken gates, add shifts or tune physical parameters
   to make the generic path pass. Only passing trajectories may be refined and
   considered for a new replay; clean timing is a separate study requiring review.

## Provenance

- [Approved fused physical repeat](https://reports.mmacklin.com/everyday-rom/performance/phase2/experiments/brush-fused-physical-repeat-profile/HANDOFF.md)
- [Separate private clean timing](https://reports.mmacklin.com/everyday-rom/performance/phase2/experiments/brush-fused-persistent-timing/HANDOFF.md)
- `sources.json` retains original finite-neck source hashes and lists the small
  shared gate/export hooks. `fused_sources.json` records this additive adapter's
  origins, runtime pin and packaged hashes. No installed runtime is modified.
