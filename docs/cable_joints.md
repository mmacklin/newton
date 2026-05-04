# Cable Joints

Newton's XPBD tendon solver is currently back at a clean baseline for
Müller et al. "Cable Joints" (SCA 2018).  This baseline intentionally removes
the capstan/finite-friction solver work so slip can be added back one case at a
time.

## Current Scope

Implemented:

- Rolling pulley links with the full angular Jacobian in the stretch row.
- Pinhole links that transfer rest length between their two adjacent spans.
- Fixed attachment links.
- Auto-computed initial rest lengths.
- Dynamic pulley bodies with revolute joints.
- No capstan friction path.  `mu` remains in the builder/model data for API
  compatibility, but XPBD tendon kernels ignore it.

Not implemented in the current baseline:

- Finite capstan friction.
- Frictional stick/slip classification.
- Separate no-slip/friction rows.
- Explicit scalar pulley angle state.
- Dynamic rerouting, merge/split, or non-circular pulley profiles.

## Link Types

| Type | Behavior |
|------|----------|
| `ROLLING` | Cable wraps around a circular body. Tangent points are recomputed from current geometry, and stretch constraints use the full linear and angular Jacobian at those tangent points. |
| `ATTACHMENT` | Cable endpoint fixed to a body-local point. |
| `PINHOLE` | Zero-radius waypoint. The two adjacent rest lengths redistribute proportionally to current span lengths, preserving their sum. |

## Solver Pipeline

The XPBD tendon path has two kernel entry points:

1. `update_tendon_attachments`
   - Runs once per tendon per solver substep, before the XPBD iterations.
   - Recomputes rolling tangent points.
   - Applies the paper's `surfaceDist(old, new)` rest-length transfer.
   - Applies pinhole rest transfer between adjacent spans.

2. `solve_tendon_segments`
   - Runs once per segment per solver iteration.
   - Solves the unilateral stretch inequality:

     ```text
     C = |x_r - x_l| - rest <= 0
     ```

   - Uses the full angular Jacobian:

     ```text
     Jw_l = -cross(r_l, n)
     Jw_r =  cross(r_r, n)
     ```

There is no separate capstan/friction kernel in this baseline.

## Formulation Notes

Each segment stores a mutable free-span rest length.  Rolling links transfer
rest length with the original paper update:

```text
rest += surfaceDist(old_left,  new_left)
rest -= surfaceDist(old_right, new_right)
```

Attachment points are stored in body-local coordinates.  At the start of a
substep, the old local contacts are transformed by the current body pose, new
tangents are computed, and the signed surface distance between those two points
updates the segment rest length.  During XPBD iterations, the stretch row
transforms the same local contacts by the current body pose every iteration.
This is what makes pulley rotation and cable motion couple immediately; the
contact point is fixed on the body during the solve rather than being a stale
world-space point.

No separate pulley-angle state is tracked.  The only rolling state is the
body-local contact point stored per segment endpoint.

Pinhole links are the only intentional slip points in this baseline.  A pinhole
preserves the sum of the two adjacent rest lengths and redistributes that sum
from current span geometry:

```text
rest_left = (rest_left + rest_right) * length_left / (length_left + length_right)
```

The stretch row remains unilateral, so slack cable is allowed.  Example tests
still track geometric total cable length as straight spans plus rolling wrap
arcs; small slack/tension differences are expected, but unbounded growth or
loss is not.

## Current Validation

The focused baseline regression suite is `newton.tests.test_tendon_capstan`.
Despite the historical filename, it now validates the no-friction baseline:

- Pinhole Atwood: the heavy side descends and the light side rises through a
  pinhole.
- Dynamic rolling pulley Atwood: the heavy side descends, the light side rises,
  and the pulley rotates from the angular Jacobian.
- `mu` ignored: changing `mu` does not change baseline motion.
- Motorized rolling pulley: a driven dynamic pulley produces slider motion
  through the no-slip cable path.
- Motorized delay regression: a driven pulley must move the cable during the
  initial rotation window.

Latest run:

```bash
uv run --extra examples python -m unittest newton.tests.test_tendon_capstan
```

Result: 10 tests passed on CPU and CUDA.

The cross-base XY table was also tested against this baseline with all pulleys
dynamic:

```bash
uv run --extra examples python -m newton.examples.cable.example_tendon_xy_table \
  --device cuda:0 --test --quiet --viewer null --num-frames 480
```

Result: passed on CUDA over the full 480-frame render horizon.  The
example now restores the historical reference route with lower table endpoints,
X/Y prismatic axes, and both drive pulleys using the original winding.  It
asserts the exact initial tangent-point fingerprint, that drive pulley rotation
produces measurable table sliding, that both drive windings do not cross
between their guide pulleys, and that all passive guide pulleys rotate.

The rolling and compound examples now have boundedness and pulley-clearance
gates over their longer render windows:

```bash
uv run --extra examples python -m newton.examples.cable.example_tendon_rolling_pulley \
  --device cuda:0 --test --quiet --viewer null --num-frames 180
uv run --extra examples python -m newton.examples.cable.example_tendon_compound_pulley \
  --device cuda:0 --test --quiet --viewer null --num-frames 220
```

Result: both passed on CUDA.  The rolling example now enables rigid
contact between the weights and pulley, uses shorter initial spans and more
conservative contact relaxation/substepping, and asserts that the light body
reaches the pulley contact neighborhood without unbounded frame jumps.

The 3D routing example also passes its 180-frame CUDA test.  It uses the
iterative tangent construction so adjacent pulleys can have non-coplanar cable
planes; its total-length check allows slack because the heavy body reaches the
ground and the taut geometric path can be shorter than the stored cable length.

The following examples are still WIP under this no-friction baseline:

- `tendon_capstan_kinematic`: expects finite-friction/free-slide/lock behavior,
  which is intentionally not implemented yet.
- `tendon_cable_machine`: runs, but still permits large stretch in the current
  authored parameters and should not be promoted until it has a tighter motion
  test.

## Repair Order

1. Keep this no-friction baseline stable.
2. Add one finite-slip case back with a small, isolated test.
3. Add capstan stick/slip projection only after the baseline tests stay green.
4. Re-render and promote complex examples only after their motion has a test
   gate that catches direction, delayed coupling, and pulley sign failures.
