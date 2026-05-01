# Cable Joints

Cable simulation for Newton's XPBD solver, implementing Müller et al.
"Cable Joints" (SCA 2018) with extensions for 3D routing and capstan
friction.

## Overview

A **tendon** is an inextensible, massless cable routed through an ordered
sequence of waypoints on rigid bodies.  Between adjacent waypoints, a
unilateral XPBD distance constraint enforces the cable segment length.
Rest length redistributes across segments as bodies move, keeping total
cable length constant.

### Link types

| Type | Description |
|------|-------------|
| `ROLLING` | Cable wraps around a cylindrical body surface.  Attachment point tracks the tangent; arc length contributes to the segment. |
| `ATTACHMENT` | Cable is fixed to a body-local point.  No sliding. |
| `PINHOLE` | Cable passes through a fixed point on the body.  Rest length can transfer between adjacent segments. |

### Builder API

```python
builder.add_tendon()
builder.add_tendon_link(body, link_type, offset, axis, radius=..., mu=..., ...)
```

Each `add_tendon()` starts a new cable.  `add_tendon_link()` appends a
waypoint.  Adjacent waypoints define cable segments, each with its own
compliance, damping, and rest length (auto-computed from initial geometry
when `rest_length=-1`).

## Architecture

```
builder.py          add_tendon(), add_tendon_link() — builds Model arrays
tendon.py           TendonLinkType enum
solver_xpbd.py      Tendon state init + per-substep kernel dispatch
tendon_kernels.py   Warp GPU kernels (geometry, rest-length distribution, XPBD solve)
cable.py            Shared visualization and cable-length test utilities
```

### Solver pipeline (per substep)

1. **`update_tendon_attachments`** — For each rolling contact, compute
   tangent points from the neighboring waypoints onto the cylinder surface.
   Iterative for 3D (non-coplanar) configurations.

2. **`distribute_tendon_rest_lengths`** — Initialize rest lengths across
   segments while preserving total cable length. This is the free-sliding
   baseline; rolling friction is applied by the rolling contact rows.

3. **`solve_tendon_segments`** — XPBD position-level constraint solve.
   Each segment enforces `length ≤ rest_length` with configurable compliance
   and damping.  Body positions and velocities are corrected via inverse-mass
   weighted deltas on the shared rigid bodies. Frictional rolling contacts add
   a no-slip coupling row scaled by the capstan tension bound.

4. **`update_tendon_coupling_rest`** — Per iteration, apply incremental
   rolling-contact rest-length transfer from accepted rim motion. Kinematic
   rolling bodies have zero inverse inertia, so their `dtheta` is zero and no
   rim-motion transfer occurs.

## Formulation decisions

This implementation deliberately separates three ideas that are easy to
conflate: cable stretch, cable redistribution, and rolling contact friction.

### Cable length is stored once

Each tendon stores a fixed total cable length:

```text
L_total = sum(segment_rest_lengths) + sum(wrap_angle * radius)
```

The mutable per-segment rest lengths are the solver's distribution of that
fixed cable length across straight spans.  Rest length can move between
neighboring spans, but the target total is not supposed to drift.  The example
tests measure the geometric total length as straight span lengths plus wrap
arcs and compare it against this stored target.

Because the segment constraint is unilateral, a slack route can have a current
taut geometric length smaller than `L_total`.  That is slack, not cable loss.
Positive error corresponds to actual stretch; negative error corresponds to
unrepresented slack length.

### Stretch and rolling are separate constraints

The stretch row for a segment is the unilateral XPBD distance constraint:

```text
C_stretch = |x_r - x_l| - rest <= 0
```

For rolling links, this row does not directly torque the pulley.  The straight
span endpoint is a tangent point, and using the stretch row's angular Jacobian
as the rolling friction model would mix geometric length enforcement with
rim-contact friction.  Instead, rolling contact is represented by a separate
constraint row that couples cable travel to rim travel:

```text
C_roll = dx - sign * R * dtheta = 0
```

Equivalently, for a segment adjacent to a frictional rolling pulley the implementation
stores the invariant:

```text
segment_length + sign * orientation * R * theta = constant
```

That rolling row has the pulley angular Jacobian.  This keeps the physics
meaning of the rows clear: stretch enforces cable length, rolling transfers
linear cable motion into pulley rotation.

### Friction limits force transfer, not geometry

Capstan friction is expressed as a force/tension admissibility condition:

```text
T_tight / T_slack <= exp(mu * theta_wrap)
```

The solver estimates the adjacent segment tensions from the lagged compliant
stretch residuals:

```text
T ~= max(length - rest, 0) / compliance
```

Those estimates are used as Coulomb-style lagged data for the friction
decision.  We chose this over copying stretch multipliers because it keeps the
friction decision local to the current stretch state and avoids extra lambda
bookkeeping.

When the capstan bound can support the requested transfer, the rolling contact
sticks.  When it cannot, transfer is reduced by the capstan stick fraction.
The same rolling row is used for dynamic and kinematic pulleys; the only
difference is the inverse mass/inertia supplied to the row.

### Incremental rolling transfer

Rolling transfer is incremental per solver iteration, not a one-time
redistribution at the start of the step.  The relevant state is:

```text
dtheta = theta_current - theta_ref
du = orientation * R * dtheta
```

After the transfer is applied, `theta_ref` is updated to the current angle.
This lets redistribution participate in each XPBD iteration rather than
lagging an entire time step behind the body solve.

### Dynamic and kinematic pulleys use the same row

The special cases follow from the same constraints and force limits:

| Case | Behavior |
|------|----------|
| `mu = 0` | No friction transfer. Cable can slide, and a dynamic pulley receives no rolling torque. |
| Finite `mu`, below critical | Capstan bound allows only partial transfer; relative slip remains. |
| Large `mu` / stick | Rolling row enforces zero relative slip: cable travel matches `R * dtheta`. |
| Dynamic pulley | Rolling transfer spins the pulley and couples its effective inertia `I/R^2` into the cable dynamics. |
| Kinematic pulley | Infinite mass/inertia means zero inverse mass/inertia; `dtheta = 0`, so high friction locks cable transfer. |

For small dynamic pulley inertia, frictionless and no-slip cable trajectories
become similar because the pulley stores little energy.  Increasing inertia
makes the rim motion and no-slip coupling easier to see.

## Examples

All examples are in `newton/examples/cable/` and support `--record` for
headless MP4 capture.

| Example | Description |
|---------|-------------|
| `tendon_pulley` | Single pulley, two weights — basic Atwood machine |
| `tendon_rolling_pulley` | Dynamic pulley on a hinge joint |
| `tendon_compound_pulley` | Compound (block-and-tackle) pulley system |
| `tendon_cable_machine` | Multi-pulley cable routing |
| `tendon_3d_routing` | Non-coplanar cylinders with iterative 3D tangent solver |
| `tendon_equilibrium` | Static equilibrium validation |
| `tendon_pinhole` | Cable through a fixed point |
| `tendon_capstan_friction` | Dynamic-pulley capstan friction: frictionless, partial grip, no-slip |
| `tendon_capstan_kinematic` | Kinematic-pulley capstan friction: free slide, partial slip, lock |

Run with:
```bash
uv sync --extra examples
uv run -m newton.examples tendon_pulley
uv run -m newton.examples tendon_pulley --record   # saves MP4
```

`render_all_examples.py` in the repo root renders all examples headless to
`~/reports/cable-sim-research/`.

## Reference implementation

The original C++ implementation is at `~/src-2/SimpleSolverExt.cpp` (Müller
et al.).  The Newton port maps roughly as:

| C++ | Newton |
|-----|--------|
| `Cable::solve()` | Three-phase kernel dispatch in `solver_xpbd.py` |
| `getTangentPointBody()` | `tangent_point_circle()` in `tendon_kernels.py` |
| `surfaceLength()` | `arc_length_on_circle()` in `tendon_kernels.py` |
| `addLink()` | `add_tendon_link()` in `builder.py` |

## Roadmap

Features remaining to port from the reference implementation.

### Not yet implemented

1. **Convex hull profiles** — Reference supports arbitrary mesh cross-sections
   by slicing with the cable plane to get a 2D convex hull.  Newton only
   supports circle profiles (cylinders).  Reference: `addLink()` lines
   240–281, `getTangentPointBody()` lines 80–93.

2. **Pinhole rest-length transfer** — Reference transfers rest length between
   segments adjacent to a pinhole based on stretch difference.  Newton's
   proportional distribution doesn't do pinhole-specific transfer.
   Reference: lines 526–557.

3. **Merge/split (dynamic rerouting)** — Dynamically add/remove rolling
   contacts as cables wrap/unwrap around bodies.  Reference: lines 429–522.

4. **Compression compliance** — Separate compliance for slack cable, allowing
   configurable buckling stiffness.  Reference: `compressionCompliance`.

5. **Rolling limit detection** — Prevent wrap angle from going negative when
   cable tries to unwrap past a pulley edge.  Reference: `inLimit` flag,
   lines 573–641.

6. **Curved cable visualization** — Smooth visual samples along cable arcs
   and catenary-like curves for slack segments.  Reference:
   `computeSamples()`.

### Already implemented

- Circle-profile rolling contacts (tangent point, arc length)
- Fixed attachments and pinhole geometry
- Unilateral XPBD distance constraints
- Rest-length redistribution (frictionless)
- Capstan friction for rolling contacts:
  `T_tight/T_slack <= exp(mu * theta)`, with kinematic lock and dynamic
  finite-friction no-slip coupling through the same rolling row.
- 3D iterative tangent computation for non-coplanar pulleys
- Auto-computed rest lengths (`rest_length=-1`)
- Dynamic pulley bodies with finite mass/inertia
- Headless MP4 recording via `ViewerGL --record`
