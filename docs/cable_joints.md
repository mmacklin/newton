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
solver_xpbd.py      Tendon state init + three-phase kernel dispatch per substep
tendon_kernels.py   Warp GPU kernels (geometry, rest-length distribution, XPBD solve)
cable.py            Shared visualization utilities (tangent-point line extraction)
```

### Solver pipeline (per substep)

1. **`compute_tendon_tangent_points`** — For each rolling contact, compute
   tangent points from the neighboring waypoints onto the cylinder surface.
   Iterative for 3D (non-coplanar) configurations.

2. **`distribute_tendon_rest_lengths`** — Redistribute rest length across
   segments proportionally to current segment length, preserving total cable
   length.  This is the frictionless-pulley model: tension equalizes across
   all segments.

3. **`solve_tendon_segments`** — XPBD position-level constraint solve.
   Each segment enforces `length ≤ rest_length` with configurable compliance
   and damping.  Body positions and velocities are corrected via inverse-mass
   weighted deltas on the shared rigid bodies.

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

1. **Capstan friction** — The `mu` parameter is stored but has no effect in
   the solver.  Friction should bound the tension ratio between adjacent
   segments at rolling/pinhole contacts via the Euler-Eytelwein equation
   (`T1/T2 ≤ exp(μθ)`).  Highest-priority missing physics feature.
   Reference: `Cable::solve()` tension clamping.

2. **Convex hull profiles** — Reference supports arbitrary mesh cross-sections
   by slicing with the cable plane to get a 2D convex hull.  Newton only
   supports circle profiles (cylinders).  Reference: `addLink()` lines
   240–281, `getTangentPointBody()` lines 80–93.

3. **Pinhole rest-length transfer** — Reference transfers rest length between
   segments adjacent to a pinhole based on stretch difference.  Newton's
   proportional distribution doesn't do pinhole-specific transfer.
   Reference: lines 526–557.

4. **Merge/split (dynamic rerouting)** — Dynamically add/remove rolling
   contacts as cables wrap/unwrap around bodies.  Reference: lines 429–522.

5. **Compression compliance** — Separate compliance for slack cable, allowing
   configurable buckling stiffness.  Reference: `compressionCompliance`.

6. **Rolling limit detection** — Prevent wrap angle from going negative when
   cable tries to unwrap past a pulley edge.  Reference: `inLimit` flag,
   lines 573–641.

7. **Curved cable visualization** — Smooth visual samples along cable arcs
   and catenary-like curves for slack segments.  Reference:
   `computeSamples()`.

### Already implemented

- Circle-profile rolling contacts (tangent point, arc length)
- Fixed attachments and pinhole geometry
- Unilateral XPBD distance constraints
- Rest-length redistribution (frictionless)
- 3D iterative tangent computation for non-coplanar pulleys
- Auto-computed rest lengths (`rest_length=-1`)
- Dynamic pulley bodies with finite mass/inertia
- Headless MP4 recording via `ViewerGL --record`
