# VBD Energy Analysis and Convergence Fixes

## Overview

This document summarizes the investigation into VBD convergence bottlenecks in Newton's cloth simulation. The root cause of poor convergence was traced to three sources of ill-conditioning in the per-vertex 3×3 Newton system, all related to contact energy formulations.

## Per-Vertex System Structure

VBD solves a local 3×3 system per vertex per iteration:

```
H_i · Δx_i = f_i
```

where:
- `H_i = (m/h²)I + H_elastic + H_bending + H_contact` (accumulated Hessian)
- `f_i = (m/h²)(y_i - x_i) + f_elastic + f_bending + f_contact` (total force)

The block Gauss-Seidel convergence rate is bounded by `(κ-1)/(κ+1)` where `κ` is the global condition number — the ratio of the largest eigenvalue of any vertex's Hessian to the smallest eigenvalue of any vertex's Hessian across the mesh.

## Stiffness Hierarchy (Before Fixes)

| Component | Hessian Eigenvalue | Source |
|-----------|-------------------|--------|
| Body-particle contact damping | **187,500,000** | `kd·ke/dt = 50·6250·600` |
| Self-contact barrier (1/d² Hessian) | **12,000,000** | `k2/d² = 50/(0.002)²` near contact |
| Self-contact friction | **3,000,000** | `μ·F_n·2/(ε·dt)` near static |
| Body-particle friction | **3,000,000** | same mechanism |
| Elastic membrane (μ) | 10,000 | Neo-Hookean membrane stiffness |
| Elastic membrane (λ) | 10,000 | Neo-Hookean area stiffness |
| Inertia (m/h²) | 5,241 | mass / dt² |
| **Bending** | **5** | Dihedral angle stiffness |

**Condition number κ ≈ 188M / 5 = 37,500,000.** Block GS convergence rate = 0.999999997. Iterations needed for 10× reduction: **~86 million.**

## Root Cause 1: Contact Damping Coefficient (c = kd·ke)

### The Problem

The body-particle contact damping energy (implicit Euler variational form):

```
E_damp = c/(2h) · (n·Δx)²
```

with coefficient `c = kd · ke` where:
- `kd = avg(soft_contact_kd, shape_material_kd) = avg(0.01, 100) = 50`
- `ke = body_particle_contact_penalty_k` (AVBD-ramped up to 6250)
- `c = 50 × 6250 = 312,500`

The Hessian eigenvalue: `c/h = 312,500 / 0.00167 = 187,500,000`.

For comparison, critical damping for a spring with stiffness ke and mass m:
```
c_crit = 2·√(m·ke) = 2·√(0.015 × 6250) ≈ 19.4
```

The actual damping was **16,000× critically overdamped**.

### The Fix

Change from `c = kd · ke` to `c = kd`:

```python
# Before:
damping_coeff = body_particle_contact_kd * body_particle_contact_ke

# After:
damping_coeff = body_particle_contact_kd
```

This keeps implicit damping (Hessian included in the per-vertex system) but with eigenvalue `kd/h = 50/0.00167 = 30,000` — comparable to elastic stiffness, not 6000× larger.

### Impact

Damping Hessian eigenvalue: **187,500,000 → 30,000** (6,250× reduction).

## Root Cause 2: Self-Contact Log-Barrier Energy

### The Problem

The self-contact energy used a reciprocal barrier in the smooth zone (`tau > d > 1e-5`):

```
E(d) = k2 · ln(d)     where k2 = 0.5 · τ² · k
dE/dd = -k2 / d
d²E/dd² = k2 / d²     ← grows unboundedly as d → 0
```

With `τ = 0.5 × collision_radius = 0.1`, `k = 10,000`:
- `k2 = 0.5 × 0.01 × 10,000 = 50`
- At `d = 0.01` (10% radius): `d²E/dd² = 50 / 0.0001 = 500,000`
- At `d = 0.001` (1% radius): `d²E/dd² = 50 / 0.000001 = 50,000,000`

### The Fix

Replace with a simple quadratic penalty:

```python
# Before (log barrier):
k2 = 0.5 * tau * tau * k
dEdD = -k2 / dis
d2E_dDdD = k2 / (dis * dis)

# After (quadratic penalty):
dEdD = -k * penetration_depth
d2E_dDdD = k
```

Energy: `E(d) = k/2 · (r - d)²`. Hessian: `d²E/dd² = k` (constant, no blowup).

### Impact

Self-contact Hessian eigenvalue: **12,000,000+ → 10,000** (1,200× reduction).

## Root Cause 3: AVBD Penalty Ramp During Iterations

### The Problem

The AVBD dual update runs inside the VBD iteration loop:

```python
# Called every iteration inside step():
k_new = min(k + beta * penetration, stiffness)
```

With `avbd_beta = 100,000`, the penalty `ke` ramps from initial value (100) to the cap (6,250) within a few iterations. This means **the optimization landscape changes between iterations** — each iteration sees a different contact stiffness, preventing convergence.

### The Fix

Set `avbd_beta = 0` (constant penalty throughout the iteration loop). The penalty stays at `k_start_body_contact` (100). Contact compliance is handled by the existing penalty stiffness, not by ramp-up.

### Impact

Removes system-changing-mid-iteration instability. Penalty stays constant at 100 instead of ramping to 6,250.

## Combined Results

### Before Fixes (from identical contact state)

```
93087 -> 63 -> 23008 -> 152 -> 33424 -> 27005 -> 29051 -> 9352 -> 36618 -> 179
```

Wild oscillation between 100s and 90,000s. H_max = 188,000,000.

### After All Fixes (same starting state, beta=0)

```
358 -> 46 -> 212 -> 41 -> 317 -> 38 -> 323 -> 39 -> 328 -> 41
```

H_max = 500,000. Residual low points ~40 (2,300× lower than baseline peaks). Still has odd/even GS color oscillation.

### Condition Number Improvement

| | Before | After |
|---|--------|-------|
| H_max | 188,000,000 | 500,000 |
| H_min (bending) | 5 | 5 |
| **Condition number κ** | **37,500,000** | **100,000** |
| GS convergence rate | 0.9999999 | 0.99998 |
| Iters for 10× reduction | 86,000,000 | 115,000 |

Still limited by the elastic/bending ratio (10,000/5 = 2,000:1) and remaining friction stiffness. The odd/even oscillation is from GS cross-color interference, which under-relaxation (alpha=0.7-0.9) can address.

## Remaining Stiffness Hierarchy (After Fixes)

| Component | Hessian Eigenvalue |
|-----------|-------------------|
| Self-contact friction (near static) | ~500,000 (from `μ·F_n·2/(ε·dt)`) |
| Contact damping (c=kd) | 30,000 |
| Elastic membrane (μ, λ) | 10,000 |
| Self-contact penalty (quadratic) | 10,000 |
| Inertia (m/h²) | 5,241 |
| Body-particle penalty | 100 |
| Bending | 5 |

The next bottleneck is **self-contact friction regularization** (`2/(friction_epsilon·dt) = 120,000` multiplier) and the **elastic/bending ratio** (2,000:1).

## Files Modified

| File | Change |
|------|--------|
| `newton/_src/solvers/vbd/rigid_vbd_kernels.py` | Contact damping: `c = kd` not `c = kd·ke` |
| `newton/_src/solvers/vbd/particle_vbd_kernels.py` | Self-contact: quadratic penalty, not log-barrier |
| Solver parameter: `avbd_beta = 0` | Kill AVBD ramp (constant penalty) |
