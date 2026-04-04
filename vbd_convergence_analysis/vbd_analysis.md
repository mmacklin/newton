# VBD Convergence Analysis

## Method Overview

Vertex Block Descent (VBD) is a block coordinate descent solver for implicit Euler time integration
in physics simulation. It was introduced in the SIGGRAPH 2024 paper by Chen et al.

### Mathematical Formulation

VBD minimizes the variational energy:

```
G(x) = (1/2h²)||x - y||²_M + E(x)
```

where:
- `x` = particle positions (unknowns)
- `y = x^t + hv^t + h²a_ext` = inertial target (forward Euler prediction)
- `h` = timestep
- `M` = mass matrix (diagonal)
- `E(x)` = total potential energy (elastic + contact + gravity)

### VBD Update Rule

For each vertex `i` in each color group (Gauss-Seidel parallel):

```
H_i = (m_i/h²)I + Σ_j ∂²E_j/∂x_i²      (3x3 local Hessian)
f_i = (m_i/h²)(y_i - x_i) + Σ_j -∂E_j/∂x_i  (3-vector local force)
Δx_i = H_i⁻¹ f_i                           (Newton step on local system)
x_i ← x_i + Δx_i
```

Each update guarantees descent of the local variational energy `G_i`.

### Key Properties
- **Unconditionally stable**: Even with 1 iteration per step
- **Block Gauss-Seidel**: Colors ensure no conflicting updates within a color
- **No global linear system**: Only 3×3 solves per vertex
- **Fixed iteration count**: No adaptive convergence criterion in the original implementation

## Newton Implementation Details

### Solver Structure (solver_vbd.py)

The Newton VBD implementation has a 3-phase structure:
1. **Initialize**: Forward integrate → compute inertia targets, detect collisions
2. **Iterate**: For each iteration, solve all color groups sequentially
3. **Finalize**: Update velocities via BDF1

### Material Models
- **StVK** (St. Venant-Kirchhoff): `ψ = μ||G||² + (λ/2)(tr G)²` with Green strain
- **Neo-Hookean membrane** (new in this fork): Stable Neo-Hookean adapted to shells
  - `ψ = (μ/2)(I_c - 2) + (λ/2)(J_s - α)²`
  - `α = 1 + μ/λ` ensures zero stress at rest

### Convergence Concerns
1. **No residual tracking**: The original solver uses a fixed iteration count with no convergence check
2. **No line search**: Paper found local backtracking costs 40% with no benefit
3. **Gauss-Seidel ordering**: Color groups processed sequentially; ordering can affect convergence
4. **Singular Hessian guard**: `|det(H)| > 1e-8` check skips degenerate vertices entirely
5. **Displacement-based**: Positions updated via accumulated displacements, not direct position writes

## Convergence Tracking Instrumentation

Added to `solver_vbd.py`:
- `track_convergence` flag on `SolverVBD`
- Per-iteration metrics: RMS/mean/max displacement, inertial residual, percentiles
- Per-step NaN detection
- `get_convergence_data()` / `reset_convergence_data()` methods

## Experimental Setup

- T-shirt mesh (unisex_shirt.usd) spawned at randomized positions/rotations
- Centimeter scale (matching franka example conventions)
- 10 substeps per frame at 60 FPS → dt = 1/600 s
- Ground plane for contact
- Multiple scenarios with different random seeds

---

## Findings

### Checkpoint 0: Baseline (Pre-Analysis)

*Results pending initial convergence test run...*

---

## Design Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-04 | Use displacement norm as primary convergence metric | Direct measure of per-iteration position change; force residual requires separate kernel |
| 2026-04-04 | Disable self-contact for initial analysis | Isolate elastic convergence behavior from contact complexity |
| 2026-04-04 | Use neo-hookean material model | This is the fork's new feature; want to analyze its convergence properties |

## Areas for Investigation

1. **Hessian approximation quality**: Are the 3×3 per-vertex Hessians well-conditioned?
2. **Convergence rate vs iteration count**: Does doubling iterations halve residual?
3. **Material model comparison**: Neo-Hookean vs StVK convergence behavior
4. **Step size / damping**: Could a relaxation factor improve convergence?
5. **Chebyshev acceleration**: Paper mentions this but Newton impl doesn't use it
6. **Position vs displacement formulation**: Current impl accumulates displacements
7. **Color group ordering**: Does it affect convergence?
8. **Stiffness sensitivity**: How does tri_ke affect convergence?
