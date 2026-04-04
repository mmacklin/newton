# VBD Convergence Analysis

## Method Overview

Vertex Block Descent (VBD) is a block coordinate descent solver for implicit Euler time integration
in physics simulation. It was introduced in the SIGGRAPH 2024 paper by Chen et al.

### Mathematical Formulation

VBD minimizes the variational energy:

```
G(x) = (1/2h^2)||x - y||^2_M + E(x)
```

where:
- `x` = particle positions (unknowns)
- `y = x^t + hv^t + h^2 a_ext` = inertial target (forward Euler prediction)
- `h` = timestep
- `M` = mass matrix (diagonal)
- `E(x)` = total potential energy (elastic + contact + gravity)

### VBD Update Rule

For each vertex `i` in each color group (Gauss-Seidel parallel):

```
H_i = (m_i/h^2)I + Sum_j d^2E_j/dx_i^2      (3x3 local Hessian)
f_i = (m_i/h^2)(y_i - x_i) + Sum_j -dE_j/dx_i  (3-vector local force)
dx_i = H_i^{-1} f_i                             (Newton step on local system)
x_i <- x_i + dx_i
```

Each update guarantees descent of the local variational energy `G_i`.

### Key Properties
- **Unconditionally stable**: Even with 1 iteration per step
- **Block Gauss-Seidel**: Colors ensure no conflicting updates within a color
- **No global linear system**: Only 3x3 solves per vertex
- **Fixed iteration count**: No adaptive convergence criterion in the original implementation

## Newton Implementation Details

### Solver Structure (solver_vbd.py)

The Newton VBD implementation has a 3-phase structure:
1. **Initialize**: Forward integrate -> compute inertia targets, detect collisions
2. **Iterate**: For each iteration, solve all color groups sequentially
3. **Finalize**: Update velocities via BDF1

### Material Models
- **StVK** (St. Venant-Kirchhoff): Standard Green strain energy
- **Neo-Hookean membrane** (new in this fork): Stable Neo-Hookean adapted to shells
  - `psi = (mu/2)(I_c - 2) + (lambda/2)(J_s - alpha)^2`
  - `alpha = 1 + mu/lambda` ensures zero stress at rest
  - SPD projection via clamping the cofactor-derivative coefficient

### Displacement Accumulation Mechanism
The solver operates on a displacement buffer rather than direct position writes:
1. `forward_step` initializes displacement = velocity * dt
2. `solve_elasticity` accumulates: displacement += H^{-1} * f
3. `apply_truncation_ts` projects: pos = pos_prev_CD + displacement

Positions are updated after each color group (Gauss-Seidel), so neighboring vertices
see the most recent position when computing their Newton step.

## Convergence Tracking Instrumentation

Added to `solver_vbd.py`:
- `track_convergence` flag on `SolverVBD`
- Per-iteration metrics: RMS/mean/max displacement, inertial residual, percentiles
- Per-step NaN detection
- `get_convergence_data()` / `reset_convergence_data()` methods

## Experimental Setup

- T-shirt mesh (unisex_shirt.usd) spawned at randomized positions/rotations
- Centimeter scale (matching franka example conventions)
- 10 substeps per frame at 60 FPS -> dt = 1/600 s
- Ground plane for contact
- 8 randomized scenarios per configuration (drop height, rotation, lateral offset)
- 30 frames per scenario (300 substeps)

---

## Findings

### Checkpoint 0: Baseline Analysis

**Critical observation**: After the first VBD iteration provides a good descent step (ratio ~0.7-0.9),
subsequent iterations show per-iteration ratios of ~1.0-1.1, indicating stagnation or slight divergence.

Example convergence curve (seed 1, iteration count=10, substep 75):
```
7.12e-06 -> 6.16e-06 -> 6.63e-06 -> 6.62e-06 -> 6.67e-06 -> 6.73e-06 -> 6.78e-06 -> 6.81e-06 -> 6.86e-06 -> 6.88e-06
Per-iter:    0.865      1.076      0.997      1.008      1.008      1.008      1.005      1.007      1.004
```

**Root cause**: Block Gauss-Seidel cross-color interference. When one color group's vertices are updated,
this changes the force balance for neighboring vertices in other color groups. Near equilibrium,
these cross-color perturbations are comparable to the Newton correction magnitude, causing oscillation.

**Impact**: 70-90% of the VBD computational budget is wasted on iterations that produce negligible improvement.

**Exception**: Dynamic scenarios with large displacements (e.g., active falling) show good monotone
convergence (ratios 0.5-0.8 per iteration) because the Newton corrections are much larger than
cross-color perturbations.

### Baseline Metrics (10 iterations, 8 scenarios, 30 frames each)
- Mean convergence ratio: 0.7274
- Median convergence ratio: 0.7425
- Mean final RMS displacement: 2.77e-03 cm
- Median final RMS displacement: 1.02e-05 cm
- NaN count: 0/8

---

### Checkpoint 1: Chebyshev Semi-Iterative Acceleration

**Implementation**: Added Chebyshev acceleration (Wang & Yang 2015, VBD paper Sec. 5.3) with the
extrapolation formula:
```
x^(n) = omega_n * (x_bar^(n) - x^(n-2)) + x^(n-2)
```

Where `x_bar^(n)` is the raw Gauss-Seidel result, and omega follows the recurrence:
- omega_1 = 1
- omega_2 = 2 / (2 - rho^2)
- omega_n = 4 / (4 - rho^2 * omega_{n-1})

Applied globally after each complete Gauss-Seidel iteration (all color groups).
Displacement buffer is updated to maintain consistency with the accelerated positions.

**Spectral radius sensitivity**:
- rho = 0.95: Too aggressive. Creates oscillations (alternating ratios 0.6 / 2.0). Median final RMS *worse* than baseline.
- rho = 0.80: Excellent. ~200x improvement in median final RMS. Monotone convergence after initial warmup.
- rho = auto: Best overall. ~1000x improvement. Adapts to local problem conditioning.

### Chebyshev 0.8 Metrics
- Mean convergence ratio: 0.2527
- Median convergence ratio: 0.0385
- Median final RMS displacement: 4.76e-08 cm (vs baseline 1.02e-05)
- NaN count: 0/4

---

### Checkpoint 2: Adaptive Spectral Radius Estimation

**Implementation**: `chebyshev_rho = "auto"` estimates rho from consecutive displacement ratios
using exponential moving average (alpha = 0.1). This avoids manual tuning while achieving
performance close to or better than the optimal manual value.

The auto-estimate converges to rho ~ 0.88, which is between the conservative 0.8 and
aggressive 0.95 values.

### Chebyshev Auto Metrics
- Mean convergence ratio: 0.2191
- Median convergence ratio: 0.0481
- Median final RMS displacement: 1.11e-08 cm (**~1000x improvement over baseline**)
- NaN count: 0/8

---

### Checkpoint 3: Stiffness Sensitivity Analysis

Tested with stiffness multipliers 1x, 5x, and 10x on both baseline and Chebyshev auto:

| Stiffness | Baseline Median Ratio | Chebyshev Auto Median Ratio |
|-----------|----------------------|---------------------------|
| 1x        | 0.74                 | 0.05                      |
| 5x        | 0.68                 | 0.01                      |
| 10x       | 0.67                 | 0.02                      |

Higher stiffness improves convergence for both methods because the inertial term (m/h^2)
maintains diagonal dominance even as elastic stiffness increases.

Chebyshev acceleration is effective across all stiffness levels with no stability issues.

---

## Design Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-04 | Use displacement norm as primary convergence metric | Direct measure of per-iteration position change; well-defined even for coupled systems |
| 2026-04-04 | Disable self-contact for initial analysis | Isolate elastic convergence behavior from contact complexity |
| 2026-04-04 | Use neo-hookean material model | This is the fork's new feature; want to analyze its convergence properties |
| 2026-04-04 | Chebyshev acceleration applied after all colors, not per-color | Matches original paper formulation; per-color would break the Gauss-Seidel structure |
| 2026-04-04 | Auto rho via exponential moving average | Simple, robust, avoids manual tuning; alpha=0.1 gives good balance of responsiveness and stability |
| 2026-04-04 | Update displacement buffer after Chebyshev | Required because the solver's truncation step reads from pos_prev_CD + displacement |

## Answered Questions

1. **Hessian approximation quality**: The 3x3 per-vertex Hessians are generally well-conditioned for the
   Neo-Hookean model thanks to the SPD projection (s_clamp). The determinant check at 1e-8 provides a
   safety net for degenerate cases.

2. **Convergence rate vs iteration count**: Doubling iterations does NOT halve residual in baseline.
   Going from 5 to 10 to 20 iterations shows diminishing returns. With Chebyshev, more iterations
   provide genuine improvement.

3. **Material model comparison**: Neo-Hookean and StVK show similar convergence characteristics.
   Neo-Hookean's SPD projection helps prevent ascent directions.

4. **Chebyshev acceleration**: Dramatically effective. This was the single biggest improvement found.

5. **Stiffness sensitivity**: Higher stiffness slightly improves baseline convergence (more diagonal dominance).
   Chebyshev acceleration is robust across stiffness scales.

## Areas for Future Investigation

1. **Hessian SPD projection**: Full eigenvalue clamp (instead of cofactor-derivative clamping)
   could improve robustness under extreme deformation.

2. **Forward-backward Gauss-Seidel**: Alternating color group order (symmetric GS) gives the
   iteration matrix real eigenvalues, potentially improving Chebyshev effectiveness.

3. **Adaptive iteration count**: Stop early when RMS displacement falls below a threshold,
   saving computation for well-converged steps (especially beneficial for near-equilibrium states).

4. **Self-contact interaction**: Need to test whether Chebyshev acceleration remains stable
   with particle_enable_self_contact=True.

5. **L-BFGS acceleration**: Could replace Chebyshev with a more general acceleration scheme
   that builds a better approximation of the inverse Hessian over iterations.

6. **Warm-starting**: Use previous timestep's solution as initial guess instead of forward Euler
   prediction. Could reduce the number of iterations needed.
