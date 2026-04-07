# VBD Convergence Investigation Handoff

## Worktree

`/home/horde/newton-vbd-convergence/` — a git worktree from the `AnkaChan/newton` fork,
branch `horde/vbd-convergence-analysis`, remote `mmacklin`.

## What Was Done

Investigated why VBD (Vertex Block Descent) solver converges poorly during cloth-ground
contact in Newton's t-shirt drop scenario. Identified three sources of ill-conditioning
in the per-vertex 3x3 Newton system, prototyped fixes, and measured their impact.

## Key Files Modified

### Kernel fixes (already applied in the worktree):

1. **`newton/_src/solvers/vbd/rigid_vbd_kernels.py` line ~760**
   - `evaluate_body_particle_contact()` — contact damping coefficient
   - Changed: `damping_coeff = body_particle_contact_kd` (was `body_particle_contact_kd * body_particle_contact_ke`)
   - Why: `kd * ke` gave c=312,500 (16,000x critically overdamped). Hessian eigenvalue `c/dt = 188M`. Plain `kd` gives c=50, eigenvalue 30K.
   - The damping energy is `E = c/(2h) * (n.dx)^2` (correct implicit Euler variational form). Force = `(c/h)(n.dx)n`, Hessian = `(c/h) n x n^T`.

2. **`newton/_src/solvers/vbd/particle_vbd_kernels.py` line ~1318**
   - `evaluate_self_contact_force_norm()` — self-contact energy model
   - Changed: replaced log-barrier (`E = k2*ln(d)`, `H = k2/d^2`) with quadratic penalty (`E = k/2*(r-d)^2`, `H = k`)
   - Why: the barrier Hessian grows as 1/d^2, reaching 12M+ near contact. Quadratic penalty has constant H=k=10,000.

### Solver parameter (not a code change, set at runtime):
3. `solver.avbd_beta = 0` — disables the AVBD penalty ramp for particle contacts
   - The ramp (`k += beta * penetration` each iteration) changes the contact stiffness mid-iteration, destabilizing convergence
   - Default `avbd_beta = 100000.0` is set in `solver_vbd.py` line ~558

### Under-relaxation (existing feature):
4. `step_length = 0.5-0.7` eliminates Newton step overshoot oscillation during contact
   - Set via `create_scenario(..., step_length=0.7)` or `solver.step_length = 0.7`

## Stiffness Hierarchy (Before vs After Fixes)

Before fixes, per-vertex Hessian eigenvalues during contact:

| Component | Before | After | Source |
|-----------|--------|-------|--------|
| Contact damping | 188,000,000 | 30,000 | `kd*ke/dt` -> `kd/dt` |
| Self-contact barrier | 12,000,000 | 10,000 | `k2/d^2` -> `k` |
| Self-contact friction | 3,000,000 | 3,000,000 | `mu*F_n*2/(eps*dt)` (unfixed) |
| Elastic membrane | 10,000 | 10,000 | Neo-Hookean mu |
| Inertia | 5,241 | 5,241 | m/dt^2 |
| Bending | 5 | 5 | Dihedral angle k |

Condition number: **37,500,000 -> ~100,000** (still limited by friction and elastic/bending ratio).

## Convergence Results

Measured from identical starting states (trajectory-based, contact regime, 10 iterations):

| Config | Before fixes | After fixes + alpha=0.5 |
|--------|-------------|------------------------|
| Per-iter curve | 93K -> 63 -> 23K -> 152 -> 33K (oscillating) | 362 -> 180 -> 90 -> 58 -> 47 -> 43 -> 42 -> 39 -> 45 -> 38 (monotonic) |
| Reduction in 10 iters | None (chaotic) | 9.5x |

## How to Verify

### Setup
```bash
cd /home/horde/newton-vbd-convergence
# The worktree already has the kernel fixes applied
```

### Test 1: Verify contact damping Hessian eigenvalue
Run a simulation to contact, check `solver.particle_hessians` eigenvalues:
```python
from vbd_convergence_analysis.run_convergence_test import create_scenario
sc = create_scenario(seed=42, iterations=10, enable_self_contact=True, drop_height_range=(5.0, 20.0))
sc['solver'].avbd_beta = 0.0
# Run 35 frames to get into contact...
# Then check: solver.particle_hessians eigenvalues should be < 500K (was 188M before fix)
```

### Test 2: Verify monotonic convergence with alpha=0.5
```python
sc = create_scenario(seed=42, iterations=10, enable_self_contact=True,
                    drop_height_range=(5.0, 20.0), step_length=0.5)
sc['solver'].avbd_beta = 0.0
# Restore a contact-state snapshot, run 1 substep with track_convergence=True
# The per-iteration residual curve should be monotonically decreasing
```

### Test 3: Substep/iteration trade-off
```bash
python3 vbd_convergence_analysis/run_substep_study.py
python3 vbd_convergence_analysis/generate_substep_report.py
# Report at vbd_convergence_analysis/substep_study_report.html
```

### Test 4: Visual verification
```bash
# Render frames already exist at:
ls vbd_convergence_analysis/render_frames/
# progression_zoomed.png shows 8-frame composite of t-shirt drop
```

## Key Physical Parameters (t-shirt scenario)

| Parameter | Value | Source |
|-----------|-------|--------|
| Elastic mu, lambda | 10,000 | `tri_ke`, `tri_ka` in `create_scenario()` |
| Bending k | 5 | `edge_ke` |
| Density | 0.02 | cloth density |
| Particle radius | 0.5 cm | collision radius |
| Self-contact margin | 0.2 cm | `particle_self_contact_radius` |
| Self-contact max_displacement | 0.085 cm | `margin * 0.85 * 0.5` |
| Contact penalty ke | 10,000 | `model.soft_contact_ke` |
| Contact damping kd | 50 | `avg(soft_contact_kd, shape_material_kd)` |
| Frame rate | 60 fps | |
| Default substeps | 10 | `sim_substeps` |
| dt per substep | 1/600 s | `1/60 / substeps` |

## Remaining Issues

1. **Self-contact friction Hessian** (`mu*F_n*2/(eps*dt)` ~3M) — unfixed. The IPC regularization `eps_u = friction_epsilon * dt` creates dt-dependent stiffness.

2. **Elastic/bending ratio** (10,000/5 = 2000:1) — fundamental to the material. Needs hierarchical solve or primal/dual splitting to address.

3. **Self-contact max_displacement clamp** (0.085 cm) — limits max cloth velocity to `0.085 * fps * substeps` cm/s. At 10 substeps = 51 cm/s, well below impact velocity (~211 cm/s for 22.7 cm drop). Not a solver issue, it's a CFL condition for the penetration-free collision detection.

## Analysis Scripts

| Script | Purpose |
|--------|---------|
| `run_convergence_test.py` | Core: `create_scenario()` and `run_scenario()` |
| `run_trajectory_convergence.py` | Trajectory-based per-iteration convergence |
| `run_rollout_comparison.py` | Full rollout per-frame residual |
| `run_substep_study.py` | Substep/iteration trade-off |
| `generate_report_v2.py` | Convergence report |
| `generate_substep_report.py` | Substep study report |

## Key Measurement: Force Residual

The convergence metric is `||nabla G(x)||` — the RMS per-vertex gradient of the implicit Euler variational energy:

```
G(x) = 1/(2h^2) * ||x - y||^2_M + E_elastic(x) + E_bending(x) + E_contact(x)
```

Computed by `compute_force_residual` kernel in `particle_vbd_kernels.py` (line ~3622). This evaluates inertia + elastic + bending + contact forces at the current positions. At the exact implicit Euler solution, this is zero.

## Documents

- `vbd_energies.md` — detailed write-up of the three fixes with math
- `vbd_analysis.md` — running analysis log (older, partially outdated)
- `HANDOFF.md` — this file
