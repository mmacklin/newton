# VBD Solver Issues

Tracking convergence and stability issues found during VBD analysis on a t-shirt self-contact benchmark (6436 vertices, 4 substeps, 10 iterations, 60 FPS, meter-scale).

**Live report:** [vbd_ablation.html](https://principles-folders-neil-relocation.trycloudflare.com/newton-vbd/vbd_ablation.html)

---

## Summary of All Fixes

Newton Main (original VBD code) is **catastrophically unstable** at 4 substeps with self-contact — all seeds produce NaN. Seven issues were identified and fixed:

### Kernel-Level Fixes (code changes)

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | Log-barrier self-contact (H = k^2/d^2, unbounded) | **Critical** — causes NaN | Quadratic penalty (H = k, constant) |
| 2 | Contact damping c = kd * ke (16,000x overdamped) | **High** — 5x jitter increase | Absolute damping c = kd |
| 3 | Friction Hessian approximation not scale-invariant | Medium — analysis only | P/D friction model implemented as alternative |
| 4 | Zero friction Hessian at rest (u == 0) | Low — minimal jitter impact | Evaluate limit: hessian_scale = 2/eps_u |

### Configuration Fixes (parameter changes)

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 5 | Self-contact buffer overflow (silent contact drops) | **Critical** — causes self-interpenetration and solver divergence | Increase buffers from 16/20 to 512/512 |
| 6 | Particle masses near zero (1e-6 kg) | **High** — 100,000:1 stiffness ratio, solver diverges after 15 iterations | Clamp min mass to 0.01 kg (10g) |
| 7 | Contact stiffness too high (ke = 1e5) | **High** — stiffness ratio still 1000:1 even with mass clamp | Reduce to ke = kf = 1e4 |

### Final Recommended Parameters

```python
# Contact stiffness
model.soft_contact_ke = 1e4        # normal contact stiffness
model.soft_contact_kf = 1e4        # friction stiffness (P/D model, = ke)
model.soft_contact_kd = 1e-2       # contact damping
model.soft_contact_mu = 0.25       # friction coefficient

# Solver
solver = SolverVBD(
    model, iterations=10,
    particle_self_contact_radius=0.002,      # 2mm collision shell
    particle_self_contact_margin=0.02,       # 20mm broad-phase margin
    particle_vertex_contact_buffer_size=512,  # was 16 (overflowed!)
    particle_edge_contact_buffer_size=512,    # was 20 (overflowed!)
    particle_collision_detection_interval=-1, # detect once per step()
    friction_epsilon=1e-4,                   # velocity threshold (m/s)
)
solver.avbd_beta = 0.0             # disable AVBD ramp
solver.k_start_body_contact = 2500 # fixed body contact penalty

# Mass clamp (post model.finalize)
mass_np = model.particle_mass.numpy()
mass_np[mass_np < 0.01] = 0.01    # 10g minimum per vertex
model.particle_mass.assign(wp.array(mass_np, dtype=wp.float32))
inv_mass_np = np.where(mass_np > 0, 1.0 / mass_np, 0.0)
model.particle_inv_mass.assign(wp.array(inv_mass_np, dtype=wp.float32))
```

### Result

With all fixes: median RMS velocity 0.0026 m/s in resting phase, 121 pops (out of 360 possible), convergence ratio 0.17 (83% reduction per 10 iterations). Best seed achieved **5 pops** — essentially at rest.

Without fixes: NaN on all seeds (Newton Main), or 0.05+ RMS with 360 pops and diverging iterations.

---

## Issue 1: Log-Barrier Self-Contact Causes Unbounded Hessian

**File:** `newton/_src/solvers/vbd/particle_vbd_kernels.py` — `evaluate_self_contact_force_norm()`

**Problem:** The original self-contact potential used a log-barrier whose Hessian scales as k^2/d^2:

```python
dEdD = -k * k / dis
d2E_dDdD = k * k / (dis * dis)
```

As distance d -> 0 between contacting surfaces, the Hessian grows without bound (10^7+ observed), dominating the 3x3 block system and stalling Gauss-Seidel convergence. With 4 substeps this is catastrophically unstable — all 4 test seeds produce NaN.

**Fix:** Quadratic penalty E = k/2 * (r - d)^2 with constant Hessian H = k:

```python
dEdD = -k * penetration_depth
d2E_dDdD = k
```

Convergence rate becomes insensitive to contact stiffness (100x change in sc_ke barely affects convergence ratio). Reduces worst-case self-contact eigenvalue by ~1200x.

**Status:** Fixed in worktree. Confirmed: reverting causes 100% NaN across all seeds.

---

## Issue 2: Contact Damping Overdamped by Factor of ke

**File:** `newton/_src/solvers/vbd/rigid_vbd_kernels.py` — body-particle contact evaluation (~line 768)

**Problem:** Body-particle contact damping coefficient was `c = kd * ke`, making the damping Hessian eigenvalue `kd * ke / dt`. With typical values (kd=0.01, ke=1e4, dt=1/240), this gives eigenvalues ~10^8 — roughly 16,000x critically overdamped. The damping term dominates the elastic term by orders of magnitude, effectively freezing contact dynamics.

```python
# Old (overdamped):
damping_coeff = body_particle_contact_kd * body_particle_contact_ke
```

**Fix:** Use absolute damping `c = kd`, giving eigenvalue kd/dt ~ 2.4 which is ~2x critical:

```python
damping_coeff = body_particle_contact_kd
```

Reduces damping Hessian eigenvalue by 6250x. Measured 5x reduction in resting-phase jitter. With the mass clamp (Issue 6), the progressive fix hierarchy shows "+ Quadratic SC penalty" (Issue 1 only) at 0.0059 RMS, while "+ Absolute damping" (Issues 1+2) drops to 0.0026 RMS.

**Status:** Fixed in worktree.

---

## Issue 3: Friction Hessian Approximation Is Not Scale-Invariant

**File:** `newton/_src/solvers/vbd/particle_vbd_kernels.py` — `compute_friction()`
**File:** `newton/_src/solvers/vbd/rigid_vbd_kernels.py` — `compute_projected_isotropic_friction()`

**Problem:** The friction function `g(u) = f1_SF_over_x(||u||) * u` has the true Jacobian:

```
dg/du = f1_SF_over_x * I  +  (d(f1_SF_over_x)/d||u||) * (u_hat x u_hat)
```

The code drops the rank-1 term for stability, keeping only the isotropic part. The **force** is scale-invariant but the **Hessian** is not: it scales as `1/s` when inputs are scaled by `s`. Since `u` is a displacement and `eps_U = friction_epsilon * dt`, the Hessian stiffness grows as `1/dt` with more substeps — an artifact of the approximation, not the physics.

The dropped rank-1 correction is always negative, so the approximation **overestimates stiffness in the slip direction**. In the Coulomb regime the true slip-direction eigenvalue is zero, but the approximation gives `mu*f_n/||u||` — entirely spurious resistance to continued sliding.

**P/D friction model (Macklin et al. 2020):** Implemented as an alternative. Uses a linear spring `kf` in the stick regime with explicit stiffness parameter independent of dt, and Coulomb in slip. The P/D model avoids the dt-scaling artifact by parameterizing stick stiffness directly. Setting `kf = ke` keeps normal/tangential stiffness balanced.

At ke=kf=1e4, the P/D model measures 0.0055 RMS vs IPC's 0.0035. P/D has better convergence rate (0.09 vs 0.17) but slightly more jitter — the tradeoff is between clean conditioning and effective stick stiffness.

**Interactive comparison:** [friction_comparison.html](https://principles-folders-neil-relocation.trycloudflare.com/newton-vbd/friction_comparison.html)

**Status:** P/D friction implemented and wired through `model.soft_contact_kf`. Selected at compile time via `VBD_FRICTION_MODEL` module variable. The IPC `eps_U = friction_epsilon * dt` formulation is correct (friction_epsilon is a velocity threshold); the Hessian approximation is the root issue, shared by both models.

---

## Issue 4: Zero Friction Hessian at Rest

**File:** `newton/_src/solvers/vbd/particle_vbd_kernels.py` — `compute_friction()`
**File:** `newton/_src/solvers/vbd/rigid_vbd_kernels.py` — `compute_projected_isotropic_friction()`

**Problem:** When tangential slip is exactly zero (`u_norm == 0`), both friction functions originally returned a zero Hessian. The regularized friction function has a well-defined nonzero derivative at the origin: f1(x)/x -> 2/eps_u as x -> 0. Zero Hessian means the solver doesn't see friction stiffness for resting contacts, allowing drift->snap-back oscillation.

**Fix:** Evaluate the limit in the `u == 0` branch. For IPC: `hessian_scale = mu * fn * 2/eps_u`. For P/D: `hessian_scale = friction_kf`. Force remains zero.

**Impact:** Minimal measurable effect on jitter in the ablation study (0.0033 vs 0.0035 RMS). The mass clamp (Issue 6) and buffer fix (Issue 5) had much larger effects.

**Status:** Fixed in worktree.

---

## Issue 5: Self-Contact Buffer Overflow (Silent Contact Drops)

**File:** `newton/_src/geometry/kernels.py` — collision detection kernels
**File:** `newton/_src/solvers/vbd/tri_mesh_collision.py` — `resize_flags` array

**Problem:** The self-contact collision detection uses fixed-size per-element buffers (`particle_vertex_contact_buffer_size` and `particle_edge_contact_buffer_size`). When a vertex or edge has more contacts than the buffer can hold, excess contacts are **silently dropped** — the `resize_flags` array is set but **never checked or reported** by the solver.

**Diagnosis:** With a 6436-vertex t-shirt mesh crumpled on the ground:
- Default buffer sizes: 16 vertex / 20 edge
- **65% of vertices** exceeded the vertex buffer (max count: 257)
- **78% of edges** exceeded the edge buffer (max count: 483)
- `resize_flags = [1, 0, 1]` — confirmed overflow in both VT and EE channels

With dropped contacts, each iteration sees a different, incomplete set of contact forces. This caused:
- The solver to **actively diverge** — residual grew from 21 to 51 over 40 iterations
- An alternating high-low residual pattern (color groups fighting each other with inconsistent contact data)
- **80 iterations was worse than 10** (reduction 0.64 vs 0.17)
- All visible self-interpenetration in videos

**Fix:** Increase buffer sizes to 512/512 (sufficient for max observed counts of 257 VT / 483 EE). Newton should add overflow detection + auto-resize or at minimum a warning.

**Status:** Fixed in ablation study configuration. Newton needs a proper fix (dynamic resize or warning).

---

## Issue 6: Particle Masses Near Zero

**Problem:** The cloth mesh has some triangles with very small area, giving per-vertex masses as low as 1e-6 kg. The VBD block-GS solve uses `m/h^2` as the inertia regularizer in the 3x3 system. With mass=1e-6 and h=1/240: `m/h^2 = 0.058`.

Contact Hessian diagonals reach 10^4-10^7 depending on ke. The stiffness ratio (contact / inertia) peaked at **38 million:1** with ke=1e5, meaning the block solve was entirely determined by contact forces with no inertial regularization. This caused the Gauss-Seidel iteration to oscillate rather than converge.

**Stiffness sweep results** (ke vs min_mass, measuring convergence ratio over 20 iterations):

| ke | min_mass | inertia (m/h^2) | ratio (ke/inertia) | conv reduction | jitter (m/s) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1e3 | 1e-1 | 5760 | 0.17 | 0.38 | 0.011 |
| **1e4** | **1e-2** | **576** | **17** | **0.07** | **0.006** |
| 1e4 | 1e-1 | 5760 | 1.7 | 0.01 | 0.007 |
| 1e5 | 1e-3 | 57.6 | 1,740 | 0.04 | 0.003 |
| 1e5 | 1e-1 | 5760 | 17 | 0.19 | 0.004 |

Sweet spot: **ke=1e4, min_mass=0.01 kg** — stiffness ratio 17:1, 93% convergence reduction, 6mm/s jitter.

**Fix:** Clamp `model.particle_mass` to a minimum of 0.01 kg (10g) after `model.finalize()`. Also update `model.particle_inv_mass` accordingly.

**Status:** Fixed in ablation study. Newton should implement a `min_particle_mass` parameter in the solver or builder.

---

## Issue 7: Contact Stiffness Too High

**Problem:** With `soft_contact_ke = 1e5` (the value used in early ablation runs), even with the mass clamp the stiffness ratio was ~1000:1. IPC friction's implicit stick stiffness `2/(friction_epsilon * dt)` reached ~480,000 at this ke level, causing convergence divergence on some seeds (conv_ratio > 9.0 observed).

Higher ke also amplifies the overdamped damping bug (Issue 2): `kd * ke` with ke=1e5 gives 10x more overdamping than ke=1e4.

**Fix:** Reduce to `soft_contact_ke = soft_contact_kf = 1e4`. This gives a stiffness ratio of ~17:1 with the mass clamp, which block-GS can handle well.

**Status:** Fixed in ablation study parameters.
