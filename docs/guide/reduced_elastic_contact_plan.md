# Reduced Elastic Contact Plan

This note captures the agreed plan for adding contact support to reduced elastic
links. It is intentionally scoped to the first useful implementation: elastic
surface samples contacting rigid or static geometry through the VBD path.

## Implementation Status

The first pass is implemented. Reduced elastic shapes now generate point-sampled
contacts from their deformed surface vertices against rigid/static shapes, VBD
uses the deformed contact point for rigid body contact forces, and a separate
modal contact pass projects the same fixed-stiffness contact model into the
elastic coordinates. The initial examples are
`basic_reduced_elastic_wall_contact`, `basic_reduced_elastic_gripper_contact`,
and `basic_reduced_elastic_scraper_contact`, each with its own behavior test.

## Goals

- Let reduced elastic links participate in contact without turning them into
  full particle or FEM bodies.
- Support a rubber end effector pressed into a wall, with visible modal
  compression/bulging.
- Support two rubber end effectors squeezing a free dynamic rigid object while
  it friction-lifts under gravity.
- Reuse the existing reduced elastic surface samples. Do not add a separate
  contact-sample API or storage for the first pass.
- Keep the solver force model consistent between the rigid body update and the
  elastic modal update, including damping and friction where applicable.

## Non-Goals For The First Pass

- Elastic-vs-elastic contact.
- Exact elastic triangle face contact, such as a rigid cone tip intersecting the
  middle of an elastic triangle.
- Hydroelastic contact area integration over the deforming elastic surface.
- Arbitrary contact-point interpolation during the solve.

The later face-contact extension should be barycentric:

```text
sample_ids = [s0, s1, s2]
weights    = [w0, w1, w2]
```

The first pass stores only one scalar sample id:

```text
sample_id = s
weight    = 1
```

## Existing Data To Reuse

Reduced elastic rendering already creates a surface cache during
`ModelBuilder.finalize()`:

- `model.elastic_shape_vertex_local`
- `model.elastic_shape_vertex_phi`
- `model.elastic_shape_shape`
- `model.elastic_shape_body`
- `model.elastic_shape_vertex_start`
- `model.elastic_shape_vertex_count`
- `model.elastic_shape_indices`

These are sufficient for point-sampled contact. Each elastic surface vertex has
the undeformed body-local point and the flattened modal values:

```text
x_local_deformed = x_local + sum_i phi_i(x_local) q_i
x_world = X_body * x_local_deformed
dx_world / dq_i = R_body * phi_i(x_local)
```

There is no separate elastic shape type today. A shape is still a normal
`GeoType.BOX`, `GeoType.MESH`, etc. Elastic ownership is identified through:

```text
shape -> model.shape_body[shape] -> model.body_elastic_index[body] >= 0
```

That identifies the elastic body, but not the surface sample that generated a
contact. The contact buffer therefore needs one scalar elastic sample id per
contact side.

## Contact Buffer Additions

Add two optional integer arrays to `Contacts`:

```text
rigid_contact_elastic_sample0
rigid_contact_elastic_sample1
```

Semantics:

- `-1` means the side is an ordinary rigid/static contact side.
- `>= 0` means the contact point for that side is an index into
  `model.elastic_shape_vertex_local` and `model.elastic_shape_vertex_phi`.
- For the first pass, only one side is expected to be elastic.
- Normal rigid contacts must write `-1` for both sides.

The existing contact shape/body arrays remain the source of truth for which body
owns each side. The new sample id is only the modal Jacobian lookup.

Store the existing `rigid_contact_point0/1` values as local points as before.
For an elastic side, store the undeformed elastic body-local surface point. The
solver will reconstruct the deformed contact point from the sample id and modal
state.

## Contact Generation

Add a reduced elastic point-vs-shape contact generation pass after the ordinary
rigid narrow phase.

For each eligible elastic surface vertex and rigid/static shape pair:

1. Get the elastic body frame and modal state.
2. Evaluate the deformed elastic sample on the fly:

   ```text
   x_e_local = elastic_shape_vertex_local[s]
             + sum_i elastic_shape_vertex_phi[s, i] q_i
   x_e_world = X_we * x_e_local
   ```

3. Transform `x_e_world` into the rigid shape local frame.
4. Evaluate the rigid shape signed distance and normal using the same SDF logic
   as `create_soft_contacts()`.
5. Let `d` be signed distance from the rigid shape surface to the elastic point
   along the rigid shape normal. Let `n` be the world normal pointing from the
   rigid shape toward the elastic point.
6. Generate a contact when:

   ```text
   d <= shape_gap_rigid + shape_gap_elastic + shape_margin_rigid + shape_margin_elastic
   ```

   or equivalently using the same margin/gap convention as existing rigid
   contacts.

7. Write the rigid side as side 0 and the elastic side as side 1 for the MVP:

   ```text
   rigid_contact_shape0 = rigid_shape
   rigid_contact_shape1 = elastic_shape
   rigid_contact_point0 = closest rigid point in rigid body local frame
   rigid_contact_point1 = undeformed elastic sample in elastic body local frame
   rigid_contact_normal = n
   rigid_contact_margin0 = shape_margin[rigid_shape]
   rigid_contact_margin1 = shape_margin[elastic_shape]
   rigid_contact_elastic_sample0 = -1
   rigid_contact_elastic_sample1 = s
   ```

The first implementation can use a simple launch over elastic surface samples
and candidate rigid shapes. For rubber-pad examples this is small. A follow-up
can reduce the pair set through deformed elastic AABBs and the existing broad
phase.

Normal rigid narrow phase must not also emit undeformed contacts for elastic
shapes. The cleanest first change is to make the ordinary contact writer skip
any shape pair where either side belongs to an elastic body, then let the elastic
surface contact pass own those pairs.

## Solver Evaluation

Any VBD rigid contact kernel that transforms contact local points to world space
must become elastic-aware.

For an ordinary side:

```text
x_world = X_body * contact_point_local
x_prev  = X_body_prev * contact_point_local
```

For an elastic side:

```text
x_local(q) = elastic_shape_vertex_local[s]
           + sum_i elastic_shape_vertex_phi[s, i] q_i
x_world = X_body * x_local(q)

x_local_prev(q_prev) = elastic_shape_vertex_local[s]
                     + sum_i elastic_shape_vertex_phi[s, i] q_prev_i
x_prev = X_body_prev * x_local_prev(q_prev)
```

This affects:

- `accumulate_body_body_contacts_per_body`
- `compute_rigid_contact_forces`

The rigid body update should still receive the full contact force and moment at
the deformed elastic contact point. That moves the floating frame as a normal
body DOF.

## Modal Contact Projection

Add a separate elastic modal contact solve pass after the joint modal solve in
each VBD iteration. Do not fold contact projection into
`solve_elastic_modes_from_joint_constraints()` for the initial implementation;
that would make the joint kernel too broad and harder to validate. The first
pass should keep the solver split explicit:

```text
solve_elastic_modes_from_joint_constraints()
solve_elastic_modes_from_contacts()
```

If the two paths later share substantial code, factor the common force/Jacobian
helpers into device functions rather than merging the launch kernels.

For each contact that touches an elastic body:

1. Reconstruct the same deformed contact point and previous contact point used by
   the rigid contact path.
2. Evaluate the same fixed-stiffness contact force model as the rigid contact
   path used for elastic contacts:

   ```text
   penetration = margin_sum - dot(n, x1 - x0)
   f_contact = normal spring + normal damping + friction
   K_contact = matching contact Hessian approximation
   ```

3. Select the force on the elastic side. With the MVP side ordering
   `shape0 = rigid`, `shape1 = elastic`, this is the side-1 force.
4. Project that force to each modal coordinate:

   ```text
   phi_world_i = R_body * elastic_shape_vertex_phi[s, i]
   Q_i = dot(phi_world_i, f_elastic)
   grad_i += -Q_i
   H_ii += phi_world_i^T K_contact phi_world_i
   ```

5. Include the same normal damping and friction terms in the modal projection
   that the rigid contact path uses. The dipper-arm debugging showed that
   projecting only stiffness while the rigid path includes damping causes slow or
   wrong convergence.

The sign convention must be locked by tests. The invariant is that increasing a
modal coordinate in the direction of contact force should reduce the modal
energy gradient.

## Follow-Up: Modal Block Assembly Optimization

Contacts, joint attachments, and future point loads all use the same projection
identity, but they do not need to share one generic implementation path. Treat
the following as the common math model for checking signs, dimensions, residuals,
and optimized kernels:

```text
F_f       Cartesian force on the elastic source point [N], shape [3]
K_f       Cartesian force derivative / stiffness [N/m], shape [3, 3]
Phi_f     local modal displacement map at the source, shape [3, mode_count]
J_f       world modal Jacobian, J_f = R_body Phi_f, shape [3, mode_count]
```

`Phi_f` is intentionally not tied to a single stored surface sample. It may be a
vertex sample, a joint endpoint sample, a barycentric interpolation of triangle
samples, or a future custom basis evaluation. The per-body modal block receives
the source contribution:

```text
g += -J_f^T F_f
H +=  J_f^T K_f J_f
```

Keep contacts and joints specialized where that is clearer or faster. Contact
code can exploit normal-force rank-1 structure and contact-specific friction
state. Joint code can exploit endpoint signs, joint type, and projection rank.
The shared requirement is that each path contributes to the same per-body modal
gradient and Hessian with consistent signs.

### Later: Optional Source Representation

A compact source buffer is not part of the near-term optimization plan. First
optimize the existing specialized contact and joint paths directly: local-space
projection, projector-aware Hessian assembly, triangular block fill, tile
accumulation, and Cholesky/LDLT solves.

After those pieces are validated, a compact source buffer may still be useful
for point loads or future interpolated face contact. It should not force joints
through a generic record format if a joint-specific kernel is simpler. A future
source record could look like:

```text
source_body
source_phi_ref       sample id, endpoint id, or interpolated-source id
source_side_sign     +1 or -1 when the endpoint convention needs it
source_force         F_f
source_hessian       K_f, or a structured/factored representation
```

For contacts, a narrower near-term data-structure improvement is to build or
sort compact per-elastic-body contact ranges so each body processes only its own
contacts. This avoids the current body-by-all-contacts scan without committing
to a general source representation.

### Structured Contact And Joint Forces

Normal contact is the common 1D cheap case. The Cartesian stiffness is rank 1:

```text
K_f = k n n^T
a = J_f^T n
H += k a a^T
```

This only asks how much each mode moves the source point along the contact
normal. Damping along the normal has the same structure and should be folded
into the scalar `k` for the implicit step. More general projected constraints
can use the same idea:

```text
K_f = D W D^T
H += (J_f^T D) W (D^T J_f)
```

where `D` contains one or more Cartesian force directions. Joint attachments are
usually not 1D:

```text
fixed translational endpoint: P = I          rank 3
prismatic endpoint:           P = I - aa^T   rank 2
normal contact:               P = n n^T      rank 1
```

So the optimization should be rank- or projector-aware, not contact-only and not
purely generic dense `3 x 3`. Prefer factored/projector forms for contact
normals and simple joint projections when they are clearer and cheaper than
materializing a dense Hessian.

### Local Space Projection

For sources on the same elastic body, rotate source terms into the body-local
frame once:

```text
F_local = R_body^T F_f
K_local = R_body^T K_f R_body
g += -Phi_f^T F_local
H +=  Phi_f^T K_local Phi_f
```

For factored 1D sources, rotate the direction instead:

```text
n_local = R_body^T n
a = Phi_f^T n_local
H += k a a^T
```

This keeps `Phi_f` local and avoids repeatedly rotating every modal column for
every force source. Apply this inside each specialized contact or joint path,
rather than requiring a single shared source-projection kernel.

### Dense Block Assembly

Assemble the modal block in local scratch storage per elastic body:

- Accumulate `g` and `H` into per-body tiles.
- Fill only one triangle of the symmetric block during assembly, then mirror
  only if a later solve path requires full storage.
- For low-rank/projector sources, use modal outer products directly instead of
  generic `3 x mode_count` by `mode_count x mode_count` multiplies.
- For dense `K_f` sources, compute `B = K_f Phi_f` and then
  `H += Phi_f^T B` using tiled operations.

Specialized contact and joint kernels should make source type explicit enough to
choose rank-1, rank-2, rank-3 projector, or generic dense paths as appropriate.

### Tile-Based Solve

Replace the fixed-sweep Gauss-Seidel dense modal solve with a small dense tile
solve:

1. Assemble the symmetric modal block tile.
2. Add modal inertia, damping, stiffness, and all source Hessian terms.
3. Compute a tile-local residual norm before the solve.
4. Factor the tile with Cholesky or LDLT.
5. Solve for the modal increment.
6. Report the post-solve residual norm and update norm.

The base reduced matrices are constant for a fixed basis:

```text
M_r = Phi^T M Phi
C_r = Phi^T C Phi
K_r = Phi^T K Phi
```

They can be precomputed. Contact and joint source Hessians remain state
dependent, so the full block generally still needs per-iteration assembly and
factorization. Since mode counts are small, the Cholesky/LDLT cost should be
minor compared with contact generation and source assembly, and it removes the
need to tune dense Gauss-Seidel sweep counts.

## Contact Stiffness

Do not use AVBD adaptive contact penalties for the initial elastic contact pass.
Use fixed contact material properties from the shapes:

- `model.shape_material_ke`
- `model.shape_material_kd`
- `model.shape_material_mu`

For the first implementation, compute an effective pair value directly from the
two shapes. Matching the current rigid VBD material convention is a reasonable
default:

```text
ke = 0.5 * (shape_material_ke[shape0] + shape_material_ke[shape1])
kd = 0.5 * (shape_material_kd[shape0] + shape_material_kd[shape1])
mu = sqrt(shape_material_mu[shape0] * shape_material_mu[shape1])
```

This removes contact warmstart and dual-update concerns from the MVP. Once
fixed-stiffness elastic contact is correct, adaptive penalties can be revisited
as a separate convergence improvement.

## Collision And Solver Ordering

Expected external loop remains:

```python
model.collide(state_in, contacts)
solver.step(state_in, state_out, control, contacts, dt)
```

The collision path should evaluate elastic body frames and modal coordinates
from `state_in.joint_q` for elastic bodies, rather than relying on possibly stale
`state_in.body_q`. Rigid/static shapes continue to use `state_in.body_q`.

Inside `SolverVBD.step()`:

1. Initialize elastic bodies and sync owner joints to body frames.
2. Initialize rigid bodies. Elastic sampled contacts use fixed shape-material
   stiffness, so they do not need AVBD contact warmstarts in the initial pass.
3. Per iteration:
   - solve rigid bodies using elastic-aware contact point reconstruction
   - solve elastic modes from joints
   - solve elastic modes from contacts
   - solve particles
4. Finalize rigid and elastic state.

## Implementation File Map

- `newton/_src/sim/contacts.py`
  - Add scalar elastic sample arrays to `Contacts`.
  - Clear/fill them with `-1`.

- `newton/_src/sim/collide.py`
  - Extend `ContactWriterData` so ordinary rigid contact writing can skip
    elastic-owned shapes or at least mark normal contacts as non-elastic.
  - Launch the reduced elastic surface contact pass.

- `newton/_src/geometry/kernels.py` or a new contact kernel module
  - Add the elastic surface point-vs-shape contact generation kernel.
  - Reuse SDF query code from `create_soft_contacts()` where possible.

- `newton/_src/solvers/vbd/rigid_vbd_kernels.py`
  - Add helper functions to evaluate contact-side world points with optional
    elastic sample ids.
  - Refactor contact force/Hessian evaluation so rigid and modal paths share the
    same force model.
  - Update rigid contact accumulation and force collection. Skip elastic-contact
    AVBD dual updates in the first pass.

- `newton/_src/solvers/vbd/reduced_elastic_kernels.py`
  - Add a separate `solve_elastic_modes_from_contacts()` kernel. Keep it
    separate from the joint modal solve for the first pass; share only small
    device helpers if needed.

- `newton/_src/solvers/vbd/solver_vbd.py`
  - Allocate/forward new contact arrays.
  - Launch elastic contact modal projection every iteration.

- `newton/tests/test_reduced_elastic_body.py`
  - Add analytic contact generation and modal projection tests.

## Tests

Start with tests that fail before implementation.

- `test_elastic_contact_arrays_default_to_nonelastic`
  - Ordinary rigid contacts have `rigid_contact_elastic_sample0/1 == -1`.

- `test_elastic_surface_contact_generation_plane`
  - A single elastic surface sample displaced through a plane generates one
    contact with the expected shape order, normal, undeformed local point, and
    scalar sample id.

- `test_elastic_surface_contact_uses_deformed_position`
  - Changing `q_i` moves the deformed sample into or out of contact without
    changing the rigid frame.

- `test_vbd_elastic_contact_modal_projection_wall`
  - One modal coordinate against a fixed wall matches the analytic one-step
    implicit solve using the same contact stiffness.

- `test_vbd_elastic_contact_damping_projection`
  - Modal contact projection includes damping consistently with the rigid
    contact path.

- `test_vbd_elastic_contact_uses_fixed_shape_stiffness`
  - Changing `ShapeConfig.ke/kd/mu` changes the elastic contact response
    directly, without relying on AVBD warmstart or dual update state.

- `test_vbd_elastic_contact_no_rest_geometry_duplicate`
  - An elastic shape does not emit both undeformed rigid narrow-phase contacts
    and deformed elastic surface contacts.

- `test_rubber_gripper_lift_smoke`
  - Two elastic pads squeeze a dynamic rigid object and lift it for a short
    sequence with bounded penetration, finite modal amplitudes, visible
    compression on both pads, and bounded relative motion under gravity.

- `test_plastic_chair_leg_stick_slip`
  - A diagonal elastic chair-leg proxy is dragged along a high-friction ground
    plane. Modal elastic energy should rise during sticking intervals, then drop
    when the contact slips.

## Examples

### Rubber End Effector Against Wall

- Rigid or kinematic actuator drives an elastic pad into a fixed wall.
- Elastic pad uses compression plus lateral bulging modes.
- Render deformed pad mesh and contact normals.
- Validation:
  - bounded penetration
  - modal compression has expected sign
  - static displacement roughly follows the series stiffness relation between
    modal stiffness and contact stiffness

### Two Rubber End Effectors Pick Up Object

- Two prismatic rigid fingers carry elastic pads.
- Pads squeeze a free dynamic rigid sphere or rounded object, then lift and
  optionally shake it.
- There is no object guide or prescribed object path. Friction must hold the
  object under gravity while the gripper pads carry the contact load.
- Validation:
  - both pads generate contacts
  - object height increases after squeeze
  - both pads have visible modal compression
  - object remains between grippers for the checked interval
  - object has bounded relative motion under lift/shake
  - modal amplitudes stay bounded

### Plastic Chair Leg Stick-Slip

- A diagonal plastic chair-leg proxy is pushed laterally along a high-friction
  ground plane, with the leg angled so the contact point loads bending modes as
  the frame advances.
- The top of the leg is driven by a kinematic rigid body or prismatic actuator;
  the bottom contact patch is an elastic surface sample set against the ground.
- The contact should stick while tangential friction can hold, building elastic
  bending energy in the modes, then slip once the accumulated tangential load
  exceeds friction.
- This is a good qualitative test for contact friction plus modal contact
  projection because the visible motion should look like a plastic chair leg
  scraping: slow elastic wind-up, sudden release, then repeated stick/slip.
- Validation:
  - tangential motion has alternating low-slip and high-slip intervals
  - modal elastic energy rises before each slip event and drops after release
  - contact penetration remains bounded
  - the average forward speed of the top drive remains larger than the average
    forward speed of the contact point during sticking intervals

## Follow-Up: Face Contact

The scalar sample-id data can be generalized without changing the math:

```text
x_local = sum_a w_a x_a
phi_i   = sum_a w_a phi_i(x_a)
```

That will support rigid feature contact against the interior of an elastic
triangle. The contact buffer can evolve from scalar sample ids to three sample
ids plus barycentric weights per side.

Do not add this until point-sampled contact is working and tested.
