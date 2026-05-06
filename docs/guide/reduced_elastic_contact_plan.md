# Reduced Elastic Contact Plan

This note captures the agreed plan for adding contact support to reduced elastic
links. It is intentionally scoped to the first useful implementation: elastic
surface samples contacting rigid or static geometry through the VBD path.

## Goals

- Let reduced elastic links participate in contact without turning them into
  full particle or FEM bodies.
- Support a rubber end effector pressed into a wall, with visible modal
  compression/bulging.
- Support two rubber end effectors squeezing and lifting a rigid object.
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
- `update_duals_body_body_contacts`
- `compute_rigid_contact_forces`

The rigid body update should still receive the full contact force and moment at
the deformed elastic contact point. That moves the floating frame as a normal
body DOF.

## Modal Contact Projection

Add an elastic modal contact solve pass after the joint modal solve in each VBD
iteration, or fold contact projection into the existing reduced elastic modal
kernel once the code is stable.

For each contact that touches an elastic body:

1. Reconstruct the same deformed contact point and previous contact point used by
   the rigid contact path.
2. Evaluate the same contact force model as rigid VBD:

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

## Contact Stiffness And AVBD Penalties

Reuse existing contact material arrays:

- `model.shape_material_ke`
- `model.shape_material_kd`
- `model.shape_material_mu`
- `body_body_contact_penalty_k`
- `body_body_contact_material_ke/kd/mu`

Elastic sampled contacts should warmstart and update duals like rigid contacts.
The only difference is how contact points are reconstructed from local data.

`update_duals_body_body_contacts()` must use the deformed elastic contact point
when computing penetration. Otherwise the adaptive penalty sees a different
constraint than the rigid/modal solve.

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
2. Initialize rigid bodies and contact warmstarts.
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
  - Update rigid contact accumulation, dual update, and force collection.

- `newton/_src/solvers/vbd/reduced_elastic_kernels.py`
  - Add `solve_elastic_modes_from_contacts()` or fold contact projection into a
    generalized modal constraint solve.

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

- `test_vbd_elastic_contact_no_rest_geometry_duplicate`
  - An elastic shape does not emit both undeformed rigid narrow-phase contacts
    and deformed elastic surface contacts.

- `test_rubber_gripper_lift_smoke`
  - Two elastic pads squeeze a rigid object and lift it for a short sequence with
    bounded penetration and finite modal amplitudes.

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
- Pads squeeze a rigid box or cylinder, then lift.
- Friction must be enabled so the object is carried through contact.
- Validation:
  - both pads generate contacts
  - object height increases after squeeze
  - object remains between grippers for the checked interval
  - modal amplitudes stay bounded

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
