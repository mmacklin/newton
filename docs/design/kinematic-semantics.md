# Kinematic semantics in Newton

Status: **Proposal**  
Scope: rigid bodies, particles, articulations, collision, coupling, and solver conformance  
Baseline audited: `newton-physics/newton` `main` at
[`fddee2be`](https://github.com/newton-physics/newton/commit/fddee2bebe749b5461a596f7624d445fc3d0e57b)
(2026-08-04)  
Kamino behavior rechecked on `main` at
[`108c8a4f`](https://github.com/newton-physics/newton/commit/108c8a4fa753ce332a185cb3817c07f93660e70f)
(2026-09-10)

Related work: [PR #3659, “Refactor VBD pose baselines”](https://github.com/newton-physics/newton/pull/3659)

## Core semantics

Kinematic bodies and particles have infinite effective inertia. As such, their
velocity does not change during a timestep. Their pose or position is naturally
integrated over the timestep from the beginning-of-step state using this
constant velocity.

```text
v(t + 1) = v(t)
q(t + 1) = advance(q(t), v(t), dt) = advance(q(t), v(t + 1), dt)
```

For particles, `advance` is vector addition. For rigid bodies, it advances the
world-space center of mass (COM) and integrates the orientation using the
agreed quaternion update while preserving the body-origin-to-COM offset.

## Decision summary

Newton should use one rule everywhere:

> **Flags define who owns motion. Mass and inertia define dynamic response.
> Inverse mass and inverse inertia are derived numerical data.**

The concrete contract is:

1. `BodyFlags.KINEMATIC` is the only way to make a rigid body kinematic.
   `mass == 0` and `body_inv_mass == 0` never classify motion.
2. Add `ParticleFlags.KINEMATIC`. Rename the existing `ParticleFlags.ACTIVE`
   bit to `ParticleFlags.DYNAMIC`, retaining `ACTIVE` as a deprecated alias so
   its numeric value and existing serialized data remain compatible.
   Particles with neither `DYNAMIC` nor `KINEMATIC` are ignored by the solver:
   they do not collide, are not rasterized to the grid, and do not participate
   in constraints. They may serve as capacity reserves or belong to another
   solver in coupled workflows.
3. A kinematic entity's beginning-of-step pose or position **and** velocity are
   authored inputs. Solvers copy the velocity to the output unchanged and do
   not apply forces to the entity. Poses and positions are
   **forward-integrated** from the beginning-of-step state and velocity. If the
   pose or position should remain constant over the step, the input velocity
   must be zero.
4. Kinematic entities still participate in contact and constraints as moving,
   infinite-mass boundaries. Their authored velocity contributes to relative
   velocity, damping, restitution, and friction.
5. A dynamic entity must have valid positive mass. A zero-mass dynamic entity
   is invalid, following a deprecation window.
6. Static rigid geometry remains a shape attached to the world
   (`shape_body == -1`).
   Newton does not need `BodyFlags.STATIC`; a stationary kinematic body is not
   static because it still has authored state and may move later.
7. Any link in an articulation may be kinematic at the model level, including
   an internal link or both endpoints. A solver must implement the absolute
   world-space boundary condition exactly or reject the model during solver
   construction with a clear capability error.

The direct answer to “is a kinematic body's pose integrated from its velocity?”
is **yes**. The user authors the velocity, and the solver follows it exactly
from the current pose. The velocity itself is copied through unchanged.
Continuous prescribed motion requires a mutually consistent pose and velocity
on every solver substep. The authored velocity should be computed from the
current pose and the target end-of-step pose.

## Why the current model is ambiguous

Rigid bodies already have explicit `DYNAMIC` and `KINEMATIC` flags, and model
finalization requires exactly one of them. However, inverse mass remains a
second de facto classifier in solver code:

- The shared rigid integrator passes `BodyFlags.KINEMATIC` through unchanged,
  but a non-kinematic zero-inverse-mass body still advances at its existing
  velocity. See
  [`solver.py`](https://github.com/newton-physics/newton/blob/fddee2bebe749b5461a596f7624d445fc3d0e57b/newton/_src/solvers/solver.py#L62-L168).
- VBD skips rigid integration and solves when effective inverse mass is zero.
  Its current previous-pose history can also turn an out-of-band pose edit into
  inferred motion. PR #3659 correctly changes this path to classify from
  `BodyFlags.KINEMATIC` and rebuild a per-step baseline.
- Semi-implicit integration honors the flag, while its own source notes that
  rigid contact does not consistently use kinematic-zeroed effective inertia.
- MuJoCo and Featherstone approximate some articulated kinematic DOFs with a
  very large armature. That is an implementation technique, not an exact public
  semantic, and it is unsuitable as the definition of kinematic motion.
- Kamino classifies `BodyFlags.KINEMATIC` and `BodyFlags.PROXY` bodies as
  immovable and uses their authored input velocity in constraint free-velocity
  terms. It copies the authored pose and velocity to the output rather than
  forward-integrating the output pose. 

[`builder.py`](https://github.com/newton-physics/newton/blob/fddee2bebe749b5461a596f7624d445fc3d0e57b/newton/_src/sim/builder.py#L8281-L8349),
but the resulting behavior depends on the solver:

- The shared particle integrator advances an active zero-inverse-mass particle
  using its existing velocity, so SemiImplicit, XPBD, and the base particle path
  treat it like a force-immune ballistic point. See
  [`integrate_particles`](https://github.com/newton-physics/newton/blob/fddee2bebe749b5461a596f7624d445fc3d0e57b/newton/_src/solvers/solver.py#L22-L58).
- VBD holds a zero-inverse-mass particle at its input position, then reconstructs
  zero velocity from zero displacement. See
  [`particle_vbd_kernels.py`](https://github.com/newton-physics/newton/blob/fddee2bebe749b5461a596f7624d445fc3d0e57b/newton/_src/solvers/vbd/particle_vbd_kernels.py#L1378-L1406).
- Style3D locally clears `ParticleFlags.ACTIVE` whenever particle mass is zero.
  See
  [`style3d/kernels.py`](https://github.com/newton-physics/newton/blob/fddee2bebe749b5461a596f7624d445fc3d0e57b/newton/_src/solvers/style3d/kernels.py#L9-L16).
- MPM assumes zero-mass particles are kinematic and prescribe the motion of the
  grid nodes to which they are rasterized. `~ParticleFlags.ACTIVE` particles are
  not rasterized to the grid and do not affect other particles.

These are observably different semantics for the same model. They also make
mass editing a hidden motion-mode switch and make coupling code guess whether
an endpoint can respond.

## Normative entity contract

“Must” in this section is intended as the solver-independent public contract.

| Entity kind | Motion owner | Output pose/position | Output velocity | Responds to force? | Contact/constraint role |
|---|---|---:|---:|---:|---|
| Dynamic rigid body | Solver | Solver-computed | Solver-computed | Yes | Finite inertia |
| Kinematic rigid body | Caller | Forward-integrated | Copy-through | No | Moving infinite-mass boundary |
| Static rigid shape (`body=-1`) | Model | No state | No state | No | Fixed world boundary |
| Dynamic particle | Solver | Solver-computed | Solver-computed | Yes | Finite mass |
| Kinematic particle | Caller | Forward-integrated | Copy-through | No | Moving infinite-mass boundary |
| Ignored particle | Caller | Undefined | Undefined | No | None; ignored |

All non-ignored output rows must be written deterministically even when
`state_in` and `state_out` are distinct. Output rows for `IGNORED` particles may
remain untouched and are undefined.

### Rigid flags

Keep the existing stored-body invariant:

```python
BodyFlags.DYNAMIC
BodyFlags.KINEMATIC
```

Exactly one is stored on every model body. `BodyFlags.PROXY` remains an
orthogonal, view-local coupling marker and must not change motion ownership.
`BodyFlags.ALL` remains a filter mask, never a stored state.

### Particle flags

Use the current bits without changing the serialized meaning of existing
dynamic particles:

```python
class ParticleFlags(IntEnum):
    IGNORED = 0
    DYNAMIC = 1 << 0
    ACTIVE = DYNAMIC       # deprecated compatibility alias
    PROXY = 1 << 1
    KINEMATIC = 1 << 2
```

The motion mask is `DYNAMIC | KINEMATIC`; a particle must not contain both.
`PROXY` is orthogonal.

- `DYNAMIC` preserves the numeric value and behavior of today's `ACTIVE` bit.
- `KINEMATIC` is an explicit moving boundary particle.
- `IGNORED` particles are ignored. They do not generate point-particle contacts.
  Solvers must not modify their attributes and should consider their state
  undefined.
- Collision, viewer, coupling, and topology code must stop using “has the old
  ACTIVE bit” as a proxy for every kind of participation. Each call site must
  ask the narrower question: dynamic, kinematic, motion-bearing, renderable, or
  point-collidable.

This alias-based layout is preferable to making “absence of KINEMATIC” mean
dynamic: it gives particles the same explicit ownership model as bodies without
renumbering the existing dynamic bit.

## State and time convention

For a kinematic entity, the caller authors the beginning-of-step state before
calling collision detection and `Solver.step()`. The pose or position is the
current geometry used by collision detection, and the velocity prescribes its
motion over the step. For one step:

```text
q_out  = advance(q_in, qd_in, dt)
qd_out = qd_in
```

and for a particle:

```text
x_out = x_in + dt * v_in
v_out = v_in
```

No force, gravity, damping, constraint impulse, or contact impulse may alter
those outputs. The authored velocity is used both to advance the kinematic body
from its input pose to its output pose and to compute relative motion against
dynamic entities. This yields consistent behavior for velocity-based and
pose-based constraints involving kinematic objects.

For a target sequence, author velocity from the current beginning-of-step state
and the next target state at the same cadence as the solver step.

For particles:

```text
v_n = (x*_{n+1} - x_n) / dt
```

For rigid bodies, use center-of-mass displacement for the linear component and
the quaternion delta for angular velocity:

```text
v_com,n = (com_world(q*_{n+1}) - com_world(q_n)) / dt
w_n     = quat_velocity(rot(q*_{n+1}), rot(q_n), dt)
```

The rigid linear component is the world-space COM velocity, matching
`State.body_qd`. `quat_velocity` must invert the same discrete rotation
convention used by `advance` so that the authored velocity reaches the target
pose.

The API should document the following edge cases explicitly:

- **Pose changes, velocity does not:** a teleport. The new pose is accepted;
  velocity and friction are not inferred from pose history. Contact still uses
  the retained authored velocity.
- **Velocity changes, pose does not:** kinematic motion from the current pose.
  This is the expected prescribed animation workflow.
- **Dynamic pose edit:** also a teleport. It establishes the dynamic body's new
  starting pose.
- **Substeps:** author a target and matching velocity for every substep. Holding
  an outer-frame pose while replaying a nonzero velocity across substeps is not
  consistent motion.

## VBD previous-pose policy

Adopt the core policy in PR #3659:

1. `body_q_prev` uses `state_in.body_q` as its per-step baseline.
2. A dynamic body's baseline is the incoming pose, `state_in.body_q`. A teleport
   therefore does not create inferred velocity or friction.
3. A kinematic body's geometry follows the authored velocity,
   `state_in.body_qd`, according to normal forward-integration rules. This
   naturally accounts for frictional motion due to the authored velocity.
4. Finalization preserves the authored kinematic `body_qd`.
5. Dynamic coupling `PROXY` bodies are forward-integrated from their `state_in`
   pose and velocity during `step()`. They have finite inertia and are subject
   to reaction impulses. The coupling hooks are responsible for adjusting
   `body_qd` so that external and coupling forces are not double-counted.

The same rule should be implemented for VBD particles:

```text
particle_q(t + 1)  = particle_q(t) + particle_qd(t) * dt
particle_qd(t + 1) = particle_qd(t) = authored velocity
```

Here, `particle_q(t)` is the incoming current pose, including any authored
teleport. This enforces consistency between velocity-dependent and
pose-dependent elastic and contact terms.
Conformance is judged by public state and physical interaction,
not by identical private buffers.

## Mass and inverse mass

Mass and inertia arrays are inertial properties, not state classifiers.

### Required invariants

- Negative mass is always invalid.
- A `DYNAMIC` body requires finite `mass > 0` and valid inertia for its enabled
  rotational degrees of freedom. A `DYNAMIC` particle requires finite
  `mass > 0`.
- A `KINEMATIC` body may retain positive mass and inertia, and a `KINEMATIC`
  particle may retain positive mass, for metadata, identification, or a future
  transition to dynamic. Either may use zero mass because the solver does not
  consume its physical response.
- `body_inv_mass`, `body_inv_inertia`, and `particle_inv_mass` are derived from
  the canonical mass/inertia values. Users must not use them to request motion
  modes.
- Solvers derive effective response arrays from flags:

  ```text
  effective_inv_mass = model_inv_mass if DYNAMIC else 0
  effective_inv_inertia = model_inv_inertia if DYNAMIC else 0
  ```

  Branches deciding whether to integrate, solve, collide, filter, or couple
  must inspect flags, not the effective numerical zero.

Newton should provide indexed inertial-property setters that update mass and
inertia together with their derived inverses. Directly editing one side of a
reciprocal pair should be documented as unsupported. This avoids contradictory
mass and inertia state during runtime model updates.

## Articulations and kinematic links

A kinematic body is an **absolute world-space boundary condition**, not a
request to freeze only the joint coordinate connecting it to its parent.

Therefore:

- A root link may be kinematic.
- An internal or tip link may be kinematic.
- Multiple links may be kinematic.
- An articulation with both endpoint links kinematic is legal when the remaining
  constraints have a solution. This is a normal two-ended boundary-value
  problem for a cable or chain.
- Two authored kinematic links that violate an enabled joint create an
  inconsistent constraint problem. Newton should surface residual/nonconvergence
  diagnostics; it must not move either authored boundary to hide the conflict.

The current generic root-only check in
[`ModelBuilder`](https://github.com/newton-physics/newton/blob/fddee2bebe749b5461a596f7624d445fc3d0e57b/newton/_src/sim/builder.py#L9772-L9788)
should be removed. The current workaround—registering an articulation before
changing endpoint flags—makes legality depend on call order and should not be
part of the API.

Maximal-coordinate solvers can treat arbitrary kinematic links as zero-response
constraint endpoints. Reduced-coordinate tree solvers need additional work:
locking the child joint does not make an internal body's world pose independent
of a dynamic parent. Until a backend can impose the absolute condition exactly,
its constructor must reject that topology with an error such as:

```text
SolverFeatherstone does not yet support non-root kinematic body 7 ('tip').
Use SolverXPBD/SolverVBD, or prescribe a supported root kinematic articulation.
```

This is an explicit capability difference, not a semantic difference. A model
accepted by two solvers must have the same meaning in both.

## Runtime transitions

Changing a motion flag is supported only through a model update followed by
`notify_model_changed` with the corresponding property flag.
The updated model determines the motion classification, while the next state
passed to `Solver.step()` supplies the initial pose and velocity.

- Dynamic → kinematic: the current pose and velocity become the first authored
  beginning-of-step state; effective inverse inertia becomes zero; per-entity
  impulses, duals, sleep state, and motion history are invalidated.
- Kinematic → dynamic: current authored pose and velocity become initial dynamic
  state; positive valid inertial properties are required; solver factorizations
  and effective inertia are rebuilt.
- Ignored particle → kinematic/dynamic: position and velocity must both be
  initialized explicitly because ignored particle state is undefined.

Add `ModelFlags.PARTICLE_PROPERTIES` and
`ModelFlags.PARTICLE_INERTIAL_PROPERTIES`, parallel to the body flags. Particle
solvers currently lack a precise public notification channel for these changes.

Changing flags or inertial arrays during CUDA graph replay remains unsupported
unless the solver explicitly advertises capture-safe refresh. The validation
should fail before replay rather than leave stale effective masses.

## Solver implementation rules

Every solver should use shared predicates or shared effective-response builders
rather than open-coding mass tests. The rules are:

1. Classify from the public motion flag.
2. Write every non-ignored output row according to the ownership contract.
3. Apply force integration and solver corrections only to dynamic rows;
   forward-integrate kinematic poses and positions from authored velocities.
4. Include kinematic rows as infinite-mass moving boundaries in contact and
   constraints.
5. Use flags for broad-phase “immovable pair” filtering.
6. Preserve kinematic velocity in restitution/friction/contact evaluation.
7. Never infer public velocity from private pose history for an authored or
   teleported entity.
8. Reject unsupported topology or entity kinds at solver construction or when
   processing a relevant model update.

### Backend work list

| Backend | Required work |
|---|---|
| Shared/SemiImplicit | Add particle motion classification; use effective rigid response in contacts; write deterministic outputs. |
| XPBD | Add particle effective inverse mass, kinematic forward integration, and velocity copy-through; audit every `ACTIVE` check; retain rigid flag-based effective inertia. |
| VBD | Adapt the PR #3659 rigid baseline policy; implement the equivalent particle baseline; replace inverse-mass motion classification with flag-based classification. |
| Featherstone | Preserve exact authored root state; remove “large armature” as the semantic guarantee; reject unsupported internal kinematic links; update shared particle path. |
| MuJoCo CPU/Warp | Preserve exact root kinematic inputs and outputs around mocap/DOF mapping; reject unsupported internal links rather than approximate them; no particle claim. |
| Kamino | Retain `KINEMATIC` infinite response and authored velocity in constraint terms; forward-integrate the kinematic output pose; stop treating dynamic `PROXY` bodies as immovable; remove zero inertia as a motion classifier. |
| Style3D | Stop converting zero mass into inactive state; branch on particle motion flags and support or explicitly reject moving kinematic boundaries. |
| Implicit MPM | Classify particle motion from flags, implement the kinematic state contract, and deprecate `mass == 0` as a motion classifier. |
| Coupled solvers | Preserve ownership and flags through views/proxies; compute effective response from the destination view's motion type; never mutate base model inertia to disable a proxy. |

## Conformance suite

A single parameterized suite should run against every backend that declares
support. Unsupported cases must test capability errors during construction or
a relevant runtime model update.

### Rigid cases

1. Positive-mass kinematic body under gravity, force, and torque: bitwise or
   tolerance-equivalent `q_out == advance(q_in, qd_in, dt)`,
   `qd_out == qd_in`.
2. Repeat with zero and large model mass: identical public trajectory and
   identical response of any contacted dynamic body.
3. Nonzero authored `qd` with unchanged `q`: pose is forward-integrated.
4. Pose teleport with unchanged `qd`: no velocity or tangential friction is
   inferred from the pose jump; refreshed contacts use the new geometry and the
   authored `qd`.
5. Kinematic–dynamic contact: dynamic body responds, kinematic body does not,
   and reported reaction is equal/opposite where the backend exposes it.
6. Dynamic body with zero mass: deprecation warning in the transition release,
   then model-validation failure.
7. Runtime dynamic ↔ kinematic transition invalidates cached solver state.
8. Root, internal, and both-endpoints-kinematic articulation cases, including
   an intentionally inconsistent target pair with a diagnostic assertion.

### Particle cases

1. `DYNAMIC`, positive mass: integrates force and gravity.
2. `KINEMATIC`, positive and zero mass: position is forward-integrated, velocity
   is copied through, and results are mass-independent.
3. Kinematic–dynamic contact and elastic adjacency: only the dynamic point
   moves; prescribed velocity affects velocity-dependent terms.
4. `IGNORED`: state remains untouched and undefined. The particle may reserve
   capacity for dynamic emission or belong to another solver whose state must
   not be modified.
5. Invalid `DYNAMIC | KINEMATIC`: model-validation failure.
6. Runtime transitions and coupled proxy variants.

Run each relevant case with separate input/output states, aliased states,
multiple substeps, reset, contact warm-start/history enabled, and CUDA graph
capture where supported. Solver tolerances may differ; ownership and state
transition assertions may not.

## Migration plan

Newton's no-breaking-change policy requires a staged transition.

### Release A: make intent explicit

- Publish this contract.
- Add `ParticleFlags.DYNAMIC`, `KINEMATIC`, and `IGNORED`; retain `ACTIVE` as a
  deprecated alias of `DYNAMIC`.
- Add common classification/effective-response helpers and the conformance
  suite.
- Adapt PR #3659 to use forward instead of backward integration.
- Warn when a dynamic body or particle has zero mass. The warning must say how
  to migrate: select `KINEMATIC`, select particle `IGNORED`, or provide positive
  inertial data.
- Keep legacy unflagged zero-mass particle behavior temporarily, clearly marked
  solver-dependent and deprecated. Explicit `KINEMATIC` uses the new contract
  immediately in every solver that accepts it.
- Remove generic root-only articulation validation and add solver capability
  validation, so supported maximal-coordinate use cases no longer depend on
  builder call order.

### Release B: remove mass-as-motion behavior

- Reject zero-mass dynamic entities at finalization.
- Remove all zero-mass motion-mode compatibility branches.
- Stop accepting direct inverse-mass edits as a motion control.
- Remove `ParticleFlags.ACTIVE` after its advertised deprecation period, or keep
  it indefinitely as a documented alias if serialization compatibility is more
  valuable than enum cleanup.

### Example migration

Before:

```python
# Solver-dependent: may move at constant velocity, freeze, or become inactive.
builder.add_particle(pos=x, vel=v, mass=0.0)
```

After:

```python
p = builder.add_particle(
    pos=x_initial,
    vel=wp.vec3(0.0, 0.0, 0.0),
    mass=0.0,
    flags=newton.ParticleFlags.KINEMATIC,
)

# Before every collision/step, author velocity at substep cadence.
state.particle_qd[p] = (x_target - state.particle_q[p]) / dt
```

For a stationary cloth pin:

```python
builder.particle_flags[p] = int(newton.ParticleFlags.KINEMATIC)
state.particle_qd[p] = wp.vec3(0.0, 0.0, 0.0)
```

For a rigid body, migrate `mass=0` intent to the existing flag:

```python
body = builder.add_body(
    xform=q_target,
    mass=0.0,
    is_kinematic=True,
)
```

Positive mass is also valid for a kinematic body and must not change its motion.

## Non-goals

- A kinematic flag is not a trajectory generator. Newton will not choose between
  pose-driven and velocity-driven motion implicitly.
- This proposal does not add a static-body state. Static rigid geometry remains
  world-attached.
- This proposal does not promise arbitrary kinematic-link support in every
  backend immediately. It requires exact support or an early explicit error.
- This proposal does not use solver-private history as a second public state.

## Acceptance criteria

The semantic cleanup is complete when:

- no solver branches on zero inverse mass to decide whether an entity is
  kinematic;
- all accepted solvers pass the same ownership/contact conformance tests;
- explicit rigid and particle kinematics produce mass-independent results;
- zero-mass dynamic entities are rejected after deprecation;
- arbitrary-link kinematic models are accepted by capable solvers and rejected
  explicitly by incapable ones;
- VBD teleports no longer generate inferred velocity/friction, while consistent
  authored kinematics retain motion-dependent contact behavior; and
- docs and examples never recommend zero mass as a kinematic switch.

## Recommended decision

Adopt this contract before adding more solver-specific fixes. PR #3659 is the
right VBD implementation direction, but it should land as one part of the
cross-solver contract: explicit flag ownership, forward-integrated kinematic
poses with copied-through velocities, per-step VBD baselines, explicit particle
motion flags, mass validation, and a shared conformance suite.
