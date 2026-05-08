# Reduced Elastic Agent Notes

These notes capture debugging lessons from the first reduced elastic body pass.
They are not a replacement for the implementation report or design document;
they are a checklist for future agentic work on this subsystem.

## Mental Model

- A reduced elastic body is a normal body slot plus a `JointType.ELASTIC` owner
  joint. The owner joint stores `[pos, quat, q0, q1, ...]` and
  `[linear, angular, qd0, qd1, ...]`.
- The body pose mirrors the owner joint's rigid frame. Modal coordinates live in
  the normal `joint_q` and `joint_qd` arrays.
- `ModalBasis` should be treated as sampled linear data: `phi[sample, mode]`.
  It should not depend on Python callables inside solver/render kernels.
- Joint attachment samples come from existing joint parent/child transforms. The
  builder flattens per-endpoint `phi` samples for the solver.
- The current solver approximation is translational point attachments. It does
  not directly project fixed/revolute angular moments into modes because endpoint
  rotations/slopes are not represented yet.

## Modal Mass Notes

- Current VBD reduced elastic updates use one scalar modal mass per mode. This is
  a diagonal approximation to the reduced mass matrix.
- `ModalGeneratorPOD` currently builds modes with ordinary sampled-displacement
  SVD, rescales each mode by maximum point displacement, then estimates
  `mode_mass[i] = sample_mass * sum_j ||phi_i(x_j)||^2`. This is not strict
  mass normalization, although it is close to a mass-orthogonal basis when sample
  weights are uniform.
- Prefer mass-normalizing POD modes when the goal is physical dynamics:
  `phi_i <- phi_i / sqrt(phi_i^T M phi_i)`. If the modes are mass-orthogonal,
  this makes the diagonal modal mass entries exactly one under the chosen lumped
  mass model and improves the interpretation of stiffness and damping.
- Mass normalization alone does not remove off-diagonal mass terms for arbitrary
  nonuniform samples or nonorthogonal exemplar bases. A future dense reduced
  model should store `M_ij = phi_i^T M phi_j` per modal basis.
- VBD can still keep the current cheap diagonal update as a Jacobi-style
  preconditioner while evaluating the RHS/residual with the full reduced mass
  matrix. Under fixed-point convergence, that targets the coupled reduced system
  rather than the diagonal approximation. The same pattern can later extend to
  dense reduced `K` and `C` matrices.

## Force Consistency Invariants

For a linear joint attachment:

```text
C = x_child - x_parent
P = projector onto constrained linear directions
f_linear = k P C + kd k P (C - C_prev) / dt
```

The elastic modal solve must see the same translational force estimate as the
rigid joint solve:

```text
Q_i = phi_i_world dot f_side
```

Equivalently, the modal gradient contribution is:

```text
grad_i += dot(f_linear, P dC/dq_i)
```

Sign convention:

- Elastic child endpoint: `dC/dq_i = +R_body phi_i`.
- Elastic parent endpoint: `dC/dq_i = -R_body phi_i`.
- The generalized force is `Q_i = -grad_i`.

Important bug we hit: the rigid VBD joint path included translational joint
damping, but the elastic modal solve initially projected only `k P C`. That made
the dipper arm creep slowly and made joint damping tuning misleading. If
`rigid_joint_linear_kd` is nonzero, the modal solve must include the damping
force and the matching diagonal damping Hessian term.

## Debugging Workflow

1. Isolate the mechanism. Disable contacts and actuation unless they are the
   behavior under test.
2. Use fixed joint stiffness when studying convergence:
   `rigid_joint_adaptive_stiffness=False`. Adaptive AVBD penalties can make
   iteration sweeps non-monotonic.
3. Do not judge convergence from joint error alone. Finite stiffness means finite
   compliance. Check an equation-of-motion residual or an update norm.
4. For static modal checks, compare `K_i q_i` against the projected joint force
   `Q_i`. For the implicit solve, include modal inertia/damping and joint damping.
5. If changing `rigid_joint_linear_kd` changes behavior, verify the elastic
   modal solve projects the same damping term as the rigid joint evaluator.
6. If a behavior looks like drift, run a zero-gravity control. If it disappears,
   it is load-driven, not free numerical drift.
7. Render after solver fixes. Viewer results exposed several bugs that scalar
   tests did not: mesh deformation, winding, interpolation artifacts, and damping
   mismatch.

## Tests To Add For Solver Changes

- One-step modal force projection against an analytic point attachment.
- Both elastic-as-child and elastic-as-parent sign cases.
- Prismatic projection case (`P = I - aa^T`).
- Joint damping projection when `joint_qd` creates endpoint velocity.
- Example `test_final()` checks for finite state, bounded residuals, bounded
  modal amplitudes, and expected visible deformation.

## Contact Plan

The current contact design is serialized in
`docs/guide/reduced_elastic_contact_plan.md`. The first pass should use existing
elastic surface samples only: deform each surface vertex on the fly during
contact generation, query it against rigid/static shapes, and store one scalar
elastic sample id on the contact side so the VBD solver can recover
`phi[sample, mode]` directly. Do not add a separate contact-sample API for this
pass. Elastic-vs-elastic and barycentric face contact are follow-ups.

## Modeling Pitfalls

- Do not add modal gravity unless the modal coordinate represents an absolute
  displacement field with a clear gravitational generalized force. For these
  floating-frame deformation modes, gravity acts on rigid body DOFs; elastic
  modes are loaded through attachments, contacts, or other forces.
- More POD modes do not automatically fix nonlinear geometric effects. Linear
  modal coordinates can approximate a finite deformation family, but the solver
  can still leave the sampled manifold unless the basis/parameterization makes
  invalid motion hard to reach.
- Avoid example-side nonlinear coordinate hacks such as `q_radial = q_twist^2`.
  If a finite deformation is needed, use independent linear modes, for example
  POD modes from exemplar configurations.
- Displacement coloring is only a visualization of `|u|`, not true strain. Do
  not interpret it as stress or strain without gradients.
- A fixed joint at one point cannot transmit a clamp moment into modal bending
  under the current point-attachment approximation. Use multiple translational
  samples or add rotational endpoint data in a future implementation.

## Report And Video Hygiene

- Use ViewerGL headless for videos when possible.
- Always update report video links with cache busting:
  `?datetime=YYYYMMDDTHHMMSSZ`.
- Sync report outputs to `/home/horde/reports/newton-reduced/` when the user is
  reviewing through `reports.mmacklin.com`.
- If using generated CSV data, write LF line endings so `git diff --check` stays
  clean.
