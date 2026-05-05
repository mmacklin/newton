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

