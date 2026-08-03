# DVI ground-contact investigation

## Symptom

The matched-timestep Kamino DVI DR Legs policy video shows visible foot-ground
interpenetration.

## Observed conditions

- Occurs with DVI at `dt=0.004 s`, two contact iterations, zero shape margin,
  and zero contact gap.
- PADMM and sparse VBD use the same collision envelope without similarly obvious
  penetration in their videos.
- The upstream standalone Kamino DVI example instead uses `dt=1/600 s`, a
  `5e-4 m` margin, a `1e-2 m` gap, four block iterations, and two contact
  iterations.
- A prior full policy test at the upstream timestep became nonfinite, so smaller
  timestep alone is not yet a validated fix for this policy workload.

## Root cause

The original matched DVI configuration under-resolved ground contact. Kamino's
contact stabilization computes a penetration correction proportional to
`gamma`; `gamma=0.015` therefore applies only a weak correction at each 4 ms
step. Two scalar Jacobi contact iterations do not compensate for that weak
correction on the driven feet.

The problem is measurable in collision geometry. During the walking phase, the
original configuration reaches 35.6 mm maximum penetration and 29.1 mm mean
frame-maximum penetration. Matched PADMM reaches 4.5 mm maximum penetration,
and sparse VBD reaches 3.3 mm.

## Selected configuration

- `gamma=0.1`
- 3x3 contact block preconditioner enabled
- six contact iterations
- four DVI block iterations
- Jacobi omega 0.45 and relaxation 0.9

This configuration completed the full 8 s rollout. It reduced maximum walking
penetration to 7.6 mm and mean frame-maximum penetration to 2.4 mm, while
increasing median CPU substep cost from 4.1 ms to 5.6 ms. Seven and eight
contact iterations did not consistently improve the worst-case metric, and
larger omega or relaxation values made it worse.
