# Evaluated hypotheses

## H-1: Insufficient contact iterations

Two DVI contact iterations at 250 Hz under-resolve the foot-ground impulses. If
false, increasing contact iterations should not materially reduce maximum and
RMS penetration.

Result: confirmed. Increasing the unpreconditioned contact solve from two to
eight iterations reduced peak walking penetration from 35.6 mm to 20.3 mm.

## H-2: Delayed contact generation

Zero shape margin and gap create contacts too late for DVI's velocity-level
update. If false, adding the upstream DR Legs contact envelope should not reduce
penetration at fixed timestep and iteration count.

Result: eliminated. A 0.5 mm margin and 10 mm gap changed peak penetration by
less than 1 mm at four and eight contact iterations.

## H-3: Weak penetration stabilization

The example value `gamma=0.015` corrects too little existing penetration per
step at 250 Hz. If false, increasing gamma at fixed timestep should not reduce
the mean and maximum penetration.

Result: confirmed. With block-preconditioned contact, increasing gamma to 0.1
reduced peak walking penetration from 12.9 mm to 8.7 mm at four iterations.
Values of 0.2 and above became unstable, so the issue cannot be addressed by
stabilization alone.

## H-4: Visual/collision geometry mismatch

The apparent penetration is mostly a visual mesh offset rather than collision
shape overlap. If false, signed collision-contact separation will show
millimeter-scale negative values at the same times.

Result: eliminated. Recomputed signed separation between the collision shapes
measured 35.6 mm of peak penetration in the original DVI configuration.

## H-5: Scalar Jacobi contact conditioning

The default scalar contact update does not resolve coupled normal and friction
impulses effectively. If false, the 3x3 contact block preconditioner should not
improve penetration at the same contact iteration count.

Result: confirmed. At four iterations and `gamma=0.015`, the block
preconditioner reduced peak walking penetration from 29.2 mm to 12.9 mm with
only a small timing increase.
