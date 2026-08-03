# Eliminated hypotheses

## Visual mesh offset

Signed collision-shape separation confirms that the penetration is physical,
not an offset between visual and collision geometry.

## Late contact generation

Using the standalone example's 0.5 mm collision margin and 10 mm contact gap did
not materially improve penetration at matched iteration counts.

## More outer block iterations as the primary remedy

Increasing the outer DVI block solve from 4 to 32 iterations reduced peak
penetration less efficiently than preconditioning the contact block itself.
