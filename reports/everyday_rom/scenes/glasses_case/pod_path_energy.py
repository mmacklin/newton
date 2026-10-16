# SPDX-License-Identifier: Apache-2.0
"""Passive piecewise-quadratic coupling around an ordered modal shape path.

Coordinates must be mass normalized. Each segment is a linear interpolation
of adjacent POD coefficient keyframes. Progress is minimized out, never driven
by time. This is an explicit shape prior, not a calibrated shell material.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathEvaluation:
    energy: float
    gradient: np.ndarray
    hessian: np.ndarray
    progress: float
    segment: int
    fraction: float
    closest: np.ndarray


class ModalPathEnergy:
    """Couple mass-normalized modes through an ordered polyline.

    Minimize ``k_perp/2 * ||q - a(s)||^2 + k_progress/2 * s^2``
    over the complete path. Each segment has an exact clamped scalar minimizer.
    The returned Hessian is exact within the active segment/interior regime,
    and positive semidefinite. Segment switches can change the Hessian; distant
    nearest-branch ties can make the envelope nonsmooth and must be diagnosed.

    Args:
        coefficients: Modal coefficient keyframes, shape (keyframes, modes).
        progress: Strictly increasing progress values starting at zero.
        k_perp: Off-path coefficient stiffness [1/s^2].
        k_progress: Restoring energy coefficient for dimensionless progress [J].
    """

    def __init__(self, coefficients, progress, k_perp, k_progress=0.0):
        self.coefficients = np.array(coefficients, dtype=np.float64, copy=True)
        self.progress = np.array(progress, dtype=np.float64, copy=True)
        self.k_perp = float(k_perp)
        self.k_progress = float(k_progress)
        a, s = self.coefficients, self.progress
        if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 1 or s.shape != (a.shape[0],):
            raise ValueError("Expected at least two coefficient keyframes and matching progress")
        if not np.isfinite(a).all() or not np.isfinite(s).all():
            raise ValueError("Path arrays must be finite")
        if s[0] != 0 or np.any(np.diff(s) <= 0):
            raise ValueError("Progress must increase strictly from zero")
        if not np.isfinite([k_perp, k_progress]).all() or k_perp <= 0 or k_progress < 0:
            raise ValueError("Require finite k_perp > 0 and k_progress >= 0")
        self.directions = np.diff(a, axis=0)
        self.steps = np.diff(s)
        if np.any(np.linalg.norm(self.directions, axis=1) <= 1e-14):
            raise ValueError("Adjacent keyframes must differ")
        self.denominator = k_perp * np.sum(self.directions**2, axis=1) + k_progress * self.steps**2
        for array in (a, s, self.directions, self.steps, self.denominator):
            array.flags.writeable = False

    def evaluate(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.coefficients.shape[1],) or not np.isfinite(q).all():
            raise ValueError("q must be a finite vector matching the path's mode count")
        residual = q - self.coefficients[:-1]
        numerator = self.k_perp * np.einsum("ij,ij->i", self.directions, residual)
        numerator -= self.k_progress * self.steps * self.progress[:-1]
        fraction = np.clip(numerator / self.denominator, 0.0, 1.0)
        closest = self.coefficients[:-1] + fraction[:, None] * self.directions
        progress = self.progress[:-1] + fraction * self.steps
        residual = q - closest
        energies = 0.5 * self.k_perp * np.sum(residual**2, axis=1) + 0.5 * self.k_progress * progress**2
        segment = int(np.argmin(energies))
        t = float(fraction[segment])
        hessian = self.k_perp * np.eye(q.size)
        if 0 < t < 1:
            d = self.directions[segment]
            hessian -= self.k_perp**2 * np.outer(d, d) / self.denominator[segment]
        return PathEvaluation(
            float(energies[segment]),
            self.k_perp * residual[segment],
            hessian,
            float(progress[segment]),
            segment,
            t,
            closest[segment].copy(),
        )
