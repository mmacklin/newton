# SPDX-License-Identifier: Apache-2.0
"""Mass-weighted POD of an ordered sequence, preserving exact clamped samples."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrderedPOD:
    rest: np.ndarray
    phi: np.ndarray
    coefficients: np.ndarray
    mass: np.ndarray
    progress: np.ndarray
    singular_values: np.ndarray
    maximum_reconstruction_error: float
    retained_energy: float

    def recover(self, coefficients):
        return self.rest + np.einsum("...m,nmc->...nc", coefficients, self.phi)

    def interpolate(self, progress):
        s = np.asarray(progress, dtype=float)
        if np.any(s < self.progress[0]) or np.any(s > self.progress[-1]):
            raise ValueError("Progress lies outside the trained path")
        i = np.minimum(np.searchsorted(self.progress, s, side="right") - 1, len(self.progress) - 2)
        t = (s - self.progress[i]) / (self.progress[i + 1] - self.progress[i])
        return self.coefficients[i] + t[..., None] * (self.coefficients[i + 1] - self.coefficients[i])


def fit_ordered_pod(rest, shapes, masses, progress, mode_count, clamped_samples=()):
    """Fit without centering the snapshots or subtracting rigid motions.

    The first shape is the undeformed reference, so its modal coefficients
    are exactly zero. Clamped samples are checked, not silently projected.
    Root-tangent preservation can be encoded by clamping two adjacent rows.
    ``shapes`` must contain absolute positions, not displacements.
    """
    rest = np.asarray(rest, dtype=np.float64)
    shapes = np.asarray(shapes, dtype=np.float64)
    mass = np.asarray(masses, dtype=np.float64)
    progress = np.asarray(progress, dtype=np.float64)
    fixed = np.asarray(clamped_samples, dtype=int)
    if rest.ndim != 2 or rest.shape[1] != 3 or shapes.ndim != 3 or shapes.shape[1:] != rest.shape:
        raise ValueError("Expected rest [samples,3] and shapes [frames,samples,3]")
    if mass.shape != (len(rest),) or np.any(mass <= 0):
        raise ValueError("Every sampled surface vertex needs positive mass")
    if progress.shape != (len(shapes),) or len(shapes) < 2 or progress[0] != 0 or np.any(np.diff(progress) <= 0):
        raise ValueError("Need increasing progress starting at zero")
    if not all(np.isfinite(x).all() for x in (rest, shapes, mass, progress)):
        raise ValueError("Training arrays must be finite")
    if np.any(fixed < 0) or np.any(fixed >= len(rest)):
        raise ValueError("Invalid clamped sample index")
    if not isinstance(mode_count, int) or isinstance(mode_count, bool) or mode_count < 1:
        raise ValueError("mode_count must be a positive integer")
    displacement = shapes - rest
    if np.max(np.abs(displacement[0])) > 1e-12:
        raise ValueError("First keyframe must equal the rest shape")
    if fixed.size and np.max(np.abs(displacement[:, fixed])) > 1e-12:
        raise ValueError("Training shape violates the clamped boundary")
    weight = np.repeat(np.sqrt(mass), 3)
    matrix = displacement.reshape(len(shapes), -1).T * weight[:, None]
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.count_nonzero(singular > max(1e-14, singular[0] * 1e-10)))
    if mode_count > rank:
        raise ValueError(f"Requested {mode_count} modes but the shape sequence has rank {rank}")
    basis = left[:, :mode_count] / weight[:, None]
    phi = basis.reshape(len(rest), 3, mode_count).transpose(0, 2, 1).copy()
    # SVD roundoff at analytically zero rows must not create a moving clamp.
    phi[fixed] = 0
    basis = phi.transpose(0, 2, 1).reshape(-1, mode_count)
    coefficients = (basis.T @ (weight[:, None] * matrix)).T
    coefficients[0] = 0
    reconstruction = (basis @ coefficients.T).T.reshape(shapes.shape) + rest
    error = float(np.max(np.linalg.norm(reconstruction - shapes, axis=-1)))
    retained = float(np.sum(singular[:mode_count] ** 2) / np.sum(singular**2))
    return OrderedPOD(rest.copy(), phi, coefficients, mass.copy(), progress.copy(), singular, error, retained)
