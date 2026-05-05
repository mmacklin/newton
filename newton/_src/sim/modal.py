# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reduced modal basis utilities for elastic links."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


class ModalBasis:
    """Sampled linear modal basis for a reduced elastic body.

    The basis stores displacement samples ``phi`` at body-local material points.
    A deformation at point ``X`` is evaluated linearly as ``sum_i phi_i(X) q_i``.
    Unknown points are evaluated by inverse-distance interpolation from the
    stored samples, keeping the runtime representation independent of how the
    modes were generated.

    Args:
        sample_points: Body-local sample points [m], shape ``[sample_count, 3]``.
        sample_phi: Translational mode values [m per modal coordinate], shape
            ``[sample_count, mode_count, 3]``.
        mode_mass: Modal masses [kg], shape ``[mode_count]``.
        mode_stiffness: Modal stiffness values [N/m], shape ``[mode_count]``.
        mode_damping: Modal damping values [N s/m], shape ``[mode_count]``.
        label: Optional basis label.
        interpolation_epsilon: Distance regularizer [m] used for inverse-distance
            interpolation.
    """

    def __init__(
        self,
        sample_points: Sequence[Sequence[float]] | np.ndarray | None = None,
        sample_phi: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        mode_mass: Sequence[float] | np.ndarray | None = None,
        mode_stiffness: Sequence[float] | np.ndarray | None = None,
        mode_damping: Sequence[float] | np.ndarray | None = None,
        label: str | None = None,
        interpolation_epsilon: float = 1.0e-8,
    ):
        self.label = label
        self.interpolation_epsilon = float(interpolation_epsilon)

        if sample_points is None:
            points = np.zeros((0, 3), dtype=np.float32)
        else:
            points = np.asarray(sample_points, dtype=np.float32)
            if points.ndim == 1:
                points = points.reshape((1, 3))
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"sample_points must have shape [sample_count, 3], got {points.shape}")

        if sample_phi is None:
            mode_count = self._infer_mode_count(mode_mass, mode_stiffness, mode_damping)
            phi = np.zeros((points.shape[0], mode_count, 3), dtype=np.float32)
        else:
            phi = np.asarray(sample_phi, dtype=np.float32)
            if phi.ndim == 2 and points.shape[0] == 1:
                phi = phi.reshape((1, phi.shape[0], 3))
            if phi.ndim != 3 or phi.shape[0] != points.shape[0] or phi.shape[2] != 3:
                raise ValueError(
                    "sample_phi must have shape [sample_count, mode_count, 3], "
                    f"got {phi.shape} for {points.shape[0]} sample points"
                )
            mode_count = int(phi.shape[1])

        self.sample_points = np.array(points, dtype=np.float32, copy=True)
        self.sample_phi = np.array(phi, dtype=np.float32, copy=True)
        self.mode_mass = self._coerce_mode_array(mode_mass, mode_count, 1.0, "mode_mass")
        self.mode_stiffness = self._coerce_mode_array(mode_stiffness, mode_count, 0.0, "mode_stiffness")
        self.mode_damping = self._coerce_mode_array(mode_damping, mode_count, 0.0, "mode_damping")

    @property
    def mode_count(self) -> int:
        """Number of modal coordinates."""
        return int(self.sample_phi.shape[1])

    @property
    def sample_count(self) -> int:
        """Number of stored sample points."""
        return int(self.sample_points.shape[0])

    @staticmethod
    def _infer_mode_count(*arrays: Sequence[float] | np.ndarray | None) -> int:
        mode_count = 0
        for values in arrays:
            if values is None:
                continue
            count = int(np.asarray(values, dtype=np.float32).reshape((-1,)).shape[0])
            if mode_count not in (0, count):
                raise ValueError(f"mode property lengths disagree: {mode_count} and {count}")
            mode_count = count
        return mode_count

    @staticmethod
    def _coerce_mode_array(
        values: Sequence[float] | np.ndarray | None,
        mode_count: int,
        default: float,
        name: str,
    ) -> np.ndarray:
        if values is None:
            return np.full(mode_count, default, dtype=np.float32)
        array = np.asarray(values, dtype=np.float32).reshape((-1,))
        if array.shape[0] != mode_count:
            raise ValueError(f"{name} must have length {mode_count}, got {array.shape[0]}")
        return np.array(array, dtype=np.float32, copy=True)

    def copy(self) -> ModalBasis:
        """Return a deep copy of this basis."""
        return ModalBasis(
            self.sample_points,
            self.sample_phi,
            self.mode_mass,
            self.mode_stiffness,
            self.mode_damping,
            label=self.label,
            interpolation_epsilon=self.interpolation_epsilon,
        )

    def find_sample(self, point: Sequence[float] | np.ndarray, tolerance: float = 1.0e-7) -> int:
        """Return the index of a matching sample point, or ``-1`` if none exists.

        Args:
            point: Body-local point [m], shape ``[3]``.
            tolerance: Matching tolerance [m].

        Returns:
            The local sample index, or ``-1``.
        """
        if self.sample_count == 0:
            return -1
        query = np.asarray(point, dtype=np.float32).reshape((3,))
        dist2 = np.sum((self.sample_points - query.reshape((1, 3))) ** 2, axis=1)
        nearest = int(np.argmin(dist2))
        return nearest if float(dist2[nearest]) <= tolerance * tolerance else -1

    def add_sample(
        self,
        point: Sequence[float] | np.ndarray,
        phi: Sequence[Sequence[float]] | np.ndarray | None = None,
        tolerance: float = 1.0e-7,
    ) -> int:
        """Add a body-local sample and return its local index.

        If ``point`` already exists, its existing index is returned. When ``phi``
        is not provided, values are interpolated from the current samples.

        Args:
            point: Body-local point [m], shape ``[3]``.
            phi: Translational mode values [m per modal coordinate], shape
                ``[mode_count, 3]``.
            tolerance: Duplicate-point tolerance [m].

        Returns:
            The local sample index.
        """
        query = np.asarray(point, dtype=np.float32).reshape((3,))
        existing = self.find_sample(query, tolerance=tolerance)
        if existing >= 0:
            return existing

        if phi is None:
            values = self.evaluate(query)
        else:
            values = np.asarray(phi, dtype=np.float32)
            if values.shape != (self.mode_count, 3):
                raise ValueError(f"phi must have shape ({self.mode_count}, 3), got {values.shape}")

        self.sample_points = np.vstack((self.sample_points, query.reshape((1, 3)))).astype(np.float32, copy=False)
        self.sample_phi = np.concatenate((self.sample_phi, values.reshape((1, self.mode_count, 3))), axis=0).astype(
            np.float32,
            copy=False,
        )
        return self.sample_count - 1

    def evaluate(self, point: Sequence[float] | np.ndarray) -> np.ndarray:
        """Evaluate modal displacement samples at a body-local point.

        Args:
            point: Body-local point [m], shape ``[3]``.

        Returns:
            Translational mode values [m per modal coordinate], shape
            ``[mode_count, 3]``.
        """
        if self.mode_count == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if self.sample_count == 0:
            return np.zeros((self.mode_count, 3), dtype=np.float32)

        query = np.asarray(point, dtype=np.float32).reshape((3,))
        delta = self.sample_points - query.reshape((1, 3))
        dist2 = np.sum(delta * delta, axis=1)
        nearest = int(np.argmin(dist2))
        exact_tol2 = max(self.interpolation_epsilon * self.interpolation_epsilon, 1.0e-14)
        if float(dist2[nearest]) <= exact_tol2:
            return np.array(self.sample_phi[nearest], dtype=np.float32, copy=True)

        eps2 = self.interpolation_epsilon * self.interpolation_epsilon
        weights = 1.0 / (dist2 + eps2)
        weights /= np.sum(weights)
        return np.einsum("s,smc->mc", weights.astype(np.float32), self.sample_phi).astype(np.float32)

    def sample_value(self, sample_index: int) -> np.ndarray:
        """Return modal values for a stored sample index.

        Args:
            sample_index: Local sample index.

        Returns:
            Translational mode values [m per modal coordinate], shape
            ``[mode_count, 3]``.
        """
        if sample_index < 0 or sample_index >= self.sample_count:
            raise IndexError(f"sample_index {sample_index} is out of range for {self.sample_count} samples")
        return np.array(self.sample_phi[sample_index], dtype=np.float32, copy=True)


class ModalGeneratorSampled:
    """Build a :class:`ModalBasis` from externally supplied sampled modes."""

    def __init__(
        self,
        sample_points: Sequence[Sequence[float]] | np.ndarray,
        sample_phi: Sequence[Sequence[Sequence[float]]] | np.ndarray,
        mode_mass: Sequence[float] | np.ndarray | None = None,
        mode_stiffness: Sequence[float] | np.ndarray | None = None,
        mode_damping: Sequence[float] | np.ndarray | None = None,
        label: str | None = None,
    ):
        self.sample_points = np.asarray(sample_points, dtype=np.float32)
        self.sample_phi = np.asarray(sample_phi, dtype=np.float32)
        self.mode_mass = None if mode_mass is None else np.asarray(mode_mass, dtype=np.float32)
        self.mode_stiffness = None if mode_stiffness is None else np.asarray(mode_stiffness, dtype=np.float32)
        self.mode_damping = None if mode_damping is None else np.asarray(mode_damping, dtype=np.float32)
        self.label = label

    def build(self) -> ModalBasis:
        """Build the sampled modal basis."""
        return ModalBasis(
            self.sample_points,
            self.sample_phi,
            self.mode_mass,
            self.mode_stiffness,
            self.mode_damping,
            label=self.label,
        )


class ModalGeneratorPOD:
    """Build linear modal basis functions with proper orthogonal decomposition.

    Args:
        sample_points: Rest sample points [m], shape ``[sample_count, 3]``.
        displacements: Exemplar displacement fields [m], shape
            ``[snapshot_count, sample_count, 3]``.
        deformed_points: Exemplar deformed points [m], shape
            ``[snapshot_count, sample_count, 3]``. Used only when
            ``displacements`` is not provided.
        mode_count: Number of POD modes to retain.
        total_mass: Total mass [kg] represented by the samples. If provided,
            modal masses are estimated by lumped sample weights.
        stiffness_scale: Heuristic stiffness scale [N/m].
        damping_ratio: Modal damping ratio.
        subtract_mean: Whether to remove the mean exemplar displacement before
            computing POD modes.
        label: Optional basis label.
    """

    def __init__(
        self,
        sample_points: Sequence[Sequence[float]] | np.ndarray,
        displacements: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        deformed_points: Sequence[Sequence[Sequence[float]]] | np.ndarray | None = None,
        mode_count: int | None = None,
        total_mass: float | None = None,
        stiffness_scale: float = 1.0,
        damping_ratio: float = 0.0,
        subtract_mean: bool = False,
        label: str | None = None,
    ):
        self.sample_points = np.asarray(sample_points, dtype=np.float32)
        if self.sample_points.ndim != 2 or self.sample_points.shape[1] != 3:
            raise ValueError(f"sample_points must have shape [sample_count, 3], got {self.sample_points.shape}")

        if displacements is None:
            if deformed_points is None:
                raise ValueError("Either displacements or deformed_points must be provided")
            deformed = np.asarray(deformed_points, dtype=np.float32)
            displacements = deformed - self.sample_points.reshape((1, self.sample_points.shape[0], 3))

        self.displacements = np.asarray(displacements, dtype=np.float32)
        if self.displacements.ndim != 3 or self.displacements.shape[1:] != self.sample_points.shape:
            raise ValueError(
                f"displacements must have shape [snapshot_count, sample_count, 3], got {self.displacements.shape}"
            )

        self.mode_count = mode_count
        self.total_mass = total_mass
        self.stiffness_scale = float(stiffness_scale)
        self.damping_ratio = float(damping_ratio)
        self.subtract_mean = bool(subtract_mean)
        self.label = label

    def build(self) -> ModalBasis:
        """Build a sampled POD modal basis."""
        snapshot_count = int(self.displacements.shape[0])
        if snapshot_count == 0:
            raise ValueError("At least one displacement snapshot is required")

        matrix = self.displacements.reshape((snapshot_count, -1)).astype(np.float64)
        if self.subtract_mean:
            matrix = matrix - np.mean(matrix, axis=0, keepdims=True)

        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        mode_count = self.mode_count or int(vt.shape[0])
        mode_count = min(int(mode_count), int(vt.shape[0]))
        if mode_count < 0:
            raise ValueError(f"mode_count must be non-negative, got {mode_count}")

        modes = vt[:mode_count].reshape((mode_count, self.sample_points.shape[0], 3)).astype(np.float32)
        for i in range(mode_count):
            scale = float(np.max(np.linalg.norm(modes[i], axis=1)))
            if scale > 0.0:
                modes[i] /= scale

        sample_phi = np.transpose(modes, (1, 0, 2))
        mode_mass = self._estimate_mode_mass(sample_phi)
        mode_stiffness = self._estimate_mode_stiffness(mode_count, singular_values)
        mode_damping = np.zeros(mode_count, dtype=np.float32)
        if self.damping_ratio > 0.0:
            for i in range(mode_count):
                mode_damping[i] = (
                    2.0
                    * self.damping_ratio
                    * math.sqrt(max(float(mode_mass[i]), 0.0) * max(float(mode_stiffness[i]), 0.0))
                )

        return ModalBasis(
            self.sample_points,
            sample_phi,
            mode_mass,
            mode_stiffness,
            mode_damping,
            label=self.label,
        )

    def _estimate_mode_mass(self, sample_phi: np.ndarray) -> np.ndarray:
        mode_count = int(sample_phi.shape[1])
        if self.total_mass is None or self.sample_points.shape[0] == 0:
            return np.ones(mode_count, dtype=np.float32)

        sample_mass = float(self.total_mass) / float(self.sample_points.shape[0])
        mode_mass = np.zeros(mode_count, dtype=np.float32)
        for i in range(mode_count):
            norm2 = np.sum(sample_phi[:, i, :] * sample_phi[:, i, :], axis=1)
            mode_mass[i] = float(sample_mass * np.sum(norm2))
        return mode_mass

    def _estimate_mode_stiffness(self, mode_count: int, singular_values: np.ndarray) -> np.ndarray:
        if mode_count == 0:
            return np.zeros(0, dtype=np.float32)

        stiffness = np.zeros(mode_count, dtype=np.float32)
        if singular_values.shape[0] > 0 and singular_values[0] > 0.0:
            energy = singular_values[:mode_count] / singular_values[0]
        else:
            energy = np.ones(mode_count, dtype=np.float64)

        for i in range(mode_count):
            # POD modes are ordered by explained displacement energy. Keep the
            # default heuristic gentle so higher-order exemplar modes are not
            # accidentally locked out before users provide calibrated values.
            stiffness[i] = float(self.stiffness_scale * max(energy[i], 1.0e-6) / ((i + 1) * (i + 1)))
        return stiffness


class ModalGeneratorFEM:
    """Build linear modal modes from nodal finite-element matrices.

    This generator solves the generalized eigenvalue problem
    ``K phi = lambda M phi`` on translational nodal DOFs, mass-normalizes the
    selected eigenvectors, and returns a sampled :class:`ModalBasis`.

    Args:
        node_positions: Rest node positions [m], shape ``[node_count, 3]``.
        mass_matrix: Nodal mass matrix [kg], shape
            ``[3 * node_count, 3 * node_count]``.
        stiffness_matrix: Nodal stiffness matrix [N/m], shape
            ``[3 * node_count, 3 * node_count]``.
        damping_matrix: Optional nodal damping matrix [N s/m], shape
            ``[3 * node_count, 3 * node_count]``.
        sample_points: Body-local points [m] where the returned basis is
            sampled. Defaults to ``node_positions``.
        sample_node_indices: Optional node index for each sample point. When
            provided, sample values are copied exactly from those FEM nodes.
            Otherwise values are interpolated from nodal modes.
        mode_count: Number of modes to retain.
        fixed_node_indices: Nodes whose translational DOFs are fixed during the
            eigen solve. Their modal displacement is zero in the returned basis.
        fixed_dof_indices: Individual nodal DOF indices fixed during the solve.
        discard_mode_count: Number of lowest eigenmodes to skip. Use this to
            drop rigid modes in free-free models.
        eigenvalue_tolerance: Minimum eigenvalue [1/s^2] retained after fixed
            DOFs and discarded modes are removed.
        damping_ratio: Modal damping ratio used when ``damping_matrix`` is not
            provided.
        label: Optional basis label.
    """

    def __init__(
        self,
        node_positions: Sequence[Sequence[float]] | np.ndarray,
        mass_matrix: Sequence[Sequence[float]] | np.ndarray,
        stiffness_matrix: Sequence[Sequence[float]] | np.ndarray,
        damping_matrix: Sequence[Sequence[float]] | np.ndarray | None = None,
        sample_points: Sequence[Sequence[float]] | np.ndarray | None = None,
        sample_node_indices: Sequence[int] | np.ndarray | None = None,
        mode_count: int | None = None,
        fixed_node_indices: Sequence[int] | np.ndarray | None = None,
        fixed_dof_indices: Sequence[int] | np.ndarray | None = None,
        discard_mode_count: int = 0,
        eigenvalue_tolerance: float = 1.0e-8,
        damping_ratio: float = 0.0,
        label: str | None = None,
    ):
        self.node_positions = np.asarray(node_positions, dtype=np.float32)
        if self.node_positions.ndim != 2 or self.node_positions.shape[1] != 3:
            raise ValueError(f"node_positions must have shape [node_count, 3], got {self.node_positions.shape}")

        self.node_count = int(self.node_positions.shape[0])
        self.dof_count = 3 * self.node_count
        self.mass_matrix = self._coerce_square_matrix(mass_matrix, self.dof_count, "mass_matrix")
        self.stiffness_matrix = self._coerce_square_matrix(stiffness_matrix, self.dof_count, "stiffness_matrix")
        self.damping_matrix = (
            None
            if damping_matrix is None
            else self._coerce_square_matrix(damping_matrix, self.dof_count, "damping_matrix")
        )

        if sample_points is None:
            self.sample_points = np.array(self.node_positions, dtype=np.float32, copy=True)
        else:
            self.sample_points = np.asarray(sample_points, dtype=np.float32)
            if self.sample_points.ndim != 2 or self.sample_points.shape[1] != 3:
                raise ValueError(f"sample_points must have shape [sample_count, 3], got {self.sample_points.shape}")

        self.sample_node_indices = None
        if sample_node_indices is not None:
            sample_node_indices = np.asarray(sample_node_indices, dtype=np.int32).reshape((-1,))
            if sample_node_indices.shape[0] != self.sample_points.shape[0]:
                raise ValueError(
                    "sample_node_indices must have one entry per sample point, "
                    f"got {sample_node_indices.shape[0]} for {self.sample_points.shape[0]} samples"
                )
            if np.any(sample_node_indices < 0) or np.any(sample_node_indices >= self.node_count):
                raise ValueError("sample_node_indices contains out-of-range node indices")
            self.sample_node_indices = sample_node_indices

        self.mode_count = None if mode_count is None else int(mode_count)
        if self.mode_count is not None and self.mode_count < 0:
            raise ValueError(f"mode_count must be non-negative, got {self.mode_count}")

        fixed: set[int] = set()
        if fixed_node_indices is not None:
            for node_value in np.asarray(fixed_node_indices, dtype=np.int32).reshape((-1,)):
                node = int(node_value)
                if node < 0 or node >= self.node_count:
                    raise ValueError(f"fixed_node_indices contains out-of-range node index {node}")
                fixed.update((3 * node, 3 * node + 1, 3 * node + 2))
        if fixed_dof_indices is not None:
            for dof_value in np.asarray(fixed_dof_indices, dtype=np.int32).reshape((-1,)):
                dof = int(dof_value)
                if dof < 0 or dof >= self.dof_count:
                    raise ValueError(f"fixed_dof_indices contains out-of-range DOF index {dof}")
                fixed.add(dof)
        self.fixed_dof_indices = np.asarray(sorted(fixed), dtype=np.int32)
        self.discard_mode_count = int(discard_mode_count)
        if self.discard_mode_count < 0:
            raise ValueError(f"discard_mode_count must be non-negative, got {self.discard_mode_count}")
        self.eigenvalue_tolerance = float(eigenvalue_tolerance)
        self.damping_ratio = float(damping_ratio)
        self.label = label

        self.eigenvalues = np.zeros(0, dtype=np.float64)
        self.frequencies = np.zeros(0, dtype=np.float64)
        self.modal_matrix = np.zeros((self.dof_count, 0), dtype=np.float64)

    @staticmethod
    def _coerce_square_matrix(matrix: Sequence[Sequence[float]] | np.ndarray, size: int, name: str) -> np.ndarray:
        array = np.asarray(matrix, dtype=np.float64)
        if array.shape != (size, size):
            raise ValueError(f"{name} must have shape ({size}, {size}), got {array.shape}")
        return 0.5 * (array + array.T)

    def build(self) -> ModalBasis:
        """Build a sampled modal basis."""
        free_dofs = np.ones(self.dof_count, dtype=bool)
        free_dofs[self.fixed_dof_indices] = False
        free = np.nonzero(free_dofs)[0]
        if free.shape[0] == 0:
            raise ValueError("At least one free DOF is required to build FEM modes")

        mass = self.mass_matrix[np.ix_(free, free)]
        stiffness = self.stiffness_matrix[np.ix_(free, free)]
        try:
            chol = np.linalg.cholesky(mass)
        except np.linalg.LinAlgError as exc:
            raise ValueError("mass_matrix must be symmetric positive definite on free DOFs") from exc

        inv_chol_stiffness = np.linalg.solve(chol, stiffness)
        standard = np.linalg.solve(chol, inv_chol_stiffness.T).T
        standard = 0.5 * (standard + standard.T)
        eigenvalues, eigenvectors = np.linalg.eigh(standard)
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        valid = eigenvalues > self.eigenvalue_tolerance
        valid_indices = np.nonzero(valid)[0]
        if self.discard_mode_count:
            valid_indices = valid_indices[self.discard_mode_count :]
        if self.mode_count is not None:
            valid_indices = valid_indices[: self.mode_count]
        if valid_indices.shape[0] == 0:
            raise ValueError("No eigenmodes remain after filtering")

        free_modes = np.linalg.solve(chol.T, eigenvectors[:, valid_indices])
        full_modes = np.zeros((self.dof_count, valid_indices.shape[0]), dtype=np.float64)
        full_modes[free, :] = free_modes

        mode_mass = np.zeros(valid_indices.shape[0], dtype=np.float64)
        mode_stiffness = np.zeros(valid_indices.shape[0], dtype=np.float64)
        mode_damping = np.zeros(valid_indices.shape[0], dtype=np.float64)
        for i in range(valid_indices.shape[0]):
            phi = full_modes[:, i]
            modal_mass = float(phi @ self.mass_matrix @ phi)
            if modal_mass <= 0.0:
                raise ValueError(f"Mode {i} has non-positive modal mass {modal_mass}")
            scale = 1.0 / math.sqrt(modal_mass)
            full_modes[:, i] *= scale
            phi = full_modes[:, i]
            mode_mass[i] = float(phi @ self.mass_matrix @ phi)
            mode_stiffness[i] = max(float(phi @ self.stiffness_matrix @ phi), 0.0)
            if self.damping_matrix is not None:
                mode_damping[i] = max(float(phi @ self.damping_matrix @ phi), 0.0)
            elif self.damping_ratio > 0.0:
                mode_damping[i] = 2.0 * self.damping_ratio * math.sqrt(mode_mass[i] * mode_stiffness[i])

        nodal_phi = np.transpose(full_modes.reshape((self.node_count, 3, -1)), (0, 2, 1))
        if self.sample_node_indices is None:
            nodal_basis = ModalBasis(
                self.node_positions,
                nodal_phi,
                mode_mass,
                mode_stiffness,
                mode_damping,
                label=self.label,
            )
            sample_phi = np.asarray([nodal_basis.evaluate(point) for point in self.sample_points], dtype=np.float32)
        else:
            sample_phi = nodal_phi[self.sample_node_indices]

        self.eigenvalues = mode_stiffness / np.maximum(mode_mass, 1.0e-12)
        self.frequencies = np.sqrt(np.maximum(self.eigenvalues, 0.0)) / (2.0 * math.pi)
        self.modal_matrix = np.array(full_modes, dtype=np.float64, copy=True)
        return ModalBasis(
            self.sample_points,
            sample_phi,
            mode_mass,
            mode_stiffness,
            mode_damping,
            label=self.label,
        )


class ModalGeneratorBeam:
    """Build sampled linear modes for a straight Euler-Bernoulli beam.

    The beam lies along local ``x`` with its midpoint at the body origin.

    Args:
        length: Beam length [m].
        half_width_y: Half width in local ``y`` [m].
        half_width_z: Half width in local ``z`` [m].
        mode_specs: Sequence of mode dictionaries. Supported ``type`` values are
            ``"axial"``, ``"bending_y"``, ``"bending_z"``, and ``"torsion"``.
        sample_count: Number of samples along the beam length.
        density: Density [kg/m^3].
        young_modulus: Young's modulus [Pa].
        shear_modulus: Shear modulus [Pa]. Defaults to ``young_modulus / 2.6``.
        area: Cross-section area [m^2]. Defaults to the rectangular section.
        area_moment_y: Second moment about local ``y`` [m^4].
        area_moment_z: Second moment about local ``z`` [m^4].
        polar_moment: Polar second moment [m^4].
        damping_ratio: Modal damping ratio.
        label: Optional basis label.
    """

    class Mode:
        """Beam mode type names."""

        AXIAL = "axial"
        BENDING_Y = "bending_y"
        BENDING_Z = "bending_z"
        TORSION = "torsion"

    class Boundary:
        """Beam boundary names used by mode specifications."""

        PINNED_PINNED = "pinned-pinned"
        CANTILEVER_TIP = "cantilever-tip"
        FIXED_FREE = "fixed-free"
        LINEAR = "linear"

    def __init__(
        self,
        length: float,
        half_width_y: float = 0.0,
        half_width_z: float = 0.0,
        mode_specs: Sequence[dict[str, Any] | str] | None = None,
        sample_count: int = 33,
        density: float = 1000.0,
        young_modulus: float = 1.0e6,
        shear_modulus: float | None = None,
        area: float | None = None,
        area_moment_y: float | None = None,
        area_moment_z: float | None = None,
        polar_moment: float | None = None,
        damping_ratio: float = 0.0,
        label: str | None = None,
    ):
        if length <= 0.0:
            raise ValueError(f"length must be positive, got {length}")
        self.length = float(length)
        self.half_width_y = float(half_width_y)
        self.half_width_z = float(half_width_z)
        self.mode_specs = [self._coerce_mode_spec(spec) for spec in (mode_specs or [self.Mode.BENDING_Z])]
        self.sample_count = max(2, int(sample_count))
        self.density = float(density)
        self.young_modulus = float(young_modulus)
        self.shear_modulus = float(shear_modulus) if shear_modulus is not None else self.young_modulus / 2.6
        self.area = float(area) if area is not None else max(4.0 * self.half_width_y * self.half_width_z, 0.0)
        self.area_moment_y = (
            float(area_moment_y)
            if area_moment_y is not None
            else (4.0 / 3.0) * self.half_width_y * self.half_width_z**3
        )
        self.area_moment_z = (
            float(area_moment_z)
            if area_moment_z is not None
            else (4.0 / 3.0) * self.half_width_z * self.half_width_y**3
        )
        self.polar_moment = (
            float(polar_moment) if polar_moment is not None else max(self.area_moment_y + self.area_moment_z, 0.0)
        )
        self.damping_ratio = float(damping_ratio)
        self.label = label

    @staticmethod
    def _coerce_mode_spec(spec: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(spec, str):
            return {"type": spec}
        result = dict(spec)
        if "type" not in result:
            raise ValueError("Each beam mode spec must include a 'type' value")
        return result

    def build(self, sample_points: Sequence[Sequence[float]] | np.ndarray | None = None) -> ModalBasis:
        """Build a sampled beam modal basis.

        Args:
            sample_points: Optional body-local sample points [m], shape
                ``[sample_count, 3]``. If omitted, centerline and rectangular
                cross-section samples are generated.

        Returns:
            A sampled modal basis.
        """
        points = self._sample_points() if sample_points is None else np.asarray(sample_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"sample_points must have shape [sample_count, 3], got {points.shape}")

        sample_phi = np.zeros((points.shape[0], len(self.mode_specs), 3), dtype=np.float32)
        for sample_index, point in enumerate(points):
            for mode_index, spec in enumerate(self.mode_specs):
                sample_phi[sample_index, mode_index] = self._evaluate_mode(spec, point)

        mode_mass = np.zeros(len(self.mode_specs), dtype=np.float32)
        mode_stiffness = np.zeros(len(self.mode_specs), dtype=np.float32)
        mode_damping = np.zeros(len(self.mode_specs), dtype=np.float32)
        for i, spec in enumerate(self.mode_specs):
            mode_mass[i], mode_stiffness[i] = self._mode_properties(spec)
            if self.damping_ratio > 0.0:
                mode_damping[i] = (
                    2.0
                    * self.damping_ratio
                    * math.sqrt(max(float(mode_mass[i]), 0.0) * max(float(mode_stiffness[i]), 0.0))
                )

        return ModalBasis(points, sample_phi, mode_mass, mode_stiffness, mode_damping, label=self.label)

    def _sample_points(self) -> np.ndarray:
        yz: list[tuple[float, float]] = [(0.0, 0.0)]
        if self.half_width_y > 0.0:
            yz.extend([(-self.half_width_y, 0.0), (self.half_width_y, 0.0)])
        if self.half_width_z > 0.0:
            yz.extend([(0.0, -self.half_width_z), (0.0, self.half_width_z)])
        if self.half_width_y > 0.0 and self.half_width_z > 0.0:
            yz.extend(
                [
                    (-self.half_width_y, -self.half_width_z),
                    (self.half_width_y, -self.half_width_z),
                    (self.half_width_y, self.half_width_z),
                    (-self.half_width_y, self.half_width_z),
                ]
            )

        unique_yz = list(dict.fromkeys(yz))
        points: list[tuple[float, float, float]] = []
        for x in np.linspace(-0.5 * self.length, 0.5 * self.length, self.sample_count):
            for y, z in unique_yz:
                points.append((float(x), y, z))
        return np.asarray(points, dtype=np.float32)

    def _evaluate_mode(self, spec: dict[str, Any], point: np.ndarray) -> np.ndarray:
        mode_type = str(spec["type"])
        order = int(spec.get("order", 1))
        boundary = str(spec.get("boundary", self._default_boundary(mode_type)))
        x = float(point[0])
        y = float(point[1])
        z = float(point[2])
        s = min(max(x + 0.5 * self.length, 0.0), self.length)

        if mode_type == self.Mode.AXIAL:
            return np.array([x / self.length, 0.0, 0.0], dtype=np.float32)

        if mode_type == self.Mode.BENDING_Y:
            phi, slope = self._bending_shape(s, order, boundary)
            return np.array([-y * slope, phi, 0.0], dtype=np.float32)

        if mode_type == self.Mode.BENDING_Z:
            phi, slope = self._bending_shape(s, order, boundary)
            return np.array([-z * slope, 0.0, phi], dtype=np.float32)

        if mode_type == self.Mode.TORSION:
            theta, _ = self._torsion_shape(s, order, boundary)
            return np.array([0.0, -z * theta, y * theta], dtype=np.float32)

        raise ValueError(f"Unsupported beam mode type '{mode_type}'")

    def _default_boundary(self, mode_type: str) -> str:
        if mode_type == self.Mode.TORSION:
            return self.Boundary.FIXED_FREE
        if mode_type == self.Mode.AXIAL:
            return self.Boundary.LINEAR
        return self.Boundary.CANTILEVER_TIP

    def _bending_shape(self, s: float, order: int, boundary: str) -> tuple[float, float]:
        if boundary == self.Boundary.PINNED_PINNED:
            k = float(order) * math.pi / self.length
            return math.sin(k * s), k * math.cos(k * s)

        if boundary == self.Boundary.CANTILEVER_TIP:
            if order != 1:
                k = (float(order) - 0.5) * math.pi / self.length
                tip = math.sin(k * self.length)
                scale = 1.0 / tip if abs(tip) > 1.0e-8 else 1.0
                return scale * math.sin(k * s), scale * k * math.cos(k * s)

            phi = (s * s * (3.0 * self.length - s)) / (2.0 * self.length**3)
            slope = (3.0 * s * (2.0 * self.length - s)) / (2.0 * self.length**3)
            return phi, slope

        raise ValueError(f"Unsupported bending boundary '{boundary}'")

    def _torsion_shape(self, s: float, order: int, boundary: str) -> tuple[float, float]:
        if boundary == self.Boundary.LINEAR:
            power = max(1, order)
            xi = s / self.length
            theta = xi**power
            dtheta = float(power) * xi ** (power - 1) / self.length
            return theta, dtheta

        if boundary == self.Boundary.FIXED_FREE:
            k = (float(order) - 0.5) * math.pi / self.length
            tip = math.sin(k * self.length)
            scale = 1.0 / tip if abs(tip) > 1.0e-8 else 1.0
            return scale * math.sin(k * s), scale * k * math.cos(k * s)

        raise ValueError(f"Unsupported torsion boundary '{boundary}'")

    def _mode_properties(self, spec: dict[str, Any]) -> tuple[float, float]:
        mode_type = str(spec["type"])
        order = int(spec.get("order", 1))
        boundary = str(spec.get("boundary", self._default_boundary(mode_type)))
        mass_per_length = self.density * self.area

        if mode_type == self.Mode.AXIAL:
            mass = mass_per_length * self.length / 12.0
            stiffness = self.young_modulus * self.area / self.length if self.area > 0.0 else 0.0
            return mass, stiffness

        if mode_type in (self.Mode.BENDING_Y, self.Mode.BENDING_Z):
            inertia = self.area_moment_z if mode_type == self.Mode.BENDING_Y else self.area_moment_y
            ei = self.young_modulus * inertia
            if boundary == self.Boundary.PINNED_PINNED:
                k = float(order) * math.pi / self.length
                mass = mass_per_length * self.length / 2.0
                stiffness = ei * k**4 * self.length / 2.0
                return mass, stiffness
            if boundary == self.Boundary.CANTILEVER_TIP and order == 1:
                mass = mass_per_length * self.length * (33.0 / 140.0)
                stiffness = 3.0 * ei / (self.length**3)
                return mass, stiffness
            k = (float(order) - 0.5) * math.pi / self.length
            mass = mass_per_length * self.length / 2.0
            stiffness = ei * k**4 * self.length / 2.0
            return mass, stiffness

        if mode_type == self.Mode.TORSION:
            rotary_mass = self.density * self.polar_moment
            gj = self.shear_modulus * self.polar_moment
            if boundary == self.Boundary.LINEAR:
                mass = rotary_mass * self.length / float(2 * order + 1)
                stiffness = gj * float(order * order) / self.length
                return mass, stiffness
            k = (float(order) - 0.5) * math.pi / self.length
            mass = rotary_mass * self.length / 2.0
            stiffness = gj * k * k * self.length / 2.0
            return mass, stiffness

        raise ValueError(f"Unsupported beam mode type '{mode_type}'")


__all__ = [
    "ModalBasis",
    "ModalGeneratorBeam",
    "ModalGeneratorFEM",
    "ModalGeneratorPOD",
    "ModalGeneratorSampled",
]
