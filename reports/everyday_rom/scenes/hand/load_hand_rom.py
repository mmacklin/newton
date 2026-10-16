# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Load the attributed derived ROM without any original NIMBLE model files."""

from pathlib import Path

import numpy as np

import newton


def load_hand_rom(path: str | Path):
    """Return a public Newton ModalBasis and derived geometry/body properties.

    No NIMBLE implementation, checkpoints, dictionaries, or pose dataset is
    needed. The shell parameters are assumed, not anatomical tissue values.
    The caller supplies a fixed wrist support and contact-only actuation.
    """
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    basis_keys = (
        "sample_points",
        "sample_phi",
        "sample_psi",
        "sample_mass",
        "mode_mass",
        "mode_stiffness",
        "mode_damping",
    )
    expected = {*basis_keys, "rest", "faces", "used", "anchor", "anchor_phi", "body_mass", "body_com", "body_inertia"}
    if set(arrays) != expected:
        raise ValueError("Unexpected fields in derived ROM")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite derived ROM")
    np.testing.assert_array_equal(arrays["rest"], arrays["sample_points"][arrays["used"]])
    basis = newton.ModalBasis(**{key: arrays[key] for key in basis_keys}, label="NIMBLE_derived_pose_POD_shell")
    return basis, {key: arrays[key] for key in expected.difference(basis_keys)}
