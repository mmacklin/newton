# SPDX-License-Identifier: Apache-2.0
"""Pure published retention helpers, without full-order experiment drivers."""

import numpy as np


def retained_band_max(distance):
    detached = np.flatnonzero(np.median(distance, axis=1) > 0.004)
    end = max(0, int(detached.min()) - 1) if len(detached) else 6
    return float(distance[:end].max()) if end else 0.0


def retained_constraints(rest, old_fixed, old_targets, rows, columns=17):
    """Add retained rows without changing an existing authored grip target."""
    rows = np.asarray(rows, dtype=int)
    if not len(rows) or np.any(rows < 0) or np.any(rows >= len(rest) // columns):
        raise ValueError("Retained row is outside the shell grid")
    added = (rows[:, None] * columns + np.arange(columns)).ravel()
    fixed = np.unique(np.r_[old_fixed, added])
    targets = rest[fixed].copy()
    targets[np.searchsorted(fixed, old_fixed)] = old_targets
    return fixed, targets
