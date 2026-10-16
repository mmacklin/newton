# SPDX-License-Identifier: Apache-2.0
"""Authored wall-field helper only; run clamped_case.py for the current scene."""

import numpy as np


class Example:
    @staticmethod
    def wall_trial_modes(points):
        """Evaluate the authored smooth wall field, including offset vertices."""
        x, y, z = np.asarray(points).T
        theta = np.arctan2(y / 0.020, z / 0.035)
        ax = np.maximum(0, 1 - (x / 0.073) ** 2) ** 2
        front = np.maximum(0, -np.sin(theta)) ** 2
        phi = np.zeros((len(points), 3, 3), np.float32)
        phi[:, 0, 1] = ax * front
        phi[:, 1, 1] = ax * front * x / 0.055
        phi[:, 2, 1] = ax * front * np.cos(2 * np.pi * x / 0.146)
        return phi
