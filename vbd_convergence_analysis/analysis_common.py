#!/usr/bin/env python3
"""Shared metric-scale configuration for VBD analysis scripts."""
from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

import newton
import newton.usd


ASSET_PATH = os.path.join(WORKTREE, "newton", "examples", "assets", "unisex_shirt.usd")

# The shirt USD is authored in centimeters (metersPerUnit = 0.01).
ASSET_METERS_PER_UNIT = 0.01

# Analysis scenarios are now expressed in SI / meter-scale units.
GRAVITY = -9.81
FRAME_DT = 1.0 / 60.0
SIM_SUBSTEPS = 4

TRI_KE = 1e4
TRI_KA = 1e4
TRI_KD = 1.5e-6
EDGE_KE = 5.0
EDGE_KD = 1e-2

# Preserve the previous areal density after converting cm-authored geometry to meters:
# 0.02 g/cm^2 = 0.2 kg/m^2.
DENSITY = 0.2

PARTICLE_RADIUS = 0.005
SOFT_CONTACT_MARGIN = 0.005

CONVERGENCE_DROP_HEIGHT_RANGE = (0.40, 1.20)
LOW_DROP_HEIGHT_RANGE = (0.05, 0.20)
LATERAL_OFFSET_RANGE = (-0.30, 0.30)

SELF_CONTACT_RADIUS = 0.002
# Margin must be wide enough that the max_displacement truncation
# (margin * 0.85 * 0.5 per substep) does not bottleneck free-fall velocity.
# 0.02m with 4 substeps @ 60fps → max_vel ≈ 2.04 m/s (covers 0.20m drop).
SELF_CONTACT_MARGIN = 0.02
SELF_CONTACT_MARGIN_WIDE = 0.02
SELF_CONTACT_REST_EXCLUSION_RADIUS = 0.005

# Self-contact friction velocity threshold (m/s). The kernel computes
# eps_U = friction_epsilon * dt, converting this velocity-space ramp width
# to a displacement-space threshold for the current substep.
FRICTION_EPSILON = 1e-4
FRICTION_EPSILON_SWEEP = (1e-4, 1e-3, 1e-2)

POP_THRESHOLD_M = 5e-4


def load_shirt_mesh_vertices() -> tuple[list[wp.vec3], np.ndarray, float]:
    """Load the shirt mesh and convert authored units to meters."""
    usd_stage = Usd.Stage.Open(ASSET_PATH)
    scale = float(UsdGeom.GetStageMetersPerUnit(usd_stage))
    usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
    shirt_mesh = newton.usd.get_mesh(usd_prim)
    vertices = [wp.vec3(*(np.array(v, dtype=np.float32) * scale)) for v in shirt_mesh.vertices]
    return vertices, shirt_mesh.indices, scale
