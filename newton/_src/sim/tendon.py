# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tendon system for cable-driven mechanisms.

Implements massless cable routing through rigid body contact points (pulleys,
pinholes, attachments) using the Cable Joints method [Müller et al. SCA 2018]
extended with capstan friction for tension-dependent force transmission.

Each tendon is an ordered sequence of waypoints on rigid bodies. Between
adjacent waypoints, a unilateral distance constraint enforces the cable
length. Rest length flows between segments as bodies rotate, and capstan
friction bounds the tension ratio at each contact.
"""

from __future__ import annotations

from enum import IntEnum


class TendonLinkType(IntEnum):
    """Type of contact between a tendon and a rigid body."""

    ROLLING = 0
    """Cable wraps around the body surface. Attachment point moves to the
    tangent; rest length updated by arc length as the body rotates."""

    ATTACHMENT = 1
    """Cable is fixed to the body at a point. Neither attachment nor rest
    length changes."""

    PINHOLE = 2
    """Cable passes through a fixed point on the body. Attachment does not
    move, but rest length transfers between adjacent segments subject to
    capstan friction bounds."""
