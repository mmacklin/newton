"""Vertical-ray coverage of the pouch mouth by the actual flap mid-surface."""

import numpy as np
from clamped_geometry import NS, NX, ROOT_WORLD, SHELL_CENTER, keyframe, shell_geometry


def opening_coverage(points):
    """Sample the full-width mouth, bounded by the two real wall lip edges.

    The model is an extrusion: each material column has the same y/z and
    symmetric x bounds. Intersect rays with its piecewise-linear side boundary,
    requiring the hit to lie above the opening plane, not behind the wall.
    """
    mid = np.asarray(points).reshape(2, NX, NS, 3).mean(axis=0)
    wall = shell_geometry()[1] + SHELL_CENTER
    rear, front = wall[-1, 0], wall[-1, -1]
    fractions = np.linspace(0.001, 0.999, 401)
    mouth = rear[None] + fractions[:, None] * (front - rear)[None]
    edge = mid[-1]
    a, b = edge[:-1], edge[1:]
    dy = b[:, 1] - a[:, 1]
    valid = np.abs(dy) > 1e-12
    t = (mouth[:, 1, None] - a[None, :, 1]) / np.where(valid, dy, 1)[None]
    hits = a[None] + t[:, :, None] * (b - a)[None]
    above = hits[:, :, 2] >= mouth[:, 2, None] - 1e-7
    mask = valid[None] & (t >= 0) & (t <= 1) & above
    half_width = np.max(np.where(mask, hits[:, :, 0], 0), axis=1)
    required = mouth[:, 0]
    fraction = np.minimum(half_width / required, 1)
    return {
        "opening_min_width_margin_m": float(np.min(half_width - required)),
        "opening_covered_fraction": float(np.mean(fraction)),
        "opening_uncovered_sections": int(np.count_nonzero(half_width < required - 1e-7)),
        "opening_sections": len(fractions),
    }


def closed_keyframe_coverage():
    return opening_coverage(keyframe(1)[0] + ROOT_WORLD)
