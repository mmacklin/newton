# SPDX-License-Identifier: Apache-2.0
"""Published gripper and progression helpers; the scene is path_banana.py."""

import numpy as np
import warp as wp
from peel_case_geometry import write_kinematic


@wp.func
def grip_pose(t: float, strip: int, path: wp.array2d[wp.vec3], driven: bool):
    frame = float(0.0)
    if driven:
        frame = wp.clamp(t * 120.0, 0.0, float(path.shape[1] - 1))
    i = wp.min(int(frame), path.shape[1] - 2)
    position = wp.lerp(path[strip, i], path[strip, i + 1], frame - float(i))
    return wp.transform(position, wp.quat_identity())


@wp.kernel
def drive_grippers(
    clock: wp.array[float],
    dt: float,
    bodies: wp.array[int],
    path: wp.array2d[wp.vec3],
    driven: bool,
    p0: wp.array[wp.transform],
    v0: wp.array[wp.spatial_vector],
    p1: wp.array[wp.transform],
    v1: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    write_kinematic(
        bodies[i], grip_pose(clock[0], i, path, driven), grip_pose(clock[0] + dt, i, path, driven), dt, p0, v0, p1, v1
    )


def peel_progression(records):
    """Geometric detachment tests with no reference to prescribed release times."""
    times = np.asarray([r["time"] for r in records])
    distance = np.asarray([r["band_median_fruit_distance_m"] for r in records]).reshape(-1, 3, 7)
    maximum = np.asarray([r["band_max_fruit_distance_m"] for r in records]).reshape(-1, 3, 7)
    detached = distance > 0.004
    detachment = np.full((3, 7), np.nan)
    retained = 0.0
    for strip in range(3):
        for band in range(7):
            stable = np.flatnonzero(np.convolve(detached[:, strip, band], np.ones(3), mode="valid") == 3)
            if len(stable):
                detachment[strip, band] = times[stable[0]]
        for frame in range(len(times)):
            free = np.flatnonzero(detached[frame, strip])
            end = max(0, int(free.min()) - 1) if len(free) else 6
            if end:
                retained = max(retained, float(maximum[frame, strip, :end].max()))
    delays = -np.diff(detachment, axis=1)
    front = np.asarray([r["peel_front_s"] for r in records])
    retreat = float(max(0, np.diff(front).max(initial=0)))
    order = bool(np.isfinite(detachment).all() and np.all(delays > 0.15))
    return {
        "geometric_detachment_times_bottom_to_top_s": [
            [float(x) if np.isfinite(x) else None for x in row] for row in detachment
        ],
        "minimum_adjacent_detachment_delay_s": float(np.nanmin(delays)) if np.isfinite(delays).any() else None,
        "retained_region_max_fruit_distance_m": retained,
        "maximum_front_retreat_fraction": retreat,
        "progressive_peel_passed": bool(order and retained < 0.0035 and retreat <= 0.05),
        "progression_definition": "Seven sampling bands per WHOLE strip, not bodies. First three consecutive frames with median signed fruit-triangle distance >4 mm define detachment. Top-first adjacent delay >0.15 s; retained region excludes the single transition band below the measured front. No release schedule enters these checks.",
    }
