# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for tendon/cable examples."""

import numpy as np
import warp as wp

from newton._src.sim.tendon import TendonLinkType


def get_tendon_attachment_worlds(solver, model, state):
    """Return current world-space tendon segment attachment points."""
    att_l = solver.tendon_seg_attachment_l.numpy()
    att_r = solver.tendon_seg_attachment_r.numpy()
    if not hasattr(solver, "tendon_seg_attachment_l_local") or solver.tendon_seg_attachment_l_local is None:
        return att_l, att_r

    local_l = solver.tendon_seg_attachment_l_local.numpy()
    local_r = solver.tendon_seg_attachment_r_local.numpy()
    seg_link_l = solver.tendon_seg_link_l.numpy()
    link_body = model.tendon_link_body.numpy()
    body_q = state.body_q.numpy()

    att_l_world = np.empty_like(att_l)
    att_r_world = np.empty_like(att_r)
    for seg, link_l in enumerate(seg_link_l):
        att_l_world[seg] = _transform_point_np(body_q[link_body[link_l]], local_l[seg])
        att_r_world[seg] = _transform_point_np(body_q[link_body[link_l + 1]], local_r[seg])

    return att_l_world, att_r_world


def get_tendon_cable_lines(solver, model, state):
    """Build line-segment arrays for tendon visualization including arc wraps."""
    att_l, att_r = get_tendon_attachment_worlds(solver, model, state)

    starts_list = []
    ends_list = []
    for i in range(model.tendon_segment_count):
        starts_list.append(att_l[i])
        ends_list.append(att_r[i])

    tendon_start = model.tendon_start.numpy()
    link_type = model.tendon_link_type.numpy()
    link_body = model.tendon_link_body.numpy()
    link_offset = model.tendon_link_offset.numpy()
    link_axis = model.tendon_link_axis.numpy()
    body_q = state.body_q.numpy()

    seg = 0
    for t in range(model.tendon_count):
        start = tendon_start[t]
        end = tendon_start[t + 1]
        num_links = end - start
        for i in range(start + 1, end - 1):
            if link_type[i] == int(TendonLinkType.ROLLING):
                b = link_body[i]
                pose = body_q[b]
                p = pose[:3]
                q = pose[3:]
                off = link_offset[i]
                ax = link_axis[i]
                t2 = 2.0 * np.cross(q[:3], off)
                center = off + q[3] * t2 + np.cross(q[:3], t2) + p
                t2n = 2.0 * np.cross(q[:3], ax)
                normal = ax + q[3] * t2n + np.cross(q[:3], t2n)

                seg_left = seg + (i - start) - 1
                seg_right = seg + (i - start)
                pt_dep = att_r[seg_left]
                pt_arr = att_l[seg_right]

                r_dep = pt_dep - center
                r_arr = pt_arr - center
                cross_val = np.dot(np.cross(r_dep, r_arr), normal)
                dot_val = np.dot(r_dep, r_arr)
                total_angle = np.arctan2(cross_val, dot_val)
                if np.isnan(total_angle):
                    continue

                n_arc = max(8, int(abs(total_angle) / 0.2))
                for j in range(n_arc):
                    frac0 = j / n_arc
                    frac1 = (j + 1) / n_arc
                    angle0 = frac0 * total_angle
                    angle1 = frac1 * total_angle
                    c0, s0 = np.cos(angle0), np.sin(angle0)
                    p0 = center + r_dep * c0 + np.cross(normal, r_dep) * s0
                    c1, s1 = np.cos(angle1), np.sin(angle1)
                    p1 = center + r_dep * c1 + np.cross(normal, r_dep) * s1
                    starts_list.append(p0)
                    ends_list.append(p1)

        seg += num_links - 1

    starts = wp.array(np.array(starts_list, dtype=np.float32), dtype=wp.vec3)
    ends = wp.array(np.array(ends_list, dtype=np.float32), dtype=wp.vec3)
    return starts, ends


def _transform_point_np(pose, point):
    p = pose[:3]
    q = pose[3:]
    t = 2.0 * np.cross(q[:3], point)
    return p + point + q[3] * t + np.cross(q[:3], t)


def _transform_vector_np(pose, vector):
    q = pose[3:]
    t = 2.0 * np.cross(q[:3], vector)
    return vector + q[3] * t + np.cross(q[:3], t)


def get_tendon_total_lengths(solver, model, state):
    """Return geometric total cable length for each tendon.

    The total includes straight segment distances plus wrap arc lengths at
    rolling links.  This is the measured counterpart of
    ``solver.tendon_total_cable``.
    """
    if model.tendon_count == 0:
        return np.zeros(0, dtype=np.float32)

    att_l, att_r = get_tendon_attachment_worlds(solver, model, state)
    tendon_start = model.tendon_start.numpy()
    link_type = model.tendon_link_type.numpy()
    link_body = model.tendon_link_body.numpy()
    link_offset = model.tendon_link_offset.numpy()
    link_axis = model.tendon_link_axis.numpy()
    link_radius = model.tendon_link_radius.numpy()
    body_q = state.body_q.numpy()

    lengths = np.zeros(model.tendon_count, dtype=np.float32)
    seg = 0
    for t in range(model.tendon_count):
        start = tendon_start[t]
        end = tendon_start[t + 1]
        num_links = end - start
        seg_base = seg
        total = 0.0

        for s in range(num_links - 1):
            total += np.linalg.norm(att_r[seg_base + s] - att_l[seg_base + s])

        for i in range(start + 1, end - 1):
            if link_type[i] != int(TendonLinkType.ROLLING):
                continue

            pose = body_q[link_body[i]]
            center = _transform_point_np(pose, link_offset[i])
            normal = _transform_vector_np(pose, link_axis[i])
            seg_left = seg_base + (i - start) - 1
            seg_right = seg_base + (i - start)
            r_left = att_r[seg_left] - center
            r_right = att_l[seg_right] - center
            cross_val = np.dot(np.cross(r_left, r_right), normal)
            dot_val = np.dot(r_left, r_right)
            theta = abs(np.arctan2(cross_val, dot_val))
            total += theta * link_radius[i]

        lengths[t] = total
        seg += num_links - 1

    return lengths


def assert_tendon_total_length(example, rel_tol=0.05, abs_tol=1.0e-3, allow_slack=False):
    """Assert that an example's geometric cable lengths stay near target.

    The check stores peak absolute and relative errors on *example* so a
    failure reports both the current error and the worst error seen so far.
    If allow_slack is true, only over-length errors fail; a loose cable can
    have a taut geometric path shorter than the stored cable length.
    """
    model = example.model
    if model.tendon_count == 0:
        return

    current = get_tendon_total_lengths(example.solver, model, example.state_0)
    expected = example.solver.tendon_total_cable.numpy()
    abs_err = np.abs(current - expected)
    rel_err = abs_err / np.maximum(np.abs(expected), 1.0e-8)

    max_abs = getattr(example, "_tendon_total_length_max_abs_error", None)
    max_rel = getattr(example, "_tendon_total_length_max_rel_error", None)
    if max_abs is None or len(max_abs) != len(abs_err):
        max_abs = np.zeros_like(abs_err)
        max_rel = np.zeros_like(rel_err)

    max_abs = np.maximum(max_abs, abs_err)
    max_rel = np.maximum(max_rel, rel_err)
    example._tendon_total_length_max_abs_error = max_abs
    example._tendon_total_length_max_rel_error = max_rel

    allowed = abs_tol + rel_tol * np.maximum(np.abs(expected), 1.0e-8)
    if allow_slack:
        failed = np.nonzero((current - expected) > allowed)[0]
    else:
        failed = np.nonzero(abs_err > allowed)[0]
    if len(failed) == 0:
        return

    details = []
    for i in failed:
        details.append(
            f"tendon {i}: current={current[i]:.6f}, expected={expected[i]:.6f}, "
            f"abs_err={abs_err[i]:.6f}, rel_err={rel_err[i]:.4%}, "
            f"max_abs={max_abs[i]:.6f}, max_rel={max_rel[i]:.4%}, allowed={allowed[i]:.6f}"
        )
    raise AssertionError("Tendon total cable length error exceeded tolerance: " + "; ".join(details))
