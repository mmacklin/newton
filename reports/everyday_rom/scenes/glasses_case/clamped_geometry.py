"""Root-clamped isometric keyframes; no kinematic map is applied to modal state."""

from functools import lru_cache

import numpy as np
from peel_case_geometry import make_basis, thick_patch

import newton

NX, NS = 37, 67
THICKNESS = 0.0016
SHELL_CENTER = np.array([0.0, 0.0, 0.052])
ROOT_LOCAL = np.array([0.0, 0.020, 0.0])
ROOT_WORLD = SHELL_CENTER + ROOT_LOCAL
EI = 2e-4
MODE_COUNT = 6


def shell_geometry():
    theta = np.linspace(np.pi / 2, 2 * np.pi - 0.68, 65)
    y, z = 0.020 * np.sin(theta), 0.035 * np.cos(theta)
    # The actual rear lip is a straight clamping strip, not a decorative overlay.
    y[:3] = 0.020
    x = np.linspace(-0.073, 0.073, NX)
    mid = np.stack(np.broadcast_arrays(x[:, None], y[None, :], z[None, :]), axis=-1)
    return thick_patch(mid, THICKNESS), mid


def material_segments():
    u = np.linspace(0, np.pi, 65)
    y = -0.0015 + 0.0215 * np.cos(u)
    z = 0.004 + 0.038 * np.sin(u)
    y = np.r_[0.020, 0.020, y]
    z = np.r_[0.0, 0.002, z]
    dy, dz = np.diff(y), np.diff(z)
    return np.hypot(dy, dz), np.arctan2(-dy, dz)


def flap_half_width(s):
    """Material half-width [m]: broad rear, concave shoulders, rounded tongue.

    The 3 mm terminal chord avoids collapsed triangles at the rounded tip.
    Width is constant on the sewn root strips and independent of fold progress.
    """
    s = np.asarray(s, dtype=float)
    # Keep the entire mouth covered before tapering below the front lip.
    flare = np.clip((s - 0.03) / 0.17, 0.0, 1.0)
    broad = 0.073 + 0.002 * flare * flare * (3.0 - 2.0 * flare)
    shoulder = np.clip((s - 0.74) / 0.20, 0.0, 1.0)
    smooth = shoulder * shoulder * (3.0 - 2.0 * shoulder)
    width = broad + (0.025 - broad) * smooth
    cap = np.clip((s - 0.94) / 0.06, 0.0, 1.0)
    rounded = 0.025 * np.sqrt(1.0 - (1.0 - (0.0015 / 0.025) ** 2) * cap**2)
    return np.where(s > 0.94, rounded, width)


def keyframe(progress):
    ds, beta = material_segments()
    angle = progress * beta
    y = np.r_[0.0, -np.cumsum(ds * np.sin(angle))]
    z = np.r_[0.0, np.cumsum(ds * np.cos(angle))]
    s = np.r_[0.0, np.cumsum(ds)] / ds.sum()
    x = np.linspace(-1.0, 1.0, NX)[:, None] * flap_half_width(s)[None, :]
    mid = np.stack(np.broadcast_arrays(x, y[None, :], z[None, :]), axis=-1)
    points, faces = thick_patch(mid, THICKNESS)
    return points, faces, mid


def root_skin_ids():
    return np.array([skin * NX * NS + ix * NS + col for skin in range(2) for ix in range(NX) for col in (0, 1)])


@lru_cache(maxsize=4)
def flap_basis(mode_count=MODE_COUNT):
    rest, faces, mid = keyframe(0.0)
    snapshots = np.stack([keyframe(t)[0] for t in np.linspace(0, 1, 25)])
    pod = newton.ModalGeneratorPOD(
        rest, deformed_points=snapshots, mode_count=mode_count, total_mass=0.018, derive_psi=False
    ).build()
    phi = pod.sample_phi.astype(float)
    # Every exemplar has identical x; remove SVD roundoff in this exact null field.
    phi[:, :, 0] = 0.0
    phi[root_skin_ids()] = 0.0
    ds, _ = material_segments()
    s = np.r_[0.0, np.cumsum(ds)]
    field = phi.reshape(2, NX, NS, mode_count, 3)
    first = np.gradient(field, s, axis=2, edge_order=2)
    second = np.gradient(first, s, axis=2, edge_order=2)
    ds_weights = np.r_[ds[0] / 2, (ds[:-1] + ds[1:]) / 2, ds[-1] / 2]
    stiffness = EI * np.einsum("s,axsic,axsjc->ij", ds_weights, second, second) / (2 * NX)
    # Translational and slope-zero root samples give a fixed material clamp.
    anchors = np.array([[x, 0.0, z] for x in np.linspace(-0.073, 0.073, NX) for z in (0.0, 0.002)])
    tip_indices = np.array([ix * NS + j for ix in (8, 28) for j in (NS - 2, NS - 1)])
    tip_points = 0.5 * (rest[tip_indices] + rest[tip_indices + NX * NS])
    tip_phi = 0.5 * (phi[tip_indices] + phi[tip_indices + NX * NS])
    points = np.concatenate([rest, anchors, tip_points])
    modes = np.concatenate([phi, np.zeros((len(anchors), mode_count, 3)), tip_phi])
    basis = make_basis(
        points,
        modes,
        0.018,
        stiffness,
        damping_ratio=0.45,
        label="root-clamped isometric POD; quadratic displacement-curvature energy",
    )
    basis.training_snapshots = snapshots
    basis.tip_samples = np.arange(len(points) - 4, len(points))
    return rest, faces, mid, basis


def angle_between(a, b):
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.rad2deg(np.arctan2(cross, dot))


def continuity(shell, flap, flap_ns=NS, shell_ns=65):
    sn = NX * shell_ns
    fn = NX * flap_ns
    sm = 0.5 * (shell[:sn] + shell[sn : 2 * sn]).reshape(NX, shell_ns, 3)
    fm = 0.5 * (flap[:fn] + flap[fn : 2 * fn]).reshape(NX, flap_ns, 3)
    st = sm[:, 0] - sm[:, 1]
    ft = fm[:, 1] - fm[:, 0]
    normal_shell = (shell[:sn] - shell[sn : 2 * sn]).reshape(NX, shell_ns, 3)[:, 0]
    normal_flap = (flap[fn : 2 * fn] - flap[:fn]).reshape(NX, flap_ns, 3)[:, 0]
    return {
        "root_gap_max_m": float(np.linalg.norm(sm[:, 0] - fm[:, 0], axis=1).max()),
        "root_tangent_error_max_degrees": float(angle_between(st, ft).max()),
        "root_normal_error_max_degrees": float(angle_between(normal_shell, normal_flap).max()),
    }


def curvature(points):
    n = NX * NS
    mid = 0.5 * (points[:n] + points[n : 2 * n]).reshape(NX, NS, 3)
    tangent = np.diff(mid, axis=1)
    # The immutable x outline is not bending; measure the extruded material fold.
    tangent[:, :, 0] = 0.0
    angles = angle_between(tangent[:, :-1], tangent[:, 1:])
    total = angles.sum(axis=1)
    # No single short material interval may carry the claimed distributed fold.
    window = 6
    peak = max(float(angles[:, i : i + window].sum(axis=1).max()) for i in range(angles.shape[1] - window + 1))
    return {
        "curvature_min_width_degrees": float(total.min()),
        "curvature_max_width_degrees": float(total.max()),
        "curvature_peak_six_segment_fraction": peak / max(float(total.max()), 1e-9),
    }


def material_orientation(points):
    """Local signed volume using material thickness, invariant to rigid rotation."""
    skins = np.asarray(points[: 2 * NX * NS], dtype=float).reshape(2, NX, NS, 3)
    mid = skins.mean(axis=0)
    director = skins[0] - skins[1]
    ex = np.diff(mid, axis=0)[:, :-1]
    es = np.diff(mid, axis=1)[:-1]
    normal = np.cross(ex, es)
    cosine = np.sum(normal * director[:-1, :-1], axis=-1) / (
        np.linalg.norm(normal, axis=-1) * np.linalg.norm(director[:-1, :-1], axis=-1)
    )
    return {
        "material_director_min_cosine": float(cosine.min()),
        "material_inverted_cells": int(np.count_nonzero(cosine <= 0)),
        "minimum_thickness_m": float(np.linalg.norm(director, axis=-1).min()),
    }
