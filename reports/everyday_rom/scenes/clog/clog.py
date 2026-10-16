# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Free perforated foam clog: whole-shoe flex, rapid release, recoil and landing."""

import numpy as np
from wearable_geometry import Assembly, Config, WearableExample, ellipsoid_mesh, grid_mesh, run_cli, thicken, tube_mesh


def distal_toe_points(vertices):
    """Round only the distal 70 mm of the physical rest mesh with a C2 taper."""
    points = np.array(vertices, dtype=float, copy=True)
    t = np.clip((points[..., 0] - 0.075) / 0.070, 0.0, 1.0)
    blend = t**3 * (10.0 - 15.0 * t + 6.0 * t**2)
    points[..., 1] *= 1.0 + 0.10 * blend
    return points


def distal_toe_mesh(mesh):
    """Return the rounded collision/rest surface before mass and Phi regeneration."""
    vertices, faces = mesh
    return distal_toe_points(vertices), faces


def outline(theta):
    return np.stack((0.005 + 0.14 * np.cos(theta), 0.057 * np.sin(theta) * (1 + 0.10 * np.cos(theta))), axis=-1)


def upper_point(u, theta):
    outer = outline(theta)
    inner = np.stack((-0.062 + 0.057 * np.cos(theta), 0.039 * np.sin(theta)), axis=-1)
    xy = (1 - u[..., None]) * outer + u[..., None] * inner
    z = 0.024 + (0.023 + 0.004 * np.cos(theta)) * u + 0.024 * np.sin(np.pi * u) * (0.65 + 0.35 * np.cos(theta))
    return np.concatenate((xy, z[..., None]), axis=-1)


def retriangulate_upper(vertices, faces, skin_face_count):
    """Improve offset-skin slivers without moving vertices or changing vent walls."""
    faces = faces.copy()

    def geometry(triangles):
        points = vertices[triangles]
        a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
        normal = np.cross(a, b)
        quality = 2 * np.sqrt(3) * np.linalg.norm(normal, axis=1) / np.sum(a * a + b * b + (a - b) ** 2, axis=1)
        return quality, normal

    for _ in range(20):
        quality, normals = geometry(faces)
        adjacent = {}
        for index, face in enumerate(faces):
            for a, b, c in (face, np.roll(face, -1), np.roll(face, -2)):
                adjacent.setdefault(tuple(sorted((a, b))), []).append((index, a, b, c))
        changed = set()
        for index in np.argsort(quality[: 2 * skin_face_count], kind="stable"):
            if quality[index] >= 0.08:
                break
            if index in changed:
                continue
            candidates = []
            for a, b, c in (faces[index], np.roll(faces[index], -1), np.roll(faces[index], -2)):
                neighbors = adjacent[tuple(sorted((a, b)))]
                if len(neighbors) != 2:
                    continue
                other, _, _, d = next(row for row in neighbors if row[0] != index)
                if other in changed or other // skin_face_count != index // skin_face_count:
                    continue
                if tuple(sorted((c, d))) in adjacent:
                    continue
                replacement = np.array(((c, d, b), (d, c, a)))
                new_quality, new_normals = geometry(replacement)
                # The oriented area vector of the patch gives the orientation
                # without letting an almost-zero-area inverted sliver dominate.
                patch_normal = normals[index] + normals[other]
                patch_normal /= np.linalg.norm(patch_normal)
                new_normals /= np.linalg.norm(new_normals, axis=1, keepdims=True)
                if np.any(new_normals @ patch_normal < 0.8):
                    continue
                gain = new_quality.min() - min(quality[index], quality[other])
                if gain > 1e-6:
                    candidates.append((gain, other, replacement))
            if candidates:
                _, other, replacement = max(candidates, key=lambda item: item[0])
                faces[[index, other]] = replacement
                changed.update((index, other))
        if not changed:
            break
    return vertices, faces


def perforated_upper(*, repair=True):
    """Annular shoe upper with 13 real round through-vents and an open heel."""
    points, faces, known = [], [], {}
    ntheta, nu = 32, 6
    # Each tile shares an eight-vertex boundary with its neighbors. Vent tiles
    # replace their center fan with a circular inner boundary, not black decals.
    boundary = np.array(((0, 0), (0.5, 0), (1, 0), (1, 0.5), (1, 1), (0.5, 1), (0, 1), (0, 0.5)))
    vents = {
        (i % ntheta, j)
        for j, columns in ((2, (-4, -2, 0, 2, 4)), (3, (-3, -1, 1, 3)), (4, (-3, -1, 1, 3)))
        for i in columns
    }

    def vertex(u, theta):
        key = (round(float(u), 9), round(float(theta % (2 * np.pi)), 9))
        if key not in known:
            known[key] = len(points)
            points.append(upper_point(np.asarray(u), np.asarray(theta)))
        return known[key]

    for j in range(nu):
        for i in range(ntheta):
            uv = boundary.copy()
            outer = [vertex((j + v) / nu, 2 * np.pi * (i + a) / ntheta) for a, v in uv]
            if (i, j) in vents:
                angles = np.arctan2(uv[:, 1] - 0.5, uv[:, 0] - 0.5)
                inner = [
                    vertex((j + 0.5 + 0.28 * np.sin(a)) / nu, 2 * np.pi * (i + 0.5 + 0.30 * np.cos(a)) / ntheta)
                    for a in angles
                ]
                for k in range(8):
                    n = (k + 1) % 8
                    faces.extend(((outer[k], outer[n], inner[n]), (outer[k], inner[n], inner[k])))
            else:
                center = vertex((j + 0.5) / nu, 2 * np.pi * (i + 0.5) / ntheta)
                for k in range(8):
                    faces.append((outer[k], outer[(k + 1) % 8], center))
    vertices, triangles = thicken(np.asarray(points), np.asarray(faces), 0.0035)
    return retriangulate_upper(vertices, triangles, len(faces)) if repair else (vertices, triangles)


def sole_mesh():
    theta = np.arange(128) * 2 * np.pi / 128
    xy = outline(theta)
    rings = []
    for scale, z in ((0.94, 0.002), (0.99, 0.005), (1.015, 0.012), (1.01, 0.021), (0.98, 0.025)):
        rings.append(np.c_[xy * scale, np.full(len(theta), z)])
    vertices, faces = grid_mesh(rings, periodic=True)
    vertices, faces = vertices.tolist(), faces.tolist()
    # Radial surface rings resolve both the beam curvature and support contact;
    # the old center-to-rim triangle fans were too coarse for whole-sole flex.
    for boundary, scale, height, reverse in ((0, 0.94, 0.002, True), (512, 0.98, 0.025, False)):
        previous = np.arange(128) + boundary
        for factor in np.linspace(0.9, 0.1, 9):
            current = np.arange(128) + len(vertices)
            vertices.extend(np.c_[xy * scale * factor, np.full(len(theta), height)])
            for j in range(128):
                k = (j + 1) % 128
                triangles = ((previous[j], previous[k], current[k]), (previous[j], current[k], current[j]))
                faces.extend(tuple(reversed(t)) if reverse else t for t in triangles)
            previous = current
        center = len(vertices)
        vertices.append([0.005, 0, height])
        for j in range(128):
            triangle = (previous[j], previous[(j + 1) % 128], center)
            faces.append(tuple(reversed(triangle)) if reverse else triangle)
    return np.asarray(vertices), np.asarray(faces, dtype=np.int32)


def modes(vertices, region):
    x, y, z = vertices.T
    length, width = 0.14, 0.057
    x = x - 0.005
    h = z - 0.026
    raw = np.zeros((len(vertices), 12, 3))
    # Euler-Bernoulli section rotations are part of the sampled *linear*
    # displacement fields. All shell, sole and strap vertices participate.
    for mode, power in ((0, 2), (1, 3), (9, 4)):
        raw[:, mode, 2] = (x / length) ** power
        raw[:, mode, 0] = -h * power * (x / length) ** (power - 1) / length
    raw[:, 2, 2] = (y / width) ** 2
    raw[:, 2, 1] = -h * 2 * y / width**2
    raw[:, 3, 2] = x * y / (length * width)
    raw[:, 3, 0] = -h * y / (length * width)
    raw[:, 3, 1] = -h * x / (length * width)
    raw[:, 4] = np.c_[-0.15 * x, -0.15 * y, h] / 0.06
    raw[:, 5] = np.c_[x, -0.15 * y, -0.15 * h] / length
    raw[:, 6] = np.c_[-0.15 * x, y, -0.15 * h] / width
    top = (region == 1) * np.clip((z - 0.023) / 0.040, 0, 1) ** 2
    raw[:, 7, 2] = top * np.exp(-(((x - 0.055) / 0.065) ** 2))
    raw[:, 8, 2] = top * np.exp(-(((x + 0.070) / 0.055) ** 2))
    raw[:, 10, 2] = top * x * y / (length * width)
    raw[:, 11, 1] = (x / length) ** 2
    raw[:, 11, 0] = -y * 2 * x / length**2
    return raw


def make_spec():
    assembly = Assembly()
    assembly.add("perforated_foam_upper", distal_toe_mesh(perforated_upper()), [0.19, 0.61, 0.47], 0.115, 1)
    assembly.add("rounded_foam_sole", distal_toe_mesh(sole_mesh()), [0.10, 0.35, 0.28], 0.105, 0)
    theta = np.linspace(0, 2 * np.pi, 129)[:-1]
    collar = upper_point(np.ones_like(theta), theta)
    assembly.add("rolled_open_heel_rim", tube_mesh(collar, 0.0024, closed=True), [0.24, 0.68, 0.52], 0.009, 1)
    xy = outline(theta)
    assembly.add(
        "sole_piping",
        distal_toe_mesh(tube_mesh(np.c_[xy * 1.014, np.full(len(theta), 0.018)], 0.0012, closed=True)),
        [0.37, 0.75, 0.60],
        0.003,
        0,
    )
    # A broad rear strap, not a simulated hinge: all parts share one ROM.
    a = np.linspace(0, np.pi, 65)
    strap = []
    for zoffset in (-0.005, 0.005):
        strap.append(np.c_[-0.035 - 0.092 * np.sin(a), 0.049 * np.cos(a), 0.048 + 0.008 * np.sin(a) + zoffset])
    assembly.add("heel_strap", thicken(*grid_mesh(strap), 0.003), [0.14, 0.45, 0.35], 0.015, 0)
    for side in (-1, 1):
        assembly.add(
            f"strap_button_{side}",
            ellipsoid_mesh((0.005, 0.0022, 0.005), (-0.035, side * 0.051, 0.048), 24, 12),
            [0.07, 0.18, 0.16],
            0.001,
            0,
        )
    return {
        "slug": "clog",
        "title": "Free foam clog: flex and recoil",
        "assembly": assembly,
        "modes": modes,
        "frequencies": [8.5, 36, 24, 20, 70, 70, 70, 110, 110, 180, 110, 28],
        "dynamic": True,
        "initial_height": 0.078,
        "rest_height": 0.063,
        "support": {"center": [0.005, 0, 0.0325], "radius": 0.0325, "half_height": 0.075},
        "pads": [{"xy": [-0.100, 0.0], "radius": 0.025}, {"xy": [0.105, 0.0], "radius": 0.028}],
        "press_clearance": 0.060,
        "indentation": 0.031,
        "schedule": [0.85, 0.90, 2.35, 0.025],
        "camera": {"position": [0.38, -0.48, 0.29], "target": [0.005, 0, 0.080], "fov": 37},
        "description": "An unclamped clog drops onto a rounded ground support, bends under heel and forefoot pads, then recoils and lands after a 25 ms release.",
        "model": "One finite-mass floating elastic body with twelve whole-shoe bending, torsion and compression modes. Ground and the visible rounded support are physical colliders. No part of the shoe is prescribed or clamped.",
        "bounds": {
            "displacement_m": 0.070,
            "edge_strain": 0.25,
            "principal_strain": 0.25,
            "release_m": 0.012,
            "control_m": 0.018,
            "minimum_deformation_m": 0.025,
            "minimum_sole_flex_m": 0.025,
            "minimum_recoil_speed_m_s": 0.08,
            "hold_chatter_m": 0.002,
            "settled_motion_m": 0.004,
            "penetration_m": 0.002,
        },
        "limitations": [
            "Generic original procedural form, not a branded product or proprietary scan.",
            "Assumed lumped component masses and modal frequencies, not calibrated foam constitutive parameters.",
            "Linear ROM, bounded indentation; no foam densification, buckling, plasticity, self-contact, or failure.",
            "Whole-body analytic beam/shell fields, not a calibrated foam constitutive model. Small-to-moderate bending is bounded by measured surface strain and rotation.",
            "The rear strap is bonded to the same whole-shoe ROM, not an articulated hinge. Twelve independent amplitudes are solved without nonlinear coordinate constraints.",
            "Surface-vertex contacts; face interiors and edge midpoints are checked independently for penetration.",
        ],
    }


class Example(WearableExample):
    """Run the contact-driven clog compression and unloading test."""

    def __init__(self, viewer=None, cfg=None):
        super().__init__(viewer, cfg or Config(), make_spec())


if __name__ == "__main__":
    run_cli(Example, "clog")
