# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Floating cap: bounded contact/release/drop; large floppy folds unvalidated."""

import numpy as np
from wearable_geometry import (
    Assembly,
    Config,
    WearableExample,
    ellipsoid_mesh,
    grid_mesh,
    remove_rigid_modes,
    run_cli,
    thicken,
    tube_mesh,
)


def crown_point(a, theta):
    # Subtle six-panel relief, not a texture or imported asset.
    seam_relief = 1 - 0.009 * np.cos(6 * theta) ** 16 * np.sin(a)
    return np.stack(
        (
            0.100 * np.sin(a) * np.cos(theta) * seam_relief,
            0.085 * np.sin(a) * np.sin(theta) * seam_relief,
            0.034 + 0.112 * np.cos(a),
        ),
        axis=-1,
    )


def crown_mesh():
    segments, rings = 96, 20
    points = [[0, 0, 0.146]]
    theta = np.arange(segments) * 2 * np.pi / segments
    for a in np.linspace(0, np.pi / 2, rings + 1)[1:]:
        points.extend(crown_point(np.full_like(theta, a), theta))
    faces = []
    for j in range(segments):
        faces.append((0, 1 + j, 1 + (j + 1) % segments))
    for i in range(rings - 1):
        for j in range(segments):
            k = (j + 1) % segments
            a, b = 1 + i * segments + j, 1 + i * segments + k
            faces.extend(((a, a + segments, b + segments), (a, b + segments, b)))
    return thicken(np.asarray(points), np.asarray(faces), 0.0012)


def brim_point(s, a):
    x = 0.10 * np.cos(a) + s * 0.085 * np.cos(a) ** 0.4
    y = (0.085 + 0.018 * s) * np.sin(a)
    z = 0.034 - 0.009 * (y / 0.103) ** 2 - 0.003 * s
    return np.stack((x, y, z), axis=-1)


def modes(vertices, region):
    x, y, z = vertices.T
    # Invert the planform numerically to evaluate the same smooth field on
    # the bill, edge piping, and stitching. At fixed y, x(s) is monotone.
    # Bracketing avoids the divergent fixed-point map at the narrow side roots.
    lower, upper = np.zeros(len(vertices)), np.full(len(vertices), 1.15)
    for _ in range(36):
        s = (lower + upper) / 2
        cosine = np.sqrt(np.maximum(1 - (y / (0.085 + 0.018 * s)) ** 2, 0))
        projected_x = 0.10 * cosine + s * 0.085 * cosine**0.4
        lower = np.where(projected_x < x, s, lower)
        upper = np.where(projected_x >= x, s, upper)
    s = (lower + upper) / 2
    bill = region == 1
    shape = bill * s**2
    raw = np.zeros((len(vertices), 5, 3))
    raw[:, 0, 2] = shape
    raw[:, 1, 2] = shape * y / 0.1
    raw[:, 2, 2] = shape * (y / 0.1) ** 2
    crown = (region == 2) * np.clip((z - 0.036) / 0.10, 0, 1) ** 2
    raw[:, 3, 2] = crown
    raw[:, 4, 1] = crown * y / 0.085
    return raw


def whole_crown_pod(vertices, region, sample_mass, mode_count=12):
    """Mass-weighted POD of geometric shell poses, not simulated cloth folds.

    Circular section bending preserves neutral-line arclength. Smooth dents
    approximate contact patches; neither area preservation nor self-contact is
    imposed. These poses are training data only, never runtime animation.
    """
    snapshots = []
    labels = []
    height = vertices[:, 2] - 0.034
    for axis in (0, 1):
        coordinate = vertices[:, axis]
        for curvature in (-2.0, -1.0, 1.0, 2.0):
            pose = vertices.astype(float).copy()
            angle = coordinate * curvature
            pose[:, axis] = np.sin(angle) / curvature - height * np.sin(angle)
            pose[:, 2] = 0.034 + (1 - np.cos(angle)) / curvature + height * np.cos(angle)
            snapshots.append(pose - vertices)
            labels.append(f"circular_section_bend_axis_{axis}_curvature_{curvature}")
    crown = region == 2
    shell_normal = np.c_[vertices[:, 0] / 0.100**2, vertices[:, 1] / 0.085**2, height / 0.112**2]
    shell_normal /= np.maximum(np.linalg.norm(shell_normal, axis=1, keepdims=True), 1e-10)
    elevation = np.clip(height / 0.112, 0, 1)
    for theta in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        center = np.array([0.067 * np.cos(theta), 0.057 * np.sin(theta), 0.110])
        patch = crown * np.exp(-np.sum((vertices - center) ** 2, axis=1) / 0.045**2) * elevation
        for depth in (-0.016, 0.016):
            snapshots.append(-depth * patch[:, None] * shell_normal)
            labels.append(f"smooth_panel_dent_{theta:.3f}_{depth}")
    for amplitude in (-0.12, -0.06, 0.06, 0.12):
        displacement = np.zeros_like(vertices)
        displacement[:, 2] = -amplitude * 0.112 * crown * elevation**2
        displacement[:, :2] = 0.25 * amplitude * (crown * elevation)[:, None] * vertices[:, :2]
        snapshots.append(displacement)
        labels.append(f"crown_meridional_compression_{amplitude}")
    # A smooth front-section bend includes the neighboring crown material.
    # Section rotations avoid the pure vertical shear of the old bill fields.
    distance = np.maximum(vertices[:, 0] - 0.080, 0)
    transverse = vertices[:, 1] / 0.1
    section_height = vertices[:, 2] - 0.030
    for mode in range(3):
        for amplitude in (-0.012, 0.012):
            displacement = np.zeros_like(vertices)
            displacement[:, 2] = amplitude * (distance / 0.105) ** 2 * transverse**mode
            displacement[:, 0] = -section_height * amplitude * 2 * distance / 0.105**2 * transverse**mode
            if mode:
                displacement[:, 1] = (
                    -section_height * amplitude * (distance / 0.105) ** 2 * mode * transverse ** (mode - 1) / 0.1
                )
            snapshots.append(displacement)
            labels.append(f"front_section_bending_{mode}_{amplitude}")
    snapshots = np.asarray(snapshots)
    projected = remove_rigid_modes(vertices, snapshots.transpose(1, 0, 2), sample_mass).transpose(1, 0, 2)
    weights = np.repeat(sample_mass, 3)
    matrix = projected.reshape(len(projected), -1).T
    left, singular, _ = np.linalg.svd(np.sqrt(weights)[:, None] * matrix, full_matrices=False)
    physical = left[:, :mode_count] / np.sqrt(weights)[:, None]
    coefficients = matrix.T @ (weights[:, None] * physical)
    recovered = (coefficients @ physical.T).reshape(projected.shape)
    scales = np.linalg.norm(physical.reshape(len(vertices), 3, mode_count), axis=1).max(axis=0)
    raw = (physical / scales).reshape(len(vertices), 3, mode_count).transpose(0, 2, 1)
    rms = np.sqrt(
        np.sum(sample_mass[None, :, None] * (projected - recovered) ** 2) / (len(projected) * sample_mass.sum())
    )
    audit = {
        "status": "Geometric POD span used by the bounded shell prototype; not a large-fold validation",
        "snapshot_count": len(snapshots),
        "retained_modes": mode_count,
        "retained_mass_weighted_energy": float(np.sum(singular[:mode_count] ** 2) / np.sum(singular**2)),
        "snapshot_weighted_rms_error_m": float(rms),
        "snapshot_max_error_m": float(np.linalg.norm(projected - recovered, axis=2).max()),
        "snapshot_labels": labels,
        "limitations": [
            "Geometric poses, not FEM equilibria or calibrated woven-fabric mechanics.",
            "Independent linear amplitudes do not enforce the nonlinear snapshot manifold.",
            "Circular section bending preserves only neutral-line arclength, not whole-hat area.",
            "No self-contact, crease model, or unsupported large-fold claim.",
        ],
    }
    return raw, snapshots, audit


def shell_stiffness_modes(assembly, vertices, sample_mass, pod):
    """Project linear membrane and dihedral bending energies into the POD span.

    Only crown and bill midsurfaces carry structural energy. Decorative trim
    retains its lumped mass and follows the same modes. Parameters are assumed
    isotropic shell properties, not a calibrated textile constitutive model.
    """
    materials = {
        "six_panel_crown": {"young_pa": 1.0e6, "thickness_m": 0.0012, "poisson": 0.3},
        "flexible_laminated_bill": {"young_pa": 2.0e7, "thickness_m": 0.0023, "poisson": 0.3},
    }
    count = pod.shape[1]
    membrane = np.zeros((count, count))
    bending = np.zeros_like(membrane)
    for part, all_faces in zip(assembly.parts, assembly.faces, strict=True):
        if part["name"] not in materials:
            continue
        material = materials[part["name"]]
        start, half = part["start"], part["count"] // 2
        faces = all_faces - start
        faces = faces[np.all(faces < half, axis=1)]
        points = (vertices[start : start + half] + vertices[start + half : start + 2 * half]) / 2
        phi = (pod[start : start + half] + pod[start + half : start + 2 * half]) / 2
        tri = points[faces]
        edge1, edge2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        normal = np.cross(edge1, edge2)
        area = np.linalg.norm(normal, axis=1) / 2
        normal /= 2 * area[:, None]
        tangent1 = edge1 / np.linalg.norm(edge1, axis=1, keepdims=True)
        tangent2 = np.cross(normal, tangent1)
        reference = np.zeros((len(faces), 2, 2))
        reference[:, 0, 0] = np.linalg.norm(edge1, axis=1)
        reference[:, 0, 1] = np.einsum("fc,fc->f", edge2, tangent1)
        reference[:, 1, 1] = np.einsum("fc,fc->f", edge2, tangent2)
        delta = np.stack((phi[faces[:, 1]] - phi[faces[:, 0]], phi[faces[:, 2]] - phi[faces[:, 0]]), axis=-1)
        gradient = np.einsum("fmci,fij->fmcj", delta, np.linalg.inv(reference))
        strain = np.stack(
            (
                np.einsum("fc,fmc->fm", tangent1, gradient[..., 0]),
                np.einsum("fc,fmc->fm", tangent2, gradient[..., 1]),
                np.einsum("fc,fmc->fm", tangent1, gradient[..., 1])
                + np.einsum("fc,fmc->fm", tangent2, gradient[..., 0]),
            ),
            axis=-1,
        )
        young, thickness, poisson = material["young_pa"], material["thickness_m"], material["poisson"]
        constitutive = (
            young
            * thickness
            / (1 - poisson**2)
            * np.array([[1, poisson, 0], [poisson, 1, 0], [0, 0, (1 - poisson) / 2]])
        )
        membrane += np.einsum("f,fmi,ij,fnj->mn", area, strain, constitutive, strain)

        adjacent, hinges = {}, []
        for face, triangle in enumerate(faces):
            for a, b in zip(triangle, np.roll(triangle, -1), strict=True):
                key = tuple(sorted((a, b)))
                if key in adjacent:
                    old_face, old_a, old_b = adjacent[key]
                    hinges.append((old_face, face, old_a, old_b))
                else:
                    adjacent[key] = (face, a, b)
        hinges = np.asarray(hinges)
        face0, face1, endpoint0, endpoint1 = hinges.T

        def dihedral(position, faces=faces, face0=face0, face1=face1, endpoint0=endpoint0, endpoint1=endpoint1):
            triangles = position[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            direction = position[endpoint1] - position[endpoint0]
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            sine = np.einsum("ec,ec->e", direction, np.cross(normals[face0], normals[face1]))
            cosine = np.einsum("ec,ec->e", normals[face0], normals[face1])
            return np.arctan2(sine, cosine)

        derivative = np.zeros((len(hinges), count))
        epsilon = 1e-6
        for mode in range(count):
            difference = dihedral(points + epsilon * phi[:, mode]) - dihedral(points - epsilon * phi[:, mode])
            derivative[:, mode] = np.arctan2(np.sin(difference), np.cos(difference)) / (2 * epsilon)
        length_squared = np.sum((points[endpoint1] - points[endpoint0]) ** 2, axis=1)
        rigidity = young * thickness**3 / (12 * (1 - poisson**2))
        hinge_weight = rigidity * 3 * length_squared / (area[face0] + area[face1])
        bending += np.einsum("e,em,en->mn", hinge_weight, derivative, derivative)

    stiffness = membrane + bending
    mass = np.einsum("n,nmc,nkc->mk", sample_mass, pod, pod)
    eigenmass, vectors = np.linalg.eigh(mass)
    assert eigenmass.min() > 0
    whitening = vectors / np.sqrt(eigenmass)
    eigenvalues, eigenvectors = np.linalg.eigh(whitening.T @ stiffness @ whitening)
    assert eigenvalues.min() > 0, "POD structural energy has an unconstrained mechanism"
    rotation = whitening @ eigenvectors
    raw = np.einsum("nmc,mk->nkc", pod, rotation)
    scale = np.linalg.norm(raw, axis=2).max(axis=0)
    rotation /= scale
    raw /= scale[None, :, None]
    audit = {
        "method": "Mass-whitened generalized eigensolve of projected plane-stress membrane plus linearized dihedral hinge energy",
        "materials": materials,
        "frequencies_hz": (np.sqrt(eigenvalues) / (2 * np.pi)).tolist(),
        "membrane_trace_N_per_m": float(np.trace(membrane)),
        "bending_trace_N_per_m": float(np.trace(bending)),
        "membrane_to_bending_trace_ratio": float(np.trace(membrane) / np.trace(bending)),
        "hinge_derivative_step_m": epsilon,
        "limitations": [
            "Small-strain isotropic shell tangent, not calibrated fabric or nonlinear area-preserving cloth energy.",
            "Crown and bill midsurfaces only; trim contributes mass but no independent stiffness.",
            "Their attachment is represented by the common POD span, not a separately resolved seam constitutive model.",
            "Geometric nonlinear membrane terms, buckling, self-contact and large folds remain unsupported.",
        ],
    }
    return raw, audit, {"membrane": membrane, "bending": bending, "mass": mass, "rotation": rotation}


def make_spec():
    assembly = Assembly()
    assembly.add("six_panel_crown", crown_mesh(), [0.68, 0.075, 0.11], 0.055, 2, 0.88)
    a = np.linspace(-1.24, 1.24, 65)
    s = np.linspace(0, 1, 19)
    bill = brim_point(s[:, None] * np.ones((1, len(a))), np.ones((len(s), 1)) * a)
    assembly.add("flexible_laminated_bill", thicken(*grid_mesh(bill), 0.0023), [0.48, 0.025, 0.055], 0.038, 1, 0.85)
    theta = np.linspace(0, 2 * np.pi, 129)[:-1]
    band = np.c_[0.100 * np.cos(theta), 0.085 * np.sin(theta), np.full(len(theta), 0.034)]
    assembly.add("sweatband_binding", tube_mesh(band, 0.0025, closed=True), [0.25, 0.025, 0.04], 0.008, 0)
    for panel in range(6):
        angles = np.linspace(0.065, np.pi / 2 - 0.025, 60)
        path = crown_point(angles, np.full_like(angles, 2 * np.pi * panel / 6))
        path[:, 2] += 0.0009
        assembly.add(f"crown_panel_seam_{panel}", tube_mesh(path, 0.00055, sides=6), [0.88, 0.29, 0.30], 0.00025, 2)
    assembly.add(
        "top_fabric_button",
        ellipsoid_mesh((0.006, 0.006, 0.003), (0, 0, 0.1475), 32, 12),
        [0.48, 0.025, 0.055],
        0.001,
        2,
    )
    for i, depth in enumerate((0.70, 0.82, 0.94)):
        path = brim_point(np.full_like(a, depth), a)
        path[:, 2] += 0.0014
        assembly.add(f"bill_stitch_row_{i}", tube_mesh(path, 0.00030, sides=6), [0.84, 0.26, 0.27], 0.0002, 1)
    edge = brim_point(np.ones_like(a), a)
    assembly.add("bill_edge_binding", tube_mesh(edge, 0.0013), [0.31, 0.015, 0.035], 0.002, 1)
    vertices, faces, sample_mass, region = assembly.finish()
    raw, snapshots, pod_audit = whole_crown_pod(vertices, region, sample_mass)
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
    rest_edges = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    rest_lengths = np.linalg.norm(rest_edges, axis=1)
    strains = [
        float(
            np.abs(np.linalg.norm(rest_edges + pose[edges[:, 1]] - pose[edges[:, 0]], axis=1) / rest_lengths - 1).max()
        )
        for pose in snapshots
    ]
    pod_audit["snapshot_max_edge_strain"] = max(strains)
    pod_audit["snapshot_edge_strains"] = strains
    pod_audit["training_pose_strain_bound"] = 0.25
    pod = raw.copy()
    raw, stiffness_audit, matrices = shell_stiffness_modes(assembly, vertices, sample_mass, pod)
    return {
        "slug": "cap",
        "dynamic": True,
        "status": "validated bounded whole-crown contact/release/drop prototype",
        "initial_height": -0.010,
        "rest_height": -0.022,
        "title": "Floating cap: bounded shell indentation",
        "assembly": assembly,
        "modes": modes,
        "raw_modes": raw,
        "pod_modes": pod,
        "stiffness_audit": stiffness_audit,
        "stiffness_matrices": matrices,
        "snapshots": snapshots,
        "pod_audit": pod_audit,
        "frequencies": stiffness_audit["frequencies_hz"],
        "press_xy": [0.015, 0],
        "press_radius": 0.045,
        "press_clearance": 0.075,
        "indentation": 0.0135,
        "validation_scope": "Whole-crown contact/release/drop at approximately 5.37 mm peak deformation; large floppy folds and woven-fabric mechanics are not validated",
        "camera": {"position": [0.48, -0.48, 0.32], "target": [0.035, 0, 0.073], "fov": 37},
        "static": [],
        "description": "A free cap drops onto physical ground, undergoes approximately 5.37 mm contact-driven crown deformation, then recovers after pad release. Large floppy folds remain unvalidated.",
        "model": "Finite-mass freely floating cap with twelve modes from thirty whole-shell poses. Stiffness follows a mass-whitened eigensolve of projected linear membrane and hinge-bending energies. No rim or rigid-frame clamp; this bounded shell prototype is not a woven-fabric model.",
        "bounds": {
            "displacement_m": 0.027,
            "edge_strain": 0.14,
            "principal_strain": 0.14,
            "release_m": 0.002,
            "control_m": 0.002,
            "minimum_deformation_m": 0.005,
            "minimum_crown_deformation_m": 0.005,
        },
        "limitations": [
            "Original procedural generic baseball cap; no proprietary geometry or logo.",
            "Laminated elastic bill and smooth crown approximation, not woven-fabric mechanics or unsupported cloth folding.",
            "Assumed isotropic shell material parameters; projected linear membrane/hinge stiffness is not calibrated woven-fabric mechanics.",
            "All-surface displacement, edge and principal-stretch guards support bounded indentation only; no self-contact, buckling, plasticity, or failure.",
            "The entire hat is a dynamic, pickable/droppable body; there is no fixed sweatband, rim, or rigid frame.",
            "Surface-vertex contact is checked using additional edge-midpoint and triangle-centroid samples.",
        ],
    }


class Example(WearableExample):
    """Validate bounded whole-crown contact/release/drop, not large cloth folds."""

    def __init__(self, viewer=None, cfg=None):
        super().__init__(viewer, cfg or Config(), make_spec())


if __name__ == "__main__":
    run_cli(Example, "cap")
