# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Private NIMBLE sampled-basis preparation and geometric diagnostics.

No model assets are embedded here. Native arrays and derived geometry must stay
under the private cache until their redistribution terms are established.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton

PRIVATE_ROOT = Path.home() / ".cache/newton-everyday-rom/nimble"
POD_ROOT = PRIVATE_ROOT / "pod"
OUTPUT_ROOT = PRIVATE_ROOT / "dynamics"


def reference_basis(rest_reference="zero"):
    """Prepare one deterministic near-mean native rest/POD without changing assets."""
    if rest_reference == "zero":
        return POD_ROOT / "basis.npz", {"name": "zero", "description": "Original zero-axis-angle native rest"}
    if rest_reference != "near-mean":
        raise ValueError("Unknown rest reference")
    directory = private_output(OUTPUT_ROOT / "near-mean-input")
    directory.mkdir(parents=True, exist_ok=True)
    basis_path, manifest = directory / "basis.npz", directory / "reference.json"
    fingerprint = {
        name: hashlib.sha256((POD_ROOT / name).read_bytes()).hexdigest() for name in ("snapshots.npz", "basis.npz")
    }
    if basis_path.exists() and manifest.exists():
        info = json.loads(manifest.read_text())
        if info.get("fingerprint") == fingerprint:
            return basis_path, info
    with np.load(POD_ROOT / "snapshots.npz", allow_pickle=False) as data:
        snapshots, poses, faces = data["train"], data["train_poses"], data["faces"]
    with np.load(POD_ROOT / "basis.npz", allow_pickle=False) as data:
        mass = data["sample_mass"].copy()
    used, inverse = np.unique(faces, return_inverse=True)
    compact_faces = inverse.reshape(-1, 3)
    distances = np.linalg.norm((poses - poses.mean(axis=0)).reshape(len(poses), -1), axis=1)
    checker = IntersectionAudit(snapshots[0, used], compact_faces, device="cpu")
    selected = None
    for index in np.argsort(distances, kind="stable"):
        if checker.count(snapshots[index, used]) == 0:
            selected = int(index)
            break
    if selected is None:
        raise AssertionError("No intersection-free native training rest was available")
    rest = snapshots[selected].copy()
    sqrt_mass = np.repeat(np.sqrt(mass.astype(float)), 3)
    displacement = (snapshots.astype(float) - rest).reshape(len(snapshots), -1)
    _, singular_values, vh = np.linalg.svd(displacement * sqrt_mass, full_matrices=False)
    phi = (vh[:48].T / sqrt_mass[:, None]).reshape(len(rest), 3, 48).transpose(0, 2, 1).astype(np.float32)
    modal_mass = np.einsum("n,nmc,nmc->m", mass.astype(float), phi.astype(float), phi.astype(float))
    np.savez_compressed(
        basis_path,
        points=rest,
        phi=phi,
        faces=faces,
        sample_mass=mass,
        mode_mass=modal_mass.astype(np.float32),
        singular_values=singular_values,
    )
    info = {
        "name": "near-mean",
        "fingerprint": fingerprint,
        "native_train_index": selected,
        "pose_distance_to_training_mean_rad": float(distances[selected]),
        "selection": "Nearest stored native training pose to the training axis-angle mean among intersection-free poses; no dynamics-based selection.",
        "basis_method": "48-mode mass-weighted SVD of all 160 saved native training positions minus the selected native rest, without mean centering.",
        "geometry": "Exact native snapshot with original faces and all original sample masses; no face deletion or corrective deformation.",
        "limitations": "Intersection-free does not certify anatomical or material validity. This is a different rest and retrained POD, not the original zero-angle basis.",
    }
    manifest.write_text(json.dumps(info, indent=2) + "\n")
    return basis_path, info


def native_audit_directory(rest_reference, stiffness_model="assumed"):
    if stiffness_model == "projected-shell":
        return OUTPUT_ROOT / "projected-shell"
    return OUTPUT_ROOT if rest_reference == "zero" else OUTPUT_ROOT / "near-mean"


def private_output(path):
    """Reject public output paths, including symlinks escaping the private cache."""
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError(f"NIMBLE-derived outputs must stay under {OUTPUT_ROOT}")
    return path


def transform_points(points, pose):
    """Recover world points from an original xyzw pose, rounded to float32."""
    p = np.asarray(points, dtype=float)
    q = np.asarray(pose[3:], dtype=float)
    t = 2 * np.cross(q[:3], p)
    return (p + q[3] * t + np.cross(q[:3], t) + pose[:3]).astype(np.float32)


def box(half, center=(0, 0, 0)):
    vertices = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], dtype=np.float32)
    vertices = (vertices * half + center).astype(np.float32)
    faces = np.array(
        [
            [0, 2, 3],
            [0, 3, 1],
            [4, 5, 7],
            [4, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [2, 6, 7],
            [2, 7, 3],
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 7],
            [1, 7, 5],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def box_sdf(points, body_pose, half):
    inv = np.r_[np.zeros(3), -body_pose[3:6], body_pose[6]]
    local = transform_points(points - body_pose[:3], inv)
    d = np.abs(local) - half
    return np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(np.max(d, axis=1), 0)


SHELL_MATERIAL = {"young_pa": 5.0e6, "thickness_m": 0.001, "poisson": 0.3}


def shell_operators(points, faces, phi):
    """Project linear triangle strain and signed dihedral derivatives at rest."""
    points, phi = np.asarray(points, float), np.asarray(phi, float)
    tri = points[faces]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    area = np.linalg.norm(cross, axis=1) / 2
    if np.any(area <= 0):
        raise ValueError("Degenerate shell rest triangle")
    normal = cross / (2 * area[:, None])
    tangent1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    tangent2 = np.cross(normal, tangent1)
    reference = np.zeros((len(faces), 2, 2))
    reference[:, 0, 0] = np.linalg.norm(e1, axis=1)
    reference[:, 0, 1] = np.einsum("fc,fc->f", e2, tangent1)
    reference[:, 1, 1] = np.einsum("fc,fc->f", e2, tangent2)
    d1, d2 = phi[faces[:, 1]] - phi[faces[:, 0]], phi[faces[:, 2]] - phi[faces[:, 0]]
    gradient = np.einsum("fmci,fij->fmcj", np.stack((d1, d2), axis=-1), np.linalg.inv(reference))
    strain = np.stack(
        (
            np.einsum("fc,fmc->fm", tangent1, gradient[..., 0]),
            np.einsum("fc,fmc->fm", tangent2, gradient[..., 1]),
            np.einsum("fc,fmc->fm", tangent1, gradient[..., 1]) + np.einsum("fc,fmc->fm", tangent2, gradient[..., 0]),
        ),
        axis=-1,
    )
    adjacent, hinges = {}, []
    for face, triangle in enumerate(faces):
        for a, b in zip(triangle, np.roll(triangle, -1), strict=True):
            key = tuple(sorted((int(a), int(b))))
            if key in adjacent:
                previous, first, second = adjacent.pop(key)
                hinges.append((previous, face, first, second))
            else:
                adjacent[key] = (face, a, b)
    if adjacent:
        raise ValueError("The private hand shell must retain its closed topology")
    hinges = np.asarray(hinges, dtype=np.int32)
    f0, f1, a, b = hinges.T
    edge = points[b] - points[a]
    length = np.linalg.norm(edge, axis=1)
    direction = edge / length[:, None]
    delta_edge = phi[b] - phi[a]
    delta_direction = (
        delta_edge - direction[:, None] * np.einsum("ec,emc->em", direction, delta_edge)[..., None]
    ) / length[:, None, None]
    delta_cross = np.cross(d1, e2[:, None]) + np.cross(e1[:, None], d2)
    delta_normal = (delta_cross - normal[:, None] * np.einsum("fc,fmc->fm", normal, delta_cross)[..., None]) / (
        2 * area[:, None, None]
    )
    n0, n1, dn0, dn1 = normal[f0], normal[f1], delta_normal[f0], delta_normal[f1]
    sine = np.einsum("ec,ec->e", direction, np.cross(n0, n1))
    cosine = np.einsum("ec,ec->e", n0, n1)
    dsine = np.einsum("emc,ec->em", delta_direction, np.cross(n0, n1)) + np.einsum(
        "ec,emc->em", direction, np.cross(dn0, n1[:, None]) + np.cross(n0[:, None], dn1)
    )
    dcosine = np.einsum("emc,ec->em", dn0, n1) + np.einsum("ec,emc->em", n0, dn1)
    bend = (cosine[:, None] * dsine - sine[:, None] * dcosine) / (sine**2 + cosine**2)[:, None]
    return strain, bend, area, 3 * length**2 / (area[f0] + area[f1]), hinges


def shell_projection(points, faces, phi, mass, *, young_pa=5.0e6):
    """Diagonalize projected membrane/bending stiffness without deleting modes."""
    if not np.isfinite(young_pa) or young_pa <= 0:
        raise ValueError("Young's modulus must be finite and positive [Pa]")
    strain, bend, area, hinge_weight, _ = shell_operators(points, faces, phi)
    material = {**SHELL_MATERIAL, "young_pa": float(young_pa)}
    young, thickness, poisson = (material[k] for k in ("young_pa", "thickness_m", "poisson"))
    constitutive = (
        young * thickness / (1 - poisson**2) * np.array([[1, poisson, 0], [poisson, 1, 0], [0, 0, (1 - poisson) / 2]])
    )
    rigidity = young * thickness**3 / (12 * (1 - poisson**2))
    membrane = np.einsum("f,fmi,ij,fnj->mn", area, strain, constitutive, strain, optimize=True)
    bending = np.einsum("e,em,en->mn", rigidity * hinge_weight, bend, bend, optimize=True)
    stiffness = membrane + bending
    gram = np.einsum("n,nic,njc->ij", mass, phi, phi)
    whitening = np.linalg.inv(np.linalg.cholesky(gram).T)
    reduced = whitening.T @ stiffness @ whitening
    eigenvalues, eigenvectors = np.linalg.eigh((reduced + reduced.T) / 2)
    if eigenvalues.min() <= 0:
        raise AssertionError("Projected shell has a nonpositive eigenvalue; no frequency floor is applied")
    transform = whitening @ eigenvectors
    # Fix eigenvector signs for deterministic state and source recovery.
    sign = np.sign(transform[np.argmax(np.abs(transform), axis=0), np.arange(len(eigenvalues))])
    transform *= sign
    result = np.einsum("nmc,mk->nkc", phi, transform)
    residual = stiffness @ transform - gram @ transform * eigenvalues[None]
    audit = {
        "material": material,
        "bending_rigidity_N_m": rigidity,
        "method": "Rest-space plane-stress triangle membrane plus linearized signed-dihedral hinge energy; generalized eigensolve in the unchanged POD span.",
        "frequency_floor": False,
        "mode_deletion": False,
        "mass": gram.tolist(),
        "membrane": membrane.tolist(),
        "bending": bending.tolist(),
        "transform_from_mass_normalized_pod": transform.tolist(),
        "eigenvalues_s2": eigenvalues.tolist(),
        "frequencies_hz": (np.sqrt(eigenvalues) / (2 * np.pi)).tolist(),
        "eigen_residual_relative": float(np.linalg.norm(residual) / np.linalg.norm(stiffness @ transform)),
        "bending_to_membrane_trace_ratio": float(np.trace(bending) / np.trace(membrane)),
        "limitations": "Uncalibrated shell-like hand ROM, not anatomical soft tissue. Linear rest tangent only; no finite-strain barrier, volumetric elasticity, or self-contact. Original all-sample mass is independent of assumed shell thickness.",
    }
    return result, np.sqrt(eigenvalues), audit


def build_hand(
    *,
    modes=24,
    frequency_min=12.0,
    frequency_max=36.0,
    damping_ratio=0.25,
    rest_reference="zero",
    stiffness_model="assumed",
    young_pa=5.0e6,
):
    """Retain the native POD span with a mass-orthonormal runtime basis."""
    if not np.isfinite(young_pa) or young_pa <= 0:
        raise ValueError("Young's modulus must be finite and positive [Pa]")
    if stiffness_model != "projected-shell" and young_pa != SHELL_MATERIAL["young_pa"]:
        raise ValueError("A custom Young's modulus requires projected-shell stiffness")
    source_path, reference = reference_basis(rest_reference)
    with np.load(source_path, allow_pickle=False) as data:
        points = data["points"].copy()
        source_phi = data["phi"][:, :modes].astype(float)
        weights = data["sample_mass"].astype(float)
        original_faces = data["faces"].copy()
    if modes < 1 or source_phi.shape[1] != modes:
        raise ValueError("Requested mode count is not present in the private POD")
    if not 0 < frequency_min <= frequency_max or damping_ratio < 0:
        raise ValueError("Frequencies must be positive and damping nonnegative")
    used, inverse = np.unique(original_faces, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    gram = np.einsum("n,nic,njc->ij", weights, source_phi, source_phi)
    change = np.linalg.inv(np.linalg.cholesky(gram).T)
    phi = np.einsum("nic,ij->njc", source_phi, change)
    omega = 2 * np.pi * np.linspace(frequency_min, frequency_max, modes)
    shell = None
    if stiffness_model == "projected-shell":
        phi, omega, shell = shell_projection(points, original_faces, phi, weights, young_pa=young_pa)
    elif stiffness_model != "assumed":
        raise ValueError("Unknown stiffness model")

    # Fit a local affine wrist displacement to couple a real six-DOF support.
    # One zero-mass anchor sample leaves all original 5,990 fields unchanged.
    wrist = used[points[used, 2] < 0.0]
    anchor = points[wrist].mean(axis=0)
    design = np.column_stack((np.ones(len(wrist)), points[wrist] - anchor))
    fit = np.linalg.lstsq(design, phi[wrist].reshape(len(wrist), -1), rcond=None)[0].reshape(4, modes, 3)
    gradient = fit[1:]
    psi_anchor = 0.5 * np.column_stack(
        (
            gradient[1, :, 2] - gradient[2, :, 1],
            gradient[2, :, 0] - gradient[0, :, 2],
            gradient[0, :, 1] - gradient[1, :, 0],
        )
    )
    points = np.vstack((points, anchor)).astype(np.float32)
    phi = np.concatenate((phi, fit[0][None])).astype(np.float32)
    psi = np.zeros_like(phi)
    psi[-1] = psi_anchor
    weights = np.r_[weights, 0.0]
    frequencies = omega / (2 * np.pi)
    basis = newton.ModalBasis(
        sample_points=points,
        sample_phi=phi,
        sample_psi=psi,
        sample_mass=weights,
        mode_stiffness=omega**2,
        mode_damping=2 * damping_ratio * omega,
        label="native_nimble_pose_pod_assumed_dynamics",
    )
    mass = float(weights.sum())
    com = (weights[:, None] * points).sum(axis=0) / mass
    x = points - com
    inertia = np.dot(weights, (x * x).sum(axis=1)) * np.eye(3) - np.einsum("n,ni,nj->ij", weights, x, x)
    normalized_gram = np.einsum("n,nic,njc->ij", weights, phi.astype(float), phi.astype(float))
    audit = {
        "mass_kg": mass,
        "com_m": com.tolist(),
        "inertia_kg_m2": inertia.tolist(),
        "source_sample_count": len(points) - 1,
        "surface_vertices": len(used),
        "triangles": len(faces),
        "unused_source_vertices": len(points) - 1 - len(used),
        "mass_orthogonality_error": float(np.max(np.abs(normalized_gram - np.eye(modes)))),
        "frequencies_hz": frequencies.tolist(),
        "damping_ratio": damping_ratio,
        "mass_model": "Original uniform 0.35 kg lumped sample mass, including unreferenced interior samples; not tissue-calibrated.",
        "stiffness_model": stiffness_model,
        "stiffness_description": "Projected membrane/bending tangent"
        if shell
        else "Assumed increasing modal frequencies, independent of POD singular values or pose variance.",
        "shell_projection": shell,
        "wrist_frame": "Affine least-squares displacement and infinitesimal curl on source wrist samples; one zero-mass attachment sample.",
        "source_basis_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_basis_path": str(source_path),
        "rest_reference": reference,
    }
    return points[used].copy(), faces, basis, used, anchor, audit


def surface_metrics(rest, vertices, faces, *, native_envelope=None):
    """Measure local surface distortion, excluding the body's rigid motion."""
    t0, t = rest[faces].astype(float), vertices[faces].astype(float)
    e0 = np.stack((t0[:, 1] - t0[:, 0], t0[:, 2] - t0[:, 1], t0[:, 0] - t0[:, 2]), axis=1)
    e = np.stack((t[:, 1] - t[:, 0], t[:, 2] - t[:, 1], t[:, 0] - t[:, 2]), axis=1)
    rest_lengths = np.linalg.norm(e0, axis=2)
    ratios = np.linalg.norm(e, axis=2) / rest_lengths
    normal0 = np.cross(e0[:, 0], -e0[:, 2])
    normal = np.cross(e[:, 0], -e[:, 2])
    a0, a = np.linalg.norm(normal0, axis=1), np.linalg.norm(normal, axis=1)
    area = a / a0
    alignment = np.einsum("nc,nc->n", normal, normal0) / np.maximum(a * a0, 1e-30)
    volume0 = np.einsum("nc,nc->n", t0[:, 0], np.cross(t0[:, 1], t0[:, 2])).sum() / 6
    volume = np.einsum("nc,nc->n", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6
    metrics = {
        "max_edge_strain": float(np.max(np.abs(ratios - 1))),
        "edge_strain_p99": float(np.quantile(np.abs(ratios - 1), 0.99)),
        "min_area_ratio": float(area.min()),
        "max_area_ratio": float(area.max()),
        "min_normal_alignment": float(alignment.min()),
        "normal_reversals": int(np.count_nonzero(alignment <= 0)),
        "degenerate_triangles": int(np.count_nonzero(a < 1e-12)),
        "signed_volume_ratio": float(volume / volume0),
        "worst_edge_triangle": int(np.argmax(np.max(np.abs(ratios - 1), axis=1))),
        "minimum_area_triangle": int(np.argmin(area)),
        "maximum_area_triangle": int(np.argmax(area)),
        "max_strain_edges_ge_0_5mm": float(np.max(np.abs(ratios[rest_lengths >= 0.0005] - 1))),
        "max_strain_edges_ge_1mm": float(np.max(np.abs(ratios[rest_lengths >= 0.001] - 1))),
        "min_area_ratio_faces_ge_0_1mm2": float(area[a0 >= 2e-7].min()),
        "max_area_ratio_faces_ge_0_1mm2": float(area[a0 >= 2e-7].max()),
    }
    if native_envelope is not None:
        excess = np.abs(ratios - 1) - native_envelope["train_edge_strain_max"]
        area_excess = np.maximum(native_envelope["train_area_min"] - area, area - native_envelope["train_area_max"])
        metrics.update(
            edges_outside_native_train_envelope=int(np.count_nonzero(excess > 1e-6) // 2),
            edges_exceeding_native_train_by_1pct=int(np.count_nonzero(excess > 0.01) // 2),
            max_edge_strain_excess_native_train=float(max(0, excess.max())),
            faces_outside_native_train_area_envelope=int(np.count_nonzero(area_excess > 1e-6)),
            faces_exceeding_native_train_area_by_1pct=int(np.count_nonzero(area_excess > 0.01)),
            max_area_ratio_excess_native_train=float(max(0, area_excess.max())),
        )
    return metrics


@wp.kernel
def audit_substep_vertices(
    rest: wp.array[wp.vec3],
    phi: wp.array2d[wp.vec3],
    joint_q: wp.array[float],
    body_q: wp.array[wp.transform],
    q_start: int,
    body: int,
    cursor: wp.array[int],
    local: wp.array[wp.vec3],
    metrics: wp.array2d[float],
    saved_q: wp.array2d[float],
    saved_pose: wp.array[wp.transform],
):
    vertex = wp.tid()
    slot = cursor[0]
    if slot < metrics.shape[0]:
        p = rest[vertex]
        for mode in range(phi.shape[1]):
            p += phi[vertex, mode] * joint_q[q_start + mode]
        local[vertex] = p
        world = wp.transform_point(body_q[body], p)
        wp.atomic_max(metrics, slot, 0, wp.max(0.0, -world[2]))
        if vertex == 0:
            saved_pose[slot] = body_q[body]
            for mode in range(phi.shape[1]):
                saved_q[slot, mode] = joint_q[q_start + mode]


@wp.kernel
def audit_substep_faces(
    rest: wp.array[wp.vec3],
    local: wp.array[wp.vec3],
    faces: wp.array2d[int],
    cursor: wp.array[int],
    metrics: wp.array2d[float],
):
    face = wp.tid()
    slot = cursor[0]
    if slot < metrics.shape[0]:
        a, b, c = faces[face, 0], faces[face, 1], faces[face, 2]
        r1, r2 = rest[b] - rest[a], rest[c] - rest[a]
        e1, e2 = local[b] - local[a], local[c] - local[a]
        strain = wp.max(wp.abs(wp.length(e1) / wp.length(r1) - 1.0), wp.abs(wp.length(e2) / wp.length(r2) - 1.0))
        strain = wp.max(strain, wp.abs(wp.length(e2 - e1) / wp.length(r2 - r1) - 1.0))
        n0, n = wp.cross(r1, r2), wp.cross(e1, e2)
        area = wp.length(n) / wp.length(n0)
        alignment = wp.dot(n, n0) / wp.max(wp.length(n) * wp.length(n0), 1.0e-30)
        wp.atomic_max(metrics, slot, 1, strain)
        wp.atomic_min(metrics, slot, 2, area)
        wp.atomic_max(metrics, slot, 3, area)
        wp.atomic_min(metrics, slot, 4, alignment)


@wp.kernel
def advance_audit_cursor(cursor: wp.array[int]):
    cursor[0] += 1


class SubstepAudit:
    """Read-only geometry audit at the integrator rate, including hidden spikes."""

    columns = ("floor_penetration_m", "max_edge_strain", "min_area_ratio", "max_area_ratio", "min_normal_alignment")

    def __init__(self, vertices, faces, phi, capacity, *, device):
        self.device = device
        self.rest = wp.array(vertices, dtype=wp.vec3, device=device)
        self.faces = wp.array(faces, dtype=int, device=device)
        self.phi = wp.array(phi, dtype=wp.vec3, device=device)
        self.local = wp.empty_like(self.rest)
        self.cursor = wp.zeros(1, dtype=int, device=device)
        initial = np.zeros((capacity, len(self.columns)), dtype=np.float32)
        initial[:, 2], initial[:, 4] = np.inf, 1.0
        self.metrics = wp.array(initial, dtype=float, device=device)
        self.q = wp.empty((capacity, phi.shape[1]), dtype=float, device=device)
        self.pose = wp.empty(capacity, dtype=wp.transform, device=device)

    def record(self, joint_q, body_q, q_start, body):
        wp.launch(
            audit_substep_vertices,
            len(self.rest),
            inputs=[
                self.rest,
                self.phi,
                joint_q,
                body_q,
                q_start,
                body,
                self.cursor,
                self.local,
                self.metrics,
                self.q,
                self.pose,
            ],
            device=self.device,
        )
        wp.launch(
            audit_substep_faces,
            len(self.faces),
            inputs=[
                self.rest,
                self.local,
                self.faces,
                self.cursor,
                self.metrics,
            ],
            device=self.device,
        )
        wp.launch(advance_audit_cursor, 1, inputs=[self.cursor], device=self.device)

    def arrays(self):
        count = int(self.cursor.numpy()[0])
        if not 0 < count <= len(self.metrics):
            raise AssertionError("Substep audit empty or capacity exceeded")
        return self.metrics.numpy()[:count], self.q.numpy()[:count], self.pose.numpy()[:count]

    def summary(self, dt):
        values, _, _ = self.arrays()
        worst = int(np.argmax(values[:, 1]))
        return {
            "samples": len(values),
            "finite": bool(np.isfinite(values).all()),
            "max_floor_penetration_m": float(values[:, 0].max()),
            "max_edge_strain": float(values[:, 1].max()),
            "min_area_ratio": float(values[:, 2].min()),
            "max_area_ratio": float(values[:, 3].max()),
            "min_normal_alignment": float(values[:, 4].min()),
            "worst_edge_time_s": (worst + 1) * dt,
            "limitations": "All integrator states; intersections, p99 strain and volume remain saved-frame audits.",
        }


@wp.kernel
def count_intersections(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    faces: wp.array2d[int],
    counts: wp.array[int],
    pair_count: wp.array[int],
    pair_keys: wp.array[int],
):
    i = wp.tid()
    a, b, c = faces[i, 0], faces[i, 1], faces[i, 2]
    p, q, r = points[a], points[b], points[c]
    lo = wp.min(wp.min(p, q), r) - wp.vec3(1e-7)
    hi = wp.max(wp.max(p, q), r) + wp.vec3(1e-7)
    query = wp.mesh_query_aabb(mesh, lo, hi)
    j = int(0)
    total = int(0)
    while wp.mesh_query_aabb_next(query, j):
        if j > i:
            d, e, f = faces[j, 0], faces[j, 1], faces[j, 2]
            adjacent = a == d or a == e or a == f or b == d or b == e or b == f or c == d or c == e or c == f
            if not adjacent:
                # Warp's Moller predicate uses a fixed 1e-6 plane epsilon.
                # Millimeters avoid false coplanarity for small SI triangles.
                hit = wp.intersect_tri_tri(
                    wp.vec3d(p) * wp.float64(1000.0),
                    wp.vec3d(q) * wp.float64(1000.0),
                    wp.vec3d(r) * wp.float64(1000.0),
                    wp.vec3d(points[d]) * wp.float64(1000.0),
                    wp.vec3d(points[e]) * wp.float64(1000.0),
                    wp.vec3d(points[f]) * wp.float64(1000.0),
                )
                total += hit
                if hit > 0:
                    slot = wp.atomic_add(pair_count, 0, 1)
                    if slot < pair_keys.shape[0]:
                        pair_keys[slot] = i * faces.shape[0] + j
    counts[i] = total


class IntersectionAudit:
    """BVH broad phase and Warp double-precision triangle intersection tests."""

    def __init__(self, vertices, faces, *, device="cpu"):
        self.device = device
        self.points = wp.array(vertices, dtype=wp.vec3, device=device)
        self.faces = wp.array(faces, dtype=int, device=device)
        self.mesh = wp.Mesh(points=self.points, indices=wp.array(faces.flatten(), dtype=int, device=device))
        self.counts = wp.zeros(len(faces), dtype=int, device=device)
        self.pair_count = wp.zeros(1, dtype=int, device=device)
        self.pair_keys = wp.empty(262144, dtype=int, device=device)

    def count(self, vertices):
        self.points.assign(np.asarray(vertices, dtype=np.float32))
        self.mesh.refit()
        self.pair_count.zero_()
        wp.launch(
            count_intersections,
            len(self.counts),
            inputs=[self.mesh.id, self.points, self.faces, self.counts, self.pair_count, self.pair_keys],
            device=self.device,
        )
        total = int(self.pair_count.numpy()[0])
        if total > len(self.pair_keys):
            raise AssertionError("Self-intersection pair audit overflowed; results cannot be certified")
        return total

    def keys(self):
        """Return all unique intersecting pairs encoded as i * triangle_count + j."""
        count = int(self.pair_count.numpy()[0])
        return np.sort(self.pair_keys.numpy()[:count])

    def pairs(self):
        keys = self.keys()
        return np.column_stack(np.divmod(keys, len(self.faces))).tolist()
