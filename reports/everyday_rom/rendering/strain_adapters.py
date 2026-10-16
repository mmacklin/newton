# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Read material-rest surfaces from existing saved everyday-ROM exports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OBJECTS = (
    "clog",
    "glasses-case",
    "toothbrush",
    "drink-bottle",
    "hand",
    "straw",
    "spatula",
    "banana",
    "headphones",
    "cup-lid",
)
ALIASES = {
    "case": "glasses-case",
    "brush": "toothbrush",
    "drinkbottle": "drink-bottle",
    "cup/lid": "cup-lid",
    "cup": "cup-lid",
}


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@dataclass
class SurfaceReference:
    """Material-rest topology with optional independent saved-state recovery.

    Args:
        mesh: Exact scene mesh name, never a positional or fuzzy name match.
        rest: Material-rest vertices [m], in the rendered vertex order.
        faces: Reference triangles, exactly matching rendered indices.
        provenance: Human-readable source and reference definition.
        local: Optional recovered local positions [m], shape (F,N,3).
        poses: Optional saved Newton body poses, shape (F,7).
    """

    mesh: str
    rest: np.ndarray
    faces: np.ndarray
    provenance: str
    local: np.ndarray | None = None
    poses: np.ndarray | None = None
    midsurface_rest: np.ndarray | None = None
    midsurface_faces: np.ndarray | None = None
    midsurface_vertex_ids: np.ndarray | None = None


class Archive:
    """Cache parsed NPZ inputs and collect provenance hashes."""

    def __init__(self, source: Path):
        self.source = Path(source)
        self.arrays = {}
        self.hashes = {}

    def read(self, name: str) -> dict[str, np.ndarray]:
        path = (self.source / name).resolve()
        if path not in self.arrays:
            self.hashes[str(path)] = digest(path)
            self.arrays[path] = load_arrays(path)
        return self.arrays[path]


def _modal(rest, phi, state, start, body, times):
    if "times" in state and not np.array_equal(state["times"], times):
        raise ValueError("State timestamps do not exactly match the rendered sequence")
    phi = np.asarray(phi)
    if phi.ndim != 3 or phi.shape[0] != len(rest) or phi.shape[2] != 3:
        raise ValueError("Reference basis vertex ordering or shape mismatch")
    start, body = int(start), int(body)
    q, poses = state["joint_q"], state["body_q"]
    if len(q) != len(times) or len(poses) != len(times) or start < 0 or start + phi.shape[1] > q.shape[1]:
        raise ValueError("State frame count or modal slice mismatch")
    if body < 0 or body >= poses.shape[1]:
        raise ValueError("Invalid elastic body")
    local = rest + np.einsum("nmc,fm->fnc", phi, q[:, start : start + phi.shape[1]])
    return local, poses[:, body]


def _mapped_faces(source_faces, ids, rendered_faces):
    """Verify every rendered triangle maps to a source material triangle."""
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid or duplicate material vertex map")
    source_set = {tuple(sorted(face)) for face in source_faces}
    if any(tuple(sorted(face)) not in source_set for face in ids[rendered_faces]):
        raise ValueError("Rendered topology does not match the material-rest mesh")


def explicit_references(manifest: Path, meta: dict) -> tuple[list[SurfaceReference], dict, dict]:
    """Read the worker-facing reference interface, including cup and lid."""
    manifest = Path(manifest).resolve()
    config = json.loads(manifest.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported reference manifest schema")
    archive = Archive(manifest.parent)
    archive.hashes[str(manifest)] = digest(manifest)

    def array(ref):
        return archive.read(ref["file"])[ref["key"]]

    surfaces = []
    for item in config["surfaces"]:
        if bool(item.get("local")) != bool(item.get("poses")):
            raise ValueError("Independent recovery requires both local and poses")
        if not item.get("provenance", "").strip():
            raise ValueError("Material-rest provenance is required")
        surfaces.append(
            SurfaceReference(
                item["mesh"],
                array(item["rest"]),
                array(item["faces"]),
                item["provenance"],
                array(item["local"]) if "local" in item else None,
                array(item["poses"]) if "poses" in item else None,
            )
        )
    uncolored = config.get("uncolored", {})
    validate_coverage(meta, surfaces, uncolored)
    return surfaces, uncolored, archive.hashes


def validate_coverage(meta: dict, surfaces: list[SurfaceReference], uncolored: dict) -> None:
    names = [item["name"] for item in meta["meshes"]]
    selected = [item.mesh for item in surfaces]
    if len(set(names)) != len(names) or len(set(selected)) != len(selected):
        raise ValueError("Duplicate scene mesh or reference name")
    if not selected or set(selected) & set(uncolored) or set(selected) | set(uncolored) != set(names):
        raise ValueError("Every mesh must be classified exactly once as colored or uncolored")
    if not all(isinstance(value, str) and value.strip() for value in uncolored.values()):
        raise ValueError("Uncolored meshes require an explicit reason")


def adapt(source: Path, object_name: str, meta: dict, scene: dict, reference_manifest: Path | None = None):
    """Return verified material references without importing simulation modules."""
    object_name = ALIASES.get(object_name, object_name)
    if object_name not in OBJECTS:
        raise ValueError(f"Unsupported object: {object_name}")
    if reference_manifest is not None:
        config = json.loads(Path(reference_manifest).read_text())
        if ALIASES.get(config.get("object"), config.get("object")) != object_name:
            raise ValueError("Reference manifest object mismatch")
        return explicit_references(reference_manifest, meta)
    archive = Archive(source)
    times, surfaces = scene["times"], []
    meshes = {item["name"]: (i, item) for i, item in enumerate(meta["meshes"])}

    def add(item, rest, faces, phi, state, start, body, provenance):
        if not np.array_equal(faces, scene[item["faces"]]):
            raise ValueError(f"Reference topology mismatch: {item['name']}")
        local, poses = _modal(rest, phi, state, start, body, times)
        surfaces.append(SurfaceReference(item["name"], rest, faces, provenance, local, poses))

    if object_name == "cup-lid":
        basis, state, mapping = (
            archive.read("basis.npz"),
            archive.read("trajectory.npz"),
            archive.read("surface_map.npz"),
        )
        rest, faces, phi = basis["rest_vertices"], basis["faces"], basis["phi"]
        ids = mapping["mesh_0_vertex_ids"]
        if not np.array_equal(ids, np.arange(len(rest))) or phi.shape != (len(rest), 4, 3):
            raise ValueError("Cup/lid contract requires identity surface map and exactly four modes")
        if not np.array_equal(state["times"], times) or state["modal_q"].shape != (len(times), 4):
            raise ValueError("Cup/lid state samples do not match the scene")
        # The owning exporter builds the lid first (body 0), then the straw.
        body = int(state.get("lid_body", 0))
        if state["body_q"].shape != (len(times), 2, 7) or body not in (0, 1):
            raise ValueError("Cup/lid body layout differs from the agreed contract")
        i, item = meshes["contact-driven-four-mode-lid"]
        if i != 0 or not np.array_equal(faces, scene[item["faces"]]):
            raise ValueError("Cup/lid skin topology mismatch")
        local = rest + np.einsum("nmc,fm->fnc", phi, state["modal_q"])
        mid = basis["mid"]
        if len(rest) != 2 * len(mid) or not np.allclose(rest[: len(mid), :2], mid[:, :2], rtol=0, atol=1e-8):
            raise ValueError("Cup/lid midsurface ordering does not match the complete skin")
        if not np.array_equal(phi[: len(mid)], phi[len(mid) :]):
            raise ValueError("Cup/lid top and bottom do not share the documented midsurface field")
        surfaces.append(
            SurfaceReference(
                item["name"],
                rest,
                faces,
                "basis.npz complete lid skin; shared linear midsurface displacement, side-wall shear is not calibrated shell strain",
                local,
                state["body_q"][:, body],
                mid,
                basis["mid_faces"],
                np.arange(len(mid), dtype=np.int32),
            )
        )
    elif object_name in ("straw", "spatula", "clog", "drink-bottle"):
        basis = archive.read("basis.npz")
        state = archive.read("trajectory.npz" if object_name == "drink-bottle" else "state.npz")
        rest, phi, faces = basis["rest_vertices"], basis["phi"], basis["faces"]
        start = state["q_start" if object_name == "drink-bottle" else "modal_q_start"]
        body = state["bottle_body" if object_name == "drink-bottle" else "elastic_body"]
        if object_name in ("straw", "spatula"):
            _, item = meshes["elastic_" + object_name]
            add(item, rest, faces, phi, state, start, body, "basis.npz: undeformed runtime elastic surface")
        else:
            mapping = archive.read("surface_map.npz") if object_name == "drink-bottle" else None
            for i, item in enumerate(meta["meshes"]):
                if mapping is not None:
                    key = f"mesh_{i}_source_vertex"
                    if key not in mapping:
                        continue
                    ids = mapping[key]
                else:
                    if "basis_vertex_start" not in item:
                        continue
                    offset, count = item["basis_vertex_start"], item["basis_vertex_count"]
                    ids = np.arange(offset, offset + count, dtype=np.int32)
                if np.any(ids < 0) or np.any(ids >= len(rest)):
                    raise ValueError("Material vertex map out of range")
                triangles = scene[item["faces"]]
                _mapped_faces(faces, ids, triangles)
                add(
                    item,
                    rest[ids],
                    triangles,
                    phi[ids],
                    state,
                    start,
                    body,
                    "basis.npz material rest, mapped by saved surface_map or basis_vertex_start/count",
                )
    elif object_name == "headphones":
        state = archive.read("original_states.npz")
        i, item = meshes["elastic_headband"]
        add(
            item,
            state["runtime_vertex_local"],
            state[f"faces_{i}"],
            state["runtime_vertex_phi"],
            state,
            state["q_start"],
            state["band_body"],
            "original_states.npz: runtime_vertex_local (accepted headphone reference)",
        )
    elif object_name == "hand":
        state = archive.read("solved_states.npz")
        _, item = meshes["native_hand"]
        add(
            item,
            state["rest"],
            state["faces"],
            state["vertex_phi"],
            state,
            state["q_start"],
            state["hand_body"],
            "solved_states.npz: undeformed registered native-hand mesh",
        )
    elif object_name == "banana" and (Path(source) / "registered_model.npz").exists():
        model, state = archive.read("registered_model.npz"), archive.read("state.npz")
        max_modes = int(model["elastic_max_mode_count"])
        phi = model["elastic_shape_vertex_phi"].reshape(-1, max_modes, 3)
        for strip, body in enumerate(state["peel_bodies"]):
            candidates = np.flatnonzero(model["elastic_shape_body"] == body)
            if len(candidates) != 1:
                raise ValueError("Expected one registered peel surface per body")
            shape = int(candidates[0])
            offset, count = (
                int(model["elastic_shape_vertex_start"][shape]),
                int(model["elastic_shape_vertex_count"][shape]),
            )
            first, length = (
                int(model["elastic_shape_index_start"][shape]),
                int(model["elastic_shape_index_count"][shape]),
            )
            # Newton elastic_shape_indices are local to each elastic shape.
            faces = model["elastic_shape_indices"][first : first + length].reshape(-1, 3)
            elastic = np.flatnonzero(model["elastic_body"] == body)
            if len(elastic) != 1:
                raise ValueError("Ambiguous peel mode ownership")
            modes = int(model["elastic_mode_count"][elastic[0]])
            _, item = meshes[f"peel_{strip}"]
            add(
                item,
                model["elastic_shape_vertex_local"][offset : offset + count],
                faces,
                phi[offset : offset + count, :modes],
                state,
                state["q_starts"][strip],
                body,
                "registered_model.npz: actual runtime peel rest mesh (not stale basis.npz)",
            )
    else:
        basis, state = archive.read("basis.npz"), archive.read("state.npz")
        for i, item in enumerate(meta["meshes"]):
            key = f"phi_{i}"
            if key not in basis or basis[key].shape[1] == 0:
                continue
            if object_name == "toothbrush":
                rest, faces = basis[f"rest_vertices_{i}"], basis[f"faces_{i}"]
                start, body = item["q_start"], item["body_index"]
            else:
                rest, faces = basis[f"rest_{i}"], scene[item["faces"]]
                start, body = basis[f"q_start_{i}"], basis[f"body_{i}"]
            add(
                item,
                rest,
                faces,
                basis[key],
                state,
                start,
                body,
                f"basis.npz: material-rest mesh {i}; all saved-state frames independently recovered",
            )
    selected = {surface.mesh for surface in surfaces}
    uncolored = {
        name: "Rigid or non-elastic scene mesh; retain normal material" for name in meshes if name not in selected
    }
    validate_coverage(meta, surfaces, uncolored)
    return surfaces, uncolored, archive.hashes
