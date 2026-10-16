# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU-only mathematical, adapter, export, and renderer binding checks."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__:
    from .render_strain_replay import annotate_frame, camera_path, create_stage
    from .strain_adapters import OBJECTS, adapt, digest, load_arrays
    from .strain_replay import prepare, validate_scene, verify, verify_combined, write_json
    from .surface_strain import components, field, jet, rest_metric, transform_points, vertex_average
else:
    from render_strain_replay import annotate_frame, camera_path, create_stage
    from strain_adapters import OBJECTS, adapt, digest, load_arrays
    from strain_replay import prepare, validate_scene, verify, verify_combined, write_json
    from surface_strain import components, field, jet, rest_metric, transform_points, vertex_average


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        self.faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    def test_large_rigid_rotations_and_translations_are_strain_free(self):
        for angle in (0, 1.5, np.pi, 5.9):
            pose = np.array([4.0, -3.0, 2.0, 0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)])
            principal, magnitude = field(self.rest, self.faces, transform_points(self.rest, pose))
            np.testing.assert_allclose(principal, 0, atol=1e-14)
            np.testing.assert_allclose(magnitude, 0, atol=1e-14)

    def test_uniform_stretch_is_engineering_not_green_strain(self):
        principal, vertices = field(self.rest, self.faces, 1.2 * self.rest)
        np.testing.assert_allclose(principal, 0.2, atol=1e-14)
        np.testing.assert_allclose(vertices, 0.2, atol=1e-14)

    def test_tension_and_compression_keep_signs(self):
        principal, vertices = field(self.rest, self.faces, self.rest * [1.1, 0.7, 1.0])
        np.testing.assert_allclose(principal, [[-0.3, 0.1]] * 2, atol=1e-14)
        np.testing.assert_allclose(vertices, 0.3, atol=1e-14)

    def test_shear_matches_independent_svd(self):
        gradient = np.array([[1, 0.4], [0, 1], [0, 0.0]])
        principal, _ = field(self.rest, self.faces, self.rest[:, :2] @ gradient.T)
        np.testing.assert_allclose(
            principal, np.tile(np.sort(np.linalg.svd(gradient, compute_uv=False)) - 1, (2, 1)), atol=1e-14
        )

    def test_current_triangle_collapse_is_minus_one_not_hidden(self):
        principal, _ = field(self.rest, self.faces, self.rest * [1, 0, 1])
        np.testing.assert_allclose(principal[:, 0], -1)

    def test_reference_degeneracy_and_unreferenced_vertices_rejected(self):
        with self.assertRaisesRegex(ValueError, "degenerate"):
            rest_metric(self.rest * [1, 0, 1], self.faces)
        with self.assertRaisesRegex(ValueError, "Unreferenced"):
            vertex_average(np.ones(2), self.faces, np.ones(2), 5)

    def test_rest_area_weighting(self):
        values = vertex_average(np.array([0.02, 0.04]), self.faces, np.array([1, 3]), 4)
        np.testing.assert_allclose(values, [0.02, 0.035, 0.035, 0.04])

    def test_disconnected_coincident_triangles_never_share_colors(self):
        rest = np.concatenate((self.rest[:3], self.rest[:3]))
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        current = rest.copy()
        current[3:] *= 1.2
        _, colors = field(rest, faces, current)
        np.testing.assert_allclose(colors, [0, 0, 0, 0.2, 0.2, 0.2], atol=1e-14)
        v, f = components(faces, 6)
        np.testing.assert_array_equal(v, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(f, [0, 1])

    def test_classical_jet_endpoints_and_clipping(self):
        np.testing.assert_array_equal(
            jet([-1, 0, 0.5, 1, 2]), [[0, 0, 0.5], [0, 0, 0.5], [0.5, 1, 0.5], [0.5, 0, 0], [0.5, 0, 0]]
        )
        with self.assertRaises(ValueError):
            jet([np.nan])


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.output = self.root / "source", self.root / "strain"
        self.source.mkdir()
        self.rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)
        # Frame zero is already strained, so an accidental first-frame reference fails.
        points = np.stack([self.rest * 1.01, self.rest * 1.05, self.rest * 1.1])
        self.arrays = {
            "times": np.arange(3) / 60,
            "p": points,
            "f": self.faces,
            "rigid": self.rest + np.array([0, 0, 2]),
            "non_geometry_evidence": np.arange(7, dtype=np.int64),
        }
        self.meta = {
            "slug": "cup-lid",
            "fps": 60,
            "camera": {
                "position": [2, -3, 2],
                "target": [0, 0, 0],
                "fov": 38,
                "track_meshes": ["skin"],
                "reference_frame": 1,
            },
            "meshes": [
                {"name": "skin", "vertices": "p", "faces": "f", "color": [0.2, 0.6, 0.4], "roughness": 0.3},
                {"name": "fixture", "vertices": "rigid", "faces": "f", "color": [0.5, 0.5, 0.5]},
            ],
        }
        np.savez_compressed(self.source / "scene.npz", **self.arrays)
        np.savez_compressed(self.source / "reference.npz", rest=self.rest, faces=self.faces)
        write_json(self.source / "scene.json", self.meta)
        write_json(self.source / "metrics.json", {"test_passed": True, "negative_control_passed": True})
        self.contract = {
            "schema_version": 1,
            "object": "cup-lid",
            "surfaces": [
                {
                    "mesh": "skin",
                    "rest": {"file": "reference.npz", "key": "rest"},
                    "faces": {"file": "reference.npz", "key": "faces"},
                    "provenance": "True material rest from runtime",
                }
            ],
            "uncolored": {"fixture": "Rigid fixture"},
        }
        self.save_contract()

    def save_contract(self):
        self.reference = self.source / "references.json"
        write_json(self.reference, self.contract)

    def run_export(self, combined=False):
        return prepare(self.source, self.output, "cup-lid", 0.06, self.reference, combined)

    def test_export_retains_every_original_array_and_full_unclipped_field(self):
        before = digest(self.source / "scene.npz")
        manifest = self.run_export()
        arrays, audit = load_arrays(self.output / "scene.npz"), load_arrays(self.output / "strain.npz")
        for name, original in self.arrays.items():
            np.testing.assert_array_equal(arrays[name], original)
        self.assertEqual(digest(self.source / "scene.npz"), before)
        self.assertGreater(audit["mesh_0_principal"][0].max(), 0.009)
        self.assertGreater(audit["mesh_0_principal"].max(), 0.09)
        self.assertEqual(manifest["surfaces"][0]["clipped_vertex_samples"], 3)
        self.assertTrue(verify(self.output)["all_fields_recomputed"])

    def test_combined_exact_geometry_time_material_and_camera_replay(self):
        self.run_export(combined=True)
        self.assertTrue(verify_combined(self.output)["identical_geometry_halves"])
        meta = json.loads((self.output / "replay/scene.json").read_text())
        arrays = load_arrays(self.output / "replay/scene.npz")
        positions, targets = camera_path(meta, arrays)
        np.testing.assert_array_equal(positions[:3], positions[3:])
        np.testing.assert_array_equal(targets[:3], targets[3:])
        np.testing.assert_array_equal(
            arrays["strain_rgb_0"][:3], np.broadcast_to(np.array([0.2, 0.6, 0.4], np.float32), (3, 3, 3))
        )

    def test_existing_vertex_colors_preserved_in_normal_half(self):
        self.arrays["original_rgb"] = np.arange(27, dtype=np.float32).reshape(3, 3, 3) / 27
        self.meta["meshes"][0]["vertex_colors"] = "original_rgb"
        np.savez_compressed(self.source / "scene.npz", **self.arrays)
        write_json(self.source / "scene.json", self.meta)
        self.run_export(combined=True)
        arrays = load_arrays(self.output / "replay/scene.npz")
        np.testing.assert_array_equal(arrays["strain_rgb_0"][:3], self.arrays["original_rgb"])

    def test_missing_mesh_classification_rejected(self):
        self.contract["uncolored"] = {}
        self.save_contract()
        with self.assertRaisesRegex(ValueError, "classified"):
            self.run_export()

    def test_duplicate_reference_rejected(self):
        self.contract["surfaces"] *= 2
        self.save_contract()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_export()

    def test_stale_topology_rejected(self):
        np.savez_compressed(self.source / "reference.npz", rest=self.rest, faces=self.faces[:, ::-1])
        with self.assertRaisesRegex(ValueError, "topology"):
            self.run_export()

    def test_exact_state_recovery_rejects_reordered_vertices(self):
        np.savez_compressed(
            self.source / "reference.npz",
            rest=self.rest,
            faces=self.faces,
            local=self.arrays["p"][:, ::-1],
            poses=np.tile([0, 0, 0, 0, 0, 0, 1.0], (3, 1)),
        )
        self.contract["surfaces"][0].update(
            local={"file": "reference.npz", "key": "local"}, poses={"file": "reference.npz", "key": "poses"}
        )
        self.save_contract()
        with self.assertRaisesRegex(ValueError, "recovery mismatch"):
            self.run_export()

    def test_stale_color_frame_count_rejected(self):
        meta, arrays = copy.deepcopy(self.meta), dict(self.arrays)
        meta["meshes"][0]["vertex_colors"] = "rgb"
        arrays["rgb"] = np.zeros((2, 3, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "every frame"):
            validate_scene(meta, arrays)
        arrays["rgb"] = np.full((3, 3, 3), np.nan)
        with self.assertRaisesRegex(ValueError, "Invalid RGB"):
            validate_scene(meta, arrays)

    def test_reject_nonpositive_scale(self):
        for scale in (0, -1, np.nan):
            with self.assertRaisesRegex(ValueError, "scale"):
                prepare(self.source, self.output, "cup-lid", scale, self.reference)

    def test_source_hash_mismatch_blocks_reuse(self):
        self.run_export()
        write_json(self.source / "metrics.json", {"test_passed": False})
        with self.assertRaisesRegex(ValueError, "Source hash"):
            verify(self.output)
        self.assertTrue(verify(self.output, check_sources=False)["verified"])

    def test_corrupt_output_blocks_verification(self):
        self.run_export()
        arrays = load_arrays(self.output / "scene.npz")
        arrays["p"][0, 0, 0] += 0.01
        np.savez_compressed(self.output / "scene.npz", **arrays)
        with self.assertRaisesRegex(ValueError, "Output hash"):
            verify(self.output)

    def test_existing_outputs_and_source_subdirectories_rejected(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):
            self.run_export()
        with self.assertRaises(ValueError):
            prepare(self.source, self.source / "nested", "cup-lid", 0.06, self.reference)

    def test_all_ten_object_names_accept_explicit_worker_contract(self):
        self.assertEqual(len(OBJECTS), 10)
        for name in OBJECTS:
            self.contract["object"] = name
            self.save_contract()
            refs, uncolored, hashes = adapt(self.source, name, self.meta, self.arrays, self.reference)
            self.assertEqual([x.mesh for x in refs], ["skin"])
            self.assertEqual(uncolored, {"fixture": "Rigid fixture"})
            self.assertIn(str(self.reference), hashes)

    def test_usd_geometry_color_material_binding_and_all_frames(self):
        try:
            from pxr import UsdGeom, UsdShade
        except ImportError:
            self.skipTest("USD optional renderer dependency unavailable")
        self.run_export(combined=True)
        target = self.output / "replay"
        meta = json.loads((target / "scene.json").read_text())
        arrays = load_arrays(target / "scene.npz")
        args = SimpleNamespace(run=target, frame=None, width=320, height=240, samples=1)
        create_stage(args, meta, arrays)
        from pxr import Usd

        stage = Usd.Stage.Open(str(target / "scene.usdc"))
        obj = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Object_0"))
        colors = obj.GetDisplayColorPrimvar()
        self.assertEqual(colors.GetInterpolation(), "vertex")
        for frame in range(6):
            np.testing.assert_array_equal(np.asarray(obj.GetPointsAttr().Get(frame)), arrays["p"][frame])
            np.testing.assert_array_equal(np.asarray(colors.Get(frame)), arrays["strain_rgb_0"][frame])
        shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Materials/Material_0/Surface"))
        self.assertTrue(shader.GetInput("diffuseColor").HasConnectedSource())
        fixture = UsdShade.Shader(stage.GetPrimAtPath("/World/Materials/Material_1/Surface"))
        self.assertFalse(fixture.GetInput("diffuseColor").HasConnectedSource())

    def test_automatic_adapters_recover_original_modal_states(self):
        phi = self.rest[:, None, :].copy()
        q = np.array([[0.01], [0.05], [0.1]], dtype=np.float32)
        poses = np.tile([0, 0, 0, 0, 0, 0, 1.0], (3, 1, 1)).astype(np.float32)
        state = {
            "times": self.arrays["times"],
            "joint_q": q,
            "body_q": poses,
            "modal_q_start": np.array(0),
            "elastic_body": np.array(0),
            "q_start": np.array(0),
            "bottle_body": np.array(0),
        }
        basis = {
            "rest_vertices": self.rest,
            "faces": self.faces,
            "phi": phi,
            "rest_0": self.rest,
            "phi_0": phi,
            "body_0": np.array(0),
            "q_start_0": np.array(0),
            "rest_vertices_0": self.rest,
            "faces_0": self.faces,
        }
        np.savez_compressed(self.source / "basis.npz", **basis)
        np.savez_compressed(self.source / "state.npz", **state)
        np.savez_compressed(self.source / "trajectory.npz", **state)
        np.savez_compressed(self.source / "surface_map.npz", mesh_0_source_vertex=np.arange(3, dtype=np.int32))
        np.savez_compressed(
            self.source / "solved_states.npz",
            **state,
            rest=self.rest,
            faces=self.faces,
            vertex_phi=phi,
            hand_body=np.array(0),
        )
        np.savez_compressed(
            self.source / "original_states.npz",
            **state,
            runtime_vertex_local=self.rest,
            runtime_vertex_phi=phi,
            faces_0=self.faces,
            band_body=np.array(0),
        )
        for name in (
            "straw",
            "spatula",
            "clog",
            "drink-bottle",
            "glasses-case",
            "toothbrush",
            "hand",
            "headphones",
            "banana",
        ):
            with self.subTest(object=name):
                meta = copy.deepcopy(self.meta)
                item = meta["meshes"][0]
                item.update(q_start=0, body_index=0, basis_vertex_start=0, basis_vertex_count=3)
                if name in ("straw", "spatula"):
                    item["name"] = "elastic_" + name
                elif name == "hand":
                    item["name"] = "native_hand"
                elif name == "headphones":
                    item["name"] = "elastic_headband"
                refs, _, _ = adapt(self.source, name, meta, self.arrays)
                self.assertEqual(len(refs), 1)
                np.testing.assert_allclose(refs[0].local, self.arrays["p"], atol=1e-7)

    def test_mapped_adapter_rejects_wrong_triangle_correspondence(self):
        np.savez_compressed(
            self.source / "basis.npz", rest_vertices=self.rest, faces=np.array([[0, 1, 1]]), phi=self.rest[:, None, :]
        )
        np.savez_compressed(self.source / "trajectory.npz", times=self.arrays["times"], q_start=0, bottle_body=0)
        np.savez_compressed(self.source / "surface_map.npz", mesh_0_source_vertex=np.arange(3, dtype=np.int32))
        with self.assertRaisesRegex(ValueError, "topology"):
            adapt(self.source, "drink-bottle", self.meta, self.arrays)

    def test_cup_lid_skin_and_midsurface_are_separate_auditable_fields(self):
        mid = self.rest.copy()
        rest = np.concatenate((mid + np.array([0, 0, 0.01]), mid - np.array([0, 0, 0.01]))).astype(np.float32)
        faces = np.array(
            [[0, 1, 2], [5, 4, 3], [0, 3, 1], [1, 3, 4], [1, 4, 2], [2, 4, 5], [2, 5, 0], [0, 5, 3]], dtype=np.int32
        )
        phi = np.zeros((6, 4, 3), dtype=np.float32)
        phi[:, 0] = np.concatenate((mid, mid))
        q = np.zeros((3, 4), dtype=np.float32)
        q[:, 0] = [0.01, 0.05, 0.1]
        local = rest + np.einsum("nmc,fm->fnc", phi, q)
        poses = np.tile([0, 0, 0, 0, 0, 0, 1.0], (3, 2, 1)).astype(np.float32)
        np.savez_compressed(
            self.source / "basis.npz", rest_vertices=rest, faces=faces, phi=phi, mid=mid, mid_faces=self.faces
        )
        np.savez_compressed(self.source / "trajectory.npz", times=self.arrays["times"], modal_q=q, body_q=poses)
        np.savez_compressed(self.source / "surface_map.npz", mesh_0_vertex_ids=np.arange(6, dtype=np.int32))
        arrays = dict(self.arrays, p=local, skin_faces=faces)
        meta = copy.deepcopy(self.meta)
        meta["meshes"][0].update(name="contact-driven-four-mode-lid", faces="skin_faces")
        meta["camera"].pop("track_meshes")
        write_json(self.source / "scene.json", meta)
        np.savez_compressed(self.source / "scene.npz", **arrays)
        manifest = prepare(self.source, self.output, "cup-lid", 0.12)
        self.assertEqual(manifest["object"], "cup-lid")
        self.assertEqual(manifest["surfaces"][0]["triangles"], 8)
        self.assertEqual(manifest["surfaces"][0]["midsurface"]["triangles"], 1)
        self.assertTrue(verify(self.output)["verified"])
        audit = load_arrays(self.output / "strain.npz")
        self.assertEqual(audit["mesh_0_principal"].shape, (3, 8, 2))
        self.assertEqual(audit["mesh_0_midsurface_principal"].shape, (3, 1, 2))

    def test_renderer_annotation_switches_phase_at_exact_boundary(self):
        self.run_export(combined=True)
        target = self.output / "replay"
        meta = json.loads((target / "scene.json").read_text())
        arrays = load_arrays(target / "scene.npz")
        image = np.full((240, 320, 3), 70, dtype=np.uint8)
        normal = annotate_frame(image, target, meta, arrays, 2)
        strain = annotate_frame(image, target, meta, arrays, 3)
        np.testing.assert_array_equal(normal[48:100], image[48:100])
        self.assertFalse(np.array_equal(strain[48:100], image[48:100]))


if __name__ == "__main__":
    unittest.main()
