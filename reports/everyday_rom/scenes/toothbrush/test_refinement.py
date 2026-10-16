# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Synthetic fresh-export tests, not a simulation or physical qualification."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from refine_render import digest, fresh_output, refine, transform_points, validate_physical, write_json
from subdivision import subdivision_embedding

HERE = Path(__file__).resolve().parent


class RefinementTests(unittest.TestCase):
    def fixture(self, directory):
        directory.mkdir()
        times = np.arange(1, 451) / 60
        poses = np.zeros((450, 2, 7), np.float32)
        poses[..., 6] = 1
        poses[:, 0, 0] = 0.004 * np.sin(times)
        poses[:, 1, 2] = 0.005 * np.cos(times)
        poses[:, :, 5] = np.float32(np.sin(0.12))
        poses[:, :, 6] = np.float32(np.cos(0.12))
        arrays, basis, meshes = {"times": times}, {}, []
        with np.load(HERE / "frozen/basis.npz", allow_pickle=False) as reference:
            for i, name in enumerate(
                ("ivory_handle", "synthetic_deforming_tuft", "synthetic_seam", "ivory_plastic_head")
            ):
                if i in (0, 3):
                    points, faces = reference[f"rest_vertices_{i}"], reference[f"faces_{i}"]
                else:
                    points = np.array([[0, 0, 0], [0.001, 0, 0], [0, 0.001, 0]], np.float32)
                    faces = np.array([[0, 1, 2]], np.int32)
                body = int(i == 3)
                vertices = np.stack([transform_points(points, pose[body]) for pose in poses])
                if i == 1:
                    vertices[:, 2, 0] += np.sin(times).astype(np.float32) * 0.0002
                basis[f"rest_vertices_{i}"] = points
                basis[f"faces_{i}"] = faces
                if i == 3:
                    basis[f"phi_{i}"] = np.zeros((len(points), 40, 3), np.float32)
                arrays[f"vertices_{i}"] = vertices
                arrays[f"faces_{i}"] = faces
                meshes.append({"name": name, "vertices": f"vertices_{i}", "faces": f"faces_{i}", "body_index": body})
        np.savez_compressed(directory / "basis.npz", **basis)
        np.savez_compressed(directory / "state.npz", times=times, body_q=poses, joint_q=np.zeros((450, 40), np.float32))
        np.savez_compressed(directory / "scene.npz", **arrays)
        write_json(directory / "scene.json", {"meshes": meshes, "fps": 60})
        write_json(
            directory / "metrics.json",
            {
                "synthetic_fixture": True,
                "test_passed": True,
                "negative_control_passed": True,
                "source_sha256": {
                    name: digest(HERE / name)
                    for name in ("toothbrush.py", "brush_glove_geometry.py", "field_coverage.py", "capacity_audit.py")
                },
            },
        )

    def test_full_synthetic_fresh_export_preserves_embedding_and_inputs(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            physical, output = root / "physical", root / "smooth"
            self.fixture(physical)
            before = {p.name: digest(p) for p in physical.iterdir()}
            audit = refine(physical, output)
            self.assertEqual(before, {p.name: digest(p) for p in physical.iterdir()})
            self.assertEqual(digest(physical / "state.npz"), digest(output / "state.npz"))
            with (
                np.load(output / "scene.npz") as smooth,
                np.load(physical / "scene.npz") as coarse,
                np.load(output / "render_embedding.npz") as embedding,
                np.load(output / "basis.npz") as basis,
            ):
                self.assertEqual(len(smooth["faces_0"]), 14080)
                self.assertEqual(len(smooth["faces_3"]), 8960)
                for i in (1, 2):
                    np.testing.assert_array_equal(smooth[f"vertices_{i}"], coarse[f"vertices_{i}"])
                    np.testing.assert_array_equal(smooth[f"faces_{i}"], coarse[f"faces_{i}"])
                for i in (0, 3):
                    weights = embedding[f"weights_{i}"]
                    self.assertGreaterEqual(float(weights.min()), 0)
                    np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-14)
                np.testing.assert_array_equal(basis["phi_3"], 0)
            self.assertTrue(
                all(row["frames"] == 450 and row["embedding_max_error_m"] < 5e-8 for row in audit["all_frame_checks"])
            )
            with self.assertRaisesRegex(ValueError, "double refinement"):
                validate_physical(output)
            with self.assertRaises(ValueError):
                fresh_output(output, (physical,))
            meta = json.loads((physical / "scene.json").read_text())
            meta["render_refinement"] = {"applications": 1}
            write_json(physical / "scene.json", meta)
            with self.assertRaisesRegex(ValueError, "double refinement"):
                validate_physical(physical)

    def test_rejects_failed_gates_foreign_source_and_input_overlap(self):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            root = Path(temp)
            write_json(root / "scene.json", {})
            write_json(root / "metrics.json", {"test_passed": False, "negative_control_passed": False})
            with self.assertRaisesRegex(ValueError, "original simulation gates"):
                validate_physical(root)
            write_json(root / "metrics.json", {"test_passed": True, "negative_control_passed": True})
            with self.assertRaisesRegex(ValueError, "source differs"):
                validate_physical(root)
            with self.assertRaisesRegex(ValueError, "overlap"):
                fresh_output(root / "render", (root,))

    def test_nonmanifold_mesh_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "closed two-manifold"):
            subdivision_embedding(np.eye(3), np.array([[0, 1, 2]]), np.zeros(3, bool))


if __name__ == "__main__":
    unittest.main()
