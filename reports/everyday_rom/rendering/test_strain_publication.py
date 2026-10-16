# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Portable replay verification and immutable sealed-source staging tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

if __package__:
    from . import strain_publication
    from . import test_strain_replay as fixtures
    from .strain_adapters import digest, load_arrays
    from .strain_publication import package_replay, stage_normal, verify_normal, verify_package
    from .strain_replay import write_json
else:
    import strain_publication
    import test_strain_replay as fixtures
    from strain_adapters import digest, load_arrays
    from strain_publication import package_replay, stage_normal, verify_normal, verify_package
    from strain_replay import write_json


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExportTests("test_export_retains_every_original_array_and_full_unclipped_field")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.run_export()
        self.plan = f.root / "plan"
        self.plan.mkdir()
        self.package = f.root / "package"
        checked = {}
        for source in (f.source, f.output):
            (source / "rtx.mp4").write_bytes(b"source-video")
            (source / "scene.usdc").write_bytes(b"source-usd")
            audit = {
                "frames": 3,
                "fps": 60,
                "max_renderer_vertex_error_m": 0,
                "max_renderer_color_error": 0,
                "inputs_sha256": {name: digest(source / name) for name in ("scene.json", "scene.npz", "metrics.json")},
                "video_sha256": digest(source / "rtx.mp4"),
                "usd_sha256": digest(source / "scene.usdc"),
            }
            write_json(source / "rtx.json", audit)
            checked.update({str(source / name): digest(source / name) for name in audit["inputs_sha256"]})
        (self.plan / "rtx.mp4").write_bytes(b"assembled-video")
        manifest = json.loads((f.output / "manifest.json").read_text())
        plan = {
            "normal_video": str(f.source / "rtx.mp4"),
            "strain_video": str(f.output / "rtx.mp4"),
            "object": "cup-lid",
            "inputs_sha256": checked,
            "schedule": {"frames_per_phase": 3, "fps": 60},
            "camera": f.meta["camera"],
            "scale": manifest["scale"],
        }
        write_json(self.plan / "video-plan.json", plan)
        write_json(
            self.plan / "concat-audit.json",
            {
                "verified": True,
                "plan_sha256": digest(self.plan / "video-plan.json"),
                "video_sha256": digest(self.plan / "rtx.mp4"),
                "probe": {"frames": 6, "fps": 60, "duration_s": 0.1, "uniform_pts": True},
                "normal_decoded_frames_identical": 3,
                "strain_decoded_frames_identical": 3,
            },
        )
        write_json(
            self.plan / "decoded-frames.json", {"normal_rgb24_sha256": ["a"] * 3, "strain_rgb24_sha256": ["b"] * 3}
        )
        with patch.object(strain_publication, "publication_entry", return_value={"performance_provenance": "separate"}):
            package_replay(self.plan, self.package)

    def test_portable_exact_arrays_and_quantitative_fields(self):
        self.assertTrue(verify_package(self.package)["all_fields_recomputed"])
        audit = json.loads((self.package / "replay/rtx.json").read_text())
        self.assertNotIn("usd_sha256", audit)
        self.assertEqual(len(audit["source_renders"]), 2)
        self.assertEqual(digest(self.fixture.source / "scene.npz"), digest(self.package / "scene.npz"))
        with patch.object(strain_publication, "decoded_frame_hashes", return_value=["a"] * 3 + ["b"] * 3):
            self.assertTrue(verify_package(self.package, True)["full_decode"])

    def test_stale_input_and_final_video_rejected(self):
        for name in ("scene.npz", "replay/rtx.mp4"):
            path = self.package / name
            original = path.read_bytes()
            path.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                verify_package(self.package)
            path.write_bytes(original)

    def rewrite_audit(self, name):
        audit_path = self.package / "replay/rtx.json"
        audit = json.loads(audit_path.read_text())
        audit["inputs_sha256"][name] = digest(self.package / "replay" / name)
        write_json(audit_path, audit)

    def test_wrong_schedule_rejected_even_after_hash_update(self):
        path = self.package / "replay/scene.npz"
        arrays = load_arrays(path)
        arrays["source_frame"][4] = 2
        np.savez_compressed(path, **arrays)
        self.rewrite_audit("scene.npz")
        with self.assertRaises(AssertionError):
            verify_package(self.package)

    def test_missing_phase_and_bad_decode_proof_rejected(self):
        path = self.package / "replay/rtx.json"
        audit = json.loads(path.read_text())
        audit["source_renders"] = audit["source_renders"][:1]
        write_json(path, audit)
        with self.assertRaisesRegex(ValueError, "in order"):
            verify_package(self.package)

    def test_changed_source_renderer_audit_rejected(self):
        path = self.package / "rtx.json"
        audit = json.loads(path.read_text())
        audit["frames"] = 4
        write_json(path, audit)
        self.rewrite_audit("../rtx.json")
        with self.assertRaisesRegex(ValueError, "sampling"):
            verify_package(self.package)

    def test_full_decode_mismatch_rejected(self):
        with patch.object(strain_publication, "decoded_frame_hashes", return_value=["wrong"] * 6):
            with self.assertRaisesRegex(ValueError, "appearance"):
                verify_package(self.package, True)

    def test_color_corruption_rejected_after_outer_hashes_updated(self):
        source = self.package / "strain"
        arrays = load_arrays(source / "scene.npz")
        arrays["strain_rgb_0"][1, 0] = [1, 0, 0]
        np.savez_compressed(source / "scene.npz", **arrays)
        manifest = json.loads((source / "manifest.json").read_text())
        manifest["outputs_sha256"]["scene.npz"] = digest(source / "scene.npz")
        write_json(source / "manifest.json", manifest)
        render = json.loads((source / "rtx.json").read_text())
        render["inputs_sha256"]["scene.npz"] = digest(source / "scene.npz")
        write_json(source / "rtx.json", render)
        for name in ("scene.npz", "manifest.json", "rtx.json"):
            self.rewrite_audit("../strain/" + name)
        path = self.package / "replay/rtx.json"
        audit = json.loads(path.read_text())
        audit["source_renders"][1]["audit_sha256"] = digest(source / "rtx.json")
        write_json(path, audit)
        with self.assertRaises(AssertionError):
            verify_package(self.package)

    def test_missing_evidence_hash_rejected(self):
        path = self.package / "replay/rtx.json"
        audit = json.loads(path.read_text())
        del audit["inputs_sha256"]["../strain/strain.npz"]
        write_json(path, audit)
        with self.assertRaisesRegex(ValueError, "omits required"):
            verify_package(self.package)


class SealTests(unittest.TestCase):
    def test_copy_preserves_complete_seal_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "sealed"
            source.mkdir()
            (source / "proof").write_bytes(b"immutable")
            write_json(source / "manifest.json", {"proof": digest(source / "proof")})
            result = stage_normal(source, root / "copy")
            self.assertEqual(result["verified_seals"]["manifest.json"]["verified_files"], 1)
            self.assertEqual(verify_normal(root / "copy")["copied_files"], 2)
            (root / "copy/proof").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_normal(root / "copy")
            (source / "proof").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Sealed source hash"):
                stage_normal(source, root / "other")


if __name__ == "__main__":
    unittest.main()
