# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for current-media and validation labels."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_report import SLUGS, compact_metrics, detail_caption, hand_pod_study, render_current, section


class ReportTests(unittest.TestCase):
    def test_replay_does_not_add_external_rainbow_legend(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "example"
            (directory / "replay").mkdir(parents=True)
            (directory / "replay/scene.json").write_text(
                json.dumps({"strain_scale": {"min": 0, "max": 0.03, "colormap": "classicaljet"}})
            )
            with (
                patch("build_report.render_current", side_effect=lambda path, audit: path == directory),
                patch("build_report.replay_current", return_value=True),
            ):
                result = section(root, "example")
            self.assertIn("videos/everyday-rom-example-normal-and-strain.mp4", result)
            self.assertIn("fixed strain scale", result)
            self.assertNotIn("strain-scale", result)
            self.assertNotIn("linear-gradient", result)

    def test_main_metrics_use_optimized_benchmark(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "toothbrush").mkdir()
            (root / "performance").mkdir()
            (root / "performance/summary.json").write_text(
                json.dumps(
                    {
                        "objects": [{"slug": "toothbrush", "median_ms": 191}],
                        "selected_candidates": {
                            "toothbrush": {"ms": 10.37, "realtime_factor": 1.61, "candidate": True}
                        },
                    }
                )
            )
            result = section(root, "toothbrush")
            self.assertIn("Optimized sim frame", result)
            self.assertNotIn("191 ms", result)
            self.assertIn("10.4 ms", result)

    def test_outdated_hand_publication_notes_are_historical(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "hand").mkdir()
            (root / "hand/PUBLICATION.md").write_text("Old two-finger sequence")
            (root / "hand/REPRODUCE.md").write_text("Current five-finger sequence")
            (root / "hand/reproduction.zip").write_bytes(b"bundle")
            (root / "hand/scene.json").write_text('{"reproduce":"obsolete-worker-command"}')
            result = section(root, "hand")
            self.assertIn("Historical publication notes", result)
            self.assertIn("Current reproduction workflow", result)
            self.assertIn("Portable source bundle", result)
            self.assertNotIn("obsolete-worker-command", result)

    def test_strain_replay_is_not_described_as_alternate_camera(self):
        label, description = detail_caption("strain", {})
        self.assertEqual(label, "Principal surface strain")
        self.assertIn("Same saved motion and camera", description)
        self.assertNotIn("alternate camera", description)
        self.assertEqual(detail_caption("detail", {})[0], "Detail view")
        self.assertEqual(
            detail_caption("strain", {"detail_label": "Specific strain scale"})[0], "Specific strain scale"
        )

    def test_retained_old_notes_are_not_labeled_as_current_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "example"
            folder.mkdir()
            (folder / "index.html").write_text("Old revision")
            (folder / "normal_manifest.json").write_text('{"files_sha256":{"scene.json":"new"}}')
            self.assertIn("Historical report notes", section(root, "example"))
            self.assertNotIn("Detailed report", section(root, "example"))

    def test_presentation_is_bound_to_the_rendered_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "example"
            folder.mkdir()
            (folder / "scene.json").write_text('{"title":"Original"}')
            (folder / "rtx.json").write_text('{"inputs_sha256":{"scene.json":"old"}}')
            (root / "presentation.json").write_text(
                json.dumps({"example": {"scene_sha256": "new", "metadata": {"title": "Revised"}}})
            )
            self.assertIn("Original", section(root, "example"))
            (folder / "rtx.json").write_text('{"inputs_sha256":{"scene.json":"new"}}')
            self.assertIn("Revised", section(root, "example"))

    def test_verified_replay_is_primary_and_stale_video_falls_back(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "example"
            replay = directory / "replay"
            replay.mkdir(parents=True)
            for folder in (directory, replay):
                (folder / "scene.json").write_text('{"status":"Validated"}')
                (folder / "scene.npz").write_bytes(b"surfaces")
                (folder / "metrics.json").write_text('{"test_passed":true,"negative_control_passed":true}')
                (folder / "rtx.mp4").write_bytes(b"video")
                audit = {
                    "video_sha256": hashlib.sha256(b"video").hexdigest(),
                    "inputs_sha256": {
                        name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                        for name in ("scene.json", "scene.npz", "metrics.json")
                    },
                }
                if folder == replay:
                    audit["artifact_type"] = "normal_then_strain_concat"
                (folder / "rtx.json").write_text(json.dumps(audit))
            result = section(root, "example")
            self.assertIn('src="videos/everyday-rom-example-normal-and-strain.mp4?', result)
            self.assertIn('download="everyday-rom-example-normal-and-strain.mp4"', result)
            self.assertIn("Principal surface strain", result)
            self.assertIn('poster="example/poster.jpg?', result)
            (replay / "rtx.mp4").write_bytes(b"changed video")
            result = section(root, "example")
            self.assertNotIn('src="videos/everyday-rom-example-normal-and-strain.mp4?', result)
            self.assertIn('src="videos/everyday-rom-example-normal.mp4?', result)

    def test_cup_replaces_mustard_without_changing_gallery_count(self):
        self.assertEqual(len(SLUGS), 10)
        self.assertIn("cup-lid", SLUGS)
        self.assertNotIn("mustard-bottle", SLUGS)

    def test_high_strain_warning_is_not_an_unqualified_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "cup-lid"
            folder.mkdir()
            (folder / "scene.json").write_text('{"status":"Contact tests pass"}')
            (folder / "scene.npz").write_bytes(b"surface")
            (folder / "metrics.json").write_text(
                json.dumps(
                    {
                        "test_passed": True,
                        "negative_control_passed": True,
                        "principal_strain_screen_passed": False,
                    }
                )
            )
            (folder / "rtx.mp4").touch()
            (folder / "rtx.json").write_text(
                json.dumps(
                    {
                        "inputs_sha256": {
                            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                            for name in ("scene.json", "scene.npz", "metrics.json")
                        }
                    }
                )
            )
            result = section(root, "cup-lid")
            self.assertIn("High-strain ROM", result)
            self.assertIn("not a material validation", result)
            self.assertNotIn('class="badge pass"', result)

    def test_accepted_banana_keeps_prior_diagnostic_as_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "banana"
            (directory / "pod-path").mkdir(parents=True)
            (directory / "pod-path/index.html").write_text("Previous WIP")
            (directory / "VALIDATION.md").write_text("Old rejected trajectory")
            (directory / "static_quadratic_diagnostic.json").write_text("{}")
            (directory / "scene.json").write_text('{"status": "Validated bounded prototype"}')
            (directory / "scene.npz").write_bytes(b"current surface")
            (directory / "metrics.json").write_text('{"test_passed": true, "negative_control_passed": true}')
            (directory / "rtx.mp4").touch()
            (directory / "rtx.json").write_text(
                json.dumps(
                    {
                        "video_sha256": "current",
                        "inputs_sha256": {
                            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                            for name in ("scene.json", "scene.npz", "metrics.json")
                        },
                    }
                )
            )
            result = section(root, "banana")
            self.assertIn('class="badge pass"', result)
            self.assertIn('href="banana/pod-path/index.html"', result)
            self.assertIn("Earlier WIP", result)
            self.assertNotIn("Latest WIP", result)
            self.assertNotIn("two progression checks still fail", result)
            self.assertIn("Historical validation notes", result)
            self.assertIn("Historical static energy diagnostic", result)

    def test_banana_path_diagnostic_is_linked_without_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "banana"
            (directory / "pod-path").mkdir(parents=True)
            (directory / "pod-path/index.html").write_text("WIP")
            (directory / "metrics.json").write_text('{"test_passed": false}')
            result = section(root, "banana")
            self.assertIn('href="banana/pod-path/index.html"', result)
            self.assertIn("Pending / not accepted", result)
            self.assertNotIn('class="badge pass"', result)

    def test_hand_presentation_sidecar_supersedes_placeholder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "hand"
            directory.mkdir()
            (directory / "scene.json").write_text('{"private": true}')
            (directory / "gallery_view.json").write_text(
                json.dumps(
                    {
                        "publication_ready": True,
                        "title": "Pose-based hand: contact and release",
                        "description": "Held wrist; two contact-driven finger pulses.",
                        "mode_count": 32,
                    }
                )
            )
            result = section(root, "hand")
            self.assertIn("Held wrist; two contact-driven finger pulses.", result)
            self.assertIn("<b>Modes:</b> 32", result)
            self.assertNotIn("checks are in progress", result)

    def test_hand_reconstruction_does_not_imply_dynamics_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "hand_pod_audit.json").write_text(
                json.dumps(
                    {
                        "vertices": 5990,
                        "training_poses": 160,
                        "heldout_poses": 48,
                        "pose_validation": [],
                    }
                )
            )
            (root / "hand_dynamics_audit.json").write_text(
                json.dumps(
                    {
                        "accepted": False,
                        "contact_response_m": 0.00942,
                        "drop_self_intersections": 170,
                        "lower_drop_trial": {
                            "accepted": False,
                            "contact_response_m": 0.00363,
                            "max_floor_penetration_m": 0.00290,
                            "drop_max_edge_strain": 0.3418,
                            "drop_min_area_ratio": 0.620,
                        },
                        "bounded_shell_trial": {
                            "accepted": True,
                            "contact_response_m": 0.003498,
                            "substep_max_edge_strain": 0.22354,
                            "substep_min_area_ratio": 0.73820,
                            "max_floor_penetration_m": 0.001859,
                        },
                    }
                )
            )
            result = hand_pod_study(root)
            self.assertIn("not validated contact dynamics", result)
            self.assertIn("Contact experiment: WIP", result)
            self.assertIn("170 intersection pairs", result)
            self.assertIn("reject this trajectory", result)
            self.assertIn("Lower-drop follow-up: still WIP", result)
            self.assertIn("34.2%", result)
            self.assertIn("no-pusher drop control fails", result)
            self.assertIn("Bounded shell follow-up: checks pass", result)
            self.assertIn("3.5 mm", result)
            self.assertIn("22.4%", result)
            self.assertIn("not a validated glove", result)

    def test_stale_or_incomplete_render_is_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "rtx.mp4").touch()
            inputs = {}
            for name in ("scene.json", "scene.npz", "metrics.json"):
                (root / name).write_bytes(b"old")
                inputs[name] = hashlib.sha256(b"old").hexdigest()
            self.assertTrue(render_current(root, {"inputs_sha256": inputs}))
            self.assertFalse(render_current(root, {"inputs_sha256": {}}))
            (root / "metrics.json").write_bytes(b"new")
            self.assertFalse(render_current(root, {"inputs_sha256": inputs}))

    def test_video_version_and_paired_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "example"
            directory.mkdir()
            (directory / "scene.json").write_text(json.dumps({"status": "validated prototype"}))
            (directory / "scene.npz").write_bytes(b"surface")
            (directory / "metrics.json").write_text(json.dumps({"test_passed": True}))
            (directory / "rtx.mp4").touch()
            audit = {
                "video_sha256": "abcd1234",
                "inputs_sha256": {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ("scene.json", "scene.npz", "metrics.json")
                },
            }
            (directory / "rtx.json").write_text(json.dumps(audit))
            result = section(root, "example")
            self.assertIn("everyday-rom-example-normal.mp4?v=abcd1234", result)
            self.assertIn("Pending / not accepted", result)
            self.assertNotIn('class="badge pass"', result)
            (root / "review.json").write_text(
                json.dumps({"example": {"status": "Reviewed bounded prototype", "blocking": False}})
            )
            result = section(root, "example")
            self.assertIn("Reviewed bounded prototype", result)
            self.assertNotIn('class="badge pass"', result)

    def test_large_arrays_are_linked_not_embedded(self):
        metrics = {"records": list(range(100)), "nested": {"records": list(range(40))}, "gravity": [0, 0, -9.81]}
        compact = compact_metrics(metrics)
        self.assertEqual(compact["gravity"], metrics["gravity"])
        self.assertIn("100 entries", compact["records"])
        self.assertIn("40 entries", compact["nested"]["records"])
        self.assertEqual(len(metrics["records"]), 100)

    def test_additional_profile_view_requires_current_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder in (
                "example",
                "example/detail",
                "example/profile/detail",
                "example/lift",
                "example/lift/detail",
                "example/strain",
                "example/wrap-side",
            ):
                directory = root / folder
                directory.mkdir(parents=True, exist_ok=True)
                meta = {"detail_label": "Tracked profile"}
                if "/lift" in folder:
                    meta["view_description"] = "Separate validated held-lift trajectory."
                (directory / "scene.json").write_text(json.dumps(meta))
                (directory / "metrics.json").write_text(
                    json.dumps({"test_passed": True, "negative_control_passed": True})
                )
                (directory / "scene.npz").write_bytes(b"same saved surface")
                (directory / "rtx.mp4").touch()
                (directory / "rtx.json").write_text(
                    json.dumps(
                        {
                            "video_sha256": "profile1234",
                            "inputs_sha256": {
                                name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                for name in ("scene.json", "metrics.json", "scene.npz")
                            },
                        }
                    )
                )
            result = section(root, "example")
            self.assertIn("everyday-rom-example-detail.mp4?v=profile1234", result)
            self.assertIn("everyday-rom-example-profile-detail.mp4?v=profile1234", result)
            self.assertIn("everyday-rom-example-lift.mp4?v=profile1234", result)
            self.assertIn("everyday-rom-example-lift-detail.mp4?v=profile1234", result)
            self.assertIn("everyday-rom-example-strain.mp4?v=profile1234", result)
            self.assertIn("everyday-rom-example-wrap-side.mp4?v=profile1234", result)
            self.assertEqual(result.count("Separate validated held-lift trajectory."), 2)
            (root / "example/profile/detail/metrics.json").write_text('{"changed": true}')
            result = section(root, "example")
            self.assertNotIn("everyday-rom-example-profile-detail.mp4", result)
            self.assertIn("everyday-rom-example-detail.mp4", result)
            lift = root / "example/lift"
            (lift / "metrics.json").write_text('{"test_passed": false, "negative_control_passed": true}')
            audit = json.loads((lift / "rtx.json").read_text())
            audit["inputs_sha256"]["metrics.json"] = hashlib.sha256((lift / "metrics.json").read_bytes()).hexdigest()
            (lift / "rtx.json").write_text(json.dumps(audit))
            self.assertNotIn("everyday-rom-example-lift.mp4", section(root, "example"))

            parent = root / "example"
            (parent / "scene.npz").write_bytes(b"revised geometry")
            audit = json.loads((parent / "rtx.json").read_text())
            audit["inputs_sha256"]["scene.npz"] = hashlib.sha256(b"revised geometry").hexdigest()
            (parent / "rtx.json").write_text(json.dumps(audit))
            result = section(root, "example")
            self.assertNotIn("everyday-rom-example-detail.mp4", result)
            self.assertNotIn("everyday-rom-example-wrap-side.mp4", result)
            self.assertIn("everyday-rom-example-normal.mp4", result)

    def test_exported_source_preferred_over_working_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "example"
            (directory / "source").mkdir(parents=True)
            (directory / "source" / "example.py").write_text("# frozen export")
            (directory / "scene.json").write_text(json.dumps({"source_files": ["example.py", "helper.py"]}))
            result = section(root, "example")
            self.assertIn('href="example/source/example.py"', result)
            self.assertIn('href="source/helper.py"', result)


if __name__ == "__main__":
    unittest.main()
