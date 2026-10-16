# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check presentation corrections without modifying frozen artifact metadata."""

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import build_report


class ReportSemanticsTests(unittest.TestCase):
    def test_actuated_hand_shows_motion_instead_of_inactive_probe(self):
        metrics = {
            "max_sampled_pusher_penetration_m": 0,
            "actuation_enabled": True,
            "per_finger": [
                {"hold_max_distal_vertex_displacement_m": 0.0113},
                {"hold_max_distal_vertex_displacement_m": 0.0169},
            ],
        }
        before = copy.deepcopy(metrics)
        result = build_report.metric_table(metrics)
        self.assertIn("Peak finger hold travel", result)
        self.assertIn("16.9 mm", result)
        self.assertNotIn("Sampled penetration", result)
        self.assertEqual(metrics, before)

    def test_contact_hand_retains_penetration_metric(self):
        result = build_report.metric_table({"max_sampled_pusher_penetration_m": 0.001})
        self.assertIn("Sampled penetration", result)
        self.assertNotIn("Peak finger hold travel", result)

    def render(self, slug, meta, *, ready=True, review=None, metrics=None, performance=None):
        root = Path("unused-report-root")
        directory = root / slug
        artifacts = {
            directory / "scene.json": meta,
            directory / "metrics.json": {"test_passed": True, "negative_control_passed": True, **(metrics or {})},
            directory / "rtx.json": {"video_sha256": "frozen1234", "inputs_sha256": {"scene.json": "unchanged"}},
            root / "review.json": {slug: review} if review else {},
            root / "performance/summary.json": performance or {},
        }
        before = copy.deepcopy(artifacts)
        with (
            patch.object(build_report, "load", side_effect=lambda path: artifacts.get(path, {})),
            patch.object(build_report, "render_current", side_effect=lambda path, audit: ready and path == directory),
        ):
            result = build_report.section(root, slug)
        self.assertEqual(artifacts, before)
        return result

    def test_model_revision_does_not_inherit_old_model_timing(self):
        result = self.render(
            "example",
            {},
            metrics={"model_revision": "wider-flap", "optimized_sim_frame_ms": 9.876},
            performance={
                "objects": [{"slug": "example", "median_ms": 87.654}],
                "selected_candidates": {"example": {"candidate": "old", "ms": 65.432, "realtime_factor": 0.25}},
            },
        )
        self.assertIn("9.876", result)
        self.assertNotIn("87.654", result)
        self.assertNotIn("65.432", result)
        self.assertIn("archived model, not this revision", result)
        self.assertNotIn("Sim frame (graph baseline)", result)

    def test_main_report_shows_optimized_timing_without_baseline(self):
        for slug in ("example", "hand"):
            metrics = {"max_sampled_pusher_penetration_m": 0.001} if slug == "hand" else {}
            result = self.render(
                slug,
                {},
                metrics=metrics,
                performance={
                    "objects": [{"slug": slug, "median_ms": 87.654}],
                    "selected_candidates": {slug: {"candidate": True, "ms": 2.345, "realtime_factor": 7.1}},
                },
            )
            self.assertIn("Optimized sim frame", result)
            self.assertIn("2.35 ms", result)
            self.assertNotIn("87.7", result)
            self.assertNotIn("baseline", result)
            self.assertNotIn("Measured optimization candidate", result)

    def test_glove_describes_horizontal_carry_and_post_release_ascent(self):
        """Replace the unsupported lift claim in both visible and accessible labels."""
        meta = {
            "title": "Hollow glove: press, recover, lift and drop",
            "description": "The cuff support lifts and releases the whole object.",
            "status": "validated bounded prototype",
            "limitations": ["Uncalibrated linear shell."],
            "parameters": {"substeps": 16},
        }
        result = self.render("glove", meta)
        self.assertIn("Hollow glove: contact, carry, release and drop", result)
        self.assertIn('aria-label="Hollow glove: contact, carry, release and drop simulation"', result)
        self.assertIn("recorded support ascent occurs after release", result)
        self.assertNotIn("lift and drop", result)
        self.assertNotIn("support lifts", result)
        self.assertIn("everyday-rom-glove-normal.mp4?v=frozen1234", result)
        self.assertIn('class="badge pass"', result)

    def test_clog_recoil_retains_support_contact(self):
        """Correct the landing caption without changing numerical validation."""
        result = self.render("clog", {"description": "The clog recoils and lands.", "status": "validated prototype"})
        self.assertIn("recoils while remaining on its rounded support", result)
        self.assertNotIn("recoils and lands", result)
        self.assertIn("<b>Simulation checks:</b> Passed", result)

    def test_cap_separates_bounded_validation_from_outstanding_intent(self):
        """Keep passed checks while displaying the larger-dynamics follow-up."""
        result = self.render("cap", {"status": "validated contact/release/drop prototype"})
        self.assertIn("bounded indentation; larger-dynamics follow-up outstanding", result)
        self.assertNotIn("validated contact/release/drop prototype", result)
        self.assertIn('<span class="badge pass">', result)
        self.assertIn("<b>Simulation checks:</b> Passed", result)

    def test_cap_dynamics_retains_validated_motion_status(self):
        """Do not apply the archived indentation label to the new dynamic run."""
        status = "validated bounded joint-mediated carry/tilt/release/impact prototype"
        result = self.render("cap", {"slug": "cap_dynamics", "status": status})
        self.assertIn(status, result)
        self.assertNotIn("larger-dynamics follow-up outstanding", result)
        self.assertIn("<b>Simulation checks:</b> Passed", result)

    def test_blocking_review_still_overrides_presentation_status(self):
        """Retain blocking review status and failed acceptance after corrections."""
        result = self.render("cap", {}, review={"blocking": True, "status": "WIP: failed review", "findings": []})
        self.assertIn("WIP: failed review", result)
        self.assertIn("Pending / not accepted", result)
        self.assertNotIn('class="badge pass"', result)

    def test_stale_media_still_remain_hidden(self):
        """Prevent semantic corrections from making stale media look current."""
        result = self.render("cap", {}, ready=False)
        self.assertIn("In progress", result)
        self.assertNotIn("<video", result)
        self.assertNotIn('class="badge pass"', result)

    def test_unrelated_metadata_are_preserved(self):
        """Leave other object descriptions and nested metadata unchanged."""
        meta = {"title": "Other object", "description": "Original description.", "parameters": {"substeps": 4}}
        result = self.render("example", meta)
        self.assertIn("Other object", result)
        self.assertIn("Original description.", result)


if __name__ == "__main__":
    unittest.main()
