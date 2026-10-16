# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Projection and overlay bounds regression tests, without a GPU."""

import unittest

import numpy as np

if __package__:
    from . import test_strain_replay as fixtures
    from .render_strain_replay import annotate_frame, legend_image
    from .strain_overlay import camera_triangles, intersects_rectangle
else:
    import test_strain_replay as fixtures
    from render_strain_replay import annotate_frame, legend_image
    from strain_overlay import camera_triangles, intersects_rectangle


class OverlayTests(unittest.TestCase):
    def test_triangle_rectangle_sat_rejects_bbox_false_positive(self):
        tri = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]])
        self.assertFalse(intersects_rectangle(tri, [1.5, 1.5, 2, 2])[0])
        self.assertTrue(intersects_rectangle(tri, [0.1, 0.1, 0.2, 0.2])[0])
        self.assertTrue(intersects_rectangle(tri, [-1, -1, 3, 3])[0])
        self.assertFalse(intersects_rectangle(tri, [3, 3, 4, 4])[0])

    def test_projection_matches_known_camera(self):
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]])
        result = camera_triangles(
            points, np.array([[0, 1, 2]]), np.array([0.0, -2.0, 0.0]), np.array([0.0, 0.0, 0.0]), 1280, 960, 90
        )
        np.testing.assert_allclose(result[0], [[640, 480], [760, 480], [640, 360]])

    def test_near_plane_clipping_keeps_finite_triangles(self):
        points = np.array([[0.0, -3.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]])
        result = camera_triangles(
            points, np.array([[0, 1, 2]]), np.array([0.0, -2.0, 0.0]), np.array([0.0, 0.0, 0.0]), 1280, 960, 90
        )
        self.assertEqual(result.shape, (2, 3, 2))
        self.assertTrue(np.isfinite(result).all())

    def test_compact_legend_does_not_paint_below_y144(self):
        fixture = fixtures.ExportTests("test_export_retains_every_original_array_and_full_unclipped_field")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.run_export()
        meta = dict(fixture.meta, color_legend="strain-legend.png")
        self.assertLessEqual(legend_image(fixture.output, meta, 1280, 960).height, 96)
        image = np.full((960, 1280, 3), 70, dtype=np.uint8)
        actual = annotate_frame(image, fixture.output, meta, fixture.arrays, 0)
        np.testing.assert_array_equal(actual[144:], image[144:])
        self.assertFalse(np.array_equal(actual[48:144], image[48:144]))

    def test_bottom_right_moves_label_and_legend_together(self):
        fixture = fixtures.ExportTests("test_export_retains_every_original_array_and_full_unclipped_field")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.run_export()
        meta = dict(fixture.meta, color_legend="strain-legend.png")
        image = np.full((960, 1280, 3), 70, dtype=np.uint8)
        actual = annotate_frame(image, fixture.output, meta, fixture.arrays, 0, "bottom-right")
        np.testing.assert_array_equal(actual[:800], image[:800])
        np.testing.assert_array_equal(actual[:, :760], image[:, :760])
        self.assertFalse(np.array_equal(actual[800:], image[800:]))


if __name__ == "__main__":
    unittest.main()
