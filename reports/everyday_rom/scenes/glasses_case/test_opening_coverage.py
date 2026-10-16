import unittest

import numpy as np
from clamped_geometry import ROOT_WORLD, keyframe
from opening_coverage import closed_keyframe_coverage, opening_coverage


class OpeningCoverageTests(unittest.TestCase):
    def test_closed_flap_covers_full_mouth_width(self):
        coverage = closed_keyframe_coverage()
        self.assertEqual(coverage["opening_uncovered_sections"], 0, coverage)
        self.assertGreaterEqual(coverage["opening_min_width_margin_m"], -1e-7)
        self.assertAlmostEqual(coverage["opening_covered_fraction"], 1)

    def test_open_flap_does_not_cover_opening(self):
        coverage = opening_coverage(keyframe(0)[0] + ROOT_WORLD)
        self.assertEqual(coverage["opening_covered_fraction"], 0)

    def test_flap_below_mouth_cannot_count_as_lid(self):
        p = keyframe(1)[0] + ROOT_WORLD - np.array([0, 0, 0.1])
        self.assertEqual(opening_coverage(p)["opening_covered_fraction"], 0)

    def test_missing_side_width_is_detected(self):
        p = keyframe(1)[0] + ROOT_WORLD
        p[:, 0] *= 0.5
        self.assertLess(opening_coverage(p)["opening_covered_fraction"], 0.6)


if __name__ == "__main__":
    unittest.main()
