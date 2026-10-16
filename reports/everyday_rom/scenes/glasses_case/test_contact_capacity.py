import unittest

import numpy as np
from contact_capacity import validate_trace


class TestContactCapacity(unittest.TestCase):
    def data(self):
        return {
            "cursor": np.array(4),
            "global_raw": np.array([0, 2, 8, 1]),
            "body_raw": np.array([[0], [2], [8], [1]]),
            "times": np.arange(4) / 240,
            "collisions": np.ones(4),
            "solves": np.ones(4),
            "body_overflow": np.zeros(4),
        }

    def check(self, data):
        return validate_trace(data, 4, 10, 9, 1 / 240)

    def test_complete(self):
        self.assertTrue(self.check(self.data())["passed"])

    def test_global_overflow_between_saved_frames(self):
        data = self.data()
        data["global_raw"][2] = 11
        with self.assertRaisesRegex(AssertionError, "Global contact overflow"):
            self.check(data)

    def test_body_overflow(self):
        data = self.data()
        data["body_raw"][2, 0] = 10
        with self.assertRaisesRegex(AssertionError, "Per-body contact overflow"):
            self.check(data)

    def test_missing_collision_or_solve(self):
        for field in ("collisions", "solves"):
            data = self.data()
            data[field][1] = 0
            with self.assertRaises(AssertionError):
                self.check(data)

    def test_missing_settling(self):
        data = self.data()
        data["cursor"] = np.array(3)
        with self.assertRaisesRegex(AssertionError, "Incomplete"):
            self.check(data)

    def test_truncated_storage(self):
        data = self.data()
        data["global_raw"] = data["global_raw"][:3]
        with self.assertRaisesRegex(AssertionError, "Truncated"):
            self.check(data)

    def test_time_gap(self):
        data = self.data()
        data["times"][2] += 1 / 240
        with self.assertRaises(AssertionError):
            self.check(data)

    def test_solver_overflow_flag(self):
        data = self.data()
        data["body_overflow"][1] = 10
        with self.assertRaisesRegex(AssertionError, "Solver reported"):
            self.check(data)
