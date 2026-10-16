"""Graph-compatible contact evidence; records every collide and solved substep."""

from pathlib import Path

import numpy as np
import warp as wp

GLOBAL_CAPACITY = 32768
RECORD_CAPACITY = 16384


@wp.kernel
def record_collide(
    count: wp.array[wp.int32],
    clock: wp.array[wp.float32],
    cursors: wp.array[wp.int32],
    rows: wp.array2d[wp.int32],
    times: wp.array[wp.float32],
):
    index = cursors[0]
    if index < rows.shape[0]:
        rows[index, 0] = count[0]
        times[index] = clock[0]
    cursors[0] = index + 1


@wp.kernel
def record_solved(
    count: wp.array[wp.int32],
    body_counts: wp.array[wp.int32],
    overflow: wp.array[wp.int32],
    cursors: wp.array[wp.int32],
    rows: wp.array2d[wp.int32],
):
    index = cursors[1]
    if index < rows.shape[0]:
        rows[index, 1] = count[0]
        rows[index, 2] = overflow[0]
        rows[index, 3] = index
        for body in range(body_counts.shape[0]):
            rows[index, 4 + body] = body_counts[body]
    cursors[1] = index + 1


def validate_counts(data, expected_steps, dt):
    """Reject missing substeps, truncated global/body lists, or recorder overflow."""
    required = {"cursors", "rows", "times", "global_capacity", "body_capacity"}
    assert required <= set(data), "Missing all-substep capacity evidence"
    rows, times = data["rows"], data["times"]
    np.testing.assert_array_equal(data["cursors"], [expected_steps, expected_steps])
    assert expected_steps > 0 and len(rows) == expected_steps == len(times)
    assert rows.ndim == 2 and rows.shape[1] >= 5
    assert np.isfinite(times).all() and times[0] == 0
    np.testing.assert_allclose(times, np.arange(expected_steps) * dt, atol=0.0002, rtol=0)
    np.testing.assert_allclose(np.diff(times), dt, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(rows[:, 3], np.arange(expected_steps))
    np.testing.assert_array_equal(rows[:, 0], rows[:, 1])
    assert (rows >= 0).all()
    assert rows[:, :2].max() < int(data["global_capacity"]), "Global overflow/saturation"
    assert rows[:, 4:].max() < int(data["body_capacity"]), "Per-body overflow/saturation"
    assert not rows[:, 2].any(), "Solver reported per-body overflow"


class ContactRecorder:
    def __init__(self, contacts, solver, clock):
        self.contacts, self.solver, self.clock = contacts, solver, clock
        self.cursors = wp.zeros(2, dtype=wp.int32, device=clock.device)
        self.rows = wp.full(
            (RECORD_CAPACITY, 4 + solver.body_body_contact_counts.shape[0]),
            -1,
            dtype=wp.int32,
            device=clock.device,
        )
        self.times = wp.full(RECORD_CAPACITY, -1.0, dtype=wp.float32, device=clock.device)

    def collide(self):
        wp.launch(
            record_collide,
            dim=1,
            inputs=[
                self.contacts.rigid_contact_count,
                self.clock,
                self.cursors,
                self.rows,
                self.times,
            ],
            device=self.clock.device,
        )

    def solved(self):
        wp.launch(
            record_solved,
            dim=1,
            inputs=[
                self.contacts.rigid_contact_count,
                self.solver.body_body_contact_counts,
                self.solver.body_body_contact_overflow_max,
                self.cursors,
                self.rows,
            ],
            device=self.clock.device,
        )

    def data(self):
        cursors = self.cursors.numpy()
        count = min(int(max(cursors)), RECORD_CAPACITY)
        return {
            "cursors": cursors,
            "rows": self.rows.numpy()[:count],
            "times": self.times.numpy()[:count],
            "global_capacity": np.int32(self.contacts.rigid_contact_max),
            "body_capacity": np.int32(self.solver.body_body_contact_buffer_pre_alloc),
        }

    def summary(self, expected_steps, dt, settling_end):
        data = self.data()
        rows = data["rows"]
        settling = min(round(settling_end / dt), len(rows))
        try:
            validate_counts(data, expected_steps, dt)
            passed = True
        except AssertionError:
            passed = False
        return {
            "passed": passed,
            "scope": "Every actual collide and solved substep, including initial settling; CUDA graph compatible",
            "global_capacity": int(data["global_capacity"]),
            "per_body_capacity": int(data["body_capacity"]),
            "expected_substeps": expected_steps,
            "collide_calls": int(data["cursors"][0]),
            "solved_calls": int(data["cursors"][1]),
            "raw_peak": int(rows[:, 0].max(initial=0)),
            "per_body_list_peaks": rows[:, 4:].max(axis=0, initial=0).tolist(),
            "per_body_overflow_peak": int(rows[:, 2].max(initial=0)),
            "settling_substeps": settling,
            "settling_raw_peak": int(rows[:settling, 0].max(initial=0)),
            "global_overflow": bool(np.any(rows[:, 0] > data["global_capacity"])),
            "per_body_overflow": bool(np.any(rows[:, 4:] > data["body_capacity"]) or np.any(rows[:, 2])),
            "row_columns": [
                "raw_after_collide",
                "raw_after_solve",
                "body_overflow_max",
                "substep_index",
                "body_counts...",
            ],
            "counts_file": "contact_counts.npz",
        }

    def validate(self, expected_steps, dt):
        validate_counts(self.data(), expected_steps, dt)

    def export(self, output):
        np.savez_compressed(Path(output) / "contact_counts.npz", **self.data())
