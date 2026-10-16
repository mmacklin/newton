"""Read-only, graph-replayed raw collision and VBD body-list counters."""

import json
from pathlib import Path

import numpy as np
import warp as wp

GLOBAL_CAPACITY = 16384
TRACE_CAPACITY = 8192


@wp.kernel
def before_solve(
    cursor: wp.array[int],
    raw: wp.array[int],
    clock: wp.array[float],
    counts: wp.array[int],
    times: wp.array[float],
    collisions: wp.array[int],
):
    row = cursor[0]
    if row < counts.shape[0]:
        counts[row] = raw[0]
        times[row] = clock[0]
        collisions[row] = collisions[row] + 1


@wp.kernel
def after_solve(
    cursor: wp.array[int],
    raw: wp.array[int],
    overflow: wp.array[int],
    bodies: wp.array2d[int],
    overflows: wp.array[int],
    solves: wp.array[int],
):
    row = cursor[0]
    if row < bodies.shape[0]:
        for body in range(raw.shape[0]):
            bodies[row, body] = raw[body]
        overflows[row] = overflow[0]
        solves[row] = solves[row] + 1
    cursor[0] = row + 1


def validate_trace(data, expected, global_capacity, body_capacity, dt):
    assert int(data["cursor"]) == expected, "Incomplete collision/solve trace"
    assert len(data["global_raw"]) == expected, "Truncated trace"
    assert expected > 0, "Empty trace"
    assert np.all(data["collisions"] == 1), "Missing or repeated collision substep"
    assert np.all(data["solves"] == 1), "Missing or repeated solve substep"
    assert data["body_raw"].shape[0] == expected, "Incomplete body trace"
    assert np.all(data["global_raw"] >= 0), "Missing raw global count"
    assert np.all(data["body_raw"] >= 0), "Missing raw body count"
    assert np.all(data["global_raw"] <= global_capacity), "Global contact overflow"
    assert np.all(data["body_raw"] <= body_capacity), "Per-body contact overflow"
    assert np.all(data["body_overflow"] == 0), "Solver reported body overflow"
    assert np.isfinite(data["times"]).all(), "Nonfinite substep time"
    assert len(data["times"]) == expected, "Incomplete time trace"
    np.testing.assert_allclose(data["times"], np.arange(expected) * dt, atol=2e-4, rtol=0)
    return {
        "passed": True,
        "expected_substeps": expected,
        "observed_substeps": expected,
        "global_capacity": global_capacity,
        "body_capacity": body_capacity,
        "global_raw_peak": int(np.max(data["global_raw"])),
        "body_raw_peaks": np.max(data["body_raw"], axis=0).tolist(),
        "overflow": False,
        "includes_settling": True,
        "scope": "Every actual collision and solve in every replayed substep, including settling; graph capture records commands, not simulated steps",
        "body_count_semantics": "Unclamped VBD list atomic counters; static and rigid kinematic bodies are ineligible. Valid only when global capacity passes.",
    }


class ContactTrace:
    def __init__(self, model, contacts, solver):
        self.device = model.device
        self.global_capacity = contacts.rigid_contact_max
        self.body_capacity = solver.body_body_contact_buffer_pre_alloc
        self.cursor = wp.zeros(1, dtype=int, device=self.device)
        self.global_raw = wp.full(TRACE_CAPACITY, -1, dtype=int, device=self.device)
        self.body_raw = wp.full((TRACE_CAPACITY, model.body_count), -1, dtype=int, device=self.device)
        self.times = wp.zeros(TRACE_CAPACITY, dtype=float, device=self.device)
        self.collisions = wp.zeros(TRACE_CAPACITY, dtype=int, device=self.device)
        self.solves = wp.zeros(TRACE_CAPACITY, dtype=int, device=self.device)
        self.body_overflow = wp.zeros(TRACE_CAPACITY, dtype=int, device=self.device)

    def collision(self, contacts, clock):
        wp.launch(
            before_solve,
            dim=1,
            inputs=[self.cursor, contacts.rigid_contact_count, clock, self.global_raw, self.times, self.collisions],
            device=self.device,
        )

    def solved(self, solver):
        wp.launch(
            after_solve,
            dim=1,
            inputs=[
                self.cursor,
                solver.body_body_contact_counts,
                solver.body_body_contact_overflow_max,
                self.body_raw,
                self.body_overflow,
                self.solves,
            ],
            device=self.device,
        )

    def data(self):
        cursor = int(self.cursor.numpy()[0])
        result = {"cursor": np.asarray(cursor)}
        for name in ("global_raw", "body_raw", "times", "collisions", "solves", "body_overflow"):
            result[name] = getattr(self, name).numpy()[:cursor]
        return result

    def check(self, expected, dt):
        return validate_trace(self.data(), expected, self.global_capacity, self.body_capacity, dt)

    def save(self, output, expected, dt):
        output = Path(output)
        data = self.data()
        np.savez_compressed(output / "contact_trace.npz", **data)
        result = validate_trace(data, expected, self.global_capacity, self.body_capacity, dt)
        (output / "contact_capacity.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
