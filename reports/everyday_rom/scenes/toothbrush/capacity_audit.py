"""Graph-safe, read-only contact observations at every collision and solve."""

import json

import numpy as np
import warp as wp


@wp.kernel
def observe_collision(
    counter: wp.array[int],
    raw: wp.array[int],
    clock: wp.array[float],
    counts: wp.array[int],
    times: wp.array[float],
    seen: wp.array[int],
):
    i = wp.atomic_add(counter, 0, 1)
    if i < counts.shape[0]:
        counts[i] = raw[0]
        times[i] = clock[0]
        seen[i] = 1


@wp.kernel
def observe_solve(
    counter: wp.array[int],
    collision_counter: wp.array[int],
    raw: wp.array[int],
    body_counts: wp.array[int],
    overflow: wp.array[int],
    counts: wp.array[int],
    per_body: wp.array2d[int],
    overflows: wp.array[int],
    paired: wp.array[int],
    seen: wp.array[int],
):
    i = wp.atomic_add(counter, 0, 1)
    if i < counts.shape[0]:
        counts[i] = raw[0]
        for body in range(body_counts.shape[0]):
            per_body[i, body] = body_counts[body]
        overflows[i] = overflow[0]
        paired[i] = collision_counter[0]
        seen[i] = 1


def validate(data, expected, global_capacity, body_capacity, dt):
    assert expected > 0, "Empty execution cannot certify capacity"
    assert int(data["collision_count"]) == expected, "Incomplete collision coverage"
    assert int(data["solve_count"]) == expected, "Incomplete solved-substep coverage"
    for key, values in data.items():
        if values.ndim:
            assert len(values) == expected, f"Incomplete history: {key}"
    assert len(data["raw_global"]) == expected, "History storage exhausted or truncated"
    assert data["raw_per_body"].shape[0] == expected, "Incomplete per-body history"
    assert np.all(data["collision_seen"] == 1), "Missing collision observations"
    assert np.all(data["solve_seen"] == 1), "Missing solve observations"
    np.testing.assert_array_equal(data["collision_pair"], np.arange(1, expected + 1))
    np.testing.assert_array_equal(data["raw_global"], data["solved_raw_global"])
    # Compare the actual accumulated float32 clock, including the first step.
    expected_times = np.empty(expected, np.float32)
    t = np.float32(0)
    for i in range(expected):
        if i <= 240 and i % 4 == 0:
            t = np.float32(0)
        expected_times[i] = t
        t = np.float32(t + np.float32(dt))
    np.testing.assert_array_equal(data["time_s"], expected_times)
    assert np.all(data["raw_global"] >= 0)
    assert np.all(data["raw_per_body"] >= 0)
    assert int(data["raw_global"].max()) <= global_capacity, "Global contact overflow"
    assert int(data["raw_per_body"].max()) <= body_capacity, "Per-body contact overflow"
    assert not np.any(data["body_overflow_max"]), "Solver reported per-body overflow"
    return {
        "passed": True,
        "expected_substeps": expected,
        "collision_observations": int(data["collision_count"]),
        "solved_substep_observations": int(data["solve_count"]),
        "rigid_contact_max": global_capacity,
        "body_contact_capacity": body_capacity,
        "raw_global_peak": int(data["raw_global"].max()),
        "raw_per_body_peak": int(data["raw_per_body"].max()),
        "raw_peak_by_body": data["raw_per_body"].max(axis=0).tolist(),
        "global_overflow": False,
        "body_overflow": False,
        "first_collision_time_s": float(data["time_s"][0]),
        "last_collision_time_s": float(data["time_s"][-1]),
        "coverage": "Every actual collide and solved substep, from time zero, including gravity settling; graph capture itself does not execute.",
        "per_body_semantics": "Unclamped SolverVBD body_body_contact_counts after each solve, for dynamic rigid and elastic bodies; only complete when global capacity passes.",
    }


class CapacityAudit:
    def __init__(self, model, contacts, solver, clock, dt, slots=16384):
        self.contacts, self.solver, self.clock, self.dt = contacts, solver, clock, dt
        self.device = model.device
        self.global_capacity = int(contacts.rigid_contact_max)
        self.body_capacity = int(solver.body_body_contact_buffer_pre_alloc)
        self.collisions = wp.zeros(1, dtype=int, device=self.device)
        self.solves = wp.zeros(1, dtype=int, device=self.device)
        self.raw = wp.zeros(slots, dtype=int, device=self.device)
        self.solved_raw = wp.zeros(slots, dtype=int, device=self.device)
        self.times = wp.zeros(slots, dtype=float, device=self.device)
        self.body = wp.zeros((slots, model.body_count), dtype=int, device=self.device)
        self.overflow = wp.zeros(slots, dtype=int, device=self.device)
        self.paired = wp.zeros(slots, dtype=int, device=self.device)
        self.collision_seen = wp.zeros(slots, dtype=int, device=self.device)
        self.solve_seen = wp.zeros(slots, dtype=int, device=self.device)

    def after_collide(self):
        wp.launch(
            observe_collision,
            dim=1,
            inputs=[
                self.collisions,
                self.contacts.rigid_contact_count,
                self.clock,
                self.raw,
                self.times,
                self.collision_seen,
            ],
            device=self.device,
        )

    def after_solve(self):
        wp.launch(
            observe_solve,
            dim=1,
            inputs=[
                self.solves,
                self.collisions,
                self.contacts.rigid_contact_count,
                self.solver.body_body_contact_counts,
                self.solver.body_body_contact_overflow_max,
                self.solved_raw,
                self.body,
                self.overflow,
                self.paired,
                self.solve_seen,
            ],
            device=self.device,
        )

    def data(self):
        collisions, solves = int(self.collisions.numpy()[0]), int(self.solves.numpy()[0])
        n = min(max(collisions, solves), len(self.raw))
        return {
            "collision_count": np.array(collisions),
            "solve_count": np.array(solves),
            "raw_global": self.raw.numpy()[:n],
            "solved_raw_global": self.solved_raw.numpy()[:n],
            "raw_per_body": self.body.numpy()[:n],
            "body_overflow_max": self.overflow.numpy()[:n],
            "time_s": self.times.numpy()[:n],
            "collision_pair": self.paired.numpy()[:n],
            "collision_seen": self.collision_seen.numpy()[:n],
            "solve_seen": self.solve_seen.numpy()[:n],
        }

    def check(self, expected):
        return validate(self.data(), expected, self.global_capacity, self.body_capacity, self.dt)

    def export(self, output, expected):
        output.mkdir(parents=True, exist_ok=True)
        data = self.data()
        np.savez_compressed(
            output / "contact_capacity.npz",
            **data,
            global_capacity=self.global_capacity,
            body_capacity=self.body_capacity,
            expected_substeps=expected,
            dt=self.dt,
        )
        try:
            result = validate(data, expected, self.global_capacity, self.body_capacity, self.dt)
        except AssertionError as error:
            result = {
                "passed": False,
                "error": str(error),
                "expected_substeps": expected,
                "collision_count": int(data["collision_count"]),
                "solve_count": int(data["solve_count"]),
            }
        (output / "contact_capacity.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
