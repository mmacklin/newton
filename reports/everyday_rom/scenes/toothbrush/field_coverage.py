"""Read-only contact coverage of the 5 longitudinal by 4 transverse tuft grid."""

import json

import numpy as np
import warp as wp


@wp.kernel
def record_tuft_contacts(
    counter: wp.array[int],
    raw: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    sample0: wp.array[int],
    sample1: wp.array[int],
    head: int,
    tool: int,
    labels: wp.array[int],
    history: wp.array2d[int],
):
    c = wp.tid()
    step = counter[0] - 1
    if c < raw[0] and step < history.shape[0]:
        sample = int(-1)
        if shape0[c] == head and shape1[c] == tool:
            sample = sample0[c]
        elif shape1[c] == head and shape0[c] == tool:
            sample = sample1[c]
        if sample >= 0 and sample < labels.shape[0]:
            tuft = labels[sample]
            if tuft >= 0:
                wp.atomic_add(history, step, tuft, 1)


def coverage_summary(records):
    selected = [r for r in records if 2.3 < r["time"] < 4.6]
    if not selected:
        return {}
    contacts = np.asarray([r["tuft_contacts_all_substeps"] for r in selected])
    deformation = np.asarray([r["tuft_tip_deformation_m"] for r in selected])
    contact_frames = np.count_nonzero(contacts, axis=0)
    visible_frames = np.count_nonzero((contacts > 0) & (deformation > 0.0005), axis=0)
    covered = visible_frames >= 6
    first = [next((r["time"] for r in selected if r["tuft_contacts_all_substeps"][i]), None) for i in range(20)]
    return {
        "layout": "5 longitudinal rows (y), 4 transverse columns (x), row-major tuft order",
        "window_s": [2.3, 4.6],
        "visibility_threshold_m": 0.0005,
        "minimum_visible_contact_frames": 6,
        "contact_frames_per_tuft": contact_frames.tolist(),
        "visible_contact_frames_per_tuft": visible_frames.tolist(),
        "peak_tip_deformation_m_per_tuft": deformation.max(axis=0).tolist(),
        "first_contact_time_s_per_tuft": first,
        "covered_tufts": int(covered.sum()),
        "covered_tufts_per_longitudinal_row": covered.reshape(5, 4).sum(axis=1).tolist(),
        "covered_tufts_per_transverse_column": covered.reshape(5, 4).sum(axis=0).tolist(),
    }


def require_field_coverage(summary):
    assert summary["covered_tufts"] == 20, "Not every tuft has sustained visible contact-driven bending"
    assert summary["covered_tufts_per_transverse_column"] == [5] * 4, "Missing transverse column coverage"
    assert summary["covered_tufts_per_longitudinal_row"] == [4] * 5, "Missing longitudinal row coverage"


class FieldAudit:
    def __init__(self, example):
        self.example = example
        model = example.model
        labels = np.full(len(model.elastic_shape_vertex_local), -1, np.int32)
        shape = np.flatnonzero(model.elastic_shape_shape.numpy() == example.elastic[0]["shape"])[0]
        start = int(model.elastic_shape_vertex_start.numpy()[shape])
        for tuft, part in enumerate(example.head_parts[2:]):
            first = start + part["start"]
            labels[first : first + part["count"]] = tuft
        self.labels = wp.array(labels, dtype=int, device=model.device)
        self.history = wp.zeros((16384, 20), dtype=int, device=model.device)

    def observe(self):
        e = self.example
        c = e.contacts
        wp.launch(
            record_tuft_contacts,
            c.rigid_contact_max,
            inputs=[
                e.capacity_audit.collisions,
                c.rigid_contact_count,
                c.rigid_contact_shape0,
                c.rigid_contact_shape1,
                c.rigid_contact_elastic_sample0,
                c.rigid_contact_elastic_sample1,
                e.elastic[0]["shape"],
                e.tool_shape,
                self.labels,
                self.history,
            ],
            device=e.model.device,
        )

    def last_frame(self):
        n = int(self.example.capacity_audit.collisions.numpy()[0])
        return self.history.numpy()[n - self.example.cfg.substeps : n].sum(axis=0)

    def export(self, output):
        n = int(self.example.capacity_audit.collisions.numpy()[0])
        np.savez_compressed(
            output / "field_contacts.npz",
            counts=self.history.numpy()[:n],
            settling_substeps=240,
            simulation_substeps=len(self.example.records) * 4,
        )
        (output / "field_coverage.json").write_text(json.dumps(coverage_summary(self.example.records), indent=2) + "\n")
