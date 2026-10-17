# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fused handle/head, 40 original modes, and public generic Newton VBD.

This reproduces the fused representation, not its private optimized dispatcher
or rigid-attachment collision routing. See FUSED_PORTABILITY.md before running.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
from brush_glove_geometry import (
    Config,
    brush_handle,
    brush_head_rom,
    coupled_mass_audit,
    run_cli,
    sampled_mass_properties,
    surface_mass,
    transform_points,
)
from subdivision import refined_plastic
from toothbrush import Example as FiniteNeckExample

import newton
import newton.solvers


def fused_mass_properties(handle, head_points, head_basis, root):
    """Combine the original finalized float32 properties in the head frame."""
    original = (
        sampled_mass_properties(handle[0], surface_mass(*handle, 0.018)),
        sampled_mass_properties(head_points, head_basis.sample_mass),
    )
    mass = np.asarray([item[0] for item in original], dtype=np.float32).astype(float)
    com = np.asarray([item[1] for item in original], dtype=np.float32).astype(float)
    inertia = np.asarray([item[2] for item in original], dtype=np.float32).astype(float)
    centers = com.copy()
    centers[0] -= root.astype(float)
    center = np.average(centers, axis=0, weights=mass)
    tensor = np.zeros((3, 3))
    for m, c, moment in zip(mass, centers, inertia, strict=True):
        d = c - center
        tensor += moment + m * ((d @ d) * np.eye(3) - np.outer(d, d))
    return float(mass.sum()), center, tensor


class FusedBuilder(newton.ModelBuilder):
    """Adapt the original construction through public builder methods only."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        handle, grip, ribs, plate, plastic, root = brush_handle(split_head=True)
        points, _, basis, _ = brush_head_rom(plastic, plate, root)
        self.fused_mass, self.fused_com, self.fused_inertia = fused_mass_properties(handle, points, basis, root)
        self.fusion_root, self.fusion_basis = root.copy(), basis
        attachments = np.vstack([mesh[0] - root for mesh in (handle, grip, ribs)])
        grasp = np.array([0, -0.105, 0.010], np.float32) - root
        attachments = np.vstack([attachments, grasp])
        # These massless samples bind rigid material and the grasp to zero Phi.
        # Never let nearest-head-sample interpolation invent attachment modes.
        zero = np.zeros((len(attachments), 40, 3), np.float32)
        self.attachment_sample_start = len(basis.sample_phi)
        self.attachment_sample_indices = basis.add_samples(attachments, phi=zero, psi=zero.copy())
        self.attachment_sample_end = len(basis.sample_phi)

    def add_body(self, **kwargs):
        if kwargs.get("label") != "dynamic_rigid_handle":
            return super().add_body(**kwargs)
        pose = wp.transform_multiply(kwargs["xform"], wp.transform(self.fusion_root, wp.quat_identity()))
        return super().add_body_elastic(
            xform=pose,
            mass=self.fused_mass,
            com=wp.vec3(*self.fused_com),
            inertia=wp.mat33(*self.fused_inertia.ravel()),
            modal_basis=self.fusion_basis,
            lock_inertia=True,
            label="fused_handle_and_head_frame",
        )

    def add_body_elastic(self, **kwargs):
        if kwargs.get("label") != "physical_elastic_head" or self.body_count != 1:
            raise ValueError("Expected exactly the original head after the fused handle")
        return 0

    def add_shape_mesh(self, body, *args, **kwargs):
        if body == 0 and kwargs.get("label") != "physical_elastic_head":
            transform = kwargs.get("xform", wp.transform_identity())
            kwargs["xform"] = wp.transform_multiply(wp.transform(-self.fusion_root, wp.quat_identity()), transform)
        return super().add_shape_mesh(body, *args, **kwargs)

    def add_joint_fixed(self, parent, child, **kwargs):
        if kwargs.get("label") == "fixed_neck_to_zero_phi_plastic_head":
            if parent != 0 or child != 0:
                raise ValueError("Fused neck must have one common owner")
            return -1
        if kwargs.get("label") == "visible_jaw_ideal_grasp_released_at_5_2s":
            kwargs["child_xform"] = wp.transform_multiply(
                wp.transform(-self.fusion_root, wp.quat_identity()), kwargs["child_xform"]
            )
        return super().add_joint_fixed(parent, child, **kwargs)


class HostArray:
    """Read-only diagnostic view, never a solver state replacement."""

    def __init__(self, array):
        self.array = array

    def numpy(self):
        return self.array.copy()


def virtual_handle(pose, velocity, root, handle_com, fused_com):
    """Recover the original handle origin and COM velocity for physical gates."""
    result = pose.copy()
    result[:3] = transform_points(-root[None], pose)[0]
    rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=float).reshape(3, 3)
    speed = velocity.copy()
    speed[:3] += np.cross(velocity[3:], rotation @ (handle_com - root - fused_com))
    return result, speed


def detailed_meshes(example):
    """Return the exact render-only two-level plastic embedding, not new mass."""
    basis = {}
    for i, mesh in enumerate(example.meshes):
        basis[f"rest_vertices_{i}"] = mesh["p"]
        basis[f"faces_{i}"] = mesh["f"]
    return refined_plastic(basis, {"meshes": example.meshes})


class Example(FiniteNeckExample):
    """Fused brush with public generic VBD; not the private timed implementation."""

    slug = "fused_toothbrush"
    default_iterations = 16
    expected_dynamic_body_count = 1
    source_files = (
        "fused_toothbrush.py",
        "toothbrush.py",
        "brush_glove_geometry.py",
        "field_coverage.py",
        "capacity_audit.py",
        "subdivision.py",
        "refine_render.py",
        "test_fused.py",
        "test_brush_glove.py",
        "fused_reference.json",
        "presentation.json",
    )

    def __init__(self, viewer=None, cfg: Config | None = None, *, construction_only=False):
        self.construction_only = construction_only
        cfg = replace(cfg or Config(iterations=16), elastic_relaxation=0.4)
        super().__init__(viewer, cfg)
        self.initial_handle_pose = self.handle_diagnostics()[0]

    def begin(self, viewer, cfg, kind):
        super().begin(viewer, cfg, kind)
        self.builder = FusedBuilder(gravity=self.gravity, up_axis="Z")

    def finish(self):
        self.root_joints = []
        self.original_handle_mesh = self.meshes[0]["p"].copy()
        for mesh in self.meshes:
            if mesh["body"] == 0 and mesh["elastic"] is None:
                mesh["p"] = mesh["p"] - self.builder.fusion_root
        if not self.construction_only:
            super().finish()
            return
        if wp.get_device().is_cuda:
            raise ValueError("Construction-only checks require the CPU device")
        self.builder.color()
        self.model = self.builder.finalize(device="cpu")
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.pipeline = newton.CollisionPipeline(self.model, reduce_contacts=False, rigid_contact_max=16384)
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.cfg.iterations,
            rigid_body_contact_buffer_size=self.contact_capacity,
            rigid_contact_k_start=self.cfg.contact_ke,
            rigid_compliant_alm=False,
            rigid_contact_hard=False,
            friction_epsilon=1e-4,
            elastic_body_relaxation=self.cfg.elastic_relaxation,
            elastic_solver_metrics=True,
            **self.solver_options(),
        )

    def handle_diagnostics(self):
        return virtual_handle(
            self.state_0.body_q.numpy()[0],
            self.state_0.body_qd.numpy()[0],
            self.builder.fusion_root,
            self.handle_com,
            self.model.body_com.numpy()[0],
        )

    def object_metrics(self, frame, displacements):
        pose, speed = self.handle_diagnostics()
        poses = np.vstack([self.state_0.body_q.numpy(), pose])
        velocities = np.vstack([self.state_0.body_qd.numpy(), speed])
        view = SimpleNamespace(**self.__dict__)
        view.handle = len(poses) - 1
        view.meshes = [dict(mesh) for mesh in self.meshes]
        view.meshes[0]["p"] = self.original_handle_mesh
        view.state_0 = SimpleNamespace(body_q=HostArray(poses), body_qd=HostArray(velocities))
        FiniteNeckExample.object_metrics(view, frame, displacements)

    def mass_audit(self):
        mass, com, inertia = (getattr(self.model, f"body_{name}").numpy()[0] for name in ("mass", "com", "inertia"))
        for actual, expected in (
            (mass, self.builder.fused_mass),
            (com, self.builder.fused_com),
            (inertia, self.builder.fused_inertia),
        ):
            np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.float32))
        result = coupled_mass_audit(
            float(mass),
            com.astype(float),
            inertia.astype(float),
            self.model.elastic_mode_mass.numpy().astype(float),
            self.model.elastic_mode_coupling_linear.numpy().astype(float),
            self.model.elastic_mode_coupling_angular.numpy().astype(float),
        )
        result.update(
            fused_mass_kg=float(mass),
            fused_com_m=com.tolist(),
            inertia_regularization_applied=False,
            fused_properties_match_original_float32_construction=True,
            actual_implicit_mass_coupling=self.solver.elastic_implicit_mass_coupling.numpy().tolist(),
        )
        return result

    def summary(self):
        result = super().summary()
        result.pop("warm_graph_mean_ms", None)
        result.pop("warm_graph_p95_ms", None)
        result.update(
            model_revision="fused-handle-head-4x16"
            if self.cfg.substeps == 4 and self.cfg.iterations == 16
            else "fused-handle-head-custom",
            solver_implementation="public generic SolverVBD",
            collision_routing="public elastic vertex contacts, including zero-Phi handle attachments",
            dynamic_body_count=1,
            finite_neck_trajectory_equivalence_claim=False,
            private_timing_equivalence_claim=False,
            timing_scope="Diagnostic stepping only; not the published private clean timing",
            handle_diagnostic_reference="Virtual original handle origin/COM; no extra body",
        )
        return result

    def scene_metadata(self):
        result = super().scene_metadata()
        result.update(
            model="One fused rigid handle/head frame, original 40 modes and 20 tufts. No finite neck joint.",
            driven="Original kinematic tool/carriage and grasp release; public generic VBD solves the fused frame and all modes.",
            rigid_fusion=True,
            diagnostic_handle_local_translation=(-self.builder.fusion_root).tolist(),
        )
        result["limitations"] += [
            "Public elastic vertex contact routing differs from the private rigid-attachment path in the accepted gallery run.",
            "CPU construction/algebra verified only. No new GPU physical qualification or private timing equivalence is claimed.",
        ]
        return result


def construction_report():
    """Construct on CPU without settling or solver steps."""
    with wp.ScopedDevice("cpu"):
        example = Example(construction_only=True)
    detailed = detailed_meshes(example)
    return {
        "scope": "CPU construction only; no integration, capture, render or physical acceptance",
        "body_count": example.model.body_count,
        "dynamic_body_count": int(np.count_nonzero(example.model.body_flags.numpy() == int(newton.BodyFlags.DYNAMIC))),
        "mode_count": example.builder.fusion_basis.mode_count,
        "tuft_count": len(example.head_parts) - 2,
        "mass_audit": example.mass_audit(),
        "detailed_meshes": {
            example.meshes[i]["name"]: {"vertices": len(mesh["points"]), "triangles": len(mesh["faces"])}
            for i, mesh in detailed.items()
        },
        "private_dispatcher": False,
        "public_elastic_contact_work_count": example.pipeline.elastic_contact_work_count,
        "physical_qualification": "pending for this public generic implementation",
    }


if __name__ == "__main__":
    if "--check-construction" in sys.argv:
        parser = argparse.ArgumentParser(description="CPU-only fused construction and algebra entry point")
        parser.add_argument("--check-construction", action="store_true", required=True)
        parser.add_argument("--output", type=Path)
        args = parser.parse_args()
        result = json.dumps(construction_report(), indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result)
        print(result, end="")
    else:
        run_cli(Example)
