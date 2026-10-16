# SPDX-License-Identifier: Apache-2.0
"""Experimental report-local passive modal-path energy in Newton's VBD block.

Uses a private solver override pinned to 8228bf6, not a public extension API.
No state projection, prescribed modes, or replacement contact solve is used.
"""

import numpy as np
import warp as wp

from newton.solvers import SolverVBD

wp.set_module_options({"enable_backward": False})


@wp.kernel
def assemble_modal_paths(
    body_ids: wp.array[wp.int32],
    body_binding: wp.array[wp.int32],
    elastic_index: wp.array[wp.int32],
    q_start: wp.array[wp.int32],
    mode_count: wp.array[wp.int32],
    frame_count: wp.array[wp.int32],
    coefficients: wp.array3d[wp.float64],
    progress: wp.array2d[wp.float64],
    stiffness: wp.array[wp.float64],
    restoring: wp.array[wp.float64],
    joint_q: wp.array[wp.float32],
    block_gradient: wp.array2d[wp.float32],
    block_matrix: wp.array3d[wp.float32],
    selected_progress: wp.array[wp.float64],
    selected_energy: wp.array[wp.float64],
):
    body = body_ids[wp.tid()]
    binding = body_binding[body]
    if binding >= 0:
        count = mode_count[binding]
        start = q_start[binding]
        k = stiffness[binding]
        ks = restoring[binding]
        best_energy = wp.float64(1.0e30)
        best_segment = int(0)
        best_t = wp.float64(0.0)
        best_denominator = wp.float64(1.0)
        best_progress = wp.float64(0.0)
        # Each segment's latent progress is eliminated analytically.
        for segment in range(frame_count[binding] - 1):
            s0 = progress[binding, segment]
            ds = progress[binding, segment + 1] - s0
            numerator = -ks * ds * s0
            denominator = ks * ds * ds
            for mode in range(count):
                a = coefficients[binding, segment, mode]
                d = coefficients[binding, segment + 1, mode] - a
                r = wp.float64(joint_q[start + mode]) - a
                numerator += k * d * r
                denominator += k * d * d
            t = wp.clamp(numerator / denominator, wp.float64(0.0), wp.float64(1.0))
            s = s0 + t * ds
            energy = wp.float64(0.5) * ks * s * s
            for mode in range(count):
                a = coefficients[binding, segment, mode]
                d = coefficients[binding, segment + 1, mode] - a
                r = wp.float64(joint_q[start + mode]) - a - t * d
                energy += wp.float64(0.5) * k * r * r
            if energy < best_energy:
                best_energy = energy
                best_segment = segment
                best_t = t
                best_denominator = denominator
                best_progress = s
        e = elastic_index[binding]
        interior = best_t > wp.float64(0.0) and best_t < wp.float64(1.0)
        for i in range(count):
            ai = coefficients[binding, best_segment, i]
            di = coefficients[binding, best_segment + 1, i] - ai
            ri = wp.float64(joint_q[start + i]) - ai - best_t * di
            block_gradient[e, 6 + i] += wp.float32(k * ri)
            for j in range(i, count):
                value = wp.float64(0.0)
                if i == j:
                    value = k
                if interior:
                    dj = coefficients[binding, best_segment + 1, j] - coefficients[binding, best_segment, j]
                    value -= k * k * di * dj / best_denominator
                block_matrix[e, 6 + i, 6 + j] += wp.float32(value)
        selected_progress[binding] = best_progress
        selected_energy[binding] = best_energy


class SolverVBDModalPath(SolverVBD):
    """Add a passive shape-path prior to selected mass-normalized elastic bodies.

    Args:
        model: Model at the pinned Newton revision, with fixed mode counts.
        paths: Mapping of body index to a CPU ModalPathEnergy reference.
        solver_options: Options forwarded unchanged to SolverVBD.

    Existing modal stiffness must be zero to avoid double-counting an
    incompatible rest energy. Damping and inertial coupling remain in Newton.
    All arrays are allocated at setup; graph replay performs no host readback.
    Body/mode/color topology must remain fixed for the solver's lifetime.
    Callers must verify the full sampled mass Gram, not only its diagonal.
    The path_selected_* fields describe the last pre-update assembly; evaluate
    the saved final coordinates separately for solved-state diagnostics.
    """

    def __init__(self, model, *, paths, **solver_options):
        if not paths:
            raise ValueError("At least one modal path is required")
        indices = model.body_elastic_index.numpy()
        counts = model.elastic_mode_count.numpy()
        starts = model.elastic_mode_start.numpy()
        joints = model.elastic_joint.numpy()
        q_starts = model.joint_q_start.numpy()
        masses = model.elastic_mode_mass.numpy()
        linear_k = model.elastic_mode_stiffness.numpy()
        maximum_frames = max(path.coefficients.shape[0] for path in paths.values())
        maximum_modes = max(path.coefficients.shape[1] for path in paths.values())
        coefficients = np.zeros((len(paths), maximum_frames, maximum_modes), dtype=np.float64)
        progress = np.zeros((len(paths), maximum_frames), dtype=np.float64)
        binding = np.full(model.body_count, -1, dtype=np.int32)
        data = {
            name: [] for name in ("elastic_index", "q_start", "mode_count", "frame_count", "stiffness", "restoring")
        }
        for index, (body, path) in enumerate(paths.items()):
            if not isinstance(body, (int, np.integer)) or isinstance(body, bool) or not 0 <= body < model.body_count:
                raise ValueError("Path keys must be valid integer body indices")
            e = int(indices[body])
            if e < 0:
                raise ValueError("Every mapped body must be elastic")
            count, start = int(counts[e]), int(starts[e])
            if path.coefficients.shape[1] != count:
                raise ValueError("Path and model mode counts differ")
            if not np.allclose(masses[start : start + count], 1, rtol=2e-5, atol=2e-5):
                raise ValueError("Path coordinates must be mass normalized")
            if np.any(linear_k[start : start + count] != 0):
                raise ValueError("Disable baseline linear modal stiffness before adding the path energy")
            if np.linalg.norm(path.coefficients[0]) > 1e-9:
                raise ValueError("First path keyframe must be the model rest configuration")
            frames = path.coefficients.shape[0]
            coefficients[index, :frames, :count] = path.coefficients
            progress[index, :frames] = path.progress
            binding[body] = index
            values = (e, int(q_starts[joints[e]]) + 7, count, frames, path.k_perp, path.k_progress)
            for name, value in zip(data, values, strict=True):
                data[name].append(value)
        super().__init__(model, **solver_options)
        self.path_body_binding = wp.array(binding, dtype=wp.int32, device=self.device)
        self.path_data = {
            name: wp.array(
                values, dtype=wp.float64 if name in ("stiffness", "restoring") else wp.int32, device=self.device
            )
            for name, values in data.items()
        }
        self.path_coefficients = wp.array(coefficients, dtype=wp.float64, device=self.device)
        self.path_progress = wp.array(progress, dtype=wp.float64, device=self.device)
        self.path_selected_progress = wp.zeros(len(paths), dtype=wp.float64, device=self.device)
        self.path_selected_energy = wp.zeros(len(paths), dtype=wp.float64, device=self.device)

    def _launch_elastic_body_solve(self, body_ids, solve_frame, state_in, state_out, relaxation, dt):
        data = self.path_data
        wp.launch(
            assemble_modal_paths,
            dim=body_ids.size,
            inputs=[
                body_ids,
                self.path_body_binding,
                data["elastic_index"],
                data["q_start"],
                data["mode_count"],
                data["frame_count"],
                self.path_coefficients,
                self.path_progress,
                data["stiffness"],
                data["restoring"],
                state_out.joint_q,
                self.elastic_body_block_grad,
                self.elastic_body_block_matrix,
                self.path_selected_progress,
                self.path_selected_energy,
            ],
            device=self.device,
        )
        super()._launch_elastic_body_solve(body_ids, solve_frame, state_in, state_out, relaxation, dt)
