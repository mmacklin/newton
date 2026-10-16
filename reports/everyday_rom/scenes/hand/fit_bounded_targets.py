"""Fit offline actuation targets with a finite surface-strain constraint.

This is target design, not a runtime deformation clamp. Runtime uses only finite
generalized forces; every solved state is audited independently.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from hand_contact import finite_strain
from hand_geometry import IntersectionAudit, surface_metrics
from individual import FINGERS, finger_patches


class StrainChart:
    def __init__(self, rest, faces, phi):
        tri = np.asarray(rest, float)[faces]
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        length = np.linalg.norm(e1, axis=1)
        x = np.sum(e1 * e2, axis=1) / length
        y = np.linalg.norm(np.cross(e1, e2), axis=1) / length
        self.f0 = np.stack((e1 / length[:, None], (e2 - e1 * (x / length)[:, None]) / y[:, None]), axis=-1)
        d1 = phi[faces[:, 1]] - phi[faces[:, 0]]
        d2 = phi[faces[:, 2]] - phi[faces[:, 0]]
        self.df = np.stack(
            (d1 / length[:, None, None], (d2 - d1 * (x / length)[:, None, None]) / y[:, None, None]), axis=-1
        )

    def evaluate(self, q, indices):
        f = self.f0[indices] + np.einsum("fmci,m->fci", self.df[indices], q)
        u, s, vt = np.linalg.svd(f, full_matrices=False)
        derivative = np.einsum("fck,fmci,fki->fkm", u, self.df[indices], vt)
        return s - 1, derivative


def fit(rest, faces, phi, native, finger, distance, bound=0.12):
    from scipy.optimize import minimize

    patches = finger_patches(rest)
    j = np.stack([phi[p].mean(axis=0).T for p in patches]).reshape(15, -1).astype(float)
    direction = (native[patches[finger]] - rest[patches[finger]]).mean(axis=0)
    direction /= np.linalg.norm(direction)
    goal = np.zeros((5, 3))
    goal[finger] = distance * direction
    matrix = phi.transpose(0, 2, 1).reshape(-1, phi.shape[1]).astype(float)
    qnative = np.linalg.lstsq(matrix, (native - rest).ravel(), rcond=None)[0]
    qnative *= distance / np.linalg.norm((native[patches[finger]] - rest[patches[finger]]).mean(axis=0))
    scale = 0.001
    # Distal-centroid fit dominates; native field is a weak regularizer.
    a = np.vstack((j * scale / 0.001, matrix * scale / np.sqrt(len(matrix)) / 0.01))
    b = np.r_[goal.ravel() / 0.001, matrix @ qnative / np.sqrt(len(matrix)) / 0.01]
    chart = StrainChart(rest, faces, phi.astype(float))
    active = np.arange(len(faces))[::100]
    x = np.zeros(phi.shape[1])
    history = []
    for outer in range(15):

        def constraints(x, active=active):
            e, _ = chart.evaluate(x * scale, active)
            return np.r_[bound - e.ravel(), bound + e.ravel()]

        def jac(x, active=active):
            _, d = chart.evaluate(x * scale, active)
            d = d.reshape(-1, len(x)) * scale
            return np.vstack((-d, d))

        result = minimize(
            lambda x: 0.5 * np.sum((a @ x - b) ** 2),
            x,
            jac=lambda x: a.T @ (a @ x - b),
            method="SLSQP",
            constraints={"type": "ineq", "fun": constraints, "jac": jac},
            options={"maxiter": 150, "ftol": 1e-10},
        )
        x = result.x
        e, _ = chart.evaluate(x * scale, np.arange(len(faces)))
        worst = np.max(np.abs(e), axis=1)
        bad = np.flatnonzero(worst > bound + 1e-6)
        history.append(
            {
                "iteration": outer,
                "optimizer_success": bool(result.success),
                "message": result.message,
                "active_faces": len(active),
                "violating_faces": len(bad),
                "max_principal_strain": float(worst.max()),
                "objective": float(result.fun),
            }
        )
        if len(bad) == 0:
            break
        active = np.union1d(active, bad)
    q = x * scale
    points = rest + np.einsum("nmc,m->nc", phi, q)
    delta = j @ q
    return q, {
        "finger": FINGERS[finger],
        "target_distance_m": distance,
        "actual_centroid_travel_m": float(np.linalg.norm(delta.reshape(5, 3)[finger])),
        "other_centroid_travel_max_m": float(
            np.max(np.linalg.norm(np.delete(delta.reshape(5, 3), finger, axis=0), axis=1))
        ),
        "max_displacement_m": float(np.linalg.norm(points - rest, axis=1).max()),
        "max_principal_strain": float(np.max(np.abs(finite_strain(rest, points, faces)))),
        "geometry": surface_metrics(rest, points, faces),
        "history": history,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--coverage", type=Path, required=True)
    p.add_argument("--modes", type=int, default=48)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.inputs / f"rom-{args.modes}.npz") as a:
        rest, faces, used, phi = (a[k] for k in ("rest", "faces", "used", "sample_phi"))
        phi = phi[used]
    with np.load(args.coverage / "paths-20.npz") as a:
        native = a["points"][:, -1, used]
    checker = IntersectionAudit(rest, faces, device="cpu")
    qs, rows = [], []
    for i in range(5):
        q, row = fit(rest, faces, phi, native[i], i, 0.012)
        row["crossings"] = checker.count(rest + np.einsum("nmc,m->nc", phi, q))
        path = rest + np.einsum("nmc,tm->tnc", phi, np.linspace(0, 1, 21)[:, None] * q)
        row["linear_path_max_principal_strain"] = float(np.abs(finite_strain(rest, path, faces)).max())
        qs.append(q)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("geometry", "history")}), flush=True)
    np.savez_compressed(args.output / "targets.npz", q=np.asarray(qs))
    (args.output / "audit.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
