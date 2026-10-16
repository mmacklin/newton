# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU-only relocation, generic scene, and unchanged final-video gate checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

if __package__:
    from . import render_scene, render_strain_replay
    from . import test_strain_replay as fixtures
    from .strain_adapters import digest
    from .strain_publication import stage_normal, verify_normal
    from .strain_replay import export_combined, verify, verify_combined, write_json
    from .verify_sources import verify_sources
else:
    import render_scene
    import render_strain_replay
    import test_strain_replay as fixtures
    from strain_adapters import digest
    from strain_publication import stage_normal, verify_normal
    from strain_replay import export_combined, verify, verify_combined, write_json
    from verify_sources import verify_sources


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExportTests("test_export_retains_every_original_array_and_full_unclipped_field")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def args(self, directory, *, normal=False, offline=False, frame=0):
        return SimpleNamespace(
            run=directory,
            normal=normal,
            offline_bundle=offline,
            frame=frame,
            width=1280,
            height=960,
            legend_anchor="top-left",
        )

    def test_generic_normal_scene_needs_no_object_adapter(self):
        f = self.fixture
        f.meta["slug"] = "arbitrary-saved-surface"
        write_json(f.source / "scene.json", f.meta)
        for name in ("reference.npz", "references.json"):
            (f.source / name).unlink()
        stage_normal(f.source, f.output)
        shutil.rmtree(f.source)
        self.assertEqual(verify_normal(f.output)["copied_files"], 3)
        meta, arrays = render_strain_replay.load_verified_scene(self.args(f.output, normal=True))
        self.assertEqual(meta["slug"], "arbitrary-saved-surface")
        np.testing.assert_array_equal(arrays["p"], f.arrays["p"])

    def test_generic_entrypoint_copies_inputs_and_delegates_exact_settings(self):
        f = self.fixture
        original = digest(f.source / "scene.npz")
        with patch.object(render_strain_replay, "main") as render:
            render_scene.main(
                [
                    str(f.source),
                    "--output",
                    str(f.output),
                    "--width",
                    "640",
                    "--height",
                    "480",
                    "--samples",
                    "24",
                    "--frame",
                    "1",
                ]
            )
            render.assert_called_once_with(
                [
                    str(f.output),
                    "--normal",
                    "--width",
                    "640",
                    "--height",
                    "480",
                    "--samples",
                    "24",
                    "--frame",
                    "1",
                ]
            )
        self.assertEqual(digest(f.source / "scene.npz"), original)
        self.assertEqual(digest(f.output / "scene.npz"), original)
        with self.assertRaisesRegex(ValueError, "new directory"):
            render_scene.main([str(f.source), "--output", str(f.output)])

    def test_full_normal_video_still_rejects_failed_physics_before_gpu(self):
        f = self.fixture
        for key in ("test_passed", "negative_control_passed"):
            metrics = {"test_passed": True, "negative_control_passed": True, key: False}
            write_json(f.source / "metrics.json", metrics)
            output = f.root / key
            stage_normal(f.source, output)
            with patch.object(render_strain_replay, "create_stage") as create:
                with patch.object(sys, "argv", ["render", str(output), "--normal"]):
                    with self.assertRaisesRegex(ValueError, "negative-control checks"):
                        render_strain_replay.main()
                create.assert_not_called()

    def test_relocated_strain_and_combined_verify_without_original_inputs(self):
        f = self.fixture
        f.run_export()
        moved = f.root / "moved"
        shutil.move(f.output, moved)
        shutil.rmtree(f.source)
        with self.assertRaises(FileNotFoundError):
            verify(moved)
        self.assertTrue(verify(moved, check_sources=False)["verified"])
        _, arrays = render_strain_replay.load_verified_scene(self.args(moved, offline=True))
        np.testing.assert_array_equal(arrays["p"], f.arrays["p"])
        replay = export_combined(moved, check_sources=False)
        self.assertTrue(verify_combined(moved)["identical_geometry_halves"])
        _, combined = render_strain_replay.load_verified_scene(self.args(replay, offline=True))
        np.testing.assert_array_equal(combined["p"][:3], combined["p"][3:])

    def test_relocated_mode_keeps_output_hash_checks(self):
        f = self.fixture
        f.run_export()
        (f.output / "scene.npz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Output hash"):
            render_strain_replay.load_verified_scene(self.args(f.output, offline=True))

    def test_relocated_mode_keeps_failed_physics_gate(self):
        f = self.fixture
        write_json(f.source / "metrics.json", {"test_passed": True, "negative_control_passed": False})
        f.run_export()
        shutil.rmtree(f.source)
        with self.assertRaisesRegex(ValueError, "negative-control checks"):
            render_strain_replay.load_verified_scene(self.args(f.output, offline=True, frame=None))

    def test_full_strain_video_requires_current_matching_overlay_audit(self):
        f = self.fixture
        f.run_export()
        audit = {
            "verified_no_projected_overlap": False,
            "style": "compact",
            "anchor": "top-left",
            "resolution": [1280, 960],
            "inputs_sha256": {"scene.npz": digest(f.output / "scene.npz")},
        }
        args = self.args(f.output, frame=None)
        write_json(f.output / "overlay-audit.json", audit)
        with self.assertRaisesRegex(ValueError, "matching clear"):
            render_strain_replay.load_verified_scene(args)
        audit["verified_no_projected_overlap"] = True
        write_json(f.output / "overlay-audit.json", audit)
        render_strain_replay.load_verified_scene(args)
        audit["inputs_sha256"]["scene.npz"] = "0" * 64
        write_json(f.output / "overlay-audit.json", audit)
        with self.assertRaisesRegex(ValueError, "stale"):
            render_strain_replay.load_verified_scene(args)

    def test_invalid_preview_frame_rejected_before_gpu(self):
        f = self.fixture
        stage_normal(f.source, f.output)
        for frame in (-1, 3):
            with self.assertRaisesRegex(ValueError, "outside saved sequence"):
                render_strain_replay.load_verified_scene(self.args(f.output, normal=True, frame=frame))

    def test_origin_pin_not_presented_as_input_runtime_identity(self):
        manifest = self.fixture.run_export()
        self.assertNotIn("base_commit", manifest)
        self.assertIn("implementation_base_commit", manifest)


class PackageTests(unittest.TestCase):
    def test_manifest_covers_packaged_sources_and_detects_mutation(self):
        package = Path(__file__).resolve().parent
        self.assertTrue(verify_sources(package)["verified"])
        with tempfile.TemporaryDirectory() as temp:
            moved = Path(temp) / "rendering"
            shutil.copytree(package, moved, ignore=shutil.ignore_patterns("__pycache__"))
            (moved / "surface_strain.py").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_sources(moved)

    def test_relocated_cli_help_and_imports_use_only_local_helpers(self):
        package = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temp:
            moved = Path(temp) / "rendering"
            shutil.copytree(package, moved, ignore=shutil.ignore_patterns("__pycache__"))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
            env.pop("PYTHONPATH", None)
            modules = (
                "render_strain_replay",
                "strain_replay",
                "strain_overlay",
                "strain_video",
                "strain_publication",
                "verify_sources",
                "render_scene",
            )
            for module in modules:
                for command in (["-m", f"rendering.{module}"], [str(moved / f"{module}.py")]):
                    result = subprocess.run(
                        [sys.executable, *command, "--help"],
                        cwd=temp,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=30,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)
            code = """
import importlib, json, pathlib, sys
names = ('render_strain_replay', 'strain_replay', 'strain_adapters', 'strain_overlay',
         'strain_video', 'strain_publication', 'surface_strain', 'verify_sources', 'render_scene')
root = pathlib.Path('rendering').resolve()
files = [str(pathlib.Path(importlib.import_module('rendering.' + name).__file__).resolve()) for name in names]
assert all(pathlib.Path(file).parent == root for file in files), files
assert not {'ovrtx', 'newton', 'warp'} & set(sys.modules)
print(json.dumps(files))
"""
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=temp,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)), 9)


if __name__ == "__main__":
    unittest.main()
