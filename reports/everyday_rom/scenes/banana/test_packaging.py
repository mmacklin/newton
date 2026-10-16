"""Check the exact derived ROM and the private extension's packaged provenance."""

import ast
import hashlib
import json
import unittest
from pathlib import Path

import banana
import build_banana_path
import numpy as np
import path_banana

ROOT = Path(__file__).resolve().parent


def extension_ast_sha256(source):
    """Normalize only legacy Warp array argument annotations, never bodies."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.arg) or not isinstance(node.annotation, ast.Call):
            continue
        annotation = node.annotation
        if (
            isinstance(annotation.func, ast.Attribute)
            and isinstance(annotation.func.value, ast.Name)
            and annotation.func.value.id == "wp"
            and annotation.func.attr in ("array", "array2d", "array3d", "array4d")
            and not annotation.args
            and len(annotation.keywords) == 1
            and annotation.keywords[0].arg == "dtype"
        ):
            node.annotation = ast.Subscript(value=annotation.func, slice=annotation.keywords[0].value, ctx=ast.Load())
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


class PackagingTests(unittest.TestCase):
    def test_all_packaged_hashes(self):
        """Verify all distributed source and derived model bytes."""
        manifest = json.loads((ROOT / "sources.json").read_text())
        for name, record in manifest["files"].items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            self.assertEqual(actual, record["packaged_sha256"], name)

    def test_private_extension_ast_equivalence(self):
        """Keep the original extension AST apart from Warp annotation syntax."""
        manifest = json.loads((ROOT / "sources.json").read_text())
        for name in ("solver_vbd_modal_path.py", "pod_path_energy.py"):
            actual = extension_ast_sha256((ROOT / name).read_text())
            self.assertEqual(actual, manifest["files"][name]["original_ast_sha256"], name)

    def test_annotation_normalization_is_narrow(self):
        for kind in ("array", "array2d", "array3d", "array4d"):
            legacy = f"def kernel(a: wp.{kind}(dtype=wp.float32)):\n    a[0] = 1\n"
            modern = f"def kernel(a: wp.{kind}[wp.float32]):\n    a[0] = 1\n"
            self.assertEqual(extension_ast_sha256(legacy), extension_ast_sha256(modern))
            for changed in (modern.replace("float32", "float64"), modern.replace("= 1", "= 2")):
                self.assertNotEqual(extension_ast_sha256(legacy), extension_ast_sha256(changed))
            constructor = f"value = wp.{kind}(dtype=wp.float32)\n"
            subscript = f"value = wp.{kind}[wp.float32]\n"
            self.assertNotEqual(extension_ast_sha256(constructor), extension_ast_sha256(subscript))
        self.assertNotEqual(
            extension_ast_sha256("def kernel(a: wp.array[wp.float32]): pass"),
            extension_ast_sha256("def kernel(a: wp.array2d[wp.float32]): pass"),
        )

    def test_local_imports(self):
        """Resolve helpers locally instead of another scene or author worktree."""
        for module in (banana, build_banana_path, path_banana):
            self.assertEqual(Path(module.__file__).resolve().parent, ROOT)

    def test_exact_published_ordered_rom(self):
        """Keep the published three-strip, 32-mode ordered path asset."""
        path = ROOT / "assets/ordered_rom.npz"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "5c7766fe2b931f3cbc06914378484933a451c832dbd7fa7296c248bb25c19f77",
        )
        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(len(data.files), 24)
            for strip, knots in enumerate((36, 37, 36)):
                self.assertEqual(data[f"phi_{strip}"].shape, (2210, 32, 3))
                self.assertEqual(data[f"coefficients_{strip}"].shape, (knots, 32))
                np.testing.assert_array_equal(data[f"coefficients_{strip}"][0], 0)
                self.assertTrue(np.all(data[f"mass_{strip}"] > 0))
                progress = data[f"progress_{strip}"]
                self.assertTrue(np.all(np.diff(progress) > 0))
                np.testing.assert_array_equal(progress[[0, -1]], [0, 1])
            self.assertTrue(all(np.isfinite(data[key]).all() for key in data.files))


if __name__ == "__main__":
    unittest.main()
