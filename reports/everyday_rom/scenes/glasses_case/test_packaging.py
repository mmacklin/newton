"""Check the packaged full-width revision and unchanged private extension."""

import ast
import hashlib
import json
import unittest
from pathlib import Path

import clamped_case
import clamped_geometry
import glasses_case
import numpy as np

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
        """Verify all distributed source bytes against the package manifest."""
        manifest = json.loads((ROOT / "sources.json").read_text())
        for name, record in manifest["files"].items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            self.assertEqual(actual, record["packaged_sha256"], name)
        coverage = manifest["files"]["opening_coverage.py"]
        self.assertEqual(coverage["packaged_sha256"], coverage["origins"][0]["sha256"])

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
        """Resolve the current full-width scene and helpers locally."""
        for module in (clamped_case, clamped_geometry, glasses_case):
            self.assertEqual(Path(module.__file__).resolve().parent, ROOT)

    def test_full_width_outline_and_original_controls(self):
        """Keep the late shoulder taper and original simulation settings."""
        np.testing.assert_allclose(
            clamped_geometry.flap_half_width(np.array([0, 0.2, 0.5, 0.73])), [0.073, 0.075, 0.075, 0.075]
        )
        config = clamped_case.Config()
        self.assertEqual((config.substeps, config.iterations, config.frames), (4, 16, 360))
        self.assertTrue(config.path_energy and config.tip_attachment and config.close_enabled)


if __name__ == "__main__":
    unittest.main()
