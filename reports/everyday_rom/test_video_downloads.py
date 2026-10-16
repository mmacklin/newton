# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Scene-named MP4 downloads preserve the audited video bytes."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import build_report


class VideoDownloadTests(unittest.TestCase):
    def test_scene_and_view_filenames_are_unique_mp4s(self):
        paths = [
            build_report.video_path(slug, view)
            for slug in build_report.SLUGS
            for view in ("", "replay", "strain", "detail", "profile/detail", "lift", "lift/detail", "wrap-side")
        ]
        names = [Path(path).name for path in paths]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.endswith(".mp4") for name in names))
        self.assertIn("everyday-rom-headphones-normal-and-strain.mp4", names)
        self.assertIn("everyday-rom-hand-normal.mp4", names)

    def test_publish_preserves_original_and_refreshes_changed_video(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "hand/replay/rtx.mp4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"original MP4 payload")
            name = "videos/everyday-rom-hand-normal-and-strain.mp4"
            assets = {name: "hand/replay/rtx.mp4"}
            for payload in (b"original MP4 payload", b"revised MP4 payload"):
                source.write_bytes(payload)
                build_report.publish_videos(root, assets)
                self.assertEqual((root / name).read_bytes(), payload)
                self.assertEqual(source.read_bytes(), payload)
                self.assertNotEqual(source.stat().st_ino, (root / name).stat().st_ino)
                manifest = json.loads((root / "videos/manifest.json").read_text())
                self.assertEqual(manifest[name]["source"], "hand/replay/rtx.mp4")
                self.assertEqual(manifest[name]["sha256"], hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
