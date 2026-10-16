# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU video reuse, concat and publication gates; no renderer required."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__:
    from . import strain_video
    from .strain_adapters import digest
    from .strain_replay import write_json
    from .strain_video import concatenate, decoded_frame_hashes, probe_video, publication_entry, validate_video_pair
else:
    import strain_video
    from strain_adapters import digest
    from strain_replay import write_json
    from strain_video import concatenate, decoded_frame_hashes, probe_video, publication_entry, validate_video_pair


def video_probe(frames=2):
    return {
        "frames": frames,
        "fps": 60.0,
        "duration_s": frames / 60,
        "uniform_pts": True,
        "stream": {
            "codec_name": "h264",
            "profile": "High",
            "width": 320,
            "height": 240,
            "pix_fmt": "yuv420p",
            "r_frame_rate": "60/1",
            "time_base": "1/15360",
            "extradata_hash": "SHA256:abc",
        },
    }


class ProbeTests(unittest.TestCase):
    def raw_probe(self):
        stream = dict(
            video_probe()["stream"], codec_type="video", avg_frame_rate="60/1", nb_read_frames="2", duration="0.033333"
        )
        return {
            "streams": [stream],
            "frames": [{"best_effort_timestamp_time": "0.000000"}, {"best_effort_timestamp_time": "0.016667"}],
        }

    def test_full_frame_probe(self):
        with patch.object(
            strain_video.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(self.raw_probe()))
        ):
            self.assertEqual(probe_video(Path("video.mp4"))["frames"], 2)

    def test_wrong_duration_nonuniform_timestamps_and_audio_rejected(self):
        for kind in ("duration", "pts", "audio"):
            raw = self.raw_probe()
            if kind == "duration":
                raw["streams"][0]["duration"] = ".05"
            elif kind == "pts":
                raw["frames"][1]["best_effort_timestamp_time"] = ".02"
            else:
                raw["streams"].append({"codec_type": "audio"})
            with (
                self.subTest(kind=kind),
                patch.object(strain_video.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(raw))),
            ):
                with self.assertRaises(ValueError):
                    probe_video(Path("video.mp4"))

    def test_rgb_frame_hash_parser(self):
        hashes = ["a" * 64, "b" * 64]
        text = "#format: frame checksums\n" + "\n".join(
            f"0, {i}, {i}, 1, 230400, {value}" for i, value in enumerate(hashes)
        )
        with patch.object(strain_video.subprocess, "run", return_value=SimpleNamespace(stdout=text)) as runner:
            self.assertEqual(decoded_frame_hashes(Path("video.mp4")), hashes)
            self.assertIn("rgb24", runner.call_args.args[0])
            self.assertIn("-vsync", runner.call_args.args[0])

    def test_copy_requires_matching_codec_headers(self):
        a, b = video_probe(), video_probe()
        self.assertEqual(validate_video_pair(a, b, 2, 60), "copy")
        b["stream"]["extradata_hash"] = "different"
        self.assertEqual(validate_video_pair(a, b, 2, 60), "lossless")
        b["stream"]["width"] = 640
        with self.assertRaises(ValueError):
            validate_video_pair(a, b, 2, 60)

    def test_fps_count_and_pixel_format_mismatch_rejected(self):
        for key, value in (("frames", 3), ("fps", 30), ("duration_s", 3)):
            a, b = video_probe(), video_probe()
            b[key] = value
            with self.assertRaises(ValueError):
                validate_video_pair(a, b, 2, 60)
        a, b = video_probe(), video_probe()
        b["stream"]["pix_fmt"] = "yuv444p"
        with self.assertRaises(ValueError):
            validate_video_pair(a, b, 2, 60)


class ConcatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.normal, self.strain = self.root / "normal.mp4", self.root / "strain.mp4"
        self.normal.write_bytes(b"mock-normal")
        self.strain.write_bytes(b"mock-strain")
        self.normal_frames, self.strain_frames = ["a" * 64, "b" * 64], ["c" * 64, "d" * 64]
        write_json(
            self.root / "decoded-frames.json",
            {"normal_rgb24_sha256": self.normal_frames, "strain_rgb24_sha256": self.strain_frames},
        )
        self.plan = {
            "schema_version": 1,
            "status": "verified_ready_for_concat",
            "object": "headphones",
            "normal_video": str(self.normal),
            "strain_video": str(self.strain),
            "normal_probe": video_probe(),
            "strain_probe": video_probe(),
            "inputs_sha256": {str(self.normal): digest(self.normal), str(self.strain): digest(self.strain)},
            "decoded_frames_sha256": digest(self.root / "decoded-frames.json"),
            "schedule": {"frames_per_phase": 2, "fps": 60},
            "scale": {"min": 0, "max": 0.03},
            "camera": {},
            "field_verification": {"accepted_headphones_reused": True},
        }
        write_json(self.root / "video-plan.json", self.plan)

    def encoder(self, command, **kwargs):
        Path(command[-1]).write_bytes(b"mock-concatenation")
        return SimpleNamespace(returncode=0)

    def run_concat(self, frames=None, probe=None, mode=None):
        with (
            patch.object(strain_video.subprocess, "run", side_effect=self.encoder),
            patch.object(strain_video, "probe_video", return_value=probe or video_probe(4)),
            patch.object(
                strain_video, "decoded_frame_hashes", return_value=frames or self.normal_frames + self.strain_frames
            ),
        ):
            return concatenate(self.root, mode)

    def test_copy_concat_exact_normal_appearance_and_publication_formatter(self):
        result = self.run_concat()
        self.assertEqual(result["mode"], "copy")
        self.assertTrue((self.root / "rtx.mp4").exists())
        self.assertFalse((self.root / "rtx.partial.mp4").exists())
        entry = publication_entry(self.root)
        self.assertEqual(entry["primary_video"], "headphones/replay/rtx.mp4")
        self.assertEqual(entry["frames"], 4)
        self.assertEqual(entry["phase_boundary_s"], 2 / 60)
        self.assertFalse(entry["new_physics_run"])
        self.assertIn("Separate from simulation timing", entry["performance_provenance"])
        self.assertIn("does not establish which runtime produced", entry["performance_provenance"])
        self.assertIn("physical validation remains as recorded in metrics.json", entry["trajectory_provenance"])

    def test_lossless_reencode_uses_original_pixel_format(self):
        self.plan["strain_probe"]["stream"]["extradata_hash"] = "different"
        write_json(self.root / "video-plan.json", self.plan)
        result = self.run_concat()
        self.assertEqual(result["mode"], "lossless")
        command = result["ffmpeg_command"]
        self.assertEqual(command[command.index("-qp") + 1], "0")
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")

    def test_incompatible_stream_copy_rejected(self):
        self.plan["strain_probe"]["stream"]["extradata_hash"] = "different"
        write_json(self.root / "video-plan.json", self.plan)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.run_concat(mode="copy")

    def test_changed_source_or_frame_audit_rejected(self):
        self.normal.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "input changed"):
            self.run_concat()

    def test_changed_normal_appearance_never_becomes_publishable(self):
        with self.assertRaisesRegex(ValueError, "appearance"):
            self.run_concat(frames=["e" * 64, *self.normal_frames[1:], *self.strain_frames])
        self.assertFalse((self.root / "rtx.mp4").exists())
        self.assertFalse((self.root / "publication.json").exists())

    def test_wrong_output_duration_is_rejected(self):
        probe = copy.deepcopy(video_probe(4))
        probe["duration_s"] = 5
        with self.assertRaisesRegex(ValueError, "duration"):
            self.run_concat(probe=probe)

    def test_publication_rejects_corrupted_final_video(self):
        self.run_concat()
        (self.root / "rtx.mp4").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "video hash"):
            publication_entry(self.root)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe not installed")
class MediaIntegrationTests(unittest.TestCase):
    def test_real_stream_copy_and_lossless_reencode_preserve_decoded_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            videos = []
            for phase, color in (("normal", "green"), ("strain", "blue")):
                video = root / f"{phase}.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        f"color=c={color}:s=64x48:r=60",
                        "-frames:v",
                        "4",
                        "-c:v",
                        "libx264",
                        "-threads",
                        "1",
                        "-crf",
                        "17",
                        "-pix_fmt",
                        "yuv420p",
                        str(video),
                    ],
                    check=True,
                )
                videos.append(video)
            probes = [probe_video(video) for video in videos]
            pixels = [decoded_frame_hashes(video) for video in videos]
            for mode in ("copy", "lossless"):
                with self.subTest(mode=mode):
                    target = root / mode
                    target.mkdir()
                    write_json(
                        target / "decoded-frames.json",
                        {"normal_rgb24_sha256": pixels[0], "strain_rgb24_sha256": pixels[1]},
                    )
                    plan = {
                        "schema_version": 1,
                        "status": "verified_ready_for_concat",
                        "object": "fixture",
                        "normal_video": str(videos[0]),
                        "strain_video": str(videos[1]),
                        "normal_probe": probes[0],
                        "strain_probe": probes[1],
                        "inputs_sha256": {str(video): digest(video) for video in videos},
                        "decoded_frames_sha256": digest(target / "decoded-frames.json"),
                        "schedule": {"frames_per_phase": 4, "fps": 60},
                        "scale": {"min": 0, "max": 0.03},
                        "camera": {},
                        "field_verification": {},
                    }
                    write_json(target / "video-plan.json", plan)
                    audit = concatenate(target, mode)
                    self.assertEqual(audit["normal_decoded_frames_identical"], 4)
                    self.assertEqual(audit["strain_decoded_frames_identical"], 4)
                    self.assertEqual(audit["probe"]["frames"], 8)


if __name__ == "__main__":
    unittest.main()
