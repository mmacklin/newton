# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render any saved scene.json/scene.npz export in a fresh owned directory."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from . import render_strain_replay
    from .strain_publication import stage_normal
else:
    import render_strain_replay
    from strain_publication import stage_normal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Saved scene directory, including original metrics.json")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory outside source; never overwrite inputs"
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--frame", type=int, help="Diagnostic still only; not physical qualification")
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.samples) <= 0:
        parser.error("Width, height and sample count must be positive")
    if args.frame is None and (args.width % 2 or args.height % 2):
        parser.error("Full yuv420p video requires even dimensions")
    for name in ("scene.json", "scene.npz", "metrics.json"):
        if not (args.source / name).is_file():
            parser.error(f"Missing saved input: {name}")
    stage_normal(args.source, args.output)
    render_args = [
        str(args.output),
        "--normal",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--samples",
        str(args.samples),
    ]
    if args.frame is not None:
        render_args.extend(("--frame", str(args.frame)))
    render_strain_replay.main(render_args)


if __name__ == "__main__":
    main()
