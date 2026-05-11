# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""AVBD 3D Heavy Rope scene recreated with Newton VBD."""

from newton.examples.vbd._avbd_common import AvbdSceneExample, main


class Example(AvbdSceneExample):
    scene_name = "heavy_rope"


if __name__ == "__main__":
    main(Example)
