# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.quiet = True

from newton.examples.basic.example_basic_reduced_elastic_chair_stick_slip import Example as ChairExample
from newton.examples.basic.example_basic_reduced_elastic_dipper import Example as DipperExample
from newton.examples.basic.example_basic_reduced_elastic_wall_contact import Example as WallExample
from newton.viewer import ViewerNull

EXAMPLES = {
    "wall": (WallExample, 60),
    "dipper": (DipperExample, 60),
    "chair": (ChairExample, 60),
}


class FastReducedElasticRepresentativeExamples:
    """Benchmark representative reduced elastic examples without rendering."""

    params = (tuple(EXAMPLES),)
    param_names = ("case",)
    repeat = 3
    number = 1

    def setup(self, case):
        example_cls, frame_count = EXAMPLES[case]
        self.frame_count = frame_count
        self.example = example_cls(ViewerNull(num_frames=frame_count), None)

        for _ in range(5):
            self.example.step()
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_step(self, case):
        del case
        for _ in range(self.frame_count):
            self.example.step()
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_final_modal_solve_residual_ratio(self, case):
        del case
        for _ in range(self.frame_count):
            self.example.step()
        wp.synchronize_device()
        return self.example.final_modal_solve_residual_ratio


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastReducedElasticRepresentativeExamples": FastReducedElasticRepresentativeExamples,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--bench", default=None, action="append", choices=benchmark_list.keys(), help="Run a single benchmark."
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
