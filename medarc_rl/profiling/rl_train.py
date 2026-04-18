"""Entry point for line-profiling prime_rl.trainer.rl.train.train.

Use in place of `-m prime_rl.trainer.rl.train` when launching the trainer.
Results are printed to stdout at process exit and written to
`<output_dir>/line_profile.txt` if OUTPUT_DIR is set.

Usage (via torchrun):
    torchrun ... -m medarc_rl.profiling.rl_train @ trainer.toml
"""

import atexit
import io
import os

import prime_rl.trainer.rl.train as _train_module
from line_profiler import LineProfiler
from pydantic_config import cli
from prime_rl.configs.trainer import TrainerConfig


def main():
    profiler = LineProfiler()
    profiler.add_function(_train_module.train)
    _train_module.train = profiler(_train_module.train)

    config = cli(TrainerConfig)

    def _dump():
        stream = io.StringIO()
        profiler.print_stats(stream=stream, output_unit=1e-3, stripzeros=True)
        output = stream.getvalue()
        print(output)
        output_dir = os.environ.get("OUTPUT_DIR") or str(config.output_dir)
        with open(os.path.join(output_dir, "line_profile.txt"), "w") as f:
            f.write(output)

    atexit.register(_dump)

    _train_module.train(config)


if __name__ == "__main__":
    main()
