from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
import wandb
import yaml
from pydantic import ValidationError
from typer import Option

from prime_rl.configs.rl import RLConfig

from medarc_rl.utils import TYPER_PASSTHROUGH_CONTEXT, _load_settings_from_toml, extra_config_args


app = typer.Typer(
    add_completion=False,
    help=(
        "Run W&B hyperparameter sweeps for PRIME-RL training locally. "
        "Pass fixed PRIME-RL config overrides as extra flags after `--`, "
        "e.g. `-- --wandb.project my-proj`."
    ),
)


@app.callback()
def _callback() -> None:
    """W&B sweep runner for PRIME-RL local training."""


def _flatten_wandb_config(config: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a (potentially nested) wandb config object to dotted keys."""
    result: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_wandb_config(value, full_key))
        else:
            result[full_key] = value
    return result


def _make_rl_agent_fn(
    config_tomls: list[Path],
    base_output_dir: Path,
    *,
    train_gpus: int,
    infer_gpus: int,
    single_gpu: bool,
    resume: bool,
    passthrough: list[str],
) -> Any:
    """Return a callable suitable for wandb.agent(function=...) that runs medarc_train rl."""

    def agent_fn() -> None:
        run_id = wandb.run.id
        output_dir = base_output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Translate sweep parameters to PRIME-RL dotted CLI overrides.
        sweep_overrides: list[str] = []
        for key, value in _flatten_wandb_config(wandb.config).items():
            sweep_overrides += [f"--{key}", str(value)]

        cmd = [
            "medarc_train",
            "rl",
            *[item for path in config_tomls for item in ("--config", str(path))],
            "--output-dir",
            str(output_dir),
        ]
        if single_gpu:
            cmd.append("--single-gpu")
        else:
            cmd += ["--train-gpus", str(train_gpus), "--infer-gpus", str(infer_gpus)]
        if resume:
            cmd.append("--resume")

        # Fixed passthrough overrides come after sweep overrides so they can win.
        all_overrides = sweep_overrides + passthrough
        if all_overrides:
            cmd += ["--", *all_overrides]

        typer.echo(f"[sweep] Run {run_id}: {' '.join(cmd)}")

        # Pass WANDB_RUN_ID so PRIME-RL's internal wandb.init() joins this run.
        env = {**os.environ, "WANDB_RUN_ID": run_id, "WANDB_RESUME": "allow"}
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Run {run_id} failed with exit code {result.returncode}")

    return agent_fn


@app.command(
    context_settings=TYPER_PASSTHROUGH_CONTEXT,
    help=(
        "Run a W&B RL sweep locally. "
        "Each trial calls `medarc_train rl` with parameters drawn from the sweep. "
        "Fixed PRIME-RL overrides can be passed after `--`, e.g. `-- --wandb.project my-proj`."
    ),
)
def rl(
    ctx: typer.Context,
    sweep_config: Annotated[Path, Option("--sweep-config", exists=True, file_okay=True, dir_okay=False, help="Path to W&B sweep YAML defining the search space and strategy.")],
    output_dir: Annotated[Path, Option("--output-dir", file_okay=False, dir_okay=True, help="Base output directory. Each trial writes to a subdirectory named by its W&B run ID.")],
    config: Annotated[list[Path] | None, Option("--config", "--config-toml", help="One or more PRIME-RL RL TOMLs. Repeat `--config` to layer files with later files overriding earlier ones.")] = None,
    train_gpus: Annotated[int, Option("--train-gpus", min=1, max=4, help="Number of GPUs for training per trial.")] = 1,
    infer_gpus: Annotated[int, Option("--infer-gpus", min=1, max=7, help="Number of GPUs for inference per trial.")] = 1,
    single_gpu: Annotated[bool, Option("--single-gpu", help="Share a single GPU between trainer and inference.")] = False,
    count: Annotated[int, Option("--count", min=1, help="Number of sweep trials to run sequentially on this agent.")] = 1,
    sweep_id: Annotated[str | None, Option("--sweep-id", help="Existing W&B sweep ID to join instead of creating a new sweep.")] = None,
    resume: Annotated[bool, Option("--resume/--no-resume", help="Resume each trial from its latest checkpoint.")] = False,
) -> None:  # fmt: skip
    config_tomls = list(config or [])
    if not config_tomls:
        raise typer.BadParameter("Missing config path. Pass one or more --config values.", param_hint="--config")

    resolved_config_paths = [p.expanduser().resolve() for p in config_tomls]
    output_dir = output_dir.expanduser().resolve()

    train_gpus = 1 if single_gpu else train_gpus
    infer_gpus = 1 if single_gpu else infer_gpus
    gpus = 1 if single_gpu else (train_gpus + infer_gpus)

    if not single_gpu and gpus < 2:
        raise typer.BadParameter(
            f"Total GPUs must be at least 2, got train_gpus ({train_gpus}) + infer_gpus ({infer_gpus}) = {gpus}.",
            param_hint="--train-gpus/--infer-gpus",
        )
    if gpus > 8:
        raise typer.BadParameter(
            f"Total GPUs must be at most 8, got train_gpus ({train_gpus}) + infer_gpus ({infer_gpus}) = {gpus}.",
            param_hint="--train-gpus/--infer-gpus",
        )

    passthrough = extra_config_args(ctx, positional_count=0)

    # Validate the base config early so the user gets errors before the sweep is created.
    # Passthrough overrides are intentionally excluded here — they may include PRIME-RL CLI flags
    # that aren't Pydantic fields (e.g. --wandb.group) and will be validated per-trial at runtime.
    try:
        base_config = _load_settings_from_toml(
            RLConfig,
            resolved_config_paths,
            output_dir=output_dir / "_validate",
            deployment={"type": "single_node", "num_train_gpus": train_gpus, "num_infer_gpus": infer_gpus},
        )
    except (ValidationError, typer.BadParameter) as e:
        raise typer.BadParameter(str(e), param_hint="--config") from e

    if single_gpu and getattr(base_config.trainer.weight_broadcast, "type", None) == "nccl":
        raise typer.BadParameter(
            "--single-gpu does not support NCCL weight broadcast. Use filesystem broadcast or 2+ GPUs.",
            param_hint="--config/--single-gpu",
        )

    sweep_cfg = yaml.safe_load(sweep_config.read_text(encoding="utf-8"))

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=sweep_cfg)
        typer.echo(f"Created sweep: {sweep_id}")
    else:
        typer.echo(f"Joining sweep: {sweep_id}")

    output_dir.mkdir(parents=True, exist_ok=True)

    agent_fn = _make_rl_agent_fn(
        resolved_config_paths,
        output_dir,
        train_gpus=train_gpus,
        infer_gpus=infer_gpus,
        single_gpu=single_gpu,
        resume=resume,
        passthrough=passthrough,
    )
    wandb.agent(sweep_id, function=agent_fn, count=count)


if __name__ == "__main__":
    app()
