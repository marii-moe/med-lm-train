from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from medarc_rl.medarc_sweep import app, _flatten_wandb_config


runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _sweep_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "sweep.yaml"
    path.write_text(
        textwrap.dedent("""\
            method: grid
            metric:
              goal: maximize
              name: reward/mean
            parameters:
              trainer.optim.lr:
                values: [1.0e-5, 3.0e-5]
        """),
        encoding="utf-8",
    )
    return path


def _build_rl_config(tmp_path: Path, *, weight_broadcast_type: str = "filesystem") -> tuple[Path, Path]:
    base = _write(
        tmp_path / "rl_base.toml",
        """
        [trainer.model]
        cp = 1

        [orchestrator]

        [inference.parallel]
        tp = 1
        dp = 1
        """,
    )
    child = _write(
        tmp_path / "rl_child.toml",
        f"""
        [inference]
        gpu_memory_utilization = 0.45

        [weight_broadcast]
        type = "{weight_broadcast_type}"
        """,
    )
    return base, child


def _cli_args(config_paths: tuple[Path, ...], *args: str) -> list[str]:
    return [*[item for path in config_paths for item in ("--config", str(path))], *args]


def _invoke_rl(tmp_path: Path, *extra: str, weight_broadcast_type: str = "filesystem") -> tuple:
    """Invoke `medarc_sweep rl` with standard args and return (result, config_paths, sweep_yaml)."""
    sweep_yaml = _sweep_yaml(tmp_path)
    config_paths = _build_rl_config(tmp_path, weight_broadcast_type=weight_broadcast_type)
    output_dir = tmp_path / "runs"

    result = runner.invoke(
        app,
        [
            "rl",
            "--sweep-config",
            str(sweep_yaml),
            "--output-dir",
            str(output_dir),
            *_cli_args(config_paths),
            *extra,
        ],
    )
    return result, config_paths, sweep_yaml


# ---------------------------------------------------------------------------
# _flatten_wandb_config unit tests
# ---------------------------------------------------------------------------


def test_flatten_flat_config() -> None:
    assert _flatten_wandb_config({"lr": 1e-4, "batch_size": 32}) == {"lr": 1e-4, "batch_size": 32}


def test_flatten_nested_config() -> None:
    result = _flatten_wandb_config({"trainer": {"optim": {"lr": 1e-4}}, "max_steps": 100})
    assert result == {"trainer.optim.lr": 1e-4, "max_steps": 100}


def test_flatten_dotted_keys_passthrough() -> None:
    # wandb may keep dotted names flat depending on version
    result = _flatten_wandb_config({"trainer.optim.lr": 3e-5})
    assert result == {"trainer.optim.lr": 3e-5}


# ---------------------------------------------------------------------------
# CLI validation tests (no wandb network calls needed)
# ---------------------------------------------------------------------------


@patch("medarc_rl.medarc_sweep.wandb")
def test_rl_rejects_missing_config(mock_wandb: MagicMock, tmp_path: Path) -> None:
    sweep_yaml = _sweep_yaml(tmp_path)
    result = runner.invoke(app, ["rl", "--sweep-config", str(sweep_yaml), "--output-dir", str(tmp_path / "runs")])
    assert result.exit_code != 0
    mock_wandb.sweep.assert_not_called()


@patch("medarc_rl.medarc_sweep.wandb")
def test_rl_rejects_total_gpus_above_eight(mock_wandb: MagicMock, tmp_path: Path) -> None:
    result, _, _ = _invoke_rl(tmp_path, "--train-gpus", "4", "--infer-gpus", "5")
    assert result.exit_code != 0
    mock_wandb.sweep.assert_not_called()


@patch("medarc_rl.medarc_sweep.wandb")
def test_rl_rejects_single_gpu_nccl(mock_wandb: MagicMock, tmp_path: Path) -> None:
    result, _, _ = _invoke_rl(tmp_path, "--single-gpu", weight_broadcast_type="nccl")
    assert result.exit_code != 0
    assert "NCCL" in result.output
    mock_wandb.sweep.assert_not_called()


@patch("medarc_rl.medarc_sweep.wandb")
def test_rl_rejects_nonexistent_sweep_config(mock_wandb: MagicMock, tmp_path: Path) -> None:
    config_paths = _build_rl_config(tmp_path)
    result = runner.invoke(
        app,
        [
            "rl",
            "--sweep-config",
            str(tmp_path / "nonexistent.yaml"),
            "--output-dir",
            str(tmp_path / "runs"),
            *_cli_args(config_paths),
        ],
    )
    assert result.exit_code != 0
    mock_wandb.sweep.assert_not_called()


# ---------------------------------------------------------------------------
# Sweep creation / agent invocation
# ---------------------------------------------------------------------------


@patch("medarc_rl.medarc_sweep.wandb")
def test_rl_creates_sweep_and_launches_agent(mock_wandb: MagicMock, tmp_path: Path) -> None:
    mock_wandb.sweep.return_value = "sweep_abc"
    result, _, _ = _invoke_rl(tmp_path, "--train-gpus", "1", "--infer-gpus", "1")
    assert result.exit_code == 0, result.output
    mock_wandb.sweep.assert_called_once()
    mock_wandb.agent.assert_called_once()
    _, agent_kwargs = mock_wandb.agent.call_args
    assert agent_kwargs["count"] == 1  # default


@patch("medarc_rl.medarc_sweep.wandb")
def test_rl_joins_existing_sweep(mock_wandb: MagicMock, tmp_path: Path) -> None:
    result, _, _ = _invoke_rl(tmp_path, "--sweep-id", "existing_sweep_99", "--train-gpus", "1", "--infer-gpus", "1")
    assert result.exit_code == 0, result.output
    mock_wandb.sweep.assert_not_called()
    mock_wandb.agent.assert_called_once()
    args, _ = mock_wandb.agent.call_args
    assert args[0] == "existing_sweep_99"


@patch("medarc_rl.medarc_sweep.wandb")
def test_rl_count_forwarded_to_agent(mock_wandb: MagicMock, tmp_path: Path) -> None:
    mock_wandb.sweep.return_value = "sweep_count"
    result, _, _ = _invoke_rl(tmp_path, "--train-gpus", "1", "--infer-gpus", "1", "--count", "5")
    assert result.exit_code == 0, result.output
    _, agent_kwargs = mock_wandb.agent.call_args
    assert agent_kwargs["count"] == 5


@patch("medarc_rl.medarc_sweep.wandb")
def test_rl_sweep_config_passed_to_wandb_sweep(mock_wandb: MagicMock, tmp_path: Path) -> None:
    mock_wandb.sweep.return_value = "sweep_cfg_check"
    _invoke_rl(tmp_path, "--train-gpus", "1", "--infer-gpus", "1")
    _, sweep_kwargs = mock_wandb.sweep.call_args
    assert "parameters" in sweep_kwargs["sweep"]
    assert "method" in sweep_kwargs["sweep"]


# ---------------------------------------------------------------------------
# agent_fn command construction
# ---------------------------------------------------------------------------


def _run_agent_fn(
    mock_wandb: MagicMock,
    mock_subprocess: MagicMock,
    tmp_path: Path,
    *extra_cli: str,
    sweep_params: dict | None = None,
    run_id: str = "run_xyz",
    weight_broadcast_type: str = "filesystem",
) -> list[str]:
    """Invoke the CLI, extract the agent_fn, run it with mocked wandb state, return the subprocess cmd."""
    mock_wandb.sweep.return_value = "sweep_for_agent"
    mock_wandb.run.id = run_id
    mock_wandb.config.items.return_value = (sweep_params or {}).items()
    mock_subprocess.return_value = MagicMock(returncode=0)

    result, _, _ = _invoke_rl(tmp_path, *extra_cli, weight_broadcast_type=weight_broadcast_type)
    assert result.exit_code == 0, result.output

    _, agent_kwargs = mock_wandb.agent.call_args
    agent_fn = agent_kwargs["function"]
    agent_fn()

    assert mock_subprocess.called
    return mock_subprocess.call_args[0][0]


@patch("medarc_rl.medarc_sweep.subprocess.run")
@patch("medarc_rl.medarc_sweep.wandb")
def test_agent_fn_calls_medarc_train_rl(mock_wandb: MagicMock, mock_subprocess: MagicMock, tmp_path: Path) -> None:
    cmd = _run_agent_fn(mock_wandb, mock_subprocess, tmp_path, "--train-gpus", "1", "--infer-gpus", "1")
    assert cmd[0] == "medarc_train"
    assert cmd[1] == "rl"


@patch("medarc_rl.medarc_sweep.subprocess.run")
@patch("medarc_rl.medarc_sweep.wandb")
def test_agent_fn_output_dir_uses_run_id(mock_wandb: MagicMock, mock_subprocess: MagicMock, tmp_path: Path) -> None:
    cmd = _run_agent_fn(mock_wandb, mock_subprocess, tmp_path, "--train-gpus", "1", "--infer-gpus", "1", run_id="myrun")
    output_dir_idx = cmd.index("--output-dir") + 1
    assert cmd[output_dir_idx].endswith("myrun")


@patch("medarc_rl.medarc_sweep.subprocess.run")
@patch("medarc_rl.medarc_sweep.wandb")
def test_agent_fn_sweep_params_become_overrides(
    mock_wandb: MagicMock, mock_subprocess: MagicMock, tmp_path: Path
) -> None:
    cmd = _run_agent_fn(
        mock_wandb,
        mock_subprocess,
        tmp_path,
        "--train-gpus",
        "1",
        "--infer-gpus",
        "1",
        sweep_params={"trainer.optim.lr": 1e-5},
    )
    assert "--" in cmd
    sep = cmd.index("--")
    overrides = cmd[sep + 1 :]
    assert "--trainer.optim.lr" in overrides
    assert "1e-05" in overrides


@patch("medarc_rl.medarc_sweep.subprocess.run")
@patch("medarc_rl.medarc_sweep.wandb")
def test_agent_fn_passes_wandb_run_id_env(mock_wandb: MagicMock, mock_subprocess: MagicMock, tmp_path: Path) -> None:
    _run_agent_fn(mock_wandb, mock_subprocess, tmp_path, "--train-gpus", "1", "--infer-gpus", "1", run_id="run_env")
    env = mock_subprocess.call_args[1]["env"]
    assert env["WANDB_RUN_ID"] == "run_env"
    assert env["WANDB_RESUME"] == "allow"


@patch("medarc_rl.medarc_sweep.subprocess.run")
@patch("medarc_rl.medarc_sweep.wandb")
def test_agent_fn_single_gpu_flag(mock_wandb: MagicMock, mock_subprocess: MagicMock, tmp_path: Path) -> None:
    cmd = _run_agent_fn(mock_wandb, mock_subprocess, tmp_path, "--single-gpu")
    assert "--single-gpu" in cmd
    assert "--train-gpus" not in cmd
    assert "--infer-gpus" not in cmd


@patch("medarc_rl.medarc_sweep.subprocess.run")
@patch("medarc_rl.medarc_sweep.wandb")
def test_agent_fn_multi_gpu_flags(mock_wandb: MagicMock, mock_subprocess: MagicMock, tmp_path: Path) -> None:
    cmd = _run_agent_fn(mock_wandb, mock_subprocess, tmp_path, "--train-gpus", "2", "--infer-gpus", "2")
    assert "--train-gpus" in cmd
    assert cmd[cmd.index("--train-gpus") + 1] == "2"
    assert "--infer-gpus" in cmd
    assert cmd[cmd.index("--infer-gpus") + 1] == "2"


@patch("medarc_rl.medarc_sweep.subprocess.run")
@patch("medarc_rl.medarc_sweep.wandb")
def test_agent_fn_resume_flag(mock_wandb: MagicMock, mock_subprocess: MagicMock, tmp_path: Path) -> None:
    cmd = _run_agent_fn(mock_wandb, mock_subprocess, tmp_path, "--train-gpus", "1", "--infer-gpus", "1", "--resume")
    assert "--resume" in cmd


@patch("medarc_rl.medarc_sweep.subprocess.run")
@patch("medarc_rl.medarc_sweep.wandb")
def test_agent_fn_raises_on_nonzero_exit(mock_wandb: MagicMock, mock_subprocess: MagicMock, tmp_path: Path) -> None:
    mock_wandb.sweep.return_value = "sweep_fail"
    mock_wandb.run.id = "run_fail"
    mock_wandb.config.items.return_value = {}.items()
    mock_subprocess.return_value = MagicMock(returncode=1)

    result, _, _ = _invoke_rl(tmp_path, "--train-gpus", "1", "--infer-gpus", "1")
    assert result.exit_code == 0, result.output

    _, agent_kwargs = mock_wandb.agent.call_args
    agent_fn = agent_kwargs["function"]
    with pytest.raises(RuntimeError, match="failed with exit code 1"):
        agent_fn()


@patch("medarc_rl.medarc_sweep.subprocess.run")
@patch("medarc_rl.medarc_sweep.wandb")
def test_agent_fn_passthrough_overrides_appended(
    mock_wandb: MagicMock, mock_subprocess: MagicMock, tmp_path: Path
) -> None:
    """Fixed overrides passed after -- on the CLI appear after sweep params."""
    mock_wandb.sweep.return_value = "sweep_passthrough"
    mock_wandb.run.id = "run_pt"
    mock_wandb.config.items.return_value = {"trainer.optim.lr": 1e-5}.items()
    mock_subprocess.return_value = MagicMock(returncode=0)

    sweep_yaml = _sweep_yaml(tmp_path)
    config_paths = _build_rl_config(tmp_path)
    output_dir = tmp_path / "runs_pt"

    result = runner.invoke(
        app,
        [
            "rl",
            "--sweep-config",
            str(sweep_yaml),
            "--output-dir",
            str(output_dir),
            *_cli_args(config_paths),
            "--train-gpus",
            "1",
            "--infer-gpus",
            "1",
            "--",
            "--wandb.group",
            "my-group",
        ],
    )
    assert result.exit_code == 0, result.output

    _, agent_kwargs = mock_wandb.agent.call_args
    agent_fn = agent_kwargs["function"]
    agent_fn()

    cmd = mock_subprocess.call_args[0][0]
    sep = cmd.index("--")
    overrides = cmd[sep + 1 :]
    assert "--trainer.optim.lr" in overrides
    assert "--wandb.group" in overrides
    assert overrides.index("--wandb.group") > overrides.index("--trainer.optim.lr")
