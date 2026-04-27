from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import Mock

from pydantic_config import cli
from prime_rl.configs.rl import RLConfig

from medarc_rl.launchers import rl_local as launcher
from medarc_rl.launchers.rl_local import _configure_dynamic_local_ports, _process_env, _shared_node_gpu_ids
from medarc_rl.utils import create_job_cache_root


def _write(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _load_rl_config(path: Path) -> RLConfig:
    return cli(RLConfig, args=["@", str(path)])


def _base_rl_config(tmp_path: Path) -> RLConfig:
    return _load_rl_config(
        _write(
            tmp_path / "rl.toml",
            """
            [trainer.model]
            cp = 1

            [orchestrator]

            [inference.parallel]
            tp = 1
            dp = 1
            """,
        )
    )


def test_create_job_cache_root_uses_unique_directory_under_slurm_tmpdir(tmp_path: Path) -> None:
    first = create_job_cache_root("47069", str(tmp_path))
    second = create_job_cache_root("47069", str(tmp_path))

    assert first.parent == tmp_path
    assert second.parent == tmp_path
    assert first != second
    assert first.name.startswith("medarc-rl-47069-")
    assert second.name.startswith("medarc-rl-47069-")
    assert first.exists()
    assert second.exists()


def test_create_job_cache_root_falls_back_to_system_tmp_without_shared_medarc_dir(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("medarc_rl.utils.tempfile.gettempdir", lambda: str(tmp_path))

    cache_root = create_job_cache_root("job/with spaces", None)

    assert cache_root.parent == tmp_path
    assert cache_root.name.startswith("medarc-rl-job_with_spaces-")
    assert cache_root.exists()
    assert not (tmp_path / "medarc").exists()


def test_shared_node_gpu_ids_split_visible_devices(tmp_path: Path, monkeypatch) -> None:
    config = _base_rl_config(tmp_path)
    config.deployment.num_train_gpus = 2
    config.deployment.num_infer_gpus = 2
    config.deployment.num_teacher_gpus = 1

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7,8")
    monkeypatch.setenv("MEDARC_SINGLE_GPU", "0")

    visible, infer, trainer, teacher = _shared_node_gpu_ids(config)

    assert visible == ["4", "5", "6", "7", "8"]
    assert infer == ["4", "5"]
    assert trainer == ["6", "7"]
    assert teacher == ["8"]


def test_shared_node_gpu_ids_single_gpu_shares_trainer_and_inference(tmp_path: Path, monkeypatch) -> None:
    config = _base_rl_config(tmp_path)
    config.deployment.num_train_gpus = 1
    config.deployment.num_infer_gpus = 1

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("MEDARC_SINGLE_GPU", "1")

    _, infer, trainer, teacher = _shared_node_gpu_ids(config)

    assert infer == ["3"]
    assert trainer == ["3"]
    assert teacher == []


def test_configure_dynamic_local_ports_updates_inference_and_orchestrator(tmp_path: Path) -> None:
    config = _base_rl_config(tmp_path)

    infer_port, teacher_port, weight_broadcast_port, rdzv_port = _configure_dynamic_local_ports(config)

    assert config.inference is not None
    assert infer_port is not None
    assert teacher_port is None
    assert weight_broadcast_port is None
    assert rdzv_port > 0
    assert config.inference.server.host == "127.0.0.1"
    assert config.inference.server.port == infer_port
    assert config.orchestrator.client.base_url == [f"http://127.0.0.1:{infer_port}/v1"]


def test_process_env_uses_isolated_cache_dirs(tmp_path: Path) -> None:
    env = _process_env(
        {"BASE": "1"},
        cache_root=tmp_path,
        cache_name="train",
        cuda_visible_devices=["2", "4"],
        extra={"EXTRA": "yes"},
    )

    assert env["BASE"] == "1"
    assert env["EXTRA"] == "yes"
    assert env["CUDA_VISIBLE_DEVICES"] == "2,4"
    assert env["XDG_CACHE_HOME"] == str(tmp_path / "xdg_train")
    assert env["TRITON_CACHE_DIR"] == str(tmp_path / "triton_train")
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "torchinductor_train")
    assert (tmp_path / "xdg_train").exists()
    assert (tmp_path / "triton_train").exists()
    assert (tmp_path / "torchinductor_train").exists()


def test_rl_delegates_to_upstream_lifecycle_with_medarc_local_launcher(tmp_path: Path, monkeypatch) -> None:
    config = _base_rl_config(tmp_path)
    original_upstream_local = Mock(name="original_upstream_local")
    observed_local_launchers = []

    def lifecycle(config_arg: RLConfig) -> None:
        observed_local_launchers.append(fake_upstream.rl_local)

    fake_upstream = types.SimpleNamespace(rl_local=original_upstream_local, rl=lifecycle)

    import prime_rl.entrypoints

    monkeypatch.setattr(prime_rl.entrypoints, "rl", fake_upstream, raising=False)

    launcher.rl(config)

    assert observed_local_launchers == [launcher.rl_local]
    assert fake_upstream.rl_local is original_upstream_local
