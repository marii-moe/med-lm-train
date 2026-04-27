# Modified from PrimeIntellect's rl_local() in PRIME-RL.
# Apache License 2.0. Copyright PrimeIntellect AI.
# Source: prime-rl/src/prime_rl/entrypoints/rl.py at the pinned submodule revision.
#
# Modifications: shared-node GPU slicing, dynamic local ports, and per-process cache isolation.

from __future__ import annotations

import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread

import tomli_w
from pydantic_config import cli
from prime_rl.configs.rl import RLConfig
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import get_log_dir
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process
from prime_rl.utils.utils import get_free_port

from medarc_rl.utils import create_job_cache_root

TRAINER_TOML = "trainer.toml"
ORCHESTRATOR_TOML = "orchestrator.toml"
INFERENCE_TOML = "inference.toml"
TEACHER_INFERENCE_TOML = "teacher_inference.toml"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_visible_gpus() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is not set; cannot determine shared-node GPU allocation.")
    gpu_ids = [part.strip() for part in raw.split(",") if part.strip()]
    if not gpu_ids:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES is invalid: {raw!r}")
    return gpu_ids


def _build_cache_env(cache_root: Path, name: str) -> dict[str, str]:
    xdg_cache = cache_root / f"xdg_{name}"
    triton_cache = cache_root / f"triton_{name}"
    torchinductor_cache = cache_root / f"torchinductor_{name}"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)
    torchinductor_cache.mkdir(parents=True, exist_ok=True)
    return {
        "XDG_CACHE_HOME": str(xdg_cache),
        "TRITON_CACHE_DIR": str(triton_cache),
        "TORCHINDUCTOR_CACHE_DIR": str(torchinductor_cache),
    }


def _process_env(
    base_env: dict[str, str],
    *,
    cache_root: Path,
    cache_name: str,
    cuda_visible_devices: list[str] | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    env = {
        **base_env,
        **_build_cache_env(cache_root, cache_name),
        "LOGURU_FORCE_COLORS": "1",
    }
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)
    if extra:
        env.update(extra)
    return env


def write_subconfigs(config: RLConfig, output_dir: Path) -> None:
    """Write resolved subconfigs to disk as TOML files without importing PRIME-RL's heavy launcher module."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / TRAINER_TOML, "wb") as f:
        tomli_w.dump(config.trainer.model_dump(exclude_none=True, mode="json"), f)

    with open(output_dir / ORCHESTRATOR_TOML, "wb") as f:
        tomli_w.dump(config.orchestrator.model_dump(exclude_none=True, mode="json"), f)

    if config.inference is not None:
        exclude_inference = {"deployment", "slurm", "output_dir", "dry_run"}
        with open(output_dir / INFERENCE_TOML, "wb") as f:
            tomli_w.dump(config.inference.model_dump(exclude=exclude_inference, exclude_none=True, mode="json"), f)

    teacher_inference = getattr(config, "teacher_inference", None)
    if teacher_inference is not None:
        with open(output_dir / TEACHER_INFERENCE_TOML, "wb") as f:
            tomli_w.dump(teacher_inference.model_dump(exclude_none=True, mode="json"), f)


def _shared_node_gpu_ids(config: RLConfig) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return visible, inference, trainer, and teacher GPU ID slices for the current shared-node allocation."""
    assert config.deployment.type == "single_node"

    visible_gpu_ids = _parse_visible_gpus()
    single_gpu_mode = _env_flag("MEDARC_SINGLE_GPU")
    num_infer_gpus = config.deployment.num_infer_gpus if config.inference is not None else 0
    num_train_gpus = config.deployment.num_train_gpus
    num_teacher_gpus = config.deployment.num_teacher_gpus or 0

    if single_gpu_mode:
        if num_teacher_gpus:
            raise RuntimeError("MEDARC_SINGLE_GPU=1 does not support teacher inference GPUs.")
        if len(visible_gpu_ids) != 1:
            raise RuntimeError(
                "MEDARC_SINGLE_GPU=1 requires exactly one visible GPU, got "
                f"{len(visible_gpu_ids)} (CUDA_VISIBLE_DEVICES={','.join(visible_gpu_ids)!r})"
            )
        if num_train_gpus != 1:
            raise RuntimeError(f"MEDARC_SINGLE_GPU=1 requires num_train_gpus=1, got {num_train_gpus}")
        infer_gpu_ids = visible_gpu_ids[:1] if config.inference is not None else []
        trainer_gpu_ids = visible_gpu_ids[:1]
        teacher_gpu_ids: list[str] = []
        return visible_gpu_ids, infer_gpu_ids, trainer_gpu_ids, teacher_gpu_ids

    expected_visible = num_train_gpus + num_infer_gpus + num_teacher_gpus
    if len(visible_gpu_ids) != expected_visible:
        raise RuntimeError(
            f"Expected {expected_visible} visible GPUs "
            f"(train={num_train_gpus}, infer={num_infer_gpus}, teacher={num_teacher_gpus}), got "
            f"{len(visible_gpu_ids)} (CUDA_VISIBLE_DEVICES={','.join(visible_gpu_ids)!r})"
        )

    offset = 0
    infer_gpu_ids = visible_gpu_ids[offset : offset + num_infer_gpus]
    offset += num_infer_gpus
    trainer_gpu_ids = visible_gpu_ids[offset : offset + num_train_gpus]
    offset += num_train_gpus
    teacher_gpu_ids = visible_gpu_ids[offset : offset + num_teacher_gpus]

    if len(trainer_gpu_ids) != num_train_gpus:
        raise RuntimeError(f"Trainer GPU slice mismatch: expected {num_train_gpus}, got {len(trainer_gpu_ids)}")
    if config.inference is not None and len(infer_gpu_ids) != num_infer_gpus:
        raise RuntimeError(f"Inference GPU slice mismatch: expected {num_infer_gpus}, got {len(infer_gpu_ids)}")
    if len(teacher_gpu_ids) != num_teacher_gpus:
        raise RuntimeError(f"Teacher GPU slice mismatch: expected {num_teacher_gpus}, got {len(teacher_gpu_ids)}")

    return visible_gpu_ids, infer_gpu_ids, trainer_gpu_ids, teacher_gpu_ids


def _configure_dynamic_local_ports(config: RLConfig) -> tuple[int | None, int | None, int | None, int]:
    """Allocate shared-node-safe ports and write them into resolved subconfigs before launch."""
    infer_port = get_free_port() if config.inference is not None else None
    teacher_port = get_free_port() if config.teacher_inference is not None else None
    weight_broadcast_is_nccl = getattr(config.trainer.weight_broadcast, "type", None) == "nccl"
    weight_broadcast_port = get_free_port() if weight_broadcast_is_nccl else None
    rdzv_port = get_free_port()

    if config.inference is not None and infer_port is not None:
        config.inference.server.host = "127.0.0.1"
        config.inference.server.port = infer_port
        if not config.orchestrator.client.is_elastic:
            config.orchestrator.client.base_url = [f"http://127.0.0.1:{infer_port}/v1"]

    if config.teacher_inference is not None and teacher_port is not None:
        config.teacher_inference.server.host = "127.0.0.1"
        config.teacher_inference.server.port = teacher_port
        if config.orchestrator.teacher_model is not None:
            config.orchestrator.teacher_model.client.base_url = [f"http://127.0.0.1:{teacher_port}/v1"]

    if weight_broadcast_port is not None:
        for weight_broadcast in (config.trainer.weight_broadcast, config.orchestrator.weight_broadcast):
            if getattr(weight_broadcast, "type", None) == "nccl":
                weight_broadcast.host = "127.0.0.1"
                weight_broadcast.port = weight_broadcast_port

    return infer_port, teacher_port, weight_broadcast_port, rdzv_port


def rl_local(config: RLConfig) -> None:
    assert config.deployment.type == "single_node"

    logger = setup_logger(
        config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=config.log.json_logging,
    )

    config_dir = config.output_dir / "configs"
    if config.dry_run:
        write_subconfigs(config, config_dir)
        logger.info(f"Wrote subconfigs to {config_dir}")
        logger.success("Dry run complete. To start an RL run locally, remove --dry-run from your command.")
        return

    visible_gpu_ids, infer_gpu_ids, trainer_gpu_ids, teacher_gpu_ids = _shared_node_gpu_ids(config)
    infer_port, teacher_port, weight_broadcast_port, rdzv_port = _configure_dynamic_local_ports(config)
    write_subconfigs(config, config_dir)
    logger.info(f"Wrote subconfigs to {config_dir}")

    start_command = sys.argv
    logger.info("Starting RL run")
    logger.debug(f"RL start command: {' '.join(start_command)}")
    logger.info(
        "GPU allocation: "
        f"infer={','.join(infer_gpu_ids) if infer_gpu_ids else '-'} "
        f"trainer={','.join(trainer_gpu_ids)} "
        f"teacher={','.join(teacher_gpu_ids) if teacher_gpu_ids else '-'} "
        f"(visible={','.join(visible_gpu_ids)}, single_gpu={_env_flag('MEDARC_SINGLE_GPU')})"
    )
    if infer_port is not None:
        logger.info(f"Selected inference port: {infer_port}")
    if teacher_port is not None:
        logger.info(f"Selected teacher inference port: {teacher_port}")
    if weight_broadcast_port is not None:
        logger.info(f"Selected weight broadcast port: {weight_broadcast_port}")
    logger.info(f"Selected torchrun rendezvous port: {rdzv_port}")

    wandb_shared_env: dict[str, str] = {}
    if config.wandb and config.wandb.shared:
        wandb_shared_env["WANDB_SHARED_MODE"] = "1"
        wandb_shared_env["WANDB_SHARED_RUN_ID"] = os.environ.get("WANDB_SHARED_RUN_ID", uuid.uuid4().hex)

    log_dir = get_log_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    cache_root = create_job_cache_root(slurm_job_id, slurm_tmpdir)

    base_env = os.environ.copy()
    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    trainer_process: Popen | None = None
    orchestrator_process: Popen | None = None

    def sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM, terminating all processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        if config.inference:
            inference_cmd = ["inference", "@", (config_dir / INFERENCE_TOML).as_posix()]
            logger.info(f"Starting inference on GPU(s) {' '.join(infer_gpu_ids)}")
            logger.debug(f"Inference start command: {' '.join(inference_cmd)}")
            with (log_dir / "inference.log").open("w") as log_file:
                inference_process = Popen(
                    inference_cmd,
                    env=_process_env(
                        base_env,
                        cache_root=cache_root,
                        cache_name="infer",
                        cuda_visible_devices=infer_gpu_ids,
                        extra={
                            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                        },
                    ),
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(inference_process)

            stop_event = Event()
            stop_events["inference"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(inference_process, stop_event, error_queue, "inference"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        else:
            if config.orchestrator.teacher_rollout_model is None:
                logger.warning(
                    "No inference config specified, skipping starting inference server. "
                    "Make sure your inference server is running."
                )
            else:
                logger.info(
                    "No inference config specified, using orchestrator.teacher_rollout_model for rollout generation."
                )

        if config.teacher_inference:
            if not teacher_gpu_ids:
                raise ValueError(
                    "teacher_inference is configured but deployment.num_teacher_gpus is not set. "
                    "Either set deployment.num_teacher_gpus to start a teacher inference server, "
                    "or omit teacher_inference and configure orchestrator.teacher_model to use an existing server."
                )

            teacher_inference_cmd = ["inference", "@", (config_dir / TEACHER_INFERENCE_TOML).as_posix()]
            logger.info(f"Starting teacher inference process on GPU(s) {' '.join(teacher_gpu_ids)}")
            logger.debug(f"Teacher inference start command: {' '.join(teacher_inference_cmd)}")
            with (log_dir / "teacher_inference.log").open("w") as log_file:
                teacher_inference_process = Popen(
                    teacher_inference_cmd,
                    env=_process_env(
                        base_env,
                        cache_root=cache_root,
                        cache_name="teacher_infer",
                        cuda_visible_devices=teacher_gpu_ids,
                        extra={
                            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                        },
                    ),
                    stdout=log_file,
                    stderr=log_file,
                )
            processes.append(teacher_inference_process)

            stop_event = Event()
            stop_events["teacher_inference"] = stop_event
            monitor_thread = Thread(
                target=monitor_process,
                args=(teacher_inference_process, stop_event, error_queue, "teacher_inference"),
                daemon=True,
            )
            monitor_thread.start()
            monitor_threads.append(monitor_thread)
        elif (
            config.trainer.loss.type == "default" and config.trainer.loss.teacher_tau > 0
        ) or config.orchestrator.teacher_model:
            logger.warning(
                "No teacher_inference config specified, skipping starting teacher inference server. "
                "Is your teacher inference server running? Make sure orchestrator.teacher_model is configured."
            )

        orchestrator_cmd = [
            "orchestrator",
            "@",
            (config_dir / ORCHESTRATOR_TOML).as_posix(),
        ]
        logger.info("Starting orchestrator process")
        logger.debug(f"Orchestrator start command: {' '.join(orchestrator_cmd)}")
        with (log_dir / "orchestrator.log").open("w") as log_file:
            orchestrator_process = Popen(
                orchestrator_cmd,
                stdout=log_file,
                stderr=log_file,
                env=_process_env(
                    base_env,
                    cache_root=cache_root,
                    cache_name="orch",
                    extra={
                        **wandb_shared_env,
                        "WANDB_SHARED_LABEL": "orchestrator",
                        "WANDB_PROGRAM": "medarc_rl.launchers.rl_local",
                        "WANDB_ARGS": json.dumps(start_command),
                    },
                ),
            )
        processes.append(orchestrator_process)

        stop_event = Event()
        stop_events["orchestrator"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(orchestrator_process, stop_event, error_queue, "orchestrator"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        trainer_cmd = [
            "torchrun",
            "--role=trainer",
            f"--rdzv-endpoint=localhost:{rdzv_port}",
            f"--rdzv-id={uuid.uuid4().hex}",
            f"--log-dir={log_dir / 'trainer' / 'torchrun'}",
            f"--local-ranks-filter={','.join(map(str, config.trainer.log.ranks_filter))}",
            "--redirect=3",
            "--tee=3",
            f"--nproc-per-node={len(trainer_gpu_ids)}",
            "-m",
            "prime_rl.trainer.rl.train",
            "@",
            (config_dir / TRAINER_TOML).as_posix(),
        ]
        logger.info(f"Starting trainer on GPU(s) {' '.join(trainer_gpu_ids)}")
        logger.debug(f"Trainer start command: {' '.join(trainer_cmd)}")
        with (log_dir / "trainer.log").open("w") as log_file:
            trainer_process = Popen(
                trainer_cmd,
                env=_process_env(
                    base_env,
                    cache_root=cache_root,
                    cache_name="train",
                    cuda_visible_devices=trainer_gpu_ids,
                    extra={
                        **wandb_shared_env,
                        "WANDB_SHARED_LABEL": "trainer",
                        "PYTHONUNBUFFERED": "1",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        "WANDB_PROGRAM": "medarc_rl.launchers.rl_local",
                        "WANDB_ARGS": json.dumps(start_command),
                    },
                ),
                stdout=log_file,
                stderr=log_file,
            )
        processes.append(trainer_process)

        stop_event = Event()
        stop_events["trainer"] = stop_event
        monitor_thread = Thread(
            target=monitor_process,
            args=(trainer_process, stop_event, error_queue, "trainer"),
            daemon=True,
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)

        logger.success("Startup complete. Showing trainer logs...")
        tail_process = Popen(
            f"tail -F '{log_dir / 'trainer.log'}' | sed -u 's/^\\[[a-zA-Z]*[0-9]*\\]://'",
            shell=True,
        )
        processes.append(tail_process)

        while not (stop_events["orchestrator"].is_set() and stop_events["trainer"].is_set()):
            if error_queue:
                logger.error(f"Error: {error_queue[0]}")
                logger.error("Terminating all processes...")
                cleanup_threads(monitor_threads)
                cleanup_processes(processes)
                sys.exit(1)
            time.sleep(1)

        if orchestrator_process.returncode != 0:
            logger.error(f"Orchestrator failed with exit code {orchestrator_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        if trainer_process.returncode != 0:
            logger.error(f"Trainer failed with exit code {trainer_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        logger.success("RL training finished!")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)

    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating all processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        raise
    finally:
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        import shutil

        shutil.rmtree(cache_root, ignore_errors=True)


def rl(config: RLConfig) -> None:
    """Run PRIME-RL's top-level RL lifecycle with MedARC's shared-node local launcher."""
    from prime_rl.entrypoints import rl as upstream_rl

    if config.slurm is not None:
        raise ValueError("medarc_rl.launchers.rl_local only supports local single-node RL configs")

    original_rl_local = upstream_rl.rl_local
    upstream_rl.rl_local = rl_local
    try:
        upstream_rl.rl(config)
    finally:
        upstream_rl.rl_local = original_rl_local


def main() -> None:
    config = cli(RLConfig)
    rl(config)


if __name__ == "__main__":
    main()
