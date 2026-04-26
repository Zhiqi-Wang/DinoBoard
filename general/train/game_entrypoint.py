from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Type

from .config import TrainJobConfig
from .extensions import TrainPipelineHooks
from .game_runner import run_game_train_job, run_train_cli

RunJobFn = Callable[[TrainJobConfig, Path, int], dict[str, Any]]
DefaultConfigFactory = Callable[[str | None], TrainJobConfig]
MainFn = Callable[[], int]


def load_default_config_from_template(
    *,
    current_file: str,
    benchmark_onnx_path: str | None = None,
) -> TrainJobConfig:
    template_path = Path(current_file).resolve().parent / "train_config.example.json"
    base = json.loads(template_path.read_text(encoding="utf-8"))
    if benchmark_onnx_path:
        base["benchmark_onnx_path"] = benchmark_onnx_path
    return TrainJobConfig.from_dict(base)


def build_game_training_entrypoints(
    *,
    current_file: str,
    description: str,
    benchmark_engine: str,
    support_benchmark_onnx: bool,
    move_policy_cls: Type[Any],
    run_selfplay_episode: Callable[[int, int, Any, str], dict[str, Any]],
    run_arena_match: Callable[[int, Any, Any], dict[str, Any]],
    run_trainer: Callable[..., dict[str, Any]] | None = None,
    build_initial_model: Callable[[TrainJobConfig, Path, int], str] | None = None,
    pipeline_hooks: TrainPipelineHooks | None = None,
    before_run_job: Callable[[TrainJobConfig], None] | None = None,
) -> tuple[RunJobFn, DefaultConfigFactory, MainFn]:
    def _run_job(config: TrainJobConfig, output_dir: Path, job_seed: int) -> dict[str, Any]:
        if before_run_job is not None:
            before_run_job(config)
        return run_game_train_job(
            config=config,
            output_dir=output_dir,
            job_seed=job_seed,
            move_policy_cls=move_policy_cls,
            run_selfplay_episode=run_selfplay_episode,
            run_arena_match=run_arena_match,
            benchmark_engine=benchmark_engine,
            run_trainer=run_trainer,
            build_initial_model=build_initial_model,
            pipeline_hooks=pipeline_hooks,
        )

    def _default_config(benchmark_onnx_path: str | None = None) -> TrainJobConfig:
        return load_default_config_from_template(
            current_file=current_file,
            benchmark_onnx_path=benchmark_onnx_path,
        )

    def _main() -> int:
        return run_train_cli(
            description=description,
            run_job=_run_job,
            default_config_factory=_default_config,
            support_benchmark_onnx=support_benchmark_onnx,
        )

    return _run_job, _default_config, _main

