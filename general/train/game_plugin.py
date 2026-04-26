from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Callable, Type

from .config import TrainJobConfig
from .extensions import TrainPipelineHooks
from .game_entrypoint import build_game_training_entrypoints
from .selfplay_worker_factory import create_worker_runners

RunJobFn = Callable[[TrainJobConfig, Path, int], dict[str, Any]]
DefaultConfigFactory = Callable[[str | None], TrainJobConfig]
MainFn = Callable[[], int]
EpisodeRunner = Callable[[int, int, Any, str], dict[str, Any]]
ArenaRunner = Callable[[int, Any, Any], dict[str, Any]]


@dataclass(slots=True)
class GameTrainPlugin:
    description: str
    benchmark_engine: str
    support_benchmark_onnx: bool
    move_policy_cls: Type[Any]
    backend_factory: Callable[[], Any]
    game_type: str
    ruleset: str
    read_shared_victory_from_raw: bool
    run_trainer: Callable[..., dict[str, Any]] | None = None
    build_initial_model: Callable[[TrainJobConfig, Path, int], str] | None = None
    pipeline_hooks: TrainPipelineHooks | None = None
    before_run_job: Callable[[TrainJobConfig], None] | None = None
    enrich_sample: Callable[[dict[str, Any], dict[str, Any]], None] | None = None
    netmcts_data_source: str | None = None
    default_value_margin_weight: float | None = None
    default_value_margin_scale: float | None = None


def build_initial_model_from_exporter(
    exporter: Callable[..., str],
    *,
    input_dim: int | None = None,
    policy_dim: int | None = None,
    model_filename: str = "selfplay_init.onnx",
) -> Callable[[TrainJobConfig, Path, int], str]:
    param_names = set(inspect.signature(exporter).parameters.keys())
    include_seed = "seed" in param_names
    include_input_dim = input_dim is not None and "input_dim" in param_names
    include_policy_dim = policy_dim is not None and "policy_dim" in param_names

    def _build(config: TrainJobConfig, artifacts_dir: Path, job_seed: int) -> str:
        kwargs: dict[str, Any] = {
            "config": config,
            "output_path": artifacts_dir / "models" / model_filename,
        }
        if include_seed:
            kwargs["seed"] = int(job_seed)
        if include_input_dim:
            kwargs["input_dim"] = int(input_dim) if input_dim is not None else None
        if include_policy_dim:
            kwargs["policy_dim"] = int(policy_dim) if policy_dim is not None else None
        return exporter(**kwargs)

    return _build


def build_selfplay_runners_from_plugin(plugin: GameTrainPlugin) -> tuple[EpisodeRunner, ArenaRunner]:
    return create_worker_runners(
        backend_factory=plugin.backend_factory,
        game_type=plugin.game_type,
        ruleset=plugin.ruleset,
        read_shared_victory_from_raw=plugin.read_shared_victory_from_raw,
        enrich_sample=plugin.enrich_sample,
        netmcts_data_source=plugin.netmcts_data_source,
        default_value_margin_weight=plugin.default_value_margin_weight,
        default_value_margin_scale=plugin.default_value_margin_scale,
    )


def build_training_entrypoints_from_plugin(
    *,
    current_file: str,
    plugin: GameTrainPlugin,
) -> tuple[RunJobFn, DefaultConfigFactory, MainFn]:
    run_selfplay_episode, run_arena_match = build_selfplay_runners_from_plugin(plugin)
    return build_game_training_entrypoints(
        current_file=current_file,
        description=plugin.description,
        benchmark_engine=plugin.benchmark_engine,
        support_benchmark_onnx=plugin.support_benchmark_onnx,
        move_policy_cls=plugin.move_policy_cls,
        run_selfplay_episode=run_selfplay_episode,
        run_arena_match=run_arena_match,
        run_trainer=plugin.run_trainer,
        build_initial_model=plugin.build_initial_model,
        pipeline_hooks=plugin.pipeline_hooks,
        before_run_job=plugin.before_run_job,
    )

