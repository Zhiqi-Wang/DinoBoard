from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import PolicyConfig, TrainJobConfig
from .pipeline_step_loop import run_training_step_loop
from .pipeline_warm_start import run_warm_start_and_init_best


@dataclass(slots=True)
class SelfplayLoopContext:
    config: TrainJobConfig
    output_dir: Path
    artifacts_dir: Path
    job_seed: int
    loop_steps: int
    episodes: int
    warm_start_episodes: int
    warm_start_engine: str
    warm_start_simulations: int
    warm_start_train_passes: int
    max_workers: int
    eval_every_steps: int
    eval_games: int
    history_best_games: int
    history_best_accept_win_rate: float
    save_latest_every_steps: int
    eval_candidate_temperature: float
    eval_opponent_temperature: float
    eval_simulations_floor: int
    worker_pool: str
    eval_workers: int
    process_pool_max_tasks_per_child: int
    diagnostics_enabled: bool
    diagnostics_zero_eps: float
    bench_model: Path | None
    benchmark_engine: str
    prepare_selfplay_policy_hook: Callable[[PolicyConfig, dict[str, Any]], PolicyConfig] | None
    prepare_eval_policy_hook: Callable[[PolicyConfig, dict[str, Any]], PolicyConfig] | None
    build_episode_context_hook: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None
    label_sample_hook: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]] | None
    postprocess_episode_payload_hook: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None
    enrich_periodic_eval_summary_hook: Callable[[dict[str, Any], dict[str, Any]], None] | None
    run_selfplay_episode: Callable[[int, int, Any, str], dict[str, Any]]
    run_arena_match: Callable[[int, Any, Any], dict[str, Any]]
    build_policy: Callable[[PolicyConfig, int], Any]
    run_trainer: Callable[..., dict[str, Any]] | None
    model_dir: Path
    latest_model_path: Path
    latest_checkpoint_path: Path
    best_model_path: Path
    best_checkpoint_path: Path
    copy_model_pair: Callable[..., None]
    write_history_best_manifest: Callable[..., None]
    executor: Any


def run_selfplay_train_loop(
    *,
    state: dict[str, Any],
    ctx: SelfplayLoopContext,
) -> None:
    run_warm_start_and_init_best(
        state=state,
        config=ctx.config,
        artifacts_dir=ctx.artifacts_dir,
        job_seed=ctx.job_seed,
        warm_start_episodes=ctx.warm_start_episodes,
        warm_start_engine=ctx.warm_start_engine,
        warm_start_simulations=ctx.warm_start_simulations,
        warm_start_train_passes=ctx.warm_start_train_passes,
        max_workers=ctx.max_workers,
        diagnostics_enabled=ctx.diagnostics_enabled,
        diagnostics_zero_eps=ctx.diagnostics_zero_eps,
        prepare_selfplay_policy_hook=ctx.prepare_selfplay_policy_hook,
        build_episode_context_hook=ctx.build_episode_context_hook,
        label_sample_hook=ctx.label_sample_hook,
        postprocess_episode_payload_hook=ctx.postprocess_episode_payload_hook,
        run_selfplay_episode=ctx.run_selfplay_episode,
        build_policy=ctx.build_policy,
        run_trainer=ctx.run_trainer,
        latest_model_path=ctx.latest_model_path,
        latest_checkpoint_path=ctx.latest_checkpoint_path,
        best_model_path=ctx.best_model_path,
        best_checkpoint_path=ctx.best_checkpoint_path,
        copy_model_pair=ctx.copy_model_pair,
        write_history_best_manifest=ctx.write_history_best_manifest,
        executor=ctx.executor,
    )

    run_training_step_loop(
        state=state,
        config=ctx.config,
        output_dir=ctx.output_dir,
        artifacts_dir=ctx.artifacts_dir,
        job_seed=ctx.job_seed,
        loop_steps=ctx.loop_steps,
        episodes=ctx.episodes,
        max_workers=ctx.max_workers,
        eval_every_steps=ctx.eval_every_steps,
        eval_games=ctx.eval_games,
        history_best_games=ctx.history_best_games,
        history_best_accept_win_rate=ctx.history_best_accept_win_rate,
        save_latest_every_steps=ctx.save_latest_every_steps,
        eval_candidate_temperature=ctx.eval_candidate_temperature,
        eval_opponent_temperature=ctx.eval_opponent_temperature,
        eval_simulations_floor=ctx.eval_simulations_floor,
        worker_pool=ctx.worker_pool,
        eval_workers=ctx.eval_workers,
        process_pool_max_tasks_per_child=ctx.process_pool_max_tasks_per_child,
        diagnostics_enabled=ctx.diagnostics_enabled,
        diagnostics_zero_eps=ctx.diagnostics_zero_eps,
        bench_model=ctx.bench_model,
        benchmark_engine=ctx.benchmark_engine,
        prepare_selfplay_policy_hook=ctx.prepare_selfplay_policy_hook,
        prepare_eval_policy_hook=ctx.prepare_eval_policy_hook,
        build_episode_context_hook=ctx.build_episode_context_hook,
        label_sample_hook=ctx.label_sample_hook,
        postprocess_episode_payload_hook=ctx.postprocess_episode_payload_hook,
        enrich_periodic_eval_summary_hook=ctx.enrich_periodic_eval_summary_hook,
        run_selfplay_episode=ctx.run_selfplay_episode,
        run_arena_match=ctx.run_arena_match,
        build_policy=ctx.build_policy,
        run_trainer=ctx.run_trainer,
        model_dir=ctx.model_dir,
        latest_model_path=ctx.latest_model_path,
        latest_checkpoint_path=ctx.latest_checkpoint_path,
        best_model_path=ctx.best_model_path,
        best_checkpoint_path=ctx.best_checkpoint_path,
        copy_model_pair=ctx.copy_model_pair,
        write_history_best_manifest=ctx.write_history_best_manifest,
        executor=ctx.executor,
    )

