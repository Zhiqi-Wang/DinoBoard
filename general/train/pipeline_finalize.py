from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from ..search_options import clone_search_options
from .config import PolicyConfig, TrainJobConfig
from .mcts_schedule import resolve_simulations
from .model_registry import promote_best_if_accepted
from .pipeline_support import build_eval_policy, run_arena_series, utc_now


def write_selfplay_artifacts(
    *,
    artifacts_dir: Path,
    selfplay_results: list[dict[str, Any]],
    total_samples_count: int,
    selfplay_elapsed_sec: float,
    episodes: int,
    loop_steps: int,
    warm_start_episodes: int,
    warm_start_engine: str,
    warm_start_train_passes: int,
    search_log: list[dict[str, Any]],
    step_summaries: list[dict[str, Any]],
    worker_pool: str,
    diagnostics_enabled: bool,
    diagnostics_zero_eps: float,
    diagnostics_shared_games: int,
    diagnostics_samples_with_z: int,
    diagnostics_zeroish_samples: int,
) -> dict[str, Any]:
    selfplay_results.sort(key=lambda x: int(x["episode"]))
    samples_per_sec = float(total_samples_count) / max(1e-9, selfplay_elapsed_sec)
    episodes_per_sec = float(len(selfplay_results)) / max(1e-9, selfplay_elapsed_sec)
    print(
        f"[train] selfplay total: episodes={len(selfplay_results)} samples={total_samples_count} "
        f"elapsed={selfplay_elapsed_sec:.2f}s samples/s={samples_per_sec:.2f}"
    )

    selfplay_games_total = len(selfplay_results)
    diagnostics_summary = {
        "enabled": diagnostics_enabled,
        "zero_value_epsilon": diagnostics_zero_eps,
        "shared_game_count": diagnostics_shared_games,
        "shared_game_rate": float(diagnostics_shared_games) / max(1, selfplay_games_total),
        "samples_with_value_target": diagnostics_samples_with_z,
        "zeroish_value_sample_count": diagnostics_zeroish_samples,
        "zeroish_value_ratio": (
            float(diagnostics_zeroish_samples) / max(1, diagnostics_samples_with_z)
            if diagnostics_enabled and diagnostics_samples_with_z > 0
            else None
        ),
    }

    (artifacts_dir / "selfplay_summary.json").write_text(
        json.dumps(
            {
                "warm_start_episodes": warm_start_episodes,
                "warm_start_engine": warm_start_engine,
                "warm_start_train_passes": warm_start_train_passes,
                "episodes_per_step": episodes,
                "steps": loop_steps,
                "target_total_episodes": warm_start_episodes + episodes * loop_steps,
                "games_finished": len(selfplay_results),
                "total_samples": total_samples_count,
                "selfplay_elapsed_sec": selfplay_elapsed_sec,
                "samples_per_sec": samples_per_sec,
                "episodes_per_sec": episodes_per_sec,
                "search_schedule": search_log,
                "step_summaries": step_summaries,
                "worker_pool": worker_pool,
                "engine_note": "self-play uses C++ engine path",
                "diagnostics": diagnostics_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "selfplay_diagnostics.json").write_text(
        json.dumps(diagnostics_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "samples_per_sec": samples_per_sec,
        "episodes_per_sec": episodes_per_sec,
        "diagnostics_summary": diagnostics_summary,
    }


def run_gating_stage(
    *,
    config: TrainJobConfig,
    job_seed: int,
    artifacts_dir: Path,
    output_dir: Path,
    loop_steps: int,
    benchmark_engine: str,
    bench_model: Path | None,
    trained_model_path: str | None,
    current_model_path: str | None,
    periodic_eval_summaries: list[dict[str, Any]],
    run_arena_match: Callable[[int, Any, Any], dict[str, Any]],
    build_policy: Callable[[PolicyConfig, int], Any],
    prepare_eval_policy_hook: Callable[[PolicyConfig, dict[str, Any]], PolicyConfig] | None,
    worker_pool: str,
    eval_workers: int,
    process_pool_max_tasks_per_child: int | None,
    eval_candidate_temperature: float,
    eval_opponent_temperature: float,
    eval_simulations_floor: int,
    candidate_manifest: dict[str, Any],
    job_status: dict[str, Any],
) -> tuple[dict[str, Any], float, float]:
    job_status["phase"] = "gating"
    job_status["phase_started_at"] = utc_now()
    (output_dir / "job_status.json").write_text(json.dumps(job_status, ensure_ascii=False, indent=2), encoding="utf-8")

    gate_games = max(2, int(config.gating.games))
    last_training_step_idx = max(0, loop_steps - 1)
    base_simulations = resolve_simulations(
        last_training_step_idx,
        config.selfplay.policy.simulations,
        config.selfplay.mcts_schedule,
        loop_steps,
    )
    candidate_model_path = str(trained_model_path) if trained_model_path else current_model_path
    candidate_policy_cfg, candidate_policy = build_eval_policy(
        base_policy_cfg=config.selfplay.policy,
        build_policy=build_policy,
        prepare_eval_policy_hook=prepare_eval_policy_hook,
        phase="gating",
        role="candidate",
        step_index=loop_steps,
        job_seed=job_seed,
        engine=str(config.selfplay.policy.engine),
        simulations=base_simulations,
        temperature=eval_candidate_temperature,
        model_path=candidate_model_path,
        search_options=clone_search_options(config.selfplay.policy.search_options),
    )
    benchmark_policy_cfg, benchmark_policy = build_eval_policy(
        base_policy_cfg=config.selfplay.policy,
        build_policy=build_policy,
        prepare_eval_policy_hook=prepare_eval_policy_hook,
        phase="gating",
        role="benchmark",
        step_index=loop_steps,
        job_seed=job_seed,
        engine=benchmark_engine,
        simulations=max(eval_simulations_floor, base_simulations),
        temperature=eval_opponent_temperature,
        model_path=str(bench_model) if bench_model is not None else None,
        search_options=clone_search_options(config.selfplay.policy.search_options),
    )

    gating_t0 = time.perf_counter()
    arena_records, win_rate = run_arena_series(
        run_arena_match=run_arena_match,
        p0=candidate_policy,
        p1=benchmark_policy,
        games=gate_games,
        max_workers=eval_workers,
        job_seed=job_seed ^ 0xDEADBEEF,
        worker_pool=worker_pool,
        phase_label="gating",
        process_pool_max_tasks_per_child=process_pool_max_tasks_per_child,
    )
    gating_summary = {
        "games": len(arena_records),
        "accept_win_rate": config.gating.accept_win_rate,
        "candidate_win_rate": win_rate,
        "accepted": win_rate >= config.gating.accept_win_rate,
        "periodic_eval": periodic_eval_summaries,
        "benchmark_model_path": str(bench_model) if bench_model is not None else None,
        "candidate_policy": {
            "engine": candidate_policy_cfg.engine,
            "simulations": base_simulations,
            "temperature": candidate_policy_cfg.temperature,
            "model_path": candidate_policy_cfg.model_path,
            "search_options": clone_search_options(candidate_policy_cfg.search_options),
        },
        "benchmark_policy": {
            "engine": benchmark_engine,
            "simulations": benchmark_policy_cfg.simulations,
            "temperature": benchmark_policy_cfg.temperature,
            "model_path": str(bench_model) if bench_model is not None else None,
            "search_options": clone_search_options(benchmark_policy_cfg.search_options),
        },
        "records": arena_records,
    }
    (artifacts_dir / "gating_summary.json").write_text(
        json.dumps(gating_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    promote_best_if_accepted(artifacts_dir, candidate_manifest, gating_summary["accepted"])
    gating_elapsed_sec = max(1e-9, time.perf_counter() - gating_t0)
    print(
        f"[train] gating done: games={len(arena_records)} "
        f"elapsed={gating_elapsed_sec:.2f}s candidate_win_rate={win_rate:.3f}"
    )
    return gating_summary, win_rate, gating_elapsed_sec

