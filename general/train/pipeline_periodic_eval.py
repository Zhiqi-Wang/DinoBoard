from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..search_options import clone_search_options
from .config import PolicyConfig, TrainJobConfig
from .mcts_schedule import resolve_simulations
from .pipeline_support import build_eval_policy, run_arena_series


def run_periodic_eval_for_step(
    *,
    state: dict[str, Any],
    config: TrainJobConfig,
    artifacts_dir: Path,
    step_idx: int,
    loop_steps: int,
    eval_games: int,
    history_best_games: int,
    history_best_accept_win_rate: float,
    eval_candidate_temperature: float,
    eval_opponent_temperature: float,
    eval_simulations_floor: int,
    worker_pool: str,
    eval_workers: int,
    process_pool_max_tasks_per_child: int,
    bench_model: Path | None,
    benchmark_engine: str,
    prepare_eval_policy_hook: Callable[[PolicyConfig, dict[str, Any]], PolicyConfig] | None,
    enrich_periodic_eval_summary_hook: Callable[[dict[str, Any], dict[str, Any]], None] | None,
    run_arena_match: Callable[[int, Any, Any], dict[str, Any]],
    build_policy: Callable[[PolicyConfig, int], Any],
    job_seed: int,
    copy_model_pair: Callable[..., None],
    write_history_best_manifest: Callable[..., None],
    best_model_path: Path,
    best_checkpoint_path: Path,
) -> None:
    base_simulations = resolve_simulations(
        step_idx,
        config.selfplay.policy.simulations,
        config.selfplay.mcts_schedule,
        max(1, int(loop_steps)),
    )
    fixed_eval_simulations = max(0, int(getattr(config.eval, "simulations_fixed", 0)))
    if fixed_eval_simulations > 0:
        eval_simulations = fixed_eval_simulations
    else:
        eval_simulations = base_simulations
    eval_opponent_simulations = max(eval_simulations_floor, eval_simulations)
    _candidate_policy_cfg, candidate_policy = build_eval_policy(
        base_policy_cfg=config.selfplay.policy,
        build_policy=build_policy,
        prepare_eval_policy_hook=prepare_eval_policy_hook,
        phase="periodic_eval",
        role="candidate",
        step_index=step_idx + 1,
        job_seed=job_seed,
        engine=str(config.selfplay.policy.engine),
        simulations=eval_simulations,
        temperature=eval_candidate_temperature,
        model_path=state["current_model_path"],
        search_options=clone_search_options(config.selfplay.policy.search_options),
    )

    benchmark_win_rate = None
    benchmark_records: list[dict[str, Any]] = []
    if str(benchmark_engine).lower() == "netmcts" and bench_model is None:
        raise RuntimeError("periodic eval requires benchmark model path when benchmark_engine=netmcts")
    benchmark_model_path = str(bench_model) if bench_model is not None else None
    _benchmark_policy_cfg, benchmark_policy = build_eval_policy(
        base_policy_cfg=config.selfplay.policy,
        build_policy=build_policy,
        prepare_eval_policy_hook=prepare_eval_policy_hook,
        phase="periodic_eval",
        role="benchmark",
        step_index=step_idx + 1,
        job_seed=job_seed,
        engine=benchmark_engine,
        simulations=eval_opponent_simulations,
        temperature=eval_opponent_temperature,
        model_path=benchmark_model_path,
        search_options=clone_search_options(config.selfplay.policy.search_options),
    )
    benchmark_records, benchmark_win_rate = run_arena_series(
        run_arena_match=run_arena_match,
        p0=candidate_policy,
        p1=benchmark_policy,
        games=max(2, eval_games),
        max_workers=eval_workers,
        job_seed=job_seed ^ ((step_idx + 1) << 8),
        worker_pool=worker_pool,
        phase_label=f"eval-benchmark step={step_idx + 1}",
        process_pool_max_tasks_per_child=process_pool_max_tasks_per_child,
    )
    print(
        f"[train] eval benchmark: step={step_idx + 1} "
        f"games={len(benchmark_records)} win_rate={benchmark_win_rate:.3f}",
        flush=True,
    )

    history_best_win_rate = None
    history_records: list[dict[str, Any]] = []
    replaced_best = False
    history_best_before = state["history_best_model_path"]
    if state["history_best_model_path"]:
        _history_policy_cfg, history_policy = build_eval_policy(
            base_policy_cfg=config.selfplay.policy,
            build_policy=build_policy,
            prepare_eval_policy_hook=prepare_eval_policy_hook,
            phase="periodic_eval",
            role="history_best",
            step_index=step_idx + 1,
            job_seed=job_seed,
            # History-best gating must compare candidate vs best under the same
            # policy engine; benchmark_engine is only for the benchmark metric.
            engine=str(config.selfplay.policy.engine),
            simulations=eval_opponent_simulations,
            temperature=eval_opponent_temperature,
            model_path=str(state["history_best_model_path"]),
            search_options=clone_search_options(config.selfplay.policy.search_options),
        )
        history_records, history_best_win_rate = run_arena_series(
            run_arena_match=run_arena_match,
            p0=candidate_policy,
            p1=history_policy,
            games=max(2, history_best_games),
            max_workers=eval_workers,
            job_seed=job_seed ^ ((step_idx + 1) << 12),
            worker_pool=worker_pool,
            phase_label=f"eval-history step={step_idx + 1}",
            process_pool_max_tasks_per_child=process_pool_max_tasks_per_child,
        )
        if history_best_win_rate >= history_best_accept_win_rate:
            copy_model_pair(
                src_model_path=state["current_model_path"],
                src_checkpoint_path=state["current_checkpoint_path"],
                dst_model_path=best_model_path,
                dst_checkpoint_path=best_checkpoint_path,
            )
            state["history_best_model_path"] = str(best_model_path)
            replaced_best = True
            write_history_best_manifest(
                step=step_idx + 1,
                model_path=state["history_best_model_path"],
                win_rate=history_best_win_rate,
                source="promoted_from_latest",
            )
        print(
            f"[train] eval history_best: step={step_idx + 1} "
            f"games={len(history_records)} win_rate={history_best_win_rate:.3f} "
            f"replace_best={replaced_best}",
            flush=True,
        )

    summary = {
        "step": step_idx + 1,
        "candidate_model_path": state["current_model_path"],
        "benchmark_model_path": str(bench_model) if bench_model is not None else None,
        "eval_simulations": int(eval_simulations),
        "eval_opponent_simulations": int(eval_opponent_simulations),
        "eval_simulations_fixed": int(fixed_eval_simulations),
        "benchmark_win_rate": benchmark_win_rate,
        "benchmark_games": len(benchmark_records),
        "history_best_model_path_before_eval": history_best_before,
        "history_best_win_rate": history_best_win_rate,
        "history_best_games": len(history_records),
        "history_best_accept_win_rate": history_best_accept_win_rate,
        "replaced_history_best": replaced_best,
        "history_best_model_path_after_eval": state["history_best_model_path"],
    }
    if enrich_periodic_eval_summary_hook is not None:
        enrich_periodic_eval_summary_hook(
            summary,
            {
                "phase": "periodic_eval",
                "step_index": step_idx + 1,
                "job_seed": job_seed,
            },
        )
    state["periodic_eval_summaries"].append(summary)
    (artifacts_dir / "periodic_eval_summary.json").write_text(
        json.dumps(state["periodic_eval_summaries"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

