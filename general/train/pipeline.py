from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .config import PolicyConfig, TrainJobConfig
from .extensions import TrainPipelineHooks
from .pipeline_finalize import run_gating_stage, write_selfplay_artifacts
from .pipeline_selfplay_loop import SelfplayLoopContext, run_selfplay_train_loop
from .pipeline_support import process_pool_kwargs, utc_now
from .metadata import ensure_dir, stable_config_hash
from .model_registry import register_candidate


SelfPlayRunner = Callable[[int, int, Any, str], dict[str, Any]]
ArenaRunner = Callable[[int, Any, Any], dict[str, Any]]
PolicyBuilder = Callable[[PolicyConfig, int], Any]
TrainerRunner = Callable[..., dict[str, Any]]
InitialModelBuilder = Callable[[TrainJobConfig, Path, int], str]


def run_train_job(
    *,
    config: TrainJobConfig,
    output_dir: Path,
    job_seed: int,
    run_selfplay_episode: SelfPlayRunner,
    run_arena_match: ArenaRunner,
    build_policy: PolicyBuilder,
    benchmark_engine: str = "netmcts",
    run_trainer: TrainerRunner | None = None,
    build_initial_model: InitialModelBuilder | None = None,
    pipeline_hooks: TrainPipelineHooks | None = None,
) -> dict[str, Any]:
    total_t0 = time.perf_counter()
    ensure_dir(output_dir)
    artifacts_dir = output_dir / "artifacts"
    ensure_dir(artifacts_dir)

    config_dict = asdict(config)
    config_hash = stable_config_hash(config_dict)
    (output_dir / "training_config.json").write_text(
        json.dumps(config_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    job_status: dict[str, Any] = {
        "phase": "init",
        "created_at": utc_now(),
        "config_hash": config_hash,
        "job_seed": job_seed,
    }
    (output_dir / "job_status.json").write_text(json.dumps(job_status, ensure_ascii=False, indent=2), encoding="utf-8")

    episodes = max(1, int(config.selfplay.episodes))
    warm_start_episodes = max(0, int(getattr(config.selfplay, "warm_start_episodes", 0)))
    warm_start_engine = str(getattr(config.selfplay, "warm_start_engine", "random") or "random")
    warm_start_simulations = max(0, int(getattr(config.selfplay, "warm_start_simulations", 0)))
    warm_start_train_passes = max(0, int(getattr(config.selfplay, "warm_start_train_passes", 0)))
    loop_steps = max(1, int(getattr(config.trainer, "steps", 0) or 1))
    max_workers = max(1, min(int(config.selfplay.max_workers), int(config.selfplay.parallel_games)))
    eval_every_steps = max(0, int(getattr(config.gating, "eval_every_steps", 0)))
    eval_workers = max(1, int(getattr(config.gating, "eval_workers", 0) or max_workers))
    eval_games = max(0, int(getattr(config.gating, "eval_games", 0) or getattr(config.gating, "games", 0)))
    history_best_games = max(0, int(getattr(config.gating, "history_best_games", 0) or eval_games))
    history_best_accept_win_rate = float(getattr(config.gating, "history_best_accept_win_rate", config.gating.accept_win_rate))
    save_latest_every_steps = max(0, int(getattr(config.gating, "save_latest_every_steps", 0)))
    eval_candidate_temperature = float(getattr(config.eval, "candidate_temperature", 0.0))
    eval_opponent_temperature = float(getattr(config.eval, "opponent_temperature", 0.0))
    eval_simulations_floor = max(1, int(getattr(config.eval, "simulations_floor", 10)))
    process_pool_max_tasks_per_child = int(getattr(config.runtime, "process_pool_max_tasks_per_child", 128))
    diagnostics_enabled = bool(getattr(config.diagnostics, "enabled", True))
    diagnostics_zero_eps = float(getattr(config.diagnostics, "zero_value_epsilon", 0.1))
    prepare_selfplay_policy_hook = pipeline_hooks.prepare_selfplay_policy if pipeline_hooks else None
    prepare_eval_policy_hook = pipeline_hooks.prepare_eval_policy if pipeline_hooks else None
    enrich_periodic_eval_summary_hook = pipeline_hooks.enrich_periodic_eval_summary if pipeline_hooks else None
    build_episode_context_hook = pipeline_hooks.build_episode_context if pipeline_hooks else None
    label_sample_hook = pipeline_hooks.label_sample if pipeline_hooks else None
    postprocess_episode_payload_hook = pipeline_hooks.postprocess_episode_payload if pipeline_hooks else None
    selfplay_results: list[dict[str, Any]] = []
    total_samples_count = 0
    search_log: list[dict[str, Any]] = []
    step_summaries: list[dict[str, Any]] = []
    selfplay_worker_pids: set[int] = set()
    selfplay_elapsed_sec = 0.0
    train_elapsed_sec = 0.0
    diagnostics_samples_with_z = 0
    diagnostics_zeroish_samples = 0
    diagnostics_shared_games = 0

    worker_pool = str(getattr(config.selfplay, "worker_pool", "thread")).lower()
    if worker_pool not in {"thread", "process"}:
        worker_pool = "thread"
    executor_cls = ProcessPoolExecutor if worker_pool == "process" else ThreadPoolExecutor
    print(f"[train] selfplay workers: pool={worker_pool} count={max_workers}")

    current_model_path = config.selfplay.policy.model_path
    if str(config.selfplay.policy.engine).lower() == "netmcts" and not current_model_path:
        if build_initial_model is None:
            raise RuntimeError("selfplay.model_path is required for netmcts unless build_initial_model is provided.")
        current_model_path = str(build_initial_model(config, artifacts_dir, job_seed))
        print(
            f"[train] selfplay init model: path={current_model_path} "
            f"hidden={int(config.trainer.hidden)} mlp_layers={int(config.trainer.mlp_layers)}",
            flush=True,
        )

    bench_model: Path | None = None
    if benchmark_engine == "netmcts":
        benchmark_path = config.benchmark_onnx_path
        if not benchmark_path:
            raise RuntimeError("benchmark_onnx_path is required when benchmark_engine=netmcts.")
        bench_model = Path(benchmark_path)
        if not bench_model.exists():
            raise RuntimeError(f"benchmark ONNX not found: {bench_model}")
    # "history_best" tracks the best model within this training run's lineage,
    # not the external benchmark model.
    history_best_model_path: str | None = None
    train_summary: dict[str, Any] = {
        "status": "skipped",
        "reason": "trainer_not_provided",
        "note": "No trainer runner provided; keep manifest contract only.",
    }
    trained_model_path: str | None = None
    current_checkpoint_path: str | None = config.resume_checkpoint_path
    next_episode_index = 0
    periodic_eval_summaries: list[dict[str, Any]] = []
    model_dir = artifacts_dir / "models"
    ensure_dir(model_dir)
    latest_model_path = model_dir / "latest_model.onnx"
    latest_checkpoint_path = model_dir / "latest_model.pt"
    best_model_path = model_dir / "best_model.onnx"
    best_checkpoint_path = model_dir / "best_model.pt"

    def _copy_model_pair(
        *,
        src_model_path: str | None,
        src_checkpoint_path: str | None,
        dst_model_path: Path,
        dst_checkpoint_path: Path,
    ) -> None:
        if not src_model_path:
            return
        shutil.copy2(src_model_path, dst_model_path)
        if src_checkpoint_path:
            src_ckpt = Path(src_checkpoint_path)
            if src_ckpt.exists():
                shutil.copy2(src_ckpt, dst_checkpoint_path)
                return
        if dst_checkpoint_path.exists():
            dst_checkpoint_path.unlink()

    def _write_history_best_manifest(
        *,
        step: int,
        model_path: str,
        win_rate: float | None,
        source: str,
    ) -> None:
        (artifacts_dir / "history_best_manifest.json").write_text(
            json.dumps(
                {
                    "updated_at": utc_now(),
                    "step": step,
                    "model_path": model_path,
                    "win_rate": win_rate,
                    "accept_win_rate": history_best_accept_win_rate,
                    "source": source,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if worker_pool == "process":
        executor = executor_cls(
            **process_pool_kwargs(max_workers, process_pool_max_tasks_per_child),
        )
    else:
        executor = executor_cls(max_workers=max_workers)
    state = {
        "job_status": job_status,
        "selfplay_results": selfplay_results,
        "total_samples_count": total_samples_count,
        "search_log": search_log,
        "step_summaries": step_summaries,
        "selfplay_worker_pids": selfplay_worker_pids,
        "selfplay_elapsed_sec": selfplay_elapsed_sec,
        "train_elapsed_sec": train_elapsed_sec,
        "diagnostics_samples_with_z": diagnostics_samples_with_z,
        "diagnostics_zeroish_samples": diagnostics_zeroish_samples,
        "diagnostics_shared_games": diagnostics_shared_games,
        "current_model_path": current_model_path,
        "current_checkpoint_path": current_checkpoint_path,
        "trained_model_path": trained_model_path,
        "next_episode_index": next_episode_index,
        "periodic_eval_summaries": periodic_eval_summaries,
        "history_best_model_path": history_best_model_path,
        "train_summary": train_summary,
    }
    try:
        run_selfplay_train_loop(
            state=state,
            ctx=SelfplayLoopContext(
                config=config,
                output_dir=output_dir,
                artifacts_dir=artifacts_dir,
                job_seed=job_seed,
                loop_steps=loop_steps,
                episodes=episodes,
                warm_start_episodes=warm_start_episodes,
                warm_start_engine=warm_start_engine,
                warm_start_simulations=warm_start_simulations,
                warm_start_train_passes=warm_start_train_passes,
                max_workers=max_workers,
                eval_every_steps=eval_every_steps,
                eval_games=eval_games,
                history_best_games=history_best_games,
                history_best_accept_win_rate=history_best_accept_win_rate,
                save_latest_every_steps=save_latest_every_steps,
                eval_candidate_temperature=eval_candidate_temperature,
                eval_opponent_temperature=eval_opponent_temperature,
                eval_simulations_floor=eval_simulations_floor,
                worker_pool=worker_pool,
                eval_workers=eval_workers,
                process_pool_max_tasks_per_child=process_pool_max_tasks_per_child,
                diagnostics_enabled=diagnostics_enabled,
                diagnostics_zero_eps=diagnostics_zero_eps,
                bench_model=bench_model,
                benchmark_engine=benchmark_engine,
                prepare_selfplay_policy_hook=prepare_selfplay_policy_hook,
                prepare_eval_policy_hook=prepare_eval_policy_hook,
                build_episode_context_hook=build_episode_context_hook,
                label_sample_hook=label_sample_hook,
                postprocess_episode_payload_hook=postprocess_episode_payload_hook,
                enrich_periodic_eval_summary_hook=enrich_periodic_eval_summary_hook,
                run_selfplay_episode=run_selfplay_episode,
                run_arena_match=run_arena_match,
                build_policy=build_policy,
                run_trainer=run_trainer,
                model_dir=model_dir,
                latest_model_path=latest_model_path,
                latest_checkpoint_path=latest_checkpoint_path,
                best_model_path=best_model_path,
                best_checkpoint_path=best_checkpoint_path,
                copy_model_pair=_copy_model_pair,
                write_history_best_manifest=_write_history_best_manifest,
                executor=executor,
            ),
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    job_status = state["job_status"]
    selfplay_results = state["selfplay_results"]
    total_samples_count = state["total_samples_count"]
    search_log = state["search_log"]
    step_summaries = state["step_summaries"]
    selfplay_worker_pids = state["selfplay_worker_pids"]
    selfplay_elapsed_sec = state["selfplay_elapsed_sec"]
    train_elapsed_sec = state["train_elapsed_sec"]
    diagnostics_samples_with_z = state["diagnostics_samples_with_z"]
    diagnostics_zeroish_samples = state["diagnostics_zeroish_samples"]
    diagnostics_shared_games = state["diagnostics_shared_games"]
    current_model_path = state["current_model_path"]
    current_checkpoint_path = state["current_checkpoint_path"]
    trained_model_path = state["trained_model_path"]
    periodic_eval_summaries = state["periodic_eval_summaries"]
    history_best_model_path = state["history_best_model_path"]
    train_summary = state["train_summary"]

    selfplay_metrics = write_selfplay_artifacts(
        artifacts_dir=artifacts_dir,
        selfplay_results=selfplay_results,
        total_samples_count=total_samples_count,
        selfplay_elapsed_sec=selfplay_elapsed_sec,
        episodes=episodes,
        loop_steps=loop_steps,
        warm_start_episodes=warm_start_episodes,
        warm_start_engine=warm_start_engine,
        warm_start_train_passes=warm_start_train_passes,
        search_log=search_log,
        step_summaries=step_summaries,
        worker_pool=worker_pool,
        diagnostics_enabled=diagnostics_enabled,
        diagnostics_zero_eps=diagnostics_zero_eps,
        diagnostics_shared_games=diagnostics_shared_games,
        diagnostics_samples_with_z=diagnostics_samples_with_z,
        diagnostics_zeroish_samples=diagnostics_zeroish_samples,
    )
    if selfplay_worker_pids:
        pid_list = sorted(selfplay_worker_pids)
        print(f"[train] selfplay worker pids: unique={len(pid_list)} pids={pid_list}")
    (artifacts_dir / "train_summary.json").write_text(
        json.dumps(train_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    candidate_id = f"candidate_{int(time.time())}"
    candidate_manifest = {
        "candidate_id": candidate_id,
        "created_at": utc_now(),
        "trainer": asdict(config.trainer),
        "train_summary": train_summary,
    }
    if trained_model_path:
        candidate_manifest["model_path"] = str(trained_model_path)
    register_candidate(artifacts_dir, candidate_manifest)

    gating_summary, win_rate, gating_elapsed_sec = run_gating_stage(
        config=config,
        job_seed=job_seed,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        loop_steps=loop_steps,
        benchmark_engine=benchmark_engine,
        bench_model=bench_model,
        trained_model_path=trained_model_path,
        current_model_path=current_model_path,
        periodic_eval_summaries=periodic_eval_summaries,
        run_arena_match=run_arena_match,
        build_policy=build_policy,
        prepare_eval_policy_hook=prepare_eval_policy_hook,
        worker_pool=worker_pool,
        eval_workers=eval_workers,
        process_pool_max_tasks_per_child=process_pool_max_tasks_per_child,
        eval_candidate_temperature=eval_candidate_temperature,
        eval_opponent_temperature=eval_opponent_temperature,
        eval_simulations_floor=eval_simulations_floor,
        candidate_manifest=candidate_manifest,
        job_status=job_status,
    )

    job_status["phase"] = "completed"
    job_status["completed_at"] = utc_now()
    job_status["result"] = {
        "samples": total_samples_count,
        "selfplay_games": len(selfplay_results),
        "gating_games": int(gating_summary.get("games", 0)),
        "selfplay_elapsed_sec": selfplay_elapsed_sec,
        "trainer_elapsed_sec": train_elapsed_sec,
        "selfplay_samples_per_sec": float(selfplay_metrics["samples_per_sec"]),
        "candidate_win_rate": win_rate,
        "accepted": gating_summary["accepted"],
    }
    total_elapsed_sec = max(1e-9, time.perf_counter() - total_t0)
    overall_samples_per_sec = float(total_samples_count) / total_elapsed_sec
    overhead_elapsed_sec = max(0.0, total_elapsed_sec - selfplay_elapsed_sec - train_elapsed_sec - gating_elapsed_sec)
    job_status["result"]["total_elapsed_sec"] = total_elapsed_sec
    job_status["result"]["overall_samples_per_sec"] = overall_samples_per_sec
    job_status["result"]["gating_elapsed_sec"] = gating_elapsed_sec
    job_status["result"]["overhead_elapsed_sec"] = overhead_elapsed_sec
    print(
        f"[train] job done: total_elapsed={total_elapsed_sec:.2f}s "
        f"overall_samples/s={overall_samples_per_sec:.2f} "
        f"overhead={overhead_elapsed_sec:.2f}s"
    )
    (output_dir / "job_status.json").write_text(json.dumps(job_status, ensure_ascii=False, indent=2), encoding="utf-8")
    return job_status
