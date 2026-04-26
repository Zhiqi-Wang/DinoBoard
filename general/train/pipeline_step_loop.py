from __future__ import annotations

import json
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .config import PolicyConfig, TrainJobConfig
from .mcts_schedule import resolve_simulations
from .pipeline_periodic_eval import run_periodic_eval_for_step
from .pipeline_support import (
    apply_episode_hooks,
    inject_temperature_decay_plugin,
    make_episode_seed,
    prepare_selfplay_submission,
    run_bounded_futures,
    utc_now,
)
from ..search_options import clone_search_options

def run_training_step_loop(
    *,
    state: dict[str, Any],
    config: TrainJobConfig,
    output_dir: Path,
    artifacts_dir: Path,
    job_seed: int,
    loop_steps: int,
    episodes: int,
    max_workers: int,
    eval_every_steps: int,
    eval_games: int,
    history_best_games: int,
    history_best_accept_win_rate: float,
    save_latest_every_steps: int,
    eval_candidate_temperature: float,
    eval_opponent_temperature: float,
    eval_simulations_floor: int,
    worker_pool: str,
    eval_workers: int,
    process_pool_max_tasks_per_child: int,
    diagnostics_enabled: bool,
    diagnostics_zero_eps: float,
    bench_model: Path | None,
    benchmark_engine: str,
    prepare_selfplay_policy_hook: Callable[[PolicyConfig, dict[str, Any]], PolicyConfig] | None,
    prepare_eval_policy_hook: Callable[[PolicyConfig, dict[str, Any]], PolicyConfig] | None,
    build_episode_context_hook: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None,
    label_sample_hook: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]] | None,
    postprocess_episode_payload_hook: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None,
    enrich_periodic_eval_summary_hook: Callable[[dict[str, Any], dict[str, Any]], None] | None,
    run_selfplay_episode: Callable[[int, int, Any, str], dict[str, Any]],
    run_arena_match: Callable[[int, Any, Any], dict[str, Any]],
    build_policy: Callable[[PolicyConfig, int], Any],
    run_trainer: Callable[..., dict[str, Any]] | None,
    model_dir: Path,
    latest_model_path: Path,
    latest_checkpoint_path: Path,
    best_model_path: Path,
    best_checkpoint_path: Path,
    copy_model_pair: Callable[..., None],
    write_history_best_manifest: Callable[..., None],
    executor,
) -> None:
    for step_idx in range(loop_steps):
        episodes_this_step = episodes
        if episodes_this_step <= 0:
            continue
        print(
            f"[train] loop step {step_idx + 1}/{loop_steps}: "
            f"episodes={episodes_this_step} model={state['current_model_path']}",
            flush=True,
        )

        state["job_status"]["phase"] = "selfplay"
        state["job_status"]["phase_started_at"] = utc_now()
        (output_dir / "job_status.json").write_text(json.dumps(state["job_status"], ensure_ascii=False, indent=2), encoding="utf-8")

        selfplay_step_t0 = time.perf_counter()
        step_simulations = resolve_simulations(
            step_idx,
            config.selfplay.policy.simulations,
            config.selfplay.mcts_schedule,
            loop_steps,
        )

        def _submit_one(local_i: int) -> tuple[Any, tuple[int, int, int, str, int]]:
            ep = state["next_episode_index"] + local_i
            ep_seed = make_episode_seed(job_seed, ep)
            simulations = step_simulations
            runtime_temperature = float(config.selfplay.policy.temperature)
            runtime_search_options = inject_temperature_decay_plugin(
                clone_search_options(config.selfplay.policy.search_options),
                temperature_initial=float(config.selfplay.exploration.temperature_initial),
                temperature_final=float(config.selfplay.exploration.temperature_final),
                temperature_decay_plies=int(getattr(config.selfplay.exploration, "temperature_decay_plies", 0)),
            )
            policy, search_hash, search_log_item = prepare_selfplay_submission(
                base_policy_cfg=config.selfplay.policy,
                build_policy=build_policy,
                prepare_selfplay_policy_hook=prepare_selfplay_policy_hook,
                phase="selfplay",
                step_index=step_idx + 1,
                episode_index=ep,
                episode_seed=ep_seed,
                job_seed=job_seed,
                engine=str(config.selfplay.policy.engine),
                simulations=simulations,
                temperature=runtime_temperature,
                model_path=state["current_model_path"],
                search_options=runtime_search_options,
                dirichlet_alpha=float(config.selfplay.exploration.dirichlet_alpha),
                dirichlet_epsilon=float(config.selfplay.exploration.dirichlet_epsilon),
                dirichlet_on_first_n_plies=int(config.selfplay.exploration.dirichlet_on_first_n_plies),
            )
            future = executor.submit(run_selfplay_episode, ep, ep_seed, policy, search_hash)
            state["search_log"].append(search_log_item)
            return future, (ep, ep_seed, simulations, search_hash, step_idx + 1)

        step_samples = 0
        step_normal_samples = 0
        step_sample_records: list[dict[str, Any]] = []
        step_samples_with_z = 0
        step_zeroish_samples = 0
        step_shared_games = 0
        step_total_plies = 0
        step_winner_p0 = 0
        step_winner_p1 = 0
        step_winner_other = 0
        for (ep, ep_seed, simulations, search_hash, step_no), result in run_bounded_futures(
            total_jobs=episodes_this_step,
            max_inflight=max_workers,
            submit_job=_submit_one,
        ):
            result = apply_episode_hooks(
                result,
                hook_ctx={
                    "phase": "selfplay",
                    "step_index": step_no,
                    "episode_index": ep,
                    "job_seed": job_seed,
                },
                build_episode_context_hook=build_episode_context_hook,
                label_sample_hook=label_sample_hook,
                postprocess_episode_payload_hook=postprocess_episode_payload_hook,
            )
            try:
                pid = int(result.get("worker_pid")) if result.get("worker_pid") is not None else None
            except (TypeError, ValueError):
                pid = None
            if pid is not None and pid > 0:
                state["selfplay_worker_pids"].add(pid)
            samples = list(result.get("samples", []))
            step_samples += len(samples)
            step_normal_samples += sum(
                1
                for srec in samples
                if float(
                    srec.get(
                        "normal_phase",
                        srec.get("phase", 0.0),
                    )
                )
                > 0.5
            )
            state["total_samples_count"] += len(samples)
            step_sample_records.extend(samples)
            plies = int(result.get("plies", len(samples)))
            step_total_plies += max(0, plies)
            winner_raw = result.get("winner")
            try:
                winner_int = int(winner_raw) if winner_raw is not None else -1
            except (TypeError, ValueError):
                winner_int = -1
            if winner_int == 0:
                step_winner_p0 += 1
            elif winner_int == 1:
                step_winner_p1 += 1
            else:
                step_winner_other += 1
            if bool(result.get("shared_victory", False)):
                state["diagnostics_shared_games"] += 1
                step_shared_games += 1
            if diagnostics_enabled:
                for srec in samples:
                    z_raw = srec.get("z")
                    if isinstance(z_raw, (int, float)):
                        state["diagnostics_samples_with_z"] += 1
                        step_samples_with_z += 1
                        if abs(float(z_raw)) <= diagnostics_zero_eps:
                            state["diagnostics_zeroish_samples"] += 1
                            step_zeroish_samples += 1
            state["selfplay_results"].append(
                {
                    "step": step_no,
                    "episode": ep,
                    "seed": ep_seed,
                    "simulations": simulations,
                    "search_params_hash": search_hash,
                    "winner": result.get("winner"),
                    "shared_victory": bool(result.get("shared_victory", False)),
                    "plies": int(result.get("plies", len(samples))),
                    "sample_count": len(samples),
                }
            )
        state["next_episode_index"] += episodes_this_step
        selfplay_step_elapsed = max(1e-9, time.perf_counter() - selfplay_step_t0)
        state["selfplay_elapsed_sec"] += selfplay_step_elapsed
        print(
            f"[train] selfplay done: step={step_idx + 1}/{loop_steps} "
            f"mcts={step_simulations} "
            f"episodes={episodes_this_step} samples={step_samples} normal_samples={step_normal_samples} "
            f"avg_plies={float(step_total_plies) / max(1, episodes_this_step):.2f} "
            f"wins(p0/p1/other)={step_winner_p0}/{step_winner_p1}/{step_winner_other} "
            f"elapsed={selfplay_step_elapsed:.2f}s samples/s={float(step_samples) / selfplay_step_elapsed:.2f}",
            flush=True,
        )

        state["job_status"]["phase"] = "train"
        state["job_status"]["phase_started_at"] = utc_now()
        (output_dir / "job_status.json").write_text(json.dumps(state["job_status"], ensure_ascii=False, indent=2), encoding="utf-8")

        trainer_cfg = deepcopy(config)
        # Keep outer-loop semantics simple and predictable:
        # one selfplay step -> one trainer step.
        # Per-step training intensity is controlled by updates_per_step.
        trainer_cfg.trainer.steps = 1
        train_step_t0 = time.perf_counter()
        if run_trainer is not None:
            train_summary = run_trainer(
                trainer_cfg,
                artifacts_dir,
                resume_checkpoint_path=state["current_checkpoint_path"],
                step_index=step_idx + 1,
                total_steps=loop_steps,
                incremental_samples=step_sample_records,
            )
            if not isinstance(train_summary, dict):
                raise TypeError("run_trainer must return dict[str, Any]")
            state["train_summary"] = train_summary
        else:
            state["train_summary"] = {
                "status": "skipped",
                "reason": "trainer_not_provided",
                "note": "No trainer runner provided; keep manifest contract only.",
            }
        train_step_elapsed = max(1e-9, time.perf_counter() - train_step_t0)
        state["train_elapsed_sec"] += train_step_elapsed
        state["train_summary"]["elapsed_sec"] = train_step_elapsed

        print(
            f"[train] trainer done: step={step_idx + 1}/{loop_steps} "
            f"status={state['train_summary'].get('status')} elapsed={train_step_elapsed:.2f}s",
            flush=True,
        )

        state["trained_model_path"] = str(state["train_summary"].get("model_path") or "") or None
        state["current_checkpoint_path"] = str(state["train_summary"].get("checkpoint_path") or "") or state["current_checkpoint_path"]
        if state["trained_model_path"]:
            state["current_model_path"] = str(state["trained_model_path"])
            state["train_summary"]["model_path"] = state["current_model_path"]
            state["trained_model_path"] = state["current_model_path"]
            copy_model_pair(
                src_model_path=state["current_model_path"],
                src_checkpoint_path=state["current_checkpoint_path"],
                dst_model_path=latest_model_path,
                dst_checkpoint_path=latest_checkpoint_path,
            )

        if save_latest_every_steps > 0 and (step_idx + 1) % save_latest_every_steps == 0 and state["current_model_path"]:
            latest_onnx = artifacts_dir / "models" / f"latest_step_{step_idx + 1}.onnx"
            shutil.copy2(state["current_model_path"], latest_onnx)
            if state["current_checkpoint_path"]:
                src_ckpt = Path(state["current_checkpoint_path"])
                if src_ckpt.exists():
                    latest_ckpt = artifacts_dir / "models" / f"latest_step_{step_idx + 1}.pt"
                    shutil.copy2(src_ckpt, latest_ckpt)
            print(f"[train] saved latest snapshot: step={step_idx + 1} path={latest_onnx}", flush=True)

        should_eval = eval_every_steps > 0 and eval_games > 0 and (step_idx + 1) % eval_every_steps == 0
        if should_eval and state["current_model_path"]:
            run_periodic_eval_for_step(
                state=state,
                config=config,
                artifacts_dir=artifacts_dir,
                step_idx=step_idx,
                loop_steps=loop_steps,
                eval_games=eval_games,
                history_best_games=history_best_games,
                history_best_accept_win_rate=history_best_accept_win_rate,
                eval_candidate_temperature=eval_candidate_temperature,
                eval_opponent_temperature=eval_opponent_temperature,
                eval_simulations_floor=eval_simulations_floor,
                worker_pool=worker_pool,
                eval_workers=eval_workers,
                process_pool_max_tasks_per_child=process_pool_max_tasks_per_child,
                bench_model=bench_model,
                benchmark_engine=benchmark_engine,
                prepare_eval_policy_hook=prepare_eval_policy_hook,
                enrich_periodic_eval_summary_hook=enrich_periodic_eval_summary_hook,
                run_arena_match=run_arena_match,
                build_policy=build_policy,
                job_seed=job_seed,
                copy_model_pair=copy_model_pair,
                write_history_best_manifest=write_history_best_manifest,
                best_model_path=best_model_path,
                best_checkpoint_path=best_checkpoint_path,
            )
            # Anti-collapse guardrail disabled: rollback prevents nopeek training
            # from making incremental progress. Standard gating (history_best
            # promotion at >=60%) in pipeline_periodic_eval.py is sufficient.
        state["step_summaries"].append(
            {
                "step": step_idx + 1,
                "episodes": episodes_this_step,
                "step_samples": step_samples,
                "step_total_plies": step_total_plies,
                "step_avg_plies": float(step_total_plies) / max(1, episodes_this_step),
                "step_winner_p0": step_winner_p0,
                "step_winner_p1": step_winner_p1,
                "step_winner_other": step_winner_other,
                "step_shared_games": step_shared_games,
                "step_shared_game_rate": float(step_shared_games) / max(1, episodes_this_step),
                "step_zeroish_value_ratio": (
                    float(step_zeroish_samples) / max(1, step_samples_with_z)
                    if diagnostics_enabled and step_samples_with_z > 0
                    else None
                ),
                "selfplay_elapsed_sec": selfplay_step_elapsed,
                "trainer_elapsed_sec": train_step_elapsed,
                "model_path_after_step": state["current_model_path"],
            }
        )

