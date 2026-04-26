from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .config import PolicyConfig, TrainJobConfig
from .pipeline_support import (
    apply_episode_hooks,
    inject_temperature_decay_plugin,
    make_episode_seed,
    prepare_selfplay_submission,
    run_bounded_futures,
)
from ..search_options import clone_search_options


def run_warm_start_and_init_best(
    *,
    state: dict[str, Any],
    config: TrainJobConfig,
    artifacts_dir: Path,
    job_seed: int,
    warm_start_episodes: int,
    warm_start_engine: str,
    warm_start_simulations: int,
    warm_start_train_passes: int,
    max_workers: int,
    diagnostics_enabled: bool,
    diagnostics_zero_eps: float,
    prepare_selfplay_policy_hook: Callable[[PolicyConfig, dict[str, Any]], PolicyConfig] | None,
    build_episode_context_hook: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None,
    label_sample_hook: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]] | None,
    postprocess_episode_payload_hook: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None,
    run_selfplay_episode: Callable[[int, int, Any, str], dict[str, Any]],
    build_policy: Callable[[PolicyConfig, int], Any],
    run_trainer: Callable[..., dict[str, Any]] | None,
    latest_model_path: Path,
    latest_checkpoint_path: Path,
    best_model_path: Path,
    best_checkpoint_path: Path,
    copy_model_pair: Callable[..., None],
    write_history_best_manifest: Callable[..., None],
    executor,
) -> None:
    if warm_start_episodes > 0:
        print(
            f"[train] warm start: episodes={warm_start_episodes} engine={warm_start_engine}",
            flush=True,
        )
        warm_t0 = time.perf_counter()

        def _submit_warm(local_i: int) -> tuple[Any, tuple[int, int, int, str, int]]:
            ep = state["next_episode_index"] + local_i
            ep_seed = make_episode_seed(job_seed, ep)
            simulations = warm_start_simulations if warm_start_simulations > 0 else max(1, int(config.selfplay.policy.simulations))
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
                phase="warm_start",
                step_index=0,
                episode_index=ep,
                episode_seed=ep_seed,
                job_seed=job_seed,
                engine=warm_start_engine,
                simulations=simulations,
                temperature=runtime_temperature,
                model_path=None,
                search_options=runtime_search_options,
                include_phase_in_search_params=True,
                dirichlet_alpha=float(config.selfplay.exploration.dirichlet_alpha),
                dirichlet_epsilon=float(config.selfplay.exploration.dirichlet_epsilon),
                dirichlet_on_first_n_plies=int(config.selfplay.exploration.dirichlet_on_first_n_plies),
            )
            future = executor.submit(run_selfplay_episode, ep, ep_seed, policy, search_hash)
            state["search_log"].append(search_log_item)
            return future, (ep, ep_seed, simulations, search_hash, 0)

        warm_samples = 0
        warm_sample_records: list[dict[str, Any]] = []
        for (ep, ep_seed, simulations, search_hash, _step_no), result in run_bounded_futures(
            total_jobs=warm_start_episodes,
            max_inflight=max_workers,
            submit_job=_submit_warm,
        ):
            result = apply_episode_hooks(
                result,
                hook_ctx={
                    "phase": "warm_start",
                    "step_index": 0,
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
            warm_samples += len(samples)
            state["total_samples_count"] += len(samples)
            warm_sample_records.extend(samples)
            if bool(result.get("shared_victory", False)):
                state["diagnostics_shared_games"] += 1
            if diagnostics_enabled:
                for srec in samples:
                    z_raw = srec.get("z")
                    if isinstance(z_raw, (int, float)):
                        state["diagnostics_samples_with_z"] += 1
                        if abs(float(z_raw)) <= diagnostics_zero_eps:
                            state["diagnostics_zeroish_samples"] += 1
            state["selfplay_results"].append(
                {
                    "step": 0,
                    "phase": "warm_start",
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

        warm_elapsed = max(1e-9, time.perf_counter() - warm_t0)
        state["selfplay_elapsed_sec"] += warm_elapsed
        state["next_episode_index"] += warm_start_episodes
        print(
            f"[train] warm start done: episodes={warm_start_episodes} samples={warm_samples} "
            f"elapsed={warm_elapsed:.2f}s samples/s={float(warm_samples) / warm_elapsed:.2f}",
            flush=True,
        )
        if warm_start_train_passes > 0:
            if run_trainer is None:
                print(
                    "[train] warm start trainer passes skipped: trainer_not_provided",
                    flush=True,
                )
            else:
                for pass_idx in range(warm_start_train_passes):
                    pass_t0 = time.perf_counter()
                    trainer_cfg = deepcopy(config)
                    # Warm-start passes should consume the collected warm data as true passes.
                    # Using steps=1 would only run a tiny fixed number of random updates and
                    # leaves most warm samples effectively unused.
                    trainer_cfg.trainer.steps = 0
                    warm_incremental = warm_sample_records if pass_idx == 0 else []
                    warm_train_summary = run_trainer(
                        trainer_cfg,
                        artifacts_dir,
                        resume_checkpoint_path=state["current_checkpoint_path"],
                        step_index=-(pass_idx + 1),
                        total_steps=warm_start_train_passes,
                        incremental_samples=warm_incremental,
                    )
                    if not isinstance(warm_train_summary, dict):
                        raise TypeError("run_trainer must return dict[str, Any]")
                    pass_elapsed = max(1e-9, time.perf_counter() - pass_t0)
                    state["train_elapsed_sec"] += pass_elapsed
                    state["trained_model_path"] = str(warm_train_summary.get("model_path") or "") or None
                    state["current_checkpoint_path"] = str(warm_train_summary.get("checkpoint_path") or "") or state["current_checkpoint_path"]
                    if state["trained_model_path"]:
                        state["current_model_path"] = state["trained_model_path"]
                    metrics = warm_train_summary.get("metrics")
                    if isinstance(metrics, dict) and metrics:
                        ft = metrics.get("final_total_loss")
                        if isinstance(ft, (int, float)):
                            print(
                                f"[train] warm start trainer pass {pass_idx + 1}/{warm_start_train_passes}: "
                                f"elapsed={pass_elapsed:.2f}s final_total_loss={float(ft):.6f}",
                                flush=True,
                            )
                        else:
                            print(
                                f"[train] warm start trainer pass {pass_idx + 1}/{warm_start_train_passes}: "
                                f"elapsed={pass_elapsed:.2f}s",
                                flush=True,
                            )
                    else:
                        print(
                            f"[train] warm start trainer pass {pass_idx + 1}/{warm_start_train_passes}: "
                            f"elapsed={pass_elapsed:.2f}s",
                            flush=True,
                        )

    if state["current_model_path"]:
        copy_model_pair(
            src_model_path=state["current_model_path"],
            src_checkpoint_path=state["current_checkpoint_path"],
            dst_model_path=latest_model_path,
            dst_checkpoint_path=latest_checkpoint_path,
        )
        copy_model_pair(
            src_model_path=state["current_model_path"],
            src_checkpoint_path=state["current_checkpoint_path"],
            dst_model_path=best_model_path,
            dst_checkpoint_path=best_checkpoint_path,
        )
        state["history_best_model_path"] = str(best_model_path)
        write_history_best_manifest(
            step=0,
            model_path=state["history_best_model_path"],
            win_rate=None,
            source="run_init_model",
        )
        print(
            f"[train] best model init: step=0 model={state['history_best_model_path']}",
            flush=True,
        )
        print(
            f"[train] latest model init: step=0 model={latest_model_path}",
            flush=True,
        )

