from __future__ import annotations

import os
import random
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable

from ..search_options import clone_search_options
from .config import PolicyConfig
from .metadata import stable_config_hash
from .policy_bridge import apply_policy_search_options_bridge

_PROCESS_POOL_MAX_TASKS_PER_CHILD_ENV = "BOARD_AI_PROCESS_MAX_TASKS_PER_CHILD"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_episode_seed(job_seed: int, episode_index: int) -> int:
    rng = random.Random((job_seed << 20) ^ (episode_index * 2654435761))
    return rng.randint(1, (1 << 63) - 1)


def winner_to_float(winner: Any, shared_victory: bool, candidate_player: int) -> float:
    if shared_victory:
        return 0.5
    if winner is None:
        return 0.5
    try:
        wid = int(winner)
    except (TypeError, ValueError):
        return 0.5
    return 1.0 if wid == candidate_player else 0.0


def init_worker_process() -> None:
    """Limit per-process native threads to avoid CPU oversubscription."""
    thread_env = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }
    for k, v in thread_env.items():
        os.environ.setdefault(k, v)


def process_pool_kwargs(max_workers: int, max_tasks_per_child: int | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_workers": max_workers,
        "initializer": init_worker_process,
    }
    if max_tasks_per_child is None:
        raw = os.getenv(_PROCESS_POOL_MAX_TASKS_PER_CHILD_ENV, "128").strip()
        try:
            recycle = int(raw)
        except ValueError:
            recycle = 128
    else:
        recycle = int(max_tasks_per_child)
    if recycle > 0:
        kwargs["max_tasks_per_child"] = recycle
    return kwargs


def apply_policy_hook(
    policy_cfg: PolicyConfig,
    hook: Callable[[PolicyConfig, dict[str, Any]], PolicyConfig] | None,
    ctx: dict[str, Any],
) -> PolicyConfig:
    if hook is None:
        return policy_cfg
    mutated = hook(policy_cfg, ctx)
    if not isinstance(mutated, PolicyConfig):
        raise TypeError("policy hook must return PolicyConfig")
    return mutated


def apply_episode_hooks(
    result: dict[str, Any],
    *,
    hook_ctx: dict[str, Any],
    build_episode_context_hook: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None,
    label_sample_hook: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]] | None,
    postprocess_episode_payload_hook: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any]:
    if postprocess_episode_payload_hook is not None:
        mutated = postprocess_episode_payload_hook(result, hook_ctx)
        if not isinstance(mutated, dict):
            raise TypeError("postprocess_episode_payload hook must return dict")
        result = mutated

    if build_episode_context_hook is None and label_sample_hook is None:
        return result

    if build_episode_context_hook is not None:
        episode_ctx = build_episode_context_hook(result, hook_ctx)
        if not isinstance(episode_ctx, dict):
            raise TypeError("build_episode_context hook must return dict")
    else:
        episode_ctx = {}

    if label_sample_hook is not None:
        samples = list(result.get("samples", []))
        for sample in samples:
            labeled = label_sample_hook(sample, episode_ctx, hook_ctx)
            if not isinstance(labeled, dict):
                raise TypeError("label_sample hook must return dict")
            sample.update(labeled)
        result["samples"] = samples
    return result


def inject_temperature_decay_plugin(
    search_options: dict[str, Any] | None,
    *,
    temperature_initial: float,
    temperature_final: float,
    temperature_decay_plies: int,
) -> dict[str, Any]:
    """Inject optional per-episode linear temperature-decay plugin settings.

    The plugin is expressed via search_options keys and interpreted by game backends
    that opt in:
    - temperature_initial
    - temperature_final
    - temperature_decay_plies
    """
    opts = clone_search_options(search_options)
    decay_plies = max(0, int(temperature_decay_plies))
    if decay_plies <= 0:
        return opts
    opts["temperature_initial"] = float(max(0.0, temperature_initial))
    opts["temperature_final"] = float(max(0.0, temperature_final))
    opts["temperature_decay_plies"] = decay_plies
    return opts


def prepare_selfplay_submission(
    *,
    base_policy_cfg: PolicyConfig,
    build_policy: Callable[[PolicyConfig, int], Any],
    prepare_selfplay_policy_hook: Callable[[PolicyConfig, dict[str, Any]], PolicyConfig] | None,
    phase: str,
    step_index: int,
    episode_index: int,
    episode_seed: int,
    job_seed: int,
    engine: str,
    simulations: int,
    temperature: float,
    model_path: str | None,
    search_options: dict[str, Any] | None,
    include_phase_in_search_params: bool = False,
    dirichlet_alpha: float | None = None,
    dirichlet_epsilon: float | None = None,
    dirichlet_on_first_n_plies: int | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    search_params: dict[str, Any] = {
        "engine": engine,
        "simulations": simulations,
        "temperature": temperature,
        "search_options": clone_search_options(search_options),
        "model_path": model_path,
    }
    if include_phase_in_search_params:
        search_params["phase"] = phase
    if dirichlet_alpha is not None:
        search_params["dirichlet_alpha"] = float(dirichlet_alpha)
    if dirichlet_epsilon is not None:
        search_params["dirichlet_epsilon"] = float(dirichlet_epsilon)
    if dirichlet_on_first_n_plies is not None:
        search_params["dirichlet_on_first_n_plies"] = int(dirichlet_on_first_n_plies)
    search_hash = stable_config_hash(search_params)

    hook_ctx = {
        "phase": phase,
        "step_index": step_index,
        "episode_index": episode_index,
        "job_seed": job_seed,
    }
    runtime_policy_cfg = deepcopy(base_policy_cfg)
    runtime_policy_cfg.engine = engine
    runtime_policy_cfg.model_path = model_path
    runtime_policy_cfg.temperature = float(temperature)
    runtime_policy_cfg.search_options = clone_search_options(search_options)
    if dirichlet_alpha is not None:
        runtime_policy_cfg.dirichlet_alpha = float(dirichlet_alpha)
    if dirichlet_epsilon is not None:
        runtime_policy_cfg.dirichlet_epsilon = float(dirichlet_epsilon)
    if dirichlet_on_first_n_plies is not None:
        runtime_policy_cfg.dirichlet_on_first_n_plies = int(dirichlet_on_first_n_plies)
    runtime_policy_cfg = apply_policy_hook(
        apply_policy_search_options_bridge(runtime_policy_cfg, hook_ctx),
        prepare_selfplay_policy_hook,
        hook_ctx,
    )
    policy = build_policy(runtime_policy_cfg, simulations)

    search_log_item: dict[str, Any] = {
        "phase": phase,
        "step": step_index,
        "episode": episode_index,
        "seed": episode_seed,
        "simulations": simulations,
        "temperature": temperature,
        "search_options": clone_search_options(search_options),
        "model_path": model_path,
        "search_params_hash": search_hash,
    }
    if dirichlet_alpha is not None:
        search_log_item["dirichlet_alpha"] = float(dirichlet_alpha)
    if dirichlet_epsilon is not None:
        search_log_item["dirichlet_epsilon"] = float(dirichlet_epsilon)
    if dirichlet_on_first_n_plies is not None:
        search_log_item["dirichlet_on_first_n_plies"] = int(dirichlet_on_first_n_plies)
    return policy, search_hash, search_log_item


def build_eval_policy(
    *,
    base_policy_cfg: PolicyConfig,
    build_policy: Callable[[PolicyConfig, int], Any],
    prepare_eval_policy_hook: Callable[[PolicyConfig, dict[str, Any]], PolicyConfig] | None,
    phase: str,
    role: str,
    step_index: int,
    job_seed: int,
    engine: str,
    simulations: int,
    temperature: float,
    model_path: str | None,
    search_options: dict[str, Any] | None,
) -> tuple[PolicyConfig, Any]:
    hook_ctx = {
        "phase": phase,
        "role": role,
        "step_index": step_index,
        "job_seed": job_seed,
    }
    policy_cfg = deepcopy(base_policy_cfg)
    policy_cfg.engine = str(engine)
    policy_cfg.simulations = int(simulations)
    policy_cfg.temperature = float(temperature)
    policy_cfg.model_path = model_path
    policy_cfg.search_options = clone_search_options(search_options)
    policy_cfg = apply_policy_hook(
        apply_policy_search_options_bridge(policy_cfg, hook_ctx),
        prepare_eval_policy_hook,
        hook_ctx,
    )
    return policy_cfg, build_policy(policy_cfg, policy_cfg.simulations)


def run_arena_series(
    *,
    run_arena_match: Callable[[int, Any, Any], dict[str, Any]],
    p0,
    p1,
    games: int,
    max_workers: int,
    job_seed: int,
    worker_pool: str = "thread",
    phase_label: str = "eval",
    process_pool_max_tasks_per_child: int | None = None,
) -> tuple[list[dict[str, Any]], float]:
    arena_records: list[dict[str, Any]] = []
    pool = str(worker_pool).lower()
    if pool == "process":
        gate_executor = ProcessPoolExecutor(
            **process_pool_kwargs(max_workers, process_pool_max_tasks_per_child),
        )
    else:
        gate_executor = ThreadPoolExecutor(max_workers=max_workers)
    with gate_executor:
        futures = {}
        for i in range(games):
            seed = make_episode_seed(job_seed ^ 0xABCDEF, i)
            if i % 2 == 0:
                left, right = p0, p1
                candidate_player = 0
            else:
                left, right = p1, p0
                candidate_player = 1
            fut = gate_executor.submit(run_arena_match, seed, left, right)
            futures[fut] = (i, seed, candidate_player)
        done_count = 0
        progress_stride = max(1, int(games) // 10)
        for fut in as_completed(futures):
            idx, seed, candidate_player = futures[fut]
            result = fut.result()
            score = winner_to_float(result.get("winner"), bool(result.get("shared_victory", False)), candidate_player)
            arena_records.append(
                {
                    "index": idx,
                    "seed": seed,
                    "candidate_player": candidate_player,
                    "winner": result.get("winner"),
                    "shared_victory": bool(result.get("shared_victory", False)),
                    "scores": result.get("scores"),
                    "candidate_score": score,
                }
            )
            done_count += 1
            if done_count % progress_stride == 0 or done_count == games:
                print(
                    f"[train] {phase_label} progress: {done_count}/{games}",
                    flush=True,
                )
    arena_records.sort(key=lambda x: int(x["index"]))
    win_rate = sum(float(r["candidate_score"]) for r in arena_records) / max(1, len(arena_records))
    return arena_records, win_rate


def run_bounded_futures(
    *,
    total_jobs: int,
    max_inflight: int,
    submit_job: Callable[[int], tuple[Any, Any]],
) -> list[tuple[Any, Any]]:
    """Run jobs with bounded concurrency and return (meta, result) pairs."""
    total = max(0, int(total_jobs))
    if total <= 0:
        return []
    inflight_limit = max(1, min(int(max_inflight), total))
    future_map: dict[Any, Any] = {}
    completed: list[tuple[Any, Any]] = []
    next_local_index = 0

    for _ in range(inflight_limit):
        future, meta = submit_job(next_local_index)
        future_map[future] = meta
        next_local_index += 1

    while future_map:
        done_set, _pending = wait(tuple(future_map.keys()), return_when=FIRST_COMPLETED)
        for done_future in done_set:
            meta = future_map.pop(done_future)
            completed.append((meta, done_future.result()))
            if next_local_index < total:
                future, next_meta = submit_job(next_local_index)
                future_map[future] = next_meta
                next_local_index += 1
    return completed

