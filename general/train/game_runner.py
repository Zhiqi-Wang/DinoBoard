from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Type

from .config import MctsSchedule, PolicyConfig, TrainJobConfig
from .extensions import TrainPipelineHooks
from .pipeline import run_train_job
from ..search_options import clone_search_options


def build_policy_from_config(policy_cfg: PolicyConfig, simulations: int, move_policy_cls: Type[Any]) -> Any:
    return move_policy_cls(
        engine=policy_cfg.engine,
        simulations=simulations,
        temperature=policy_cfg.temperature,
        dirichlet_alpha=policy_cfg.dirichlet_alpha,
        dirichlet_epsilon=policy_cfg.dirichlet_epsilon,
        dirichlet_on_first_n_plies=policy_cfg.dirichlet_on_first_n_plies,
        model_path=policy_cfg.model_path,
        search_options=clone_search_options(policy_cfg.search_options),
    )


def run_game_train_job(
    *,
    config: TrainJobConfig,
    output_dir: Path,
    job_seed: int,
    move_policy_cls: Type[Any],
    run_selfplay_episode: Callable[[int, int, Any, str], dict[str, Any]],
    run_arena_match: Callable[[int, Any, Any], dict[str, Any]],
    benchmark_engine: str,
    run_trainer: Callable[..., dict[str, Any]] | None = None,
    build_initial_model: Callable[[TrainJobConfig, Path, int], str] | None = None,
    pipeline_hooks: TrainPipelineHooks | None = None,
) -> dict[str, Any]:
    effective_benchmark_engine = str(
        config.benchmark_engine if config.benchmark_engine is not None else benchmark_engine
    )

    def _build_policy(policy_cfg: PolicyConfig, simulations: int) -> Any:
        return build_policy_from_config(policy_cfg, simulations, move_policy_cls)

    return run_train_job(
        config=config,
        output_dir=output_dir,
        job_seed=job_seed,
        run_selfplay_episode=run_selfplay_episode,
        run_arena_match=run_arena_match,
        build_policy=_build_policy,
        benchmark_engine=effective_benchmark_engine,
        run_trainer=run_trainer,
        build_initial_model=build_initial_model,
        pipeline_hooks=pipeline_hooks,
    )


def run_train_cli(
    *,
    description: str,
    run_job: Callable[[TrainJobConfig, Path, int], dict[str, Any]],
    default_config_factory: Callable[[str | None], TrainJobConfig],
    support_benchmark_onnx: bool,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="", help="Path to training config json.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for job artifacts.")
    parser.add_argument("--seed", type=int, default=20260323, help="Job seed.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Override selfplay episodes per step when > 0 (total episodes = episodes * steps).",
    )
    parser.add_argument("--workers", type=int, default=0, help="Override selfplay workers/games when > 0.")
    parser.add_argument(
        "--worker-pool",
        type=str,
        default="",
        choices=["thread", "process"],
        help="Override selfplay worker pool type.",
    )
    parser.add_argument("--warm-start-episodes", type=int, default=0, help="Warm-start selfplay episodes before step loop.")
    parser.add_argument("--warm-start-engine", type=str, default="", help="Warm-start engine (e.g., heuristic).")
    parser.add_argument(
        "--warm-start-simulations",
        type=int,
        default=0,
        help="Warm-start simulations per move when > 0.",
    )
    parser.add_argument(
        "--warm-start-train-passes",
        type=int,
        default=0,
        help="Warm-start trainer passes on warm-start buffer when > 0.",
    )
    parser.add_argument("--gating-games", type=int, default=0, help="Override gating games when > 0.")
    parser.add_argument("--eval-every", type=int, default=0, help="Run periodic eval every N steps when > 0.")
    parser.add_argument("--eval-workers", type=int, default=0, help="Periodic eval worker count when > 0.")
    parser.add_argument("--eval-games", type=int, default=0, help="Periodic eval games against benchmark when > 0.")
    parser.add_argument(
        "--history-best-games",
        type=int,
        default=0,
        help="Periodic eval games against historical best when > 0.",
    )
    parser.add_argument(
        "--promote-best-win-rate",
        type=float,
        default=0.0,
        help="Win-rate threshold to replace historical best when > 0.",
    )
    parser.add_argument("--save-every", type=int, default=0, help="Save latest model snapshot every N steps when > 0.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override trainer batch size when > 0.")
    parser.add_argument("--epochs", type=int, default=0, help="Override trainer epochs when > 0.")
    parser.add_argument("--hidden", type=int, default=0, help="Override trainer hidden size when > 0.")
    parser.add_argument("--mlp-layers", type=int, default=0, help="Override trainer MLP layers when > 0.")
    parser.add_argument("--updates-per-step", type=int, default=0, help="Override trainer updates per step when > 0.")
    parser.add_argument("--buffer-size", type=int, default=0, help="Override trainer buffer size when > 0.")
    parser.add_argument("--steps", type=int, default=0, help="Override trainer steps when > 0.")
    parser.add_argument(
        "--schedule-start-simulations",
        type=int,
        default=0,
        help="Override mcts schedule start simulations when > 0.",
    )
    parser.add_argument(
        "--schedule-end-simulations",
        type=int,
        default=0,
        help="Override mcts schedule end simulations when > 0.",
    )
    parser.add_argument(
        "--process-max-tasks-per-child",
        type=int,
        default=-1,
        help="Override process pool max tasks per child (<=0 keeps config/environment).",
    )
    parser.add_argument(
        "--eval-candidate-temp",
        type=float,
        default=999.0,
        help="Override eval/gating candidate temperature.",
    )
    parser.add_argument(
        "--eval-opponent-temp",
        type=float,
        default=999.0,
        help="Override eval/gating benchmark/history temperature.",
    )
    parser.add_argument(
        "--eval-sim-floor",
        type=int,
        default=0,
        help="Override minimum simulations for eval/gating opponents when > 0.",
    )
    parser.add_argument(
        "--eval-simulations",
        type=int,
        default=0,
        help="Override periodic eval simulations for candidate/opponents when > 0 (fixed value).",
    )
    parser.add_argument(
        "--diag-enable",
        type=str,
        default="",
        choices=["", "true", "false"],
        help="Enable/disable selfplay diagnostics output.",
    )
    parser.add_argument(
        "--diag-zero-epsilon",
        type=float,
        default=-1.0,
        help="Override epsilon for zero-ish value label ratio when >= 0.",
    )
    parser.add_argument(
        "--selfplay-dirichlet-alpha",
        type=float,
        default=-1.0,
        help="Override selfplay dirichlet alpha when >= 0.",
    )
    parser.add_argument(
        "--selfplay-dirichlet-epsilon",
        type=float,
        default=-1.0,
        help="Override selfplay dirichlet epsilon when >= 0.",
    )
    parser.add_argument(
        "--selfplay-dirichlet-first-n-plies",
        type=int,
        default=-1,
        help="Override selfplay dirichlet_on_first_n_plies when >= 0.",
    )
    parser.add_argument(
        "--selfplay-temp-initial",
        type=float,
        default=999.0,
        help="Override selfplay exploration.temperature_initial.",
    )
    parser.add_argument(
        "--selfplay-temp-final",
        type=float,
        default=999.0,
        help="Override selfplay exploration.temperature_final.",
    )
    parser.add_argument(
        "--selfplay-temp-decay-plies",
        type=int,
        default=-1,
        help="Override selfplay exploration.temperature_decay_plies when >= 0.",
    )
    if support_benchmark_onnx:
        parser.add_argument(
            "--benchmark-onnx",
            type=str,
            default="",
            help="Current ONNX model used as benchmark in gating.",
        )
    args = parser.parse_args()

    benchmark_onnx = args.benchmark_onnx if support_benchmark_onnx else ""
    if args.config:
        cfg = TrainJobConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
        if support_benchmark_onnx and benchmark_onnx:
            cfg.benchmark_onnx_path = benchmark_onnx
    else:
        cfg = default_config_factory(benchmark_onnx or None)

    if args.episodes > 0:
        cfg.selfplay.episodes = int(args.episodes)
    if args.workers > 0:
        workers = int(args.workers)
        cfg.selfplay.parallel_games = workers
        cfg.selfplay.max_workers = workers
        # Keep warm-start/selfplay/eval worker counts aligned by default.
        # Users can still override eval workers explicitly with --eval-workers.
        if int(args.eval_workers) <= 0:
            cfg.gating.eval_workers = workers
    if args.worker_pool:
        cfg.selfplay.worker_pool = str(args.worker_pool)
    if args.warm_start_episodes > 0:
        cfg.selfplay.warm_start_episodes = int(args.warm_start_episodes)
    if args.warm_start_engine:
        cfg.selfplay.warm_start_engine = str(args.warm_start_engine)
    if args.warm_start_simulations > 0:
        cfg.selfplay.warm_start_simulations = int(args.warm_start_simulations)
    if args.warm_start_train_passes > 0:
        cfg.selfplay.warm_start_train_passes = int(args.warm_start_train_passes)
    if args.gating_games > 0:
        cfg.gating.games = int(args.gating_games)
    if args.eval_every > 0:
        cfg.gating.eval_every_steps = int(args.eval_every)
    if args.eval_workers > 0:
        cfg.gating.eval_workers = int(args.eval_workers)
    if args.eval_games > 0:
        cfg.gating.eval_games = int(args.eval_games)
    if args.history_best_games > 0:
        cfg.gating.history_best_games = int(args.history_best_games)
    if args.promote_best_win_rate > 0:
        cfg.gating.history_best_accept_win_rate = float(args.promote_best_win_rate)
    if args.save_every > 0:
        cfg.gating.save_latest_every_steps = int(args.save_every)
    if args.batch_size > 0:
        cfg.trainer.batch_size = int(args.batch_size)
    if args.epochs > 0:
        cfg.trainer.epochs = int(args.epochs)
    if args.hidden > 0:
        cfg.trainer.hidden = int(args.hidden)
    if args.mlp_layers > 0:
        cfg.trainer.mlp_layers = int(args.mlp_layers)
    if args.updates_per_step > 0:
        cfg.trainer.updates_per_step = int(args.updates_per_step)
    if args.buffer_size > 0:
        cfg.trainer.buffer_size = int(args.buffer_size)
    if args.steps > 0:
        cfg.trainer.steps = int(args.steps)
    if args.schedule_start_simulations > 0 or args.schedule_end_simulations > 0:
        if cfg.selfplay.mcts_schedule is None:
            base_sim = max(1, int(cfg.selfplay.policy.simulations))
            cfg.selfplay.mcts_schedule = MctsSchedule(
                type="linear",
                start_simulations=base_sim,
                end_simulations=base_sim,
            )
        if args.schedule_start_simulations > 0:
            cfg.selfplay.mcts_schedule.start_simulations = int(args.schedule_start_simulations)
        if args.schedule_end_simulations > 0:
            cfg.selfplay.mcts_schedule.end_simulations = int(args.schedule_end_simulations)
    if args.process_max_tasks_per_child > 0:
        cfg.runtime.process_pool_max_tasks_per_child = int(args.process_max_tasks_per_child)
    if args.eval_candidate_temp != 999.0:
        cfg.eval.candidate_temperature = float(args.eval_candidate_temp)
    if args.eval_opponent_temp != 999.0:
        cfg.eval.opponent_temperature = float(args.eval_opponent_temp)
    if args.eval_sim_floor > 0:
        cfg.eval.simulations_floor = int(args.eval_sim_floor)
    if args.eval_simulations > 0:
        cfg.eval.simulations_fixed = int(args.eval_simulations)
    if args.diag_enable == "true":
        cfg.diagnostics.enabled = True
    elif args.diag_enable == "false":
        cfg.diagnostics.enabled = False
    if args.diag_zero_epsilon >= 0.0:
        cfg.diagnostics.zero_value_epsilon = float(args.diag_zero_epsilon)
    if args.selfplay_dirichlet_alpha >= 0.0:
        cfg.selfplay.exploration.dirichlet_alpha = float(args.selfplay_dirichlet_alpha)
    if args.selfplay_dirichlet_epsilon >= 0.0:
        cfg.selfplay.exploration.dirichlet_epsilon = float(args.selfplay_dirichlet_epsilon)
    if args.selfplay_dirichlet_first_n_plies >= 0:
        cfg.selfplay.exploration.dirichlet_on_first_n_plies = int(args.selfplay_dirichlet_first_n_plies)
    if args.selfplay_temp_initial != 999.0:
        cfg.selfplay.exploration.temperature_initial = float(args.selfplay_temp_initial)
    if args.selfplay_temp_final != 999.0:
        cfg.selfplay.exploration.temperature_final = float(args.selfplay_temp_final)
    if args.selfplay_temp_decay_plies >= 0:
        cfg.selfplay.exploration.temperature_decay_plies = int(args.selfplay_temp_decay_plies)
    if support_benchmark_onnx and not cfg.benchmark_onnx_path:
        raise RuntimeError("--benchmark-onnx is required for this training entry.")

    status = run_job(cfg, output_dir=Path(args.output), job_seed=int(args.seed))
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0
