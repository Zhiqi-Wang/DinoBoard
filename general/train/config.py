from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..search_options import clone_search_options


@dataclass(slots=True)
class MctsSchedule:
    """Linear MCTS sim count vs outer training step (warm start excluded).

    Effective linear range is fixed globally in general/train:
    - start step = 0
    - end step = total training steps
    """

    type: str = "linear"
    start_simulations: int = 120
    end_simulations: int = 800


@dataclass(slots=True)
class PolicyConfig:
    engine: str = "heuristic"
    simulations: int = 400
    time_budget_ms: int = 0
    temperature: float = 0.0
    dirichlet_alpha: float = 0.0
    dirichlet_epsilon: float = 0.0
    dirichlet_on_first_n_plies: int = 0
    enable_tail_solve: bool = True
    model_path: str | None = None
    search_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExplorationConfig:
    temperature_initial: float = 1.0
    temperature_final: float = 0.1
    # Optional per-game in-episode linear decay (by ply). 0 disables.
    temperature_decay_plies: int = 0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    dirichlet_on_first_n_plies: int = 2


@dataclass(slots=True)
class SelfPlayConfig:
    episodes: int = 50
    parallel_games: int = 8
    max_workers: int = 8
    worker_pool: str = "thread"
    warm_start_episodes: int = 0
    warm_start_engine: str = "random"
    warm_start_simulations: int = 0
    warm_start_train_passes: int = 0
    mcts_schedule: MctsSchedule | None = None
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


@dataclass(slots=True)
class TrainerConfig:
    batch_size: int = 512
    epochs: int = 1
    learning_rate: float = 3e-4
    hidden: int = 128
    mlp_layers: int = 2
    updates_per_step: int = 1
    buffer_size: int = 100000
    steps: int = 0
    value_late_weight: float = 0.6


@dataclass(slots=True)
class GatingConfig:
    games: int = 40
    accept_win_rate: float = 0.55
    eval_every_steps: int = 0
    eval_workers: int = 0
    eval_games: int = 0
    history_best_games: int = 0
    history_best_accept_win_rate: float = 0.55
    save_latest_every_steps: int = 0


@dataclass(slots=True)
class EvalPolicyConfig:
    candidate_temperature: float = 0.0
    opponent_temperature: float = 0.0
    simulations_floor: int = 10
    simulations_fixed: int = 0


@dataclass(slots=True)
class RuntimeConfig:
    process_pool_max_tasks_per_child: int = 128


@dataclass(slots=True)
class DiagnosticsConfig:
    enabled: bool = True
    zero_value_epsilon: float = 0.1


@dataclass(slots=True)
class TrainJobConfig:
    game_type: str = "game"
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    eval: EvalPolicyConfig = field(default_factory=EvalPolicyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    benchmark_engine: str | None = None
    benchmark_onnx_path: str | None = None
    resume_checkpoint_path: str | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TrainJobConfig":
        sp_raw = dict(d.get("selfplay") or {})
        pol_raw = dict(sp_raw.get("policy") or {})
        sch_raw = sp_raw.get("mcts_schedule")
        schedule = None
        if isinstance(sch_raw, dict):
            schedule = MctsSchedule(
                type=str(sch_raw.get("type", "linear")),
                start_simulations=int(sch_raw.get("start_simulations", 120)),
                end_simulations=int(sch_raw.get("end_simulations", 800)),
            )
        exp_raw = dict(sp_raw.get("exploration") or {})
        exploration = ExplorationConfig(
            temperature_initial=float(exp_raw.get("temperature_initial", 1.0)),
            temperature_final=float(exp_raw.get("temperature_final", 0.1)),
            temperature_decay_plies=max(0, int(exp_raw.get("temperature_decay_plies", 0))),
            dirichlet_alpha=float(exp_raw.get("dirichlet_alpha", 0.3)),
            dirichlet_epsilon=float(exp_raw.get("dirichlet_epsilon", 0.25)),
            dirichlet_on_first_n_plies=int(exp_raw.get("dirichlet_on_first_n_plies", 2)),
        )
        policy = PolicyConfig(
            engine=str(pol_raw.get("engine", "heuristic")),
            simulations=int(pol_raw.get("simulations", 400)),
            time_budget_ms=int(pol_raw.get("time_budget_ms", 0)),
            temperature=float(pol_raw.get("temperature", 0.0)),
            dirichlet_alpha=float(pol_raw.get("dirichlet_alpha", 0.0)),
            dirichlet_epsilon=float(pol_raw.get("dirichlet_epsilon", 0.0)),
            dirichlet_on_first_n_plies=int(pol_raw.get("dirichlet_on_first_n_plies", 0)),
            enable_tail_solve=bool(pol_raw.get("enable_tail_solve", True)),
            model_path=pol_raw.get("model_path"),
            search_options=clone_search_options(pol_raw.get("search_options")),
        )
        selfplay = SelfPlayConfig(
            episodes=int(sp_raw.get("episodes", 50)),
            parallel_games=int(sp_raw.get("parallel_games", 8)),
            max_workers=int(sp_raw.get("max_workers", sp_raw.get("parallel_games", 8))),
            worker_pool=str(sp_raw.get("worker_pool", "thread")),
            warm_start_episodes=int(sp_raw.get("warm_start_episodes", 0)),
            warm_start_engine=str(sp_raw.get("warm_start_engine", "random")),
            warm_start_simulations=int(sp_raw.get("warm_start_simulations", 0)),
            warm_start_train_passes=int(sp_raw.get("warm_start_train_passes", 0)),
            mcts_schedule=schedule,
            exploration=exploration,
            policy=policy,
        )
        tr_raw = dict(d.get("trainer") or {})
        trainer = TrainerConfig(
            batch_size=int(tr_raw.get("batch_size", 512)),
            epochs=int(tr_raw.get("epochs", 1)),
            learning_rate=float(tr_raw.get("learning_rate", 3e-4)),
            hidden=int(tr_raw.get("hidden", 128)),
            mlp_layers=int(tr_raw.get("mlp_layers", 2)),
            updates_per_step=int(tr_raw.get("updates_per_step", 1)),
            buffer_size=int(tr_raw.get("buffer_size", 100000)),
            steps=int(tr_raw.get("steps", 0)),
            value_late_weight=float(tr_raw.get("value_late_weight", 0.6)),
        )
        g_raw = dict(d.get("gating") or {})
        gating = GatingConfig(
            games=int(g_raw.get("games", 40)),
            accept_win_rate=float(g_raw.get("accept_win_rate", 0.55)),
            eval_every_steps=int(g_raw.get("eval_every_steps", 0)),
            eval_workers=int(g_raw.get("eval_workers", 0)),
            eval_games=int(g_raw.get("eval_games", 0)),
            history_best_games=int(g_raw.get("history_best_games", 0)),
            history_best_accept_win_rate=float(g_raw.get("history_best_accept_win_rate", g_raw.get("accept_win_rate", 0.55))),
            save_latest_every_steps=int(g_raw.get("save_latest_every_steps", 0)),
        )
        eval_raw = dict(d.get("eval") or {})
        eval_cfg = EvalPolicyConfig(
            candidate_temperature=float(eval_raw.get("candidate_temperature", 0.0)),
            opponent_temperature=float(eval_raw.get("opponent_temperature", 0.0)),
            simulations_floor=max(1, int(eval_raw.get("simulations_floor", 10))),
            simulations_fixed=max(0, int(eval_raw.get("simulations_fixed", 0))),
        )
        runtime_raw = dict(d.get("runtime") or {})
        runtime_cfg = RuntimeConfig(
            process_pool_max_tasks_per_child=int(runtime_raw.get("process_pool_max_tasks_per_child", 128)),
        )
        diag_raw = dict(d.get("diagnostics") or {})
        diagnostics_cfg = DiagnosticsConfig(
            enabled=bool(diag_raw.get("enabled", True)),
            zero_value_epsilon=max(0.0, float(diag_raw.get("zero_value_epsilon", 0.1))),
        )
        return TrainJobConfig(
            game_type=str(d.get("game_type", "game")),
            selfplay=selfplay,
            trainer=trainer,
            gating=gating,
            eval=eval_cfg,
            runtime=runtime_cfg,
            diagnostics=diagnostics_cfg,
            benchmark_engine=(str(d.get("benchmark_engine")) if d.get("benchmark_engine") is not None else None),
            benchmark_onnx_path=d.get("benchmark_onnx_path"),
            resume_checkpoint_path=d.get("resume_checkpoint_path"),
        )
