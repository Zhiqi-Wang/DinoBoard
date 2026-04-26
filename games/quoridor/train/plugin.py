from __future__ import annotations

import math

from general.search_options import clone_search_options
from general.train.config import PolicyConfig, TrainJobConfig
from general.train.cpp_training_backend import MovePolicy, build_search_options_backend_factory
from general.train.extensions import TrainPipelineHooks
from general.train.game_plugin import GameTrainPlugin, build_initial_model_from_exporter
from general.train.plugin_loader import build_train_exports_from_current_game

from .constants import QUORIDOR_INPUT_DIM, QUORIDOR_POLICY_DIM
from .torch_trainer import export_initial_policy_onnx, run_torch_train

_HEURISTIC_RANDOM_ACTION_PROB_KEY = "heuristic_random_action_prob"
# Keep warm-start targets deterministic by default; exploration comes from
# NetMCTS temperature + Dirichlet during selfplay, not from random heuristic noise.
_DEFAULT_WARM_START_HEURISTIC_RANDOM_ACTION_PROB = 0.0
_DEFAULT_VALUE_MARGIN_WEIGHT = 0.1
_DEFAULT_WIN_MARGIN_SCALE = 8.0


def _assert_cpp_dims() -> None:
    try:
        import cpp_quoridor_engine_v1 as cpp_quoridor_engine  # type: ignore
    except Exception:
        return
    feature_dim_getter = getattr(cpp_quoridor_engine, "feature_dim", None)
    action_space_getter = getattr(cpp_quoridor_engine, "action_space", None)
    if callable(feature_dim_getter):
        cpp_feature_dim = int(feature_dim_getter())
        if cpp_feature_dim != QUORIDOR_INPUT_DIM:
            raise RuntimeError(
                f"quoridor feature dim mismatch: python={QUORIDOR_INPUT_DIM}, cpp={cpp_feature_dim}"
            )
    if callable(action_space_getter):
        cpp_action_space = int(action_space_getter())
        if cpp_action_space != QUORIDOR_POLICY_DIM:
            raise RuntimeError(
                f"quoridor policy dim mismatch: python={QUORIDOR_POLICY_DIM}, cpp={cpp_action_space}"
            )


def _validate_runtime(config: TrainJobConfig) -> None:
    _assert_cpp_dims()
    want_netmcts = str(config.selfplay.policy.engine).lower() == "netmcts"
    if not want_netmcts:
        return
    try:
        import cpp_quoridor_engine_v1 as cpp_quoridor_engine  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "strict mode: selfplay.policy.engine=netmcts but cpp_quoridor_engine_v1 cannot be imported."
        ) from e
    onnx_enabled = getattr(cpp_quoridor_engine, "onnx_enabled", None)
    if not callable(onnx_enabled) or not onnx_enabled():
        raise RuntimeError(
            "strict mode: selfplay.policy.engine=netmcts but cpp_quoridor_engine_v1 has no ONNX support. "
            "请使用 games/quoridor/debug_service/setup_env.ps1 -WithOnnx 重新构建环境。"
        )


def _enrich_quoridor_sample(sample: dict[str, object], rec: dict[str, object]) -> None:
    features = rec.get("features")
    if isinstance(features, list) and features:
        sample["features"] = [float(v) for v in features[:QUORIDOR_INPUT_DIM]]


def _prepare_quoridor_selfplay_policy(policy_cfg: PolicyConfig, ctx: dict[str, object]) -> PolicyConfig:
    phase = str(ctx.get("phase", "")).lower()
    engine = str(policy_cfg.engine).lower()
    if phase != "warm_start" or engine != "heuristic":
        return policy_cfg
    options = clone_search_options(policy_cfg.search_options)
    # Force deterministic heuristic targets during warm start to avoid
    # action execution / policy-label mismatch when random override is enabled.
    options[_HEURISTIC_RANDOM_ACTION_PROB_KEY] = _DEFAULT_WARM_START_HEURISTIC_RANDOM_ACTION_PROB
    policy_cfg.search_options = options
    return policy_cfg


def _build_quoridor_episode_context(raw_episode: dict[str, object], _ctx: dict[str, object]) -> dict[str, object]:
    label_params = raw_episode.get("label_params") or {}
    if not isinstance(label_params, dict):
        label_params = {}
    try:
        margin_weight = float(label_params.get("value_margin_weight", _DEFAULT_VALUE_MARGIN_WEIGHT))
    except (TypeError, ValueError):
        margin_weight = _DEFAULT_VALUE_MARGIN_WEIGHT
    margin_weight = max(0.0, min(1.0, margin_weight))
    try:
        margin_scale = float(label_params.get("value_margin_scale", _DEFAULT_WIN_MARGIN_SCALE))
    except (TypeError, ValueError):
        margin_scale = _DEFAULT_WIN_MARGIN_SCALE
    margin_scale = max(1e-6, margin_scale)

    winner_raw = raw_episode.get("winner")
    try:
        winner = int(winner_raw) if winner_raw is not None else None
    except (TypeError, ValueError):
        winner = None
    shared = bool(raw_episode.get("shared_victory", False))
    try:
        win_margin_steps = max(0.0, float(raw_episode.get("win_margin_steps", 0.0)))
    except (TypeError, ValueError):
        win_margin_steps = 0.0

    if shared or winner is None:
        value_targets = {0: 0.0, 1: 0.0}
    else:
        base_targets = {winner: 1.0, 1 - winner: -1.0}
        margin = math.tanh(win_margin_steps / margin_scale)
        margin_targets = {winner: margin, 1 - winner: -margin}
        value_targets = {
            0: float((1.0 - margin_weight) * base_targets[0] + margin_weight * margin_targets[0]),
            1: float((1.0 - margin_weight) * base_targets[1] + margin_weight * margin_targets[1]),
        }

    plies = max(1, int(raw_episode.get("plies", 1)))
    return {
        "plies": plies,
        "value_targets": value_targets,
    }


def _label_quoridor_sample(
    sample: dict[str, object], episode_ctx: dict[str, object], _ctx: dict[str, object]
) -> dict[str, object]:
    try:
        player = int(sample.get("player", 0))
    except (TypeError, ValueError):
        player = 0
    try:
        plies = max(1, int(episode_ctx.get("plies", 1)))
    except (TypeError, ValueError):
        plies = 1
    denom = max(1, plies - 1)
    try:
        ply = int(sample.get("ply", 0))
    except (TypeError, ValueError):
        ply = 0
    value_targets = episode_ctx.get("value_targets", {})
    if not isinstance(value_targets, dict):
        value_targets = {}
    return {
        "z": float(value_targets.get(player, 0.0)),
        "phase": float(max(0.0, min(1.0, float(ply) / float(denom)))),
    }


TRAIN_PLUGIN = GameTrainPlugin(
    description="Run baseline training job for Quoridor.",
    benchmark_engine="heuristic",
    support_benchmark_onnx=False,
    move_policy_cls=MovePolicy,
    backend_factory=build_search_options_backend_factory("cpp_quoridor_engine_v1"),
    game_type="quoridor",
    ruleset="quoridor_v1",
    read_shared_victory_from_raw=True,
    run_trainer=run_torch_train,
    build_initial_model=build_initial_model_from_exporter(
        export_initial_policy_onnx,
        input_dim=QUORIDOR_INPUT_DIM,
        policy_dim=QUORIDOR_POLICY_DIM,
    ),
    before_run_job=_validate_runtime,
    enrich_sample=_enrich_quoridor_sample,
    pipeline_hooks=TrainPipelineHooks(
        prepare_selfplay_policy=_prepare_quoridor_selfplay_policy,
        build_episode_context=_build_quoridor_episode_context,
        label_sample=_label_quoridor_sample,
    ),
    netmcts_data_source="netmcts_selfplay",
    default_value_margin_weight=_DEFAULT_VALUE_MARGIN_WEIGHT,
    default_value_margin_scale=_DEFAULT_WIN_MARGIN_SCALE,
)

run_job, _default_config, main, run_selfplay_episode_payload, run_arena_match = build_train_exports_from_current_game(
    __file__
)


if __name__ == "__main__":
    raise SystemExit(main())

