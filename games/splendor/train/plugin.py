from __future__ import annotations

from general.train.config import TrainJobConfig
from general.train.cpp_training_backend import MovePolicy, build_search_options_backend_factory
from general.train.extensions import build_score_margin_pipeline_hooks
from general.train.game_plugin import GameTrainPlugin, build_initial_model_from_exporter
from general.train.plugin_loader import build_train_exports_from_current_game

from .constants import (
    DEFAULT_LABEL_VERSION,
    DEFAULT_VALUE_MARGIN_SCALE,
    DEFAULT_VALUE_MARGIN_WEIGHT,
    FEATURE_PENDING_RETURNS_INDEX,
    FEATURE_STAGE_CHOOSE_NOBLE_INDEX,
    FEATURE_STAGE_NORMAL_INDEX,
    FEATURE_RETURN_PHASE_INDEX,
    SPLENDOR_INPUT_DIM,
    SPLENDOR_POLICY_DIM,
)
from .torch_trainer import export_initial_policy_onnx, run_torch_train


def _assert_cpp_dims() -> None:
    try:
        import cpp_splendor_engine_v1 as cpp_splendor_engine  # type: ignore
    except Exception:
        return
    feature_dim_getter = getattr(cpp_splendor_engine, "feature_dim", None)
    action_space_getter = getattr(cpp_splendor_engine, "action_space", None)
    if callable(feature_dim_getter):
        cpp_feature_dim = int(feature_dim_getter())
        if cpp_feature_dim != SPLENDOR_INPUT_DIM:
            raise RuntimeError(
                f"splendor feature dim mismatch: python={SPLENDOR_INPUT_DIM}, cpp={cpp_feature_dim}"
            )
    if callable(action_space_getter):
        cpp_action_space = int(action_space_getter())
        if cpp_action_space != SPLENDOR_POLICY_DIM:
            raise RuntimeError(
                f"splendor policy dim mismatch: python={SPLENDOR_POLICY_DIM}, cpp={cpp_action_space}"
            )


def _can_use_netmcts() -> bool:
    try:
        import cpp_splendor_engine_v1 as cpp_splendor_engine  # type: ignore
    except Exception:
        return False
    onnx_enabled = getattr(cpp_splendor_engine, "onnx_enabled", None)
    return bool(callable(onnx_enabled) and onnx_enabled())


def _validate_nopeek_search_options(config: TrainJobConfig) -> None:
    if str(config.selfplay.policy.engine).lower() != "netmcts":
        return
    opts = dict(config.selfplay.policy.search_options or {})
    stop_on_draw_transition = bool(opts.get("stop_on_draw_transition", False))
    enable_draw_chance = bool(opts.get("enable_draw_chance", False))
    try:
        chance_expand_cap = int(opts.get("chance_expand_cap", 0))
    except (TypeError, ValueError):
        chance_expand_cap = 0
    if stop_on_draw_transition and enable_draw_chance and chance_expand_cap >= 1:
        return
    raise RuntimeError(
        "strict mode: splendor netmcts training requires chance-aware hidden-info search "
        "(search_options.stop_on_draw_transition=true, enable_draw_chance=true, chance_expand_cap>=1). "
        "Refusing to run with peek-prone configuration."
    )


def _validate_runtime(config: TrainJobConfig) -> None:
    _assert_cpp_dims()
    want_netmcts = str(config.selfplay.policy.engine).lower() == "netmcts"
    if want_netmcts and not _can_use_netmcts():
        raise RuntimeError(
            "strict mode: selfplay.policy.engine=netmcts but cpp_splendor_engine_v1 has no ONNX support. "
            "Rebuild environment with ONNX enabled (e.g. setup_game_env.ps1 -WithOnnx -AutoDiscoverOnnx)."
        )
    _validate_nopeek_search_options(config)


def _enrich_splendor_sample(sample: dict[str, object], rec: dict[str, object]) -> None:
    features = rec.get("features")
    if isinstance(features, list) and features:
        normalized = [float(v) for v in features[:SPLENDOR_INPUT_DIM]]
        sample["features"] = normalized
        if len(normalized) == SPLENDOR_INPUT_DIM:
            sample["return_phase"] = float(normalized[FEATURE_RETURN_PHASE_INDEX])
            sample["choose_noble_phase"] = float(normalized[FEATURE_STAGE_CHOOSE_NOBLE_INDEX])
            sample["normal_phase"] = float(normalized[FEATURE_STAGE_NORMAL_INDEX])
            sample["pending_returns_norm"] = float(normalized[FEATURE_PENDING_RETURNS_INDEX])


TRAIN_PLUGIN = GameTrainPlugin(
    description="Run baseline training job for Splendor.",
    benchmark_engine="heuristic",
    support_benchmark_onnx=False,
    move_policy_cls=MovePolicy,
    backend_factory=build_search_options_backend_factory("cpp_splendor_engine_v1"),
    game_type="splendor",
    ruleset="splendor_v1",
    read_shared_victory_from_raw=True,
    run_trainer=run_torch_train,
    build_initial_model=build_initial_model_from_exporter(
        export_initial_policy_onnx,
        input_dim=SPLENDOR_INPUT_DIM,
        policy_dim=SPLENDOR_POLICY_DIM,
    ),
    pipeline_hooks=build_score_margin_pipeline_hooks(
        default_label_version=DEFAULT_LABEL_VERSION,
        default_margin_weight=DEFAULT_VALUE_MARGIN_WEIGHT,
        default_margin_scale=DEFAULT_VALUE_MARGIN_SCALE,
    ),
    before_run_job=_validate_runtime,
    enrich_sample=_enrich_splendor_sample,
    netmcts_data_source="netmcts_selfplay",
    default_value_margin_weight=DEFAULT_VALUE_MARGIN_WEIGHT,
    default_value_margin_scale=DEFAULT_VALUE_MARGIN_SCALE,
)

run_job, _default_config, main, run_selfplay_episode_payload, run_arena_match = build_train_exports_from_current_game(
    __file__
)


if __name__ == "__main__":
    raise SystemExit(main())

