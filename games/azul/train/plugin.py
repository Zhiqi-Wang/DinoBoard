from __future__ import annotations

from general.train.cpp_training_backend import MovePolicy, build_flexible_search_options_backend_factory
from general.train.extensions import build_score_margin_pipeline_hooks
from general.train.game_plugin import GameTrainPlugin, build_initial_model_from_exporter
from general.train.plugin_loader import build_train_exports_from_current_game

from .constants import (
    AZUL_INPUT_DIM,
    AZUL_POLICY_DIM,
    DEFAULT_LABEL_VERSION,
    DEFAULT_VALUE_MARGIN_SCALE,
    DEFAULT_VALUE_MARGIN_WEIGHT,
)
from .torch_trainer import export_initial_policy_onnx, run_torch_train


def _enrich_sample(sample: dict[str, object], rec: dict[str, object]) -> None:
    features = rec.get("features")
    if isinstance(features, list):
        sample["features"] = [float(v) for v in features]


TRAIN_PLUGIN = GameTrainPlugin(
    description="Run MVP training job for Azul.",
    benchmark_engine="netmcts",
    support_benchmark_onnx=True,
    move_policy_cls=MovePolicy,
    backend_factory=build_flexible_search_options_backend_factory("cpp_azul_engine_v7"),
    game_type="azul",
    ruleset="azul_v1",
    read_shared_victory_from_raw=True,
    run_trainer=run_torch_train,
    build_initial_model=build_initial_model_from_exporter(
        export_initial_policy_onnx,
        input_dim=AZUL_INPUT_DIM,
        policy_dim=AZUL_POLICY_DIM,
    ),
    pipeline_hooks=build_score_margin_pipeline_hooks(
        default_label_version=DEFAULT_LABEL_VERSION,
        default_margin_weight=DEFAULT_VALUE_MARGIN_WEIGHT,
        default_margin_scale=DEFAULT_VALUE_MARGIN_SCALE,
    ),
    enrich_sample=_enrich_sample,
    netmcts_data_source="netmcts_selfplay",
    default_value_margin_weight=DEFAULT_VALUE_MARGIN_WEIGHT,
    default_value_margin_scale=DEFAULT_VALUE_MARGIN_SCALE,
)

run_job, _default_config, main, run_selfplay_episode_payload, run_arena_match = build_train_exports_from_current_game(
    __file__
)


if __name__ == "__main__":
    raise SystemExit(main())

