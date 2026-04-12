from __future__ import annotations

from general.train.cpp_training_backend import MovePolicy, build_classic_backend_factory
from general.train.game_plugin import GameTrainPlugin, build_initial_model_from_exporter
from general.train.plugin_loader import build_train_exports_from_current_game

from .torch_trainer import export_initial_policy_onnx, run_torch_train


def _enrich_tictactoe_sample(sample: dict[str, object], rec: dict[str, object]) -> None:
    board = rec.get("board")
    if isinstance(board, list) and len(board) == 9:
        sample["board"] = [int(x) for x in board]


TRAIN_PLUGIN = GameTrainPlugin(
    description="Run MVP training job for TicTacToe.",
    benchmark_engine="heuristic",
    support_benchmark_onnx=False,
    move_policy_cls=MovePolicy,
    backend_factory=build_classic_backend_factory("cpp_tictactoe_engine_v7"),
    game_type="tictactoe",
    ruleset="tictactoe_v1",
    read_shared_victory_from_raw=False,
    run_trainer=run_torch_train,
    build_initial_model=build_initial_model_from_exporter(export_initial_policy_onnx),
    enrich_sample=_enrich_tictactoe_sample,
    netmcts_data_source="netmcts_selfplay",
)

run_job, _default_config, main, run_selfplay_episode_payload, run_arena_match = build_train_exports_from_current_game(
    __file__
)


if __name__ == "__main__":
    raise SystemExit(main())

