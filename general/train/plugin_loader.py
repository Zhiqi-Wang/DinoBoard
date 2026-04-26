from __future__ import annotations

import importlib
from pathlib import Path

from .game_plugin import (
    GameTrainPlugin,
    build_selfplay_runners_from_plugin,
    build_training_entrypoints_from_plugin,
)


def infer_game_name_from_train_file(current_file: str) -> str:
    path = Path(current_file).resolve()
    # .../games/<game>/train/<file>.py
    return path.parents[1].name


def load_train_plugin(current_file: str) -> GameTrainPlugin:
    game = infer_game_name_from_train_file(current_file)
    module = importlib.import_module(f"games.{game}.train.plugin")
    plugin = getattr(module, "TRAIN_PLUGIN", None)
    if not isinstance(plugin, GameTrainPlugin):
        raise RuntimeError(f"games.{game}.train.plugin must expose TRAIN_PLUGIN: GameTrainPlugin")
    return plugin


def build_training_entrypoints_from_current_game(current_file: str):
    plugin = load_train_plugin(current_file)
    return build_training_entrypoints_from_plugin(
        current_file=current_file,
        plugin=plugin,
    )


def build_selfplay_runners_from_current_game(current_file: str):
    plugin = load_train_plugin(current_file)
    return build_selfplay_runners_from_plugin(plugin)


def build_train_exports_from_current_game(current_file: str):
    run_job, default_config_factory, main = build_training_entrypoints_from_current_game(current_file)
    run_selfplay_episode_payload, run_arena_match = build_selfplay_runners_from_current_game(current_file)
    return run_job, default_config_factory, main, run_selfplay_episode_payload, run_arena_match

