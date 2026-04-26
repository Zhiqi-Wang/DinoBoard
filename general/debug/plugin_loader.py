from __future__ import annotations

import importlib
from pathlib import Path

from .cpp_extension_setup import build_standard_game_cpp_setup_kwargs
from .game_plugin import GameDebugPlugin, create_debug_app_from_plugin


def infer_game_name_from_debug_file(current_file: str) -> str:
    path = Path(current_file).resolve()
    # .../games/<game>/debug_service/<file>.py
    return path.parents[1].name


def load_debug_plugin(current_file: str) -> GameDebugPlugin:
    game = infer_game_name_from_debug_file(current_file)
    module = importlib.import_module(f"games.{game}.debug_service.plugin")
    plugin = getattr(module, "DEBUG_PLUGIN", None)
    if not isinstance(plugin, GameDebugPlugin):
        raise RuntimeError(f"games.{game}.debug_service.plugin must expose DEBUG_PLUGIN: GameDebugPlugin")
    return plugin


def create_debug_app_from_current_game(current_file: str):
    plugin = load_debug_plugin(current_file)
    return create_debug_app_from_plugin(current_file=current_file, plugin=plugin)


def build_cpp_setup_kwargs_from_current_game(*, current_file: str, root: Path) -> dict[str, object]:
    plugin = load_debug_plugin(current_file)
    if not plugin.cpp_extension_name or not plugin.cpp_package_name:
        raise RuntimeError(
            "DEBUG_PLUGIN must provide cpp_extension_name and cpp_package_name for setup.py template."
        )
    return build_standard_game_cpp_setup_kwargs(
        root=root,
        game=plugin.game_name,
        extension_name=plugin.cpp_extension_name,
        package_name=plugin.cpp_package_name,
        include_action_constraint=bool(plugin.cpp_include_action_constraint),
    )

