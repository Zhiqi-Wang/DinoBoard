from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .cpp_backend_factory import create_cpp_session_backend_from_module_name


@dataclass(slots=True)
class GameDebugPlugin:
    game_name: str
    title: str
    version: str
    backend_factory: Callable[[], Any]
    action_id_min: int
    action_id_max: int
    allowed_engines: tuple[str, ...]
    default_engine: str
    default_simulations: int
    require_model_exists_for_engines: tuple[str, ...] = ()
    cpp_extension_name: str | None = None
    cpp_package_name: str | None = None
    cpp_include_action_constraint: bool = False


def build_standard_game_debug_plugin(
    *,
    game_name: str,
    game_display_name: str,
    version: str,
    action_id_max: int,
    cpp_extension_name: str,
    cpp_package_name: str,
    default_simulations: int = 200,
    action_id_min: int = 0,
    allowed_engines: tuple[str, ...] = ("heuristic", "netmcts"),
    default_engine: str = "netmcts",
    require_model_exists_for_engines: tuple[str, ...] = ("netmcts",),
    cpp_include_action_constraint: bool = False,
) -> GameDebugPlugin:
    return GameDebugPlugin(
        game_name=game_name,
        title=f"DinoBoard Intelligence {game_display_name} Debug Service",
        version=version,
        backend_factory=lambda: create_cpp_session_backend_from_module_name(cpp_extension_name),
        action_id_min=action_id_min,
        action_id_max=action_id_max,
        allowed_engines=allowed_engines,
        default_engine=default_engine,
        default_simulations=default_simulations,
        require_model_exists_for_engines=require_model_exists_for_engines,
        cpp_extension_name=cpp_extension_name,
        cpp_package_name=cpp_package_name,
        cpp_include_action_constraint=cpp_include_action_constraint,
    )


def create_debug_app_from_plugin(*, current_file: str, plugin: GameDebugPlugin):
    from .entrypoint import create_game_debug_app

    return create_game_debug_app(
        current_file=current_file,
        game_name=plugin.game_name,
        title=plugin.title,
        version=plugin.version,
        backend=plugin.backend_factory(),
        action_id_min=int(plugin.action_id_min),
        action_id_max=int(plugin.action_id_max),
        allowed_engines=tuple(plugin.allowed_engines),
        default_engine=str(plugin.default_engine),
        default_simulations=int(plugin.default_simulations),
        require_model_exists_for_engines=tuple(plugin.require_model_exists_for_engines),
    )

