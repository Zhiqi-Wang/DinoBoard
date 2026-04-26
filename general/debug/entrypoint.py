from __future__ import annotations

from pathlib import Path
from typing import Any

from .app_factory import create_debug_service_app
from .model_path_resolver import resolve_default_model_path


def create_game_debug_app(
    *,
    current_file: str,
    game_name: str,
    title: str,
    version: str,
    backend: Any,
    action_id_min: int,
    action_id_max: int,
    allowed_engines: tuple[str, ...],
    default_engine: str,
    default_simulations: int,
    require_model_exists_for_engines: tuple[str, ...] = (),
    fixed_default_model_path: str | Path | None = None,
):
    project_dir = Path(current_file).resolve().parents[3]
    web_dir = Path(current_file).resolve().parent / "web"
    default_model_path = (
        Path(fixed_default_model_path).resolve()
        if fixed_default_model_path
        else resolve_default_model_path(project_dir, game_name)
    )
    return create_debug_service_app(
        title=title,
        version=version,
        project_dir=project_dir,
        web_dir=web_dir,
        backend=backend,
        action_id_min=action_id_min,
        action_id_max=action_id_max,
        allowed_engines=allowed_engines,
        default_engine=default_engine,
        default_simulations=default_simulations,
        default_model_path=default_model_path,
        require_model_exists_for_engines=require_model_exists_for_engines,
    )

