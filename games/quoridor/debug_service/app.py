from __future__ import annotations

from pathlib import Path

from general.debug.entrypoint import create_game_debug_app
from general.debug.plugin_loader import load_debug_plugin

_PLUGIN = load_debug_plugin(__file__)
_FIXED_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "best_model.onnx"

app = create_game_debug_app(
    current_file=__file__,
    game_name=_PLUGIN.game_name,
    title=_PLUGIN.title,
    version=_PLUGIN.version,
    backend=_PLUGIN.backend_factory(),
    action_id_min=int(_PLUGIN.action_id_min),
    action_id_max=int(_PLUGIN.action_id_max),
    allowed_engines=tuple(_PLUGIN.allowed_engines),
    default_engine=str(_PLUGIN.default_engine),
    default_simulations=int(_PLUGIN.default_simulations),
    require_model_exists_for_engines=tuple(_PLUGIN.require_model_exists_for_engines),
    fixed_default_model_path=_FIXED_DEFAULT_MODEL_PATH,
)

