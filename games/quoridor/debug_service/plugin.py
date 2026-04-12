from __future__ import annotations

from general.debug.game_plugin import build_standard_game_debug_plugin


DEBUG_PLUGIN = build_standard_game_debug_plugin(
    game_name="quoridor",
    game_display_name="Quoridor",
    version="0.1.0",
    action_id_max=208,
    cpp_extension_name="cpp_quoridor_engine_v1",
    cpp_package_name="cpp-quoridor-engine-v1",
    default_simulations=400,
)

