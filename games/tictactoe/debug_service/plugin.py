from __future__ import annotations

from general.debug.game_plugin import build_standard_game_debug_plugin


DEBUG_PLUGIN = build_standard_game_debug_plugin(
    game_name="tictactoe",
    game_display_name="TicTacToe",
    version="0.1.0",
    action_id_max=8,
    cpp_extension_name="cpp_tictactoe_engine_v7",
    cpp_package_name="cpp-tictactoe-engine-v7",
    default_simulations=500,
)

