from __future__ import annotations

from general.debug.game_plugin import build_standard_game_debug_plugin

from games.splendor.train.constants import SPLENDOR_POLICY_DIM

DEBUG_PLUGIN = build_standard_game_debug_plugin(
    game_name="splendor",
    game_display_name="Splendor",
    version="0.1.0",
    action_id_max=SPLENDOR_POLICY_DIM - 1,
    cpp_extension_name="cpp_splendor_engine_v1",
    cpp_package_name="cpp-splendor-engine-v1",
    default_simulations=200,
)

