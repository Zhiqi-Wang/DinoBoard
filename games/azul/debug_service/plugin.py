from __future__ import annotations

from general.debug.game_plugin import build_standard_game_debug_plugin

from games.azul.train.constants import AZUL_POLICY_DIM

DEBUG_PLUGIN = build_standard_game_debug_plugin(
    game_name="azul",
    game_display_name="Azul",
    version="0.2.0",
    action_id_max=AZUL_POLICY_DIM - 1,
    cpp_extension_name="cpp_azul_engine_v7",
    cpp_package_name="cpp-azul-engine-v7",
    default_simulations=200,
    cpp_include_action_constraint=True,
)

