#pragma once

#include <vector>

#include "core/game_interfaces.h"
#include "azul_state.h"

namespace board_ai::azul {

struct SearchSpecializationConfig {
  int max_depth = 128;
  bool stop_on_round_transition = true;
  int chance_expand_cap = 10;
  bool leaf_override_enabled = false;
};

class AzulRules final : public IGameRules {
 public:
  explicit AzulRules(SearchSpecializationConfig cfg = {});

  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;
  UndoToken do_action_fast(IGameState& state, ActionId action) const override;
  void undo_action(IGameState& state, const UndoToken& token) const override;

 private:
  static int decode_source(ActionId action);
  static int decode_color(ActionId action);
  static int decode_target_line(ActionId action);
  static int wall_col_for_color(int row, int color_idx);
  static int score_wall_placement(const PlayerState& p, int row, int col);
  static int apply_final_bonus(AzulState& state, int pid);
  static bool will_round_end_after_action(const AzulState& state, ActionId action);
  static void apply_action_no_undo(AzulState& state, ActionId action);
  static void apply_round_settlement(AzulState& state);

  static const AzulState* as_azul_state(const IGameState& state);
  static AzulState* as_azul_state(IGameState& state);

  SearchSpecializationConfig cfg_{};
};

}  // namespace board_ai::azul
