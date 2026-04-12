#pragma once

#include <vector>

#include "core/game_interfaces.h"
#include "splendor_state.h"

namespace board_ai::splendor {

class SplendorRules final : public IGameRules {
 public:
  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;
  UndoToken do_action_fast(IGameState& state, ActionId action) const override;
  void undo_action(IGameState& state, const UndoToken& token) const override;

  static std::vector<ActionId> legal_actions_data(const SplendorData& data);
  static bool is_terminal_data(const SplendorData& data);
  static SplendorData apply_action_copy(const SplendorData& src, ActionId action);

 private:
  static const SplendorState* as_state(const IGameState& state);
  static SplendorState* as_state(IGameState& state);
};

}  // namespace board_ai::splendor
