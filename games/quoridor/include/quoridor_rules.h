#pragma once

#include <vector>

#include "core/game_interfaces.h"
#include "quoridor_state.h"

namespace board_ai::quoridor {

class QuoridorRules final : public IGameRules {
 public:
  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;
  UndoToken do_action_fast(IGameState& state, ActionId action) const override;
  void undo_action(IGameState& state, const UndoToken& token) const override;

  static bool has_path_to_goal(const QuoridorState& state, int player);
  static int shortest_path_distance(const QuoridorState& state, int player);

 private:
  static const QuoridorState* as_state(const IGameState& state);
  static QuoridorState* as_state(IGameState& state);
};

}  // namespace board_ai::quoridor

