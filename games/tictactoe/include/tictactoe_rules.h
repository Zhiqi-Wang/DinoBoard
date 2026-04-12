#pragma once

#include <vector>

#include "core/game_interfaces.h"
#include "tictactoe_state.h"

namespace board_ai::tictactoe {

class TicTacToeRules final : public IGameRules {
 public:
  bool validate_action(const IGameState& state, ActionId action) const override;
  std::vector<ActionId> legal_actions(const IGameState& state) const override;
  UndoToken do_action_fast(IGameState& state, ActionId action) const override;
  void undo_action(IGameState& state, const UndoToken& token) const override;

 private:
  static const TicTacToeState* as_state(const IGameState& state);
  static TicTacToeState* as_state(IGameState& state);
  static int evaluate_winner(const TicTacToeState& state);
};

}  // namespace board_ai::tictactoe
