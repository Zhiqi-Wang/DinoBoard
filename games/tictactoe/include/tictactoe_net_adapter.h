#pragma once

#include <vector>

#include "tictactoe_state.h"
#include "infer/feature_encoder.h"
#include "search/net_mcts.h"

namespace board_ai::tictactoe {

class TicTacToeFeatureEncoder final : public infer::IFeatureEncoder {
 public:
  int action_space() const override { return 9; }
  int feature_dim() const override { return 27; }

  bool encode(
      const IGameState& state,
      int perspective_player,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* features,
      std::vector<float>* legal_mask) const override;
};

class TicTacToeStateValueModel final : public search::IStateValueModel {
 public:
  int current_player(const IGameState& state) const override;
  bool is_terminal(const IGameState& state) const override;
  float terminal_value_for_player(const IGameState& state, int perspective_player) const override;
};

}  // namespace board_ai::tictactoe
