#pragma once

#include <vector>

#include "infer/feature_encoder.h"
#include "quoridor_state.h"
#include "search/net_mcts.h"

namespace board_ai::quoridor {

constexpr int kFeatureDim = 294;

class QuoridorFeatureEncoder final : public infer::IFeatureEncoder {
 public:
  int action_space() const override { return kActionSpace; }
  int feature_dim() const override { return kFeatureDim; }

  bool encode(
      const IGameState& state,
      int perspective_player,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* features,
      std::vector<float>* legal_mask) const override;
  ActionId canonicalize_action(ActionId action, int perspective_player) const override;
  ActionId decanonicalize_action(ActionId canonical_action, int perspective_player) const override;
};

class QuoridorStateValueModel final : public search::IStateValueModel {
 public:
  int current_player(const IGameState& state) const override;
  bool is_terminal(const IGameState& state) const override;
  float terminal_value_for_player(const IGameState& state, int perspective_player) const override;
};

}  // namespace board_ai::quoridor

