#pragma once

#include <vector>

#include "core/game_interfaces.h"

namespace board_ai::infer {

class IFeatureEncoder {
 public:
  virtual ~IFeatureEncoder() = default;
  virtual int action_space() const = 0;
  virtual int feature_dim() const = 0;
  virtual bool encode(
      const IGameState& state,
      int perspective_player,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* features,
      std::vector<float>* legal_mask) const = 0;
  // Optional action-space canonicalization hooks for perspective-based encoders.
  // Default behavior keeps action ids unchanged.
  virtual ActionId canonicalize_action(ActionId action, int perspective_player) const {
    (void)perspective_player;
    return action;
  }
  virtual ActionId decanonicalize_action(ActionId canonical_action, int perspective_player) const {
    (void)perspective_player;
    return canonical_action;
  }
};

}  // namespace board_ai::infer
