#pragma once

#include <memory>
#include <vector>

#include "core/game_interfaces.h"

namespace board_ai {

class IActionConstraint {
 public:
  virtual ~IActionConstraint() = default;
  virtual bool enabled_for(const IGameState& state, int actor) const = 0;
  virtual bool allow(const IGameState& state, int actor, ActionId action) const = 0;
};

class ConstraintPipeline {
 public:
  void add(std::shared_ptr<IActionConstraint> constraint);
  std::vector<ActionId> filter_actions(
      const IGameState& state,
      int actor,
      const std::vector<ActionId>& legal_actions) const;

 private:
  std::vector<std::shared_ptr<IActionConstraint>> constraints_;
};

}  // namespace board_ai
