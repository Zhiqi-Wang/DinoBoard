#include "core/action_constraint.h"

namespace board_ai {

void ConstraintPipeline::add(std::shared_ptr<IActionConstraint> constraint) {
  if (constraint) {
    constraints_.push_back(std::move(constraint));
  }
}

std::vector<ActionId> ConstraintPipeline::filter_actions(
    const IGameState& state,
    int actor,
    const std::vector<ActionId>& legal_actions) const {
  std::vector<ActionId> filtered = legal_actions;

  for (const auto& constraint : constraints_) {
    if (!constraint->enabled_for(state, actor)) {
      continue;
    }

    std::vector<ActionId> next;
    next.reserve(filtered.size());
    for (ActionId action : filtered) {
      if (constraint->allow(state, actor, action)) {
        next.push_back(action);
      }
    }
    filtered = std::move(next);
  }

  // Safety fallback: never return empty if legal actions were available.
  if (filtered.empty() && !legal_actions.empty()) {
    return legal_actions;
  }
  return filtered;
}

}  // namespace board_ai
