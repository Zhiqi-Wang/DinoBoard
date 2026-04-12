#pragma once

#include <cstdint>
#include <vector>

#include "core/game_interfaces.h"

namespace board_ai::search {

class INetMctsTraversalLimiter;

struct NetMctsConfig {
  int simulations = 200;
  float c_puct = 1.4f;
  int max_depth = 128;
  float value_clip = 1.0f;
  float root_dirichlet_alpha = 0.0f;
  float root_dirichlet_epsilon = 0.0f;
  INetMctsTraversalLimiter* traversal_limiter = nullptr;
};

struct NetMctsStats {
  int simulations_done = 0;
  std::int64_t expanded_nodes = 0;
  double nodes_per_sec = 0.0;
  double best_action_value = 0.0;
  std::vector<ActionId> root_actions{};
  std::vector<int> root_action_visits{};
};

ActionId select_action_from_visits(
    const std::vector<ActionId>& actions,
    const std::vector<int>& visits,
    double temperature,
    std::uint64_t rng_seed,
    ActionId fallback_action);

class IPolicyValueEvaluator {
 public:
  virtual ~IPolicyValueEvaluator() = default;
  // Return priors for legal_actions and value for perspective_player.
  virtual bool evaluate(
      const IGameState& state,
      int perspective_player,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* priors,
      float* value) const = 0;
};

class IStateValueModel {
 public:
  virtual ~IStateValueModel() = default;
  virtual int current_player(const IGameState& state) const = 0;
  virtual bool is_terminal(const IGameState& state) const = 0;
  virtual float terminal_value_for_player(const IGameState& state, int perspective_player) const = 0;
};

enum class TraversalStopAction {
  kFallbackToDefaultLeaf = 0,
  kUseLeafValue = 1,
  kContinue = 2,
};

struct TraversalStopResult {
  TraversalStopAction action = TraversalStopAction::kFallbackToDefaultLeaf;
  float leaf_value = 0.0f;
};

class INetMctsTraversalLimiter {
 public:
  virtual ~INetMctsTraversalLimiter() = default;
  // Return true to stop traversal at current state and evaluate as leaf.
  virtual bool should_stop(const IGameState& root_state, const IGameState& current_state, int depth) const = 0;
  // Whether should_stop_with_parent needs parent reconstruction on each depth.
  virtual bool requires_parent_for_stop() const { return false; }
  // Optional parent-aware stop predicate for stochastic transitions.
  virtual bool should_stop_with_parent(
      const IGameState& root_state,
      const IGameState& current_state,
      const IGameState* parent_state,
      ActionId parent_action,
      int depth) const {
    (void)parent_state;
    (void)parent_action;
    return should_stop(root_state, current_state, depth);
  }
  // Unified stop hook. Can return kContinue after mutating current_state to allow
  // search to continue from a sampled transition outcome.
  virtual TraversalStopResult on_traversal_stop(
      const IGameState& root_state,
      IGameState& current_state,
      const IGameState* parent_state,
      ActionId parent_action,
      int depth,
      const IGameRules& rules,
      const IStateValueModel& value_model,
      const IPolicyValueEvaluator& evaluator) const {
    TraversalStopResult out{};
    float leaf_value = 0.0f;
    if (on_truncation_leaf(
            root_state,
            current_state,
            parent_state,
            parent_action,
            depth,
            rules,
            value_model,
            evaluator,
            &leaf_value)) {
      out.action = TraversalStopAction::kUseLeafValue;
      out.leaf_value = leaf_value;
    }
    return out;
  }
  // Optional hook for chance-aware truncation.
  // Return true when `out_leaf_value` is filled and should be used directly.
  // Return false to fallback to default leaf expansion/evaluation.
  virtual bool on_truncation_leaf(
      const IGameState& root_state,
      const IGameState& current_state,
      const IGameState* parent_state,
      ActionId parent_action,
      int depth,
      const IGameRules& rules,
      const IStateValueModel& value_model,
      const IPolicyValueEvaluator& evaluator,
      float* out_leaf_value) const {
    (void)root_state;
    (void)current_state;
    (void)parent_state;
    (void)parent_action;
    (void)depth;
    (void)rules;
    (void)value_model;
    (void)evaluator;
    (void)out_leaf_value;
    return false;
  }
};

class NetMcts {
 public:
  explicit NetMcts(NetMctsConfig cfg = {});

  ActionId search_root(
      const IGameState& root,
      const IGameRules& rules,
      const IStateValueModel& value_model,
      const IPolicyValueEvaluator& evaluator,
      NetMctsStats* stats = nullptr) const;

 private:
  NetMctsConfig cfg_{};
};

}  // namespace board_ai::search
