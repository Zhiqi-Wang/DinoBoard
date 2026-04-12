#include "search/net_mcts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>

namespace board_ai::search {

namespace {

class SplitMix64Engine {
 public:
  using result_type = std::uint64_t;

  explicit SplitMix64Engine(std::uint64_t seed) : state_(seed) {}

  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return UINT64_MAX; }

  result_type operator()() {
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
  }

 private:
  std::uint64_t state_;
};

struct Edge {
  ActionId action = -1;
  float prior = 0.0f;
  int child = -1;
  StateHash64 child_state_hash = 0;
  bool has_child_state_hash = false;
  std::vector<std::pair<StateHash64, int>> chance_children{};
  int visit_count = 0;
  float value_sum = 0.0f;
};

struct Node {
  int to_play = 0;
  bool expanded = false;
  int visit_count = 0;
  float value_sum = 0.0f;
  std::vector<Edge> edges{};
};

static float clip_value(float v, float lim) {
  const float cap = std::max(0.0f, lim);
  return std::max(-cap, std::min(cap, v));
}

static void apply_root_dirichlet_noise(Node& root, float alpha, float epsilon, std::uint64_t seed) {
  if (root.edges.empty()) {
    return;
  }
  const float a = std::max(0.0f, alpha);
  const float eps = std::max(0.0f, std::min(1.0f, epsilon));
  if (a <= 1e-8f || eps <= 1e-8f) {
    return;
  }
  SplitMix64Engine rng(seed);
  std::gamma_distribution<float> gamma(a, 1.0f);
  std::vector<float> noise(root.edges.size(), 0.0f);
  float sum = 0.0f;
  for (size_t i = 0; i < root.edges.size(); ++i) {
    const float g = std::max(0.0f, gamma(rng));
    noise[i] = g;
    sum += g;
  }
  if (sum <= 1e-12f) {
    return;
  }
  for (size_t i = 0; i < root.edges.size(); ++i) {
    const float n = noise[i] / sum;
    const float p = std::max(0.0f, root.edges[i].prior);
    root.edges[i].prior = (1.0f - eps) * p + eps * n;
  }
}

}  // namespace

NetMcts::NetMcts(NetMctsConfig cfg) : cfg_(cfg) {}

ActionId select_action_from_visits(
    const std::vector<ActionId>& actions,
    const std::vector<int>& visits,
    double temperature,
    std::uint64_t rng_seed,
    ActionId fallback_action) {
  if (actions.empty() || actions.size() != visits.size()) return fallback_action;
  if (temperature <= 1e-6) {
    int best_visit = std::numeric_limits<int>::min();
    std::vector<size_t> best_indices;
    best_indices.reserve(actions.size());
    for (size_t i = 0; i < visits.size(); ++i) {
      if (visits[i] > best_visit) {
        best_visit = visits[i];
        best_indices.clear();
        best_indices.push_back(i);
      } else if (visits[i] == best_visit) {
        best_indices.push_back(i);
      }
    }
    if (best_indices.empty()) {
      return fallback_action;
    }
    if (best_indices.size() == 1) {
      return actions[best_indices.front()];
    }
    SplitMix64Engine rng(rng_seed);
    std::uniform_int_distribution<size_t> pick(0, best_indices.size() - 1);
    return actions[best_indices[pick(rng)]];
  }

  const double inv_t = 1.0 / std::max(1e-6, temperature);
  std::vector<double> weights(actions.size(), 0.0);
  double sum_w = 0.0;
  for (size_t i = 0; i < visits.size(); ++i) {
    const double base = static_cast<double>(std::max(0, visits[i]));
    const double w = (base > 0.0) ? std::pow(base, inv_t) : 0.0;
    weights[i] = std::isfinite(w) ? w : 0.0;
    sum_w += weights[i];
  }
  if (sum_w <= 1e-12) return fallback_action;

  SplitMix64Engine rng(rng_seed);
  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
  return actions[dist(rng)];
}

ActionId NetMcts::search_root(
    const IGameState& root,
    const IGameRules& rules,
    const IStateValueModel& value_model,
    const IPolicyValueEvaluator& evaluator,
    NetMctsStats* stats) const {
  const auto legal_root = rules.legal_actions(root);
  if (legal_root.empty()) {
    if (stats) {
      *stats = {};
    }
    return -1;
  }

  const auto t0 = std::chrono::steady_clock::now();

  std::vector<Node> nodes;
  nodes.reserve(static_cast<size_t>(std::max(512, cfg_.simulations * 2)));
  nodes.push_back(Node{value_model.current_player(root), false, 0, 0.0f, {}});

  auto expand_node = [&](Node& node, const IGameState& state) -> float {
    const auto legal = rules.legal_actions(state);
    if (legal.empty()) {
      node.expanded = true;
      node.edges.clear();
      return value_model.terminal_value_for_player(state, node.to_play);
    }
    std::vector<float> priors;
    float v = 0.0f;
    const bool ok = evaluator.evaluate(state, node.to_play, legal, &priors, &v);
    if (!ok || priors.size() != legal.size()) {
      priors.assign(legal.size(), 1.0f / static_cast<float>(legal.size()));
      v = 0.0f;
    }

    float sum = 0.0f;
    for (float p : priors) {
      sum += std::max(0.0f, p);
    }
    if (sum <= 1e-8f) {
      const float uniform = 1.0f / static_cast<float>(legal.size());
      priors.assign(legal.size(), uniform);
    } else {
      for (float& p : priors) {
        p = std::max(0.0f, p) / sum;
      }
    }

    node.edges.clear();
    node.edges.reserve(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
      Edge edge{};
      edge.action = legal[i];
      edge.prior = priors[i];
      node.edges.push_back(std::move(edge));
    }
    node.expanded = true;
    return clip_value(v, cfg_.value_clip);
  };

  // Pre-expand root.
  (void)expand_node(nodes[0], root);
  apply_root_dirichlet_noise(
      nodes[0],
      cfg_.root_dirichlet_alpha,
      cfg_.root_dirichlet_epsilon,
      static_cast<std::uint64_t>(t0.time_since_epoch().count()) ^
          static_cast<std::uint64_t>(value_model.current_player(root) + 13));

  const int simulations = std::max(1, cfg_.simulations);
  for (int sim = 0; sim < simulations; ++sim) {
    std::unique_ptr<IGameState> sim_state = root.clone_state();
    std::vector<int> path_nodes;
    std::vector<int> path_edges;
    std::vector<ActionId> path_actions;
    std::vector<UndoToken> path_undos;
    path_nodes.reserve(static_cast<size_t>(cfg_.max_depth + 2));
    path_edges.reserve(static_cast<size_t>(cfg_.max_depth + 2));
    path_actions.reserve(static_cast<size_t>(cfg_.max_depth + 2));
    path_undos.reserve(static_cast<size_t>(cfg_.max_depth + 2));

    int cur_idx = 0;
    path_nodes.push_back(cur_idx);

    float leaf_value = 0.0f;
    int depth = 0;
    bool skip_limiter_once = false;
    while (depth < cfg_.max_depth) {
      Node& cur = nodes[cur_idx];
      if (!skip_limiter_once && cfg_.traversal_limiter != nullptr) {
        std::unique_ptr<IGameState> parent_state{};
        const IGameState* parent_ptr = nullptr;
        ActionId parent_action = -1;
        const bool need_parent = cfg_.traversal_limiter->requires_parent_for_stop();
        if (need_parent && !path_undos.empty() && !path_actions.empty()) {
          parent_state = sim_state->clone_state();
          rules.undo_action(*parent_state, path_undos.back());
          parent_ptr = parent_state.get();
          parent_action = path_actions.back();
        }
        if (cfg_.traversal_limiter->should_stop_with_parent(root, *sim_state, parent_ptr, parent_action, depth)) {
          if (parent_ptr == nullptr && !path_undos.empty() && !path_actions.empty()) {
            parent_state = sim_state->clone_state();
            rules.undo_action(*parent_state, path_undos.back());
            parent_ptr = parent_state.get();
            parent_action = path_actions.back();
          }
          const TraversalStopResult stop_result = cfg_.traversal_limiter->on_traversal_stop(
              root,
              *sim_state,
              parent_ptr,
              parent_action,
              depth,
              rules,
              value_model,
              evaluator);
          if (stop_result.action == TraversalStopAction::kContinue) {
            if (!path_edges.empty() && path_nodes.size() >= 2) {
              const int parent_idx = path_nodes[path_nodes.size() - 2];
              const int parent_edge_idx = path_edges.back();
              Edge& parent_edge = nodes[parent_idx].edges[parent_edge_idx];
              const StateHash64 sampled_hash = sim_state->state_hash(true);
              int rerouted_child = -1;
              if (parent_edge.child >= 0 && parent_edge.has_child_state_hash &&
                  parent_edge.child_state_hash == sampled_hash) {
                rerouted_child = parent_edge.child;
              } else {
                for (const auto& entry : parent_edge.chance_children) {
                  if (entry.first == sampled_hash) {
                    rerouted_child = entry.second;
                    break;
                  }
                }
                if (rerouted_child < 0) {
                  nodes.push_back(Node{value_model.current_player(*sim_state), false, 0, 0.0f, {}});
                  rerouted_child = static_cast<int>(nodes.size()) - 1;
                  if (parent_edge.child < 0 || !parent_edge.has_child_state_hash) {
                    parent_edge.child = rerouted_child;
                    parent_edge.child_state_hash = sampled_hash;
                    parent_edge.has_child_state_hash = true;
                  } else {
                    parent_edge.chance_children.push_back({sampled_hash, rerouted_child});
                  }
                }
              }
              if (rerouted_child >= 0 && rerouted_child != cur_idx) {
                cur_idx = rerouted_child;
                path_nodes.back() = cur_idx;
              }
            }
            skip_limiter_once = true;
            continue;
          }
          if (stop_result.action == TraversalStopAction::kUseLeafValue) {
            leaf_value = clip_value(stop_result.leaf_value, cfg_.value_clip);
          } else {
            leaf_value = expand_node(cur, *sim_state);
          }
          break;
        }
      }
      skip_limiter_once = false;
      if (value_model.is_terminal(*sim_state)) {
        leaf_value = value_model.terminal_value_for_player(*sim_state, cur.to_play);
        break;
      }
      if (!cur.expanded) {
        leaf_value = expand_node(cur, *sim_state);
        break;
      }
      if (cur.edges.empty()) {
        leaf_value = value_model.terminal_value_for_player(*sim_state, cur.to_play);
        break;
      }

      int best_edge = 0;
      float best_score = -std::numeric_limits<float>::infinity();
      const float sqrt_parent = std::sqrt(static_cast<float>(std::max(1, cur.visit_count)));
      for (int ei = 0; ei < static_cast<int>(cur.edges.size()); ++ei) {
        const Edge& e = cur.edges[ei];
        float q = 0.0f;
        if (e.visit_count > 0) {
          q = e.value_sum / static_cast<float>(e.visit_count);
        }
        const float u =
            cfg_.c_puct * e.prior * sqrt_parent /
            (1.0f + static_cast<float>(e.visit_count));
        const float score = q + u;
        if (score > best_score) {
          best_score = score;
          best_edge = ei;
        }
      }

      // Avoid holding references into `nodes` across `nodes.push_back()`:
      // pushing can reallocate the `nodes` vector and invalidate references.
      const ActionId chosen_action = cur.edges[best_edge].action;
      const UndoToken undo_tok = rules.do_action_fast(*sim_state, chosen_action);
      const int next_player = value_model.current_player(*sim_state);
      const StateHash64 next_hash = sim_state->state_hash(true);
      int chosen_child = nodes[cur_idx].edges[best_edge].child;

      auto bind_edge_to_child = [&](int edge_idx, int child_idx, StateHash64 outcome_hash) {
        Edge& edge = nodes[cur_idx].edges[edge_idx];
        edge.child = child_idx;
        edge.child_state_hash = outcome_hash;
        edge.has_child_state_hash = true;
      };

      if (chosen_child < 0) {
        nodes.push_back(Node{next_player, false, 0, 0.0f, {}});
        chosen_child = static_cast<int>(nodes.size()) - 1;
        bind_edge_to_child(best_edge, chosen_child, next_hash);
      } else {
        Edge& edge = nodes[cur_idx].edges[best_edge];
        if (!edge.has_child_state_hash) {
          edge.child_state_hash = next_hash;
          edge.has_child_state_hash = true;
        } else if (edge.child_state_hash != next_hash) {
          int matched_child = -1;
          for (const auto& item : edge.chance_children) {
            if (item.first == next_hash) {
              matched_child = item.second;
              break;
            }
          }
          if (matched_child >= 0) {
            chosen_child = matched_child;
          } else {
            nodes.push_back(Node{next_player, false, 0, 0.0f, {}});
            chosen_child = static_cast<int>(nodes.size()) - 1;
            edge.chance_children.push_back({next_hash, chosen_child});
          }
        }
      }
      cur_idx = chosen_child;
      path_edges.push_back(best_edge);
      path_actions.push_back(chosen_action);
      path_undos.push_back(undo_tok);
      path_nodes.push_back(cur_idx);
      depth += 1;
    }

    // Backup as negamax values.
    float v = clip_value(leaf_value, cfg_.value_clip);
    for (int i = static_cast<int>(path_nodes.size()) - 1; i >= 0; --i) {
      const int node_idx = path_nodes[static_cast<size_t>(i)];
      Node& n = nodes[node_idx];
      n.visit_count += 1;
      n.value_sum += v;
      // Convert `v` (perspective of current node's `to_play`) to the parent node's
      // perspective. Only negate when `to_play` changes.
      if (i > 0) {
        const int parent_idx = path_nodes[static_cast<size_t>(i - 1)];
        const bool same_player = (nodes[parent_idx].to_play == n.to_play);
        const float parent_v = same_player ? v : -v;
        const int parent_edge_idx = path_edges[static_cast<size_t>(i - 1)];
        Edge& parent_edge = nodes[parent_idx].edges[parent_edge_idx];
        parent_edge.visit_count += 1;
        parent_edge.value_sum += parent_v;
        v = parent_v;
      }
    }
  }

  const Node& root_node = nodes[0];
  int best_edge = 0;
  int best_visit = -1;
  for (int ei = 0; ei < static_cast<int>(root_node.edges.size()); ++ei) {
    const Edge& e = root_node.edges[ei];
    const int vc = e.visit_count;
    if (vc > best_visit) {
      best_visit = vc;
      best_edge = ei;
    }
  }

  if (stats) {
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::max(1e-9, std::chrono::duration<double>(t1 - t0).count());
    stats->simulations_done = simulations;
    stats->expanded_nodes = static_cast<std::int64_t>(nodes.size());
    stats->nodes_per_sec = static_cast<double>(nodes.size()) / sec;
    stats->root_actions.clear();
    stats->root_action_visits.clear();
    stats->root_actions.reserve(root_node.edges.size());
    stats->root_action_visits.reserve(root_node.edges.size());
    for (const Edge& e : root_node.edges) {
      stats->root_actions.push_back(e.action);
      stats->root_action_visits.push_back(e.visit_count);
    }
    const Edge& best = root_node.edges[best_edge];
    float q = 0.0f;
    if (best.visit_count > 0) {
      q = best.value_sum / static_cast<float>(best.visit_count);
    }
    stats->best_action_value = static_cast<double>(clip_value(q, cfg_.value_clip));
  }
  return root_node.edges[best_edge].action;
}

}  // namespace board_ai::search
