#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "infer/onnx_policy_value_evaluator.h"
#include "search/net_mcts.h"
#include "search/root_noise.h"
#include "search/search_options_common.h"
#include "search/temperature_schedule.h"
#include "splendor_net_adapter.h"
#include "splendor_rules.h"
#include "splendor_state.h"

namespace {

using board_ai::ActionId;
using board_ai::IGameRules;
using board_ai::IGameState;
using board_ai::splendor::SplendorData;
using board_ai::splendor::SplendorRules;
using board_ai::splendor::SplendorState;
using board_ai::splendor::SplendorFeatureEncoder;
using board_ai::splendor::SplendorStateValueModel;
using board_ai::splendor::kActionSpace;
using board_ai::splendor::kBuyFaceupOffset;
using board_ai::splendor::kBuyFaceupCount;
using board_ai::splendor::kBuyReservedOffset;
using board_ai::splendor::kBuyReservedCount;
using board_ai::splendor::kFeatureDim;
using board_ai::splendor::kPassAction;
using board_ai::splendor::kReserveFaceupOffset;
using board_ai::splendor::kReserveFaceupCount;
using board_ai::splendor::kReserveDeckOffset;
using board_ai::splendor::kReserveDeckCount;
using board_ai::splendor::kTakeThreeOffset;
using board_ai::splendor::kTakeThreeCount;
using board_ai::splendor::kTakeTwoDifferentOffset;
using board_ai::splendor::kTakeTwoDifferentCount;
using board_ai::splendor::kTakeOneOffset;
using board_ai::splendor::kTakeOneCount;
using board_ai::splendor::kTakeTwoSameOffset;
using board_ai::splendor::kTakeTwoSameCount;
using board_ai::splendor::kChooseNobleOffset;
using board_ai::splendor::kChooseNobleCount;
using board_ai::infer::OnnxPolicyValueEvaluator;
using board_ai::search::NetMcts;
using board_ai::search::NetMctsConfig;
using board_ai::search::NetMctsStats;
using board_ai::search::select_action_from_visits;
using board_ai::search::INetMctsTraversalLimiter;
using board_ai::search::IPolicyValueEvaluator;
using board_ai::search::IStateValueModel;
using board_ai::search::TraversalStopAction;
using board_ai::search::TraversalStopResult;

struct SplendorSearchOptions {
  bool stop_on_draw_transition = true;
  bool enable_draw_chance = true;
  int chance_expand_cap = 10;
  bool enable_tail_solve = false;
  int tail_solve_start_ply = 40;
  int tail_solve_node_budget = 10000000;
  int tail_solve_time_ms = 0;
  int tail_solve_depth_limit = 5;
  double tail_solve_score_diff_weight = 0.01;
  board_ai::search::CommonSearchOptions common{};
};

struct EventRec {
  int ply = 0;
  int actor = 0;
  int action_id = 0;
  bool forced = false;
};

struct Session {
  SplendorState state;
  SplendorRules rules;
  int human_player = 0;
  std::vector<SplendorState> timeline_states{};
  std::vector<EventRec> timeline_events{};
  int cursor = 0;

  Session(std::uint64_t seed, int human) : state(), rules(), human_player(human), cursor(0) {
    state.reset_with_seed(seed);
    timeline_states.push_back(state);
  }
};

static std::unordered_map<std::int64_t, std::unique_ptr<Session>> g_sessions;
static std::int64_t g_next_handle = 1;

Session* get_session_from_handle(PyObject* handle_obj) {
  const long long handle = PyLong_AsLongLong(handle_obj);
  if (PyErr_Occurred()) return nullptr;
  auto it = g_sessions.find(static_cast<std::int64_t>(handle));
  if (it == g_sessions.end()) {
    PyErr_SetString(PyExc_KeyError, "session handle not found");
    return nullptr;
  }
  return it->second.get();
}

std::vector<double> visit_probs(const std::vector<ActionId>& actions, const std::vector<int>& visits) {
  std::vector<double> out(actions.size(), 0.0);
  double s = 0.0;
  for (int v : visits) s += static_cast<double>(std::max(0, v));
  if (s <= 1e-12) {
    if (!out.empty()) {
      const double u = 1.0 / static_cast<double>(out.size());
      for (double& x : out) x = u;
    }
    return out;
  }
  for (size_t i = 0; i < out.size() && i < visits.size(); ++i) {
    out[i] = static_cast<double>(std::max(0, visits[i])) / s;
  }
  return out;
}

bool parse_splendor_search_options(PyObject* obj, SplendorSearchOptions* out) {
  if (!out) return false;
  if (!board_ai::search::parse_common_search_options(obj, &out->common)) return false;
  if (obj == nullptr || obj == Py_None) return true;
  if (!board_ai::search::parse_bool_option(obj, "stop_on_draw_transition", &out->stop_on_draw_transition)) return false;
  if (!board_ai::search::parse_bool_option(obj, "enable_draw_chance", &out->enable_draw_chance)) return false;
  if (!board_ai::search::parse_int_option(obj, "chance_expand_cap", &out->chance_expand_cap, 1)) return false;
  if (!board_ai::search::parse_bool_option(obj, "enable_tail_solve", &out->enable_tail_solve)) return false;
  if (!board_ai::search::parse_int_option(obj, "tail_solve_start_ply", &out->tail_solve_start_ply, 0)) return false;
  if (!board_ai::search::parse_int_option(obj, "tail_solve_node_budget", &out->tail_solve_node_budget, 1)) return false;
  if (!board_ai::search::parse_int_option(obj, "tail_solve_time_ms", &out->tail_solve_time_ms, 0)) return false;
  if (!board_ai::search::parse_int_option(obj, "tail_solve_depth_limit", &out->tail_solve_depth_limit, 0)) return false;
  if (!board_ai::search::parse_nonnegative_double_option(
          obj, "tail_solve_score_diff_weight", &out->tail_solve_score_diff_weight)) {
    return false;
  }
  return true;
}

SplendorState with_hidden_randomized(const SplendorState& base, std::mt19937_64* rng) {
  SplendorState sampled = base;
  if (!rng) return sampled;
  SplendorData data = sampled.persistent.data();
  for (int t = 0; t < 3; ++t) {
    auto& deck = data.decks[static_cast<size_t>(t)];
    if (!deck.empty()) {
      std::shuffle(deck.begin(), deck.end(), *rng);
    }
  }
  data.draw_nonce ^= static_cast<std::uint64_t>((*rng)());
  auto node = std::make_shared<board_ai::splendor::SplendorPersistentNode>();
  node->action_from_parent = -1;
  node->materialized = std::make_shared<SplendorData>(std::move(data));
  sampled.persistent = board_ai::splendor::SplendorPersistentState(node);
  sampled.undo_stack.clear();
  return sampled;
}

class SplendorNetMctsTraversalLimiter final : public INetMctsTraversalLimiter {
 public:
  explicit SplendorNetMctsTraversalLimiter(const SplendorSearchOptions& options) : options_(options) {}

  bool should_stop(const IGameState& root_state, const IGameState& current_state, int /*depth*/) const override {
    const auto* root = dynamic_cast<const SplendorState*>(&root_state);
    const auto* cur = dynamic_cast<const SplendorState*>(&current_state);
    if (!root || !cur) return false;
    if (!options_.stop_on_draw_transition) return false;
    return cur->persistent.data().draw_nonce != root->persistent.data().draw_nonce;
  }

  bool requires_parent_for_stop() const override { return true; }

  bool should_stop_with_parent(
      const IGameState& root_state,
      const IGameState& current_state,
      const IGameState* parent_state,
      ActionId parent_action,
      int depth) const override {
    (void)parent_action;
    if (!options_.stop_on_draw_transition) return false;
    const auto* cur = dynamic_cast<const SplendorState*>(&current_state);
    const auto* pre = dynamic_cast<const SplendorState*>(parent_state);
    if (cur && pre) {
      return cur->persistent.data().draw_nonce != pre->persistent.data().draw_nonce;
    }
    return should_stop(root_state, current_state, depth);
  }

  TraversalStopResult on_traversal_stop(
      const IGameState& root_state,
      IGameState& current_state,
      const IGameState* parent_state,
      ActionId parent_action,
      int depth,
      const IGameRules& rules,
      const IStateValueModel& value_model,
      const IPolicyValueEvaluator& evaluator) const override {
    (void)root_state;
    (void)depth;
    (void)value_model;
    (void)evaluator;
    TraversalStopResult out{};
    if (!options_.stop_on_draw_transition || !options_.enable_draw_chance) return out;
    auto* cur = dynamic_cast<SplendorState*>(&current_state);
    const auto* pre = dynamic_cast<const SplendorState*>(parent_state);
    const auto* splendor_rules = dynamic_cast<const SplendorRules*>(&rules);
    if (!cur || !pre || !splendor_rules) return out;
    if (parent_action < 0) return out;
    const auto& cur_d = cur->persistent.data();
    const auto& pre_d = pre->persistent.data();
    if (cur_d.draw_nonce == pre_d.draw_nonce) return out;
    const int cap = std::max(1, options_.chance_expand_cap);
    std::uint64_t key = static_cast<std::uint64_t>(pre->state_hash(false));
    key ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(parent_action)) + 0x9e3779b97f4a7c15ULL + (key << 6U) +
           (key >> 2U);
    auto& pool = chance_pool_[key];
    if (static_cast<int>(pool.size()) < cap) {
      SplendorState sampled = with_hidden_randomized(*pre, &chance_rng_);
      if (splendor_rules->validate_action(sampled, parent_action)) {
        splendor_rules->do_action_fast(sampled, parent_action);
        sampled.undo_stack.clear();
        pool.push_back(std::move(sampled));
      }
    }
    if (pool.empty()) return out;
    const size_t idx = static_cast<size_t>(chance_rng_() % static_cast<std::uint64_t>(pool.size()));
    *cur = pool[idx];
    out.action = TraversalStopAction::kContinue;
    return out;
  }

 private:
  SplendorSearchOptions options_{};
  mutable std::unordered_map<std::uint64_t, std::vector<SplendorState>> chance_pool_{};
  mutable std::mt19937_64 chance_rng_{0xB19F3A5D7E991233ULL};
};

int effective_card_cost_after_bonus(const board_ai::splendor::SplendorCard& card, const SplendorData& d, int player) {
  int need = 0;
  for (int c = 0; c < 5; ++c) {
    const int raw = static_cast<int>(card.cost[static_cast<size_t>(c)]);
    const int bonus = static_cast<int>(d.player_bonuses[static_cast<size_t>(player)][static_cast<size_t>(c)]);
    need += std::max(0, raw - bonus);
  }
  return need;
}

int affordable_deficit_after_gems(const board_ai::splendor::SplendorCard& card, const SplendorData& d, int player) {
  int deficit = 0;
  for (int c = 0; c < 5; ++c) {
    const int raw = static_cast<int>(card.cost[static_cast<size_t>(c)]);
    const int bonus = static_cast<int>(d.player_bonuses[static_cast<size_t>(player)][static_cast<size_t>(c)]);
    const int need = std::max(0, raw - bonus);
    const int have = static_cast<int>(d.player_gems[static_cast<size_t>(player)][static_cast<size_t>(c)]);
    deficit += std::max(0, need - have);
  }
  const int gold = static_cast<int>(d.player_gems[static_cast<size_t>(player)][5]);
  return std::max(0, deficit - gold);
}

int gold_needed_for_purchase(const board_ai::splendor::SplendorCard& card, const SplendorData& d, int player) {
  int uncovered = 0;
  for (int c = 0; c < 5; ++c) {
    const int raw = static_cast<int>(card.cost[static_cast<size_t>(c)]);
    const int bonus = static_cast<int>(d.player_bonuses[static_cast<size_t>(player)][static_cast<size_t>(c)]);
    const int need = std::max(0, raw - bonus);
    const int have = static_cast<int>(d.player_gems[static_cast<size_t>(player)][static_cast<size_t>(c)]);
    uncovered += std::max(0, need - have);
  }
  const int gold = static_cast<int>(d.player_gems[static_cast<size_t>(player)][5]);
  return std::max(0, std::min(uncovered, gold));
}

int total_tokens_of_player(const SplendorData& d, int player) {
  int total = 0;
  for (int c = 0; c < 6; ++c) {
    total += static_cast<int>(d.player_gems[static_cast<size_t>(player)][static_cast<size_t>(c)]);
  }
  return total;
}

bool has_affordable_buy(const SplendorData& d, int player) {
  const auto& cards = board_ai::splendor::splendor_card_pool();
  for (int tier = 0; tier < 3; ++tier) {
    for (int slot = 0; slot < d.tableau_size[static_cast<size_t>(tier)]; ++slot) {
      const int cid = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
      if (affordable_deficit_after_gems(cards[static_cast<size_t>(cid)], d, player) == 0) return true;
    }
  }
  for (int slot = 0; slot < d.reserved_size[static_cast<size_t>(player)]; ++slot) {
    const int cid = d.reserved[static_cast<size_t>(player)][static_cast<size_t>(slot)];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
    if (affordable_deficit_after_gems(cards[static_cast<size_t>(cid)], d, player) == 0) return true;
  }
  return false;
}

int projected_token_gain(const SplendorData& d, ActionId action) {
  if (action >= kTakeThreeOffset && action < kTakeThreeOffset + kTakeThreeCount) return 3;
  if (action >= kTakeTwoDifferentOffset && action < kTakeTwoDifferentOffset + kTakeTwoDifferentCount) return 2;
  if (action >= kTakeOneOffset && action < kTakeOneOffset + kTakeOneCount) return 1;
  if (action >= kTakeTwoSameOffset && action < kTakeTwoSameOffset + kTakeTwoSameCount) return 2;
  if (action >= kReserveFaceupOffset && action < kReserveFaceupOffset + kReserveFaceupCount) {
    return d.bank[5] > 0 ? 1 : 0;
  }
  if (action >= kReserveDeckOffset && action < kReserveDeckOffset + kReserveDeckCount) {
    return d.bank[5] > 0 ? 1 : 0;
  }
  return 0;
}

double noble_progress_score(const SplendorData& d, int player) {
  const auto& nobles = board_ai::splendor::splendor_nobles();
  if (d.nobles_size <= 0) return 0.0;
  int best_missing = std::numeric_limits<int>::max();
  for (int i = 0; i < d.nobles_size; ++i) {
    const int nid = d.nobles[static_cast<size_t>(i)];
    if (nid < 0 || nid >= static_cast<int>(nobles.size())) continue;
    int missing = 0;
    for (int c = 0; c < 5; ++c) {
      const int req = static_cast<int>(nobles[static_cast<size_t>(nid)][static_cast<size_t>(c)]);
      const int have = static_cast<int>(d.player_bonuses[static_cast<size_t>(player)][static_cast<size_t>(c)]);
      missing += std::max(0, req - have);
    }
    best_missing = std::min(best_missing, missing);
  }
  if (best_missing == std::numeric_limits<int>::max()) return 0.0;
  return 1.0 / (1.0 + static_cast<double>(best_missing));
}

int noble_missing_for_player(const SplendorData& d, int player, int noble_slot) {
  const auto& nobles = board_ai::splendor::splendor_nobles();
  if (noble_slot < 0 || noble_slot >= d.nobles_size) return 99;
  const int nid = d.nobles[static_cast<size_t>(noble_slot)];
  if (nid < 0 || nid >= static_cast<int>(nobles.size())) return 99;
  int missing = 0;
  for (int c = 0; c < 5; ++c) {
    const int req = static_cast<int>(nobles[static_cast<size_t>(nid)][static_cast<size_t>(c)]);
    const int have = static_cast<int>(d.player_bonuses[static_cast<size_t>(player)][static_cast<size_t>(c)]);
    missing += std::max(0, req - have);
  }
  return missing;
}

double token_color_need_score(const SplendorData& d, int player, int color, int copies) {
  if (color < 0 || color >= 5 || copies <= 0) return 0.0;
  const auto& cards = board_ai::splendor::splendor_card_pool();
  const int op = 1 - player;
  double score = 0.0;

  auto score_card = [&](const board_ai::splendor::SplendorCard& card, double point_scale, double urgency_scale) {
    const int raw = static_cast<int>(card.cost[static_cast<size_t>(color)]);
    const int bonus = static_cast<int>(d.player_bonuses[static_cast<size_t>(player)][static_cast<size_t>(color)]);
    const int have = static_cast<int>(d.player_gems[static_cast<size_t>(player)][static_cast<size_t>(color)]);
    const int missing_this_color = std::max(0, std::max(0, raw - bonus) - have);
    if (missing_this_color <= 0) return;
    const int deficit = affordable_deficit_after_gems(card, d, player);
    const int covered = std::min(copies, missing_this_color);
    score += static_cast<double>(covered) * (point_scale + static_cast<double>(card.points) * 0.16);
    if (deficit <= 2) score += urgency_scale;
    if (deficit == 1) score += urgency_scale * 0.8;
  };

  for (int tier = 0; tier < 3; ++tier) {
    for (int slot = 0; slot < d.tableau_size[static_cast<size_t>(tier)]; ++slot) {
      const int cid = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
      score_card(cards[static_cast<size_t>(cid)], 0.18, 0.22);
    }
  }
  for (int slot = 0; slot < d.reserved_size[static_cast<size_t>(player)]; ++slot) {
    const int cid = d.reserved[static_cast<size_t>(player)][static_cast<size_t>(slot)];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
    score_card(cards[static_cast<size_t>(cid)], 0.28, 0.32);
  }

  int scarce_pressure = 0;
  for (int tier = 0; tier < 3; ++tier) {
    for (int slot = 0; slot < d.tableau_size[static_cast<size_t>(tier)]; ++slot) {
      const int cid = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
      const auto& card = cards[static_cast<size_t>(cid)];
      const int op_deficit = affordable_deficit_after_gems(card, d, op);
      if (op_deficit > 1) continue;
      const int raw = static_cast<int>(card.cost[static_cast<size_t>(color)]);
      const int op_bonus = static_cast<int>(d.player_bonuses[static_cast<size_t>(op)][static_cast<size_t>(color)]);
      const int op_have = static_cast<int>(d.player_gems[static_cast<size_t>(op)][static_cast<size_t>(color)]);
      if (std::max(0, std::max(0, raw - op_bonus) - op_have) > 0) scarce_pressure += 1;
    }
  }
  score += static_cast<double>(scarce_pressure) * 0.08;
  return score;
}

double noble_choice_bias(const SplendorData& d, int player, int noble_slot) {
  if (noble_slot < 0 || noble_slot >= d.nobles_size) return -1e6;
  const int op = 1 - player;
  const int op_missing = noble_missing_for_player(d, op, noble_slot);
  const int my_missing = noble_missing_for_player(d, player, noble_slot);
  double bias = 0.35;
  bias += static_cast<double>(std::max(0, 4 - std::min(op_missing, 4))) * 0.25;
  bias += static_cast<double>(std::max(0, 4 - std::min(my_missing, 4))) * 0.05;
  return bias;
}

double heuristic_eval(const SplendorData& d, int player) {
  if (d.terminal) {
    if (d.winner < 0 || d.shared_victory) return 0.0;
    return d.winner == player ? 10000.0 : -10000.0;
  }
  const int op = 1 - player;
  const double points_term = static_cast<double>(d.player_points[player] - d.player_points[op]) * 8.0;
  int bonus_p = 0;
  int bonus_o = 0;
  int color_gems_p = 0;
  int color_gems_o = 0;
  for (int c = 0; c < 5; ++c) {
    bonus_p += d.player_bonuses[player][c];
    bonus_o += d.player_bonuses[op][c];
    color_gems_p += d.player_gems[player][c];
    color_gems_o += d.player_gems[op][c];
  }
  const int gold_p = static_cast<int>(d.player_gems[static_cast<size_t>(player)][5]);
  const int gold_o = static_cast<int>(d.player_gems[static_cast<size_t>(op)][5]);

  double reserved_progress = 0.0;
  const auto& cards = board_ai::splendor::splendor_card_pool();
  for (int slot = 0; slot < d.reserved_size[static_cast<size_t>(player)]; ++slot) {
    const int cid = d.reserved[static_cast<size_t>(player)][static_cast<size_t>(slot)];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
    const auto& card = cards[static_cast<size_t>(cid)];
    const int deficit = affordable_deficit_after_gems(card, d, player);
    reserved_progress += static_cast<double>(card.points) * 0.85;
    reserved_progress += (deficit == 0 ? 1.3 : 0.0);
    reserved_progress -= static_cast<double>(deficit) * 0.95;
  }
  for (int slot = 0; slot < d.reserved_size[static_cast<size_t>(op)]; ++slot) {
    const int cid = d.reserved[static_cast<size_t>(op)][static_cast<size_t>(slot)];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
    const auto& card = cards[static_cast<size_t>(cid)];
    const int deficit = affordable_deficit_after_gems(card, d, op);
    reserved_progress -= static_cast<double>(card.points) * 0.75;
    reserved_progress -= (deficit == 0 ? 0.9 : 0.0);
  }
  const int my_reserved_slots = static_cast<int>(d.reserved_size[static_cast<size_t>(player)]);
  const int op_reserved_slots = static_cast<int>(d.reserved_size[static_cast<size_t>(op)]);
  reserved_progress -= static_cast<double>(my_reserved_slots) * 0.55;
  reserved_progress += static_cast<double>(op_reserved_slots) * 0.22;
  if (my_reserved_slots >= 2 && !has_affordable_buy(d, player)) {
    reserved_progress -= static_cast<double>(my_reserved_slots - 1) * 0.9;
  }
  if (my_reserved_slots >= 3 && !has_affordable_buy(d, player)) {
    reserved_progress -= 1.2;
  }

  double buy_now_pressure = 0.0;
  for (int tier = 0; tier < 3; ++tier) {
    for (int slot = 0; slot < d.tableau_size[static_cast<size_t>(tier)]; ++slot) {
      const int cid = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
      const auto& card = cards[static_cast<size_t>(cid)];
      const int my_deficit = affordable_deficit_after_gems(card, d, player);
      const int op_deficit = affordable_deficit_after_gems(card, d, op);
      if (my_deficit == 0) buy_now_pressure += 1.0 + static_cast<double>(card.points) * 0.7;
      if (my_deficit <= 2) buy_now_pressure += 0.3 + static_cast<double>(card.points) * 0.25;
      if (op_deficit == 0) buy_now_pressure -= 0.4 + static_cast<double>(card.points) * 0.45;
    }
  }

  const double noble_term = (noble_progress_score(d, player) - noble_progress_score(d, op)) * 2.2;
  const double engine_term = static_cast<double>(bonus_p - bonus_o) * 0.85;
  const double gem_term = static_cast<double>(color_gems_p - color_gems_o) * 0.16 +
                          static_cast<double>(gold_p - gold_o) * 0.32;
  const double endgame_push =
      (d.player_points[player] >= 12 ? 1.8 : 0.0) - (d.player_points[op] >= 12 ? 1.4 : 0.0);
  const double return_penalty =
      (d.current_player == player ? static_cast<double>(std::max(0, d.pending_returns)) * 0.9 : 0.0);

  return points_term + engine_term + gem_term + reserved_progress + buy_now_pressure + noble_term + endgame_push -
         return_penalty;
}

double tactical_action_bias(const SplendorState& state, const SplendorRules& rules, int player, ActionId action) {
  const auto& d = state.persistent.data();
  const int op = 1 - player;
  const auto& cards = board_ai::splendor::splendor_card_pool();
  const int total_tokens = total_tokens_of_player(d, player);
  const int gained_tokens = projected_token_gain(d, action);
  const int overflow_after_action = std::max(0, total_tokens + gained_tokens - 10);
  const bool buy_available = has_affordable_buy(d, player);
  const double token_need_scale = total_tokens >= 9 ? 0.22 : (total_tokens >= 7 ? 0.5 : 1.0);
  double bias = 0.0;

  if (action == kPassAction) bias -= 5.0;

  if (action >= kBuyFaceupOffset && action < kBuyFaceupOffset + kBuyFaceupCount) {
    const int idx = action - kBuyFaceupOffset;
    const int tier = idx / 4;
    const int slot = idx % 4;
    if (tier >= 0 && tier < 3 && slot >= 0 && slot < d.tableau_size[static_cast<size_t>(tier)]) {
      const int cid = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      if (cid >= 0 && cid < static_cast<int>(cards.size())) {
        const auto& card = cards[static_cast<size_t>(cid)];
        bias += static_cast<double>(card.points) * 1.2 + 0.7;
        const int gold_spent = gold_needed_for_purchase(card, d, player);
        if (gold_spent > 0) {
          const int gold_now = static_cast<int>(d.player_gems[static_cast<size_t>(player)][5]);
          double gold_penalty = static_cast<double>(gold_spent) * 0.48;
          if (card.points <= 1) gold_penalty += 0.28;
          if (gold_now <= gold_spent) gold_penalty += 0.18;
          if (card.points >= 3) gold_penalty -= 0.18;
          bias -= std::max(0.1, gold_penalty);
        }
      }
    }
  } else if (action >= kBuyReservedOffset && action < kBuyReservedOffset + kBuyReservedCount) {
    const int slot = action - kBuyReservedOffset;
    if (slot >= 0 && slot < d.reserved_size[static_cast<size_t>(player)]) {
      const int cid = d.reserved[static_cast<size_t>(player)][static_cast<size_t>(slot)];
      if (cid >= 0 && cid < static_cast<int>(cards.size())) {
        const auto& card = cards[static_cast<size_t>(cid)];
        bias += static_cast<double>(card.points) * 1.35 + 0.8;
        const int gold_spent = gold_needed_for_purchase(card, d, player);
        if (gold_spent > 0) {
          const int gold_now = static_cast<int>(d.player_gems[static_cast<size_t>(player)][5]);
          double gold_penalty = static_cast<double>(gold_spent) * 0.48;
          if (card.points <= 1) gold_penalty += 0.28;
          if (gold_now <= gold_spent) gold_penalty += 0.18;
          if (card.points >= 3) gold_penalty -= 0.18;
          bias -= std::max(0.1, gold_penalty);
        }
      }
    }
  } else if (action >= kReserveFaceupOffset && action < kReserveFaceupOffset + kReserveFaceupCount) {
    const int idx = action - kReserveFaceupOffset;
    const int tier = idx / 4;
    const int slot = idx % 4;
    if (tier >= 0 && tier < 3 && slot >= 0 && slot < d.tableau_size[static_cast<size_t>(tier)]) {
      const int cid = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      if (cid >= 0 && cid < static_cast<int>(cards.size())) {
        const auto& card = cards[static_cast<size_t>(cid)];
        const int op_deficit = affordable_deficit_after_gems(card, d, op);
        const int my_cost = effective_card_cost_after_bonus(card, d, player);
        const bool deny = (op_deficit <= 1 && card.points >= 2);
        const int my_reserved_slots = static_cast<int>(d.reserved_size[static_cast<size_t>(player)]);
        bias += static_cast<double>(card.points) * 0.25;
        bias += deny ? 0.9 : 0.0;
        bias += (my_cost <= 4 ? 0.25 : 0.0);
        bias -= static_cast<double>(my_reserved_slots) * 0.8;
        if (my_reserved_slots >= 2) bias -= 0.9;
        if (!buy_available && my_reserved_slots >= 1) bias -= 0.6;
        if (my_cost >= 7) bias -= 0.7;
        if (total_tokens <= 2 && !deny && my_cost > 5) bias -= 0.8;
      }
    }
  } else if (action >= kReserveDeckOffset && action < kReserveDeckOffset + kReserveDeckCount) {
    const int my_reserved_slots = static_cast<int>(d.reserved_size[static_cast<size_t>(player)]);
    bias += 0.05;  // keep a weak default value for unknown deck reserve.
    bias -= static_cast<double>(my_reserved_slots) * 0.85;
    if (!buy_available && my_reserved_slots >= 1) bias -= 0.6;
  } else if (action >= kTakeThreeOffset && action < kTakeThreeOffset + kTakeThreeCount) {
    bias += 0.25;
    const int idx = action - kTakeThreeOffset;
    static constexpr std::array<std::array<int, 3>, 10> kTakeThreeCombos{{
        {{0, 1, 2}},
        {{0, 1, 3}},
        {{0, 1, 4}},
        {{0, 2, 3}},
        {{0, 2, 4}},
        {{0, 3, 4}},
        {{1, 2, 3}},
        {{1, 2, 4}},
        {{1, 3, 4}},
        {{2, 3, 4}},
    }};
    for (int color : kTakeThreeCombos[static_cast<size_t>(idx)]) {
      bias += token_color_need_score(d, player, color, 1) * token_need_scale;
    }
  } else if (action >= kTakeTwoDifferentOffset && action < kTakeTwoDifferentOffset + kTakeTwoDifferentCount) {
    bias += 0.18;
    static constexpr std::array<std::array<int, 2>, 10> kTakeTwoDifferentCombos{{
        {{0, 1}},
        {{0, 2}},
        {{0, 3}},
        {{0, 4}},
        {{1, 2}},
        {{1, 3}},
        {{1, 4}},
        {{2, 3}},
        {{2, 4}},
        {{3, 4}},
    }};
    const int idx = action - kTakeTwoDifferentOffset;
    for (int color : kTakeTwoDifferentCombos[static_cast<size_t>(idx)]) {
      bias += token_color_need_score(d, player, color, 1) * token_need_scale;
    }
  } else if (action >= kTakeOneOffset && action < kTakeOneOffset + kTakeOneCount) {
    const int color = action - kTakeOneOffset;
    bias += 0.08 + token_color_need_score(d, player, color, 1) * token_need_scale;
  } else if (action >= kTakeTwoSameOffset && action < kTakeTwoSameOffset + kTakeTwoSameCount) {
    const int color = action - kTakeTwoSameOffset;
    bias += 0.10 + token_color_need_score(d, player, color, 2) * 0.85 * token_need_scale;
  } else if (action >= kChooseNobleOffset && action < kChooseNobleOffset + kChooseNobleCount) {
    const int noble_slot = action - kChooseNobleOffset;
    bias += noble_choice_bias(d, player, noble_slot);
  }
  if (gained_tokens > 0) {
    bias -= static_cast<double>(overflow_after_action) * 2.6;
    if (total_tokens >= 8) {
      bias -= static_cast<double>(total_tokens - 7) * 0.55;
    }
    if (total_tokens >= 10) {
      bias -= 5.0;
    }
    if (buy_available) {
      bias -= 1.4 + static_cast<double>(gained_tokens) * 0.25;
    }
  }
  if (buy_available &&
      ((action >= kBuyFaceupOffset && action < kBuyFaceupOffset + kBuyFaceupCount) ||
       (action >= kBuyReservedOffset && action < kBuyReservedOffset + kBuyReservedCount))) {
    bias += 1.1 + (total_tokens >= 8 ? 0.6 : 0.0);
  }
  if (d.pending_returns > 0 && action >= board_ai::splendor::kReturnTokenOffset &&
      action < board_ai::splendor::kReturnTokenOffset + board_ai::splendor::kReturnTokenCount) {
    const int token = action - board_ai::splendor::kReturnTokenOffset;
    bias += 0.2;
    if (token == 5) bias += 0.18;
    if (token >= 0 && token < 5) {
      bias -= token_color_need_score(d, player, token, 1) * 0.6;
    }
  }
  return bias;
}

ActionId choose_heuristic_action(const SplendorState& state, const SplendorRules& rules) {
  const auto legal = rules.legal_actions(state);
  if (legal.empty()) return kPassAction;
  const int player = state.persistent.data().current_player;
  ActionId best = legal.front();
  double best_score = -1e18;
  for (ActionId a : legal) {
    const auto nxt = state.persistent.advance(a);
    double s = heuristic_eval(nxt.data(), player);
    s += tactical_action_bias(state, rules, player, a);
    if (s > best_score) {
      best_score = s;
      best = a;
    }
  }
  return best;
}

struct SplendorTailSolveResult {
  ActionId chosen_action = -1;
  double chosen_value = 0.0;
  double nodes = 0.0;
  double cutoffs = 0.0;
  double elapsed_ms = 0.0;
  bool budget_hit = false;
  bool solved = false;
};

constexpr std::array<std::array<int, 3>, 10> kTailTakeThreeCombos{{
    {{0, 1, 2}},
    {{0, 1, 3}},
    {{0, 1, 4}},
    {{0, 2, 3}},
    {{0, 2, 4}},
    {{0, 3, 4}},
    {{1, 2, 3}},
    {{1, 2, 4}},
    {{1, 3, 4}},
    {{2, 3, 4}},
}};

constexpr std::array<std::array<int, 2>, 10> kTailTakeTwoDifferentCombos{{
    {{0, 1}},
    {{0, 2}},
    {{0, 3}},
    {{0, 4}},
    {{1, 2}},
    {{1, 3}},
    {{1, 4}},
    {{2, 3}},
    {{2, 4}},
    {{3, 4}},
}};

constexpr std::array<int, 5> kTailTakeOneColors{{0, 1, 2, 3, 4}};

struct TailCompactState {
  int current_player = 0;
  int plies = 0;
  int final_round_remaining = -1;
  std::int8_t stage = static_cast<std::int8_t>(board_ai::splendor::SplendorTurnStage::kNormal);
  int pending_returns = 0;
  std::array<std::int8_t, 3> pending_noble_slots{{-1, -1, -1}};
  std::int8_t pending_nobles_size = 0;
  int winner = -1;
  bool terminal = false;
  bool shared_victory = false;

  std::array<std::int8_t, board_ai::splendor::kTokenTypes> bank{};
  std::array<std::array<std::int8_t, board_ai::splendor::kTokenTypes>, board_ai::splendor::kPlayers> player_gems{};
  std::array<std::array<std::int8_t, board_ai::splendor::kColorCount>, board_ai::splendor::kPlayers> player_bonuses{};
  std::array<std::int16_t, board_ai::splendor::kPlayers> player_points{};
  std::array<std::int16_t, board_ai::splendor::kPlayers> player_cards_count{};
  std::array<std::int16_t, board_ai::splendor::kPlayers> player_nobles_count{};
  std::array<std::array<std::int16_t, 3>, board_ai::splendor::kPlayers> reserved{};
  std::array<std::array<std::int8_t, 3>, board_ai::splendor::kPlayers> reserved_visible{};
  std::array<std::int8_t, board_ai::splendor::kPlayers> reserved_size{};
  std::array<int, 3> deck_counts{{0, 0, 0}};
  std::array<std::array<std::int16_t, 4>, 3> tableau{};
  std::array<std::int8_t, 3> tableau_size{};
  std::array<std::int16_t, 3> nobles{};
  std::int8_t nobles_size = 0;
};

TailCompactState make_tail_compact_state(const SplendorData& d) {
  TailCompactState s{};
  s.current_player = d.current_player;
  s.plies = d.plies;
  s.final_round_remaining = d.final_round_remaining;
  s.stage = d.stage;
  s.pending_returns = d.pending_returns;
  s.pending_noble_slots = d.pending_noble_slots;
  s.pending_nobles_size = d.pending_nobles_size;
  s.winner = d.winner;
  s.terminal = d.terminal;
  s.shared_victory = d.shared_victory;
  s.bank = d.bank;
  s.player_gems = d.player_gems;
  s.player_bonuses = d.player_bonuses;
  s.player_points = d.player_points;
  s.player_cards_count = d.player_cards_count;
  s.player_nobles_count = d.player_nobles_count;
  s.reserved = d.reserved;
  s.reserved_visible = d.reserved_visible;
  s.reserved_size = d.reserved_size;
  for (int t = 0; t < 3; ++t) {
    s.deck_counts[static_cast<size_t>(t)] = static_cast<int>(d.decks[static_cast<size_t>(t)].size());
  }
  s.tableau = d.tableau;
  s.tableau_size = d.tableau_size;
  s.nobles = d.nobles;
  s.nobles_size = d.nobles_size;
  return s;
}

std::uint64_t tail_state_hash(const TailCompactState& s) {
  std::uint64_t h = 0x9e3779b97f4a7c15ULL;
  auto mix = [&](std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    x ^= (x >> 31U);
    h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
  };
  mix(static_cast<std::uint64_t>(s.current_player + 1));
  mix(static_cast<std::uint64_t>(s.plies + 3));
  mix(static_cast<std::uint64_t>(static_cast<int>(s.stage) + 5));
  mix(static_cast<std::uint64_t>(s.pending_returns + 7));
  mix(static_cast<std::uint64_t>(s.final_round_remaining + 11));
  mix(static_cast<std::uint64_t>(s.winner + 13));
  mix(static_cast<std::uint64_t>(s.terminal ? 17 : 19));
  mix(static_cast<std::uint64_t>(s.shared_victory ? 23 : 29));
  for (int i = 0; i < 3; ++i) mix(static_cast<std::uint64_t>(s.pending_noble_slots[static_cast<size_t>(i)] + 31 + i));
  mix(static_cast<std::uint64_t>(s.pending_nobles_size + 41));
  for (int i = 0; i < board_ai::splendor::kTokenTypes; ++i) mix(static_cast<std::uint64_t>(s.bank[static_cast<size_t>(i)] + 43 + i));
  for (int p = 0; p < board_ai::splendor::kPlayers; ++p) {
    for (int i = 0; i < board_ai::splendor::kTokenTypes; ++i) mix(static_cast<std::uint64_t>(s.player_gems[p][static_cast<size_t>(i)] + 53 + p * 11 + i));
    for (int i = 0; i < board_ai::splendor::kColorCount; ++i) mix(static_cast<std::uint64_t>(s.player_bonuses[p][static_cast<size_t>(i)] + 71 + p * 7 + i));
    mix(static_cast<std::uint64_t>(s.player_points[static_cast<size_t>(p)] + 89 + p));
    mix(static_cast<std::uint64_t>(s.player_cards_count[static_cast<size_t>(p)] + 97 + p));
    mix(static_cast<std::uint64_t>(s.player_nobles_count[static_cast<size_t>(p)] + 101 + p));
    mix(static_cast<std::uint64_t>(s.reserved_size[static_cast<size_t>(p)] + 107 + p));
    for (int i = 0; i < 3; ++i) {
      mix(static_cast<std::uint64_t>(s.reserved[p][static_cast<size_t>(i)] + 113 + p * 5 + i));
      mix(static_cast<std::uint64_t>(s.reserved_visible[p][static_cast<size_t>(i)] + 127 + p * 3 + i));
    }
  }
  for (int t = 0; t < 3; ++t) {
    mix(static_cast<std::uint64_t>(s.deck_counts[static_cast<size_t>(t)] + 137 + t));
    mix(static_cast<std::uint64_t>(s.tableau_size[static_cast<size_t>(t)] + 149 + t));
    for (int i = 0; i < 4; ++i) mix(static_cast<std::uint64_t>(s.tableau[static_cast<size_t>(t)][static_cast<size_t>(i)] + 157 + t * 7 + i));
  }
  mix(static_cast<std::uint64_t>(s.nobles_size + 181));
  for (int i = 0; i < 3; ++i) mix(static_cast<std::uint64_t>(s.nobles[static_cast<size_t>(i)] + 191 + i));
  return h;
}

inline bool tail_can_afford(const TailCompactState& s, int player, const board_ai::splendor::SplendorCard& card) {
  int need_gold = 0;
  for (int c = 0; c < 5; ++c) {
    const int remaining = std::max(
        0,
        static_cast<int>(card.cost[static_cast<size_t>(c)]) - static_cast<int>(s.player_bonuses[player][static_cast<size_t>(c)]) -
            static_cast<int>(s.player_gems[player][static_cast<size_t>(c)]));
    need_gold += remaining;
  }
  return need_gold <= static_cast<int>(s.player_gems[player][5]);
}

void tail_legal_actions(const TailCompactState& s, std::vector<ActionId>& out) {
  out.clear();
  if (s.terminal) return;
  const int player = s.current_player;
  const auto stage = static_cast<board_ai::splendor::SplendorTurnStage>(s.stage);
  if (stage == board_ai::splendor::SplendorTurnStage::kReturnTokens || s.pending_returns > 0) {
    for (int c = 0; c < board_ai::splendor::kTokenTypes; ++c) {
      if (s.player_gems[player][static_cast<size_t>(c)] > 0) out.push_back(board_ai::splendor::kReturnTokenOffset + c);
    }
    if (out.empty()) out.push_back(board_ai::splendor::kPassAction);
    return;
  }
  if (stage == board_ai::splendor::SplendorTurnStage::kChooseNoble) {
    for (int i = 0; i < s.pending_nobles_size; ++i) {
      const int slot = s.pending_noble_slots[static_cast<size_t>(i)];
      if (slot >= 0) out.push_back(board_ai::splendor::kChooseNobleOffset + slot);
    }
    if (out.empty()) out.push_back(board_ai::splendor::kPassAction);
    return;
  }

  const auto& cards = board_ai::splendor::splendor_card_pool();
  for (int t = 0; t < 3; ++t) {
    const int n = s.tableau_size[static_cast<size_t>(t)];
    for (int slot = 0; slot < n && slot < 4; ++slot) {
      const int cid = s.tableau[static_cast<size_t>(t)][static_cast<size_t>(slot)];
      if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
      if (tail_can_afford(s, player, cards[static_cast<size_t>(cid)])) out.push_back(kBuyFaceupOffset + t * 4 + slot);
    }
  }
  for (int idx = 0; idx < s.reserved_size[static_cast<size_t>(player)] && idx < 3; ++idx) {
    const int cid = s.reserved[static_cast<size_t>(player)][static_cast<size_t>(idx)];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) continue;
    if (tail_can_afford(s, player, cards[static_cast<size_t>(cid)])) out.push_back(kBuyReservedOffset + idx);
  }
  if (s.reserved_size[static_cast<size_t>(player)] < 3) {
    for (int t = 0; t < 3; ++t) {
      const int n = s.tableau_size[static_cast<size_t>(t)];
      for (int slot = 0; slot < n && slot < 4; ++slot) {
        const int cid = s.tableau[static_cast<size_t>(t)][static_cast<size_t>(slot)];
        if (cid >= 0) out.push_back(kReserveFaceupOffset + t * 4 + slot);
      }
      if (s.deck_counts[static_cast<size_t>(t)] > 0) out.push_back(kReserveDeckOffset + t);
    }
  }

  for (int i = 0; i < 10; ++i) {
    const auto& comb = kTailTakeThreeCombos[static_cast<size_t>(i)];
    if (s.bank[static_cast<size_t>(comb[0])] > 0 && s.bank[static_cast<size_t>(comb[1])] > 0 && s.bank[static_cast<size_t>(comb[2])] > 0) {
      out.push_back(kTakeThreeOffset + i);
    }
  }
  for (int i = 0; i < 10; ++i) {
    const auto& comb = kTailTakeTwoDifferentCombos[static_cast<size_t>(i)];
    if (s.bank[static_cast<size_t>(comb[0])] > 0 && s.bank[static_cast<size_t>(comb[1])] > 0) out.push_back(kTakeTwoDifferentOffset + i);
  }
  for (int i = 0; i < 5; ++i) {
    const int c = kTailTakeOneColors[static_cast<size_t>(i)];
    if (s.bank[static_cast<size_t>(c)] > 0) out.push_back(kTakeOneOffset + i);
  }
  for (int c = 0; c < 5; ++c) {
    if (s.bank[static_cast<size_t>(c)] >= 4) out.push_back(kTakeTwoSameOffset + c);
  }
  if (out.empty()) out.push_back(kPassAction);
}

double tail_action_order_score_compact(const TailCompactState& s, ActionId aid, int root_player) {
  const auto& cards = board_ai::splendor::splendor_card_pool();
  const int actor = s.current_player;
  double v = 0.0;
  if (aid >= kBuyFaceupOffset && aid < (kBuyFaceupOffset + kBuyFaceupCount)) {
    const int idx = aid - kBuyFaceupOffset;
    const int tier = idx / 4;
    const int slot = idx % 4;
    if (tier >= 0 && tier < 3 && slot >= 0 && slot < s.tableau_size[static_cast<size_t>(tier)]) {
      const int cid = s.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      if (cid >= 0 && cid < static_cast<int>(cards.size())) v += 8.0 * static_cast<double>(cards[static_cast<size_t>(cid)].points);
    }
    v += 6.0;
  } else if (aid >= kBuyReservedOffset && aid < (kBuyReservedOffset + kBuyReservedCount)) {
    const int slot = aid - kBuyReservedOffset;
    if (slot >= 0 && slot < s.reserved_size[static_cast<size_t>(actor)]) {
      const int cid = s.reserved[static_cast<size_t>(actor)][static_cast<size_t>(slot)];
      if (cid >= 0 && cid < static_cast<int>(cards.size())) v += 8.0 * static_cast<double>(cards[static_cast<size_t>(cid)].points);
    }
    v += 5.0;
  } else if (aid >= kReserveFaceupOffset && aid < (kReserveFaceupOffset + kReserveFaceupCount)) {
    v += 1.5;
  } else if (aid >= kReserveDeckOffset && aid < (kReserveDeckOffset + kReserveDeckCount)) {
    v += 1.2;
  } else if (aid >= kTakeThreeOffset && aid < (kTakeThreeOffset + kTakeThreeCount)) {
    v += 0.8;
  } else if (aid >= kTakeTwoDifferentOffset && aid < (kTakeTwoDifferentOffset + kTakeTwoDifferentCount)) {
    v += 0.6;
  } else if (aid >= kTakeOneOffset && aid < (kTakeOneOffset + kTakeOneCount)) {
    v += 0.2;
  } else if (aid >= kTakeTwoSameOffset && aid < (kTakeTwoSameOffset + kTakeTwoSameCount)) {
    v += 0.4;
  } else if (aid >= kChooseNobleOffset && aid < (kChooseNobleOffset + kChooseNobleCount)) {
    v += 10.0;
  } else if (aid >= board_ai::splendor::kReturnTokenOffset &&
             aid < (board_ai::splendor::kReturnTokenOffset + board_ai::splendor::kReturnTokenCount)) {
    v -= 0.5;
  }
  if (actor != root_player) v = -v;
  return v;
}

inline bool is_tail_reserve_deck_action(ActionId aid) {
  return aid >= kReserveDeckOffset && aid < (kReserveDeckOffset + kReserveDeckCount);
}

bool tail_apply_action(TailCompactState& s, ActionId action) {
  if (s.terminal) return false;
  const auto& cards = board_ai::splendor::splendor_card_pool();
  const auto& nobles = board_ai::splendor::splendor_nobles();
  const int player = s.current_player;
  bool bought_card = false;
  const auto stage = static_cast<board_ai::splendor::SplendorTurnStage>(s.stage);

  auto noble_met = [&](int noble_slot) {
    if (noble_slot < 0 || noble_slot >= s.nobles_size) return false;
    const int nid = s.nobles[static_cast<size_t>(noble_slot)];
    if (nid < 0 || nid >= static_cast<int>(nobles.size())) return false;
    for (int c = 0; c < 5; ++c) {
      if (s.player_bonuses[player][static_cast<size_t>(c)] < nobles[static_cast<size_t>(nid)][static_cast<size_t>(c)]) return false;
    }
    return true;
  };
  auto clear_pending_nobles = [&]() {
    s.pending_noble_slots = {{-1, -1, -1}};
    s.pending_nobles_size = 0;
  };
  auto claim_noble = [&](int noble_slot) {
    if (!noble_met(noble_slot)) return;
    for (int j = noble_slot + 1; j < s.nobles_size; ++j) s.nobles[static_cast<size_t>(j - 1)] = s.nobles[static_cast<size_t>(j)];
    s.nobles[static_cast<size_t>(s.nobles_size - 1)] = -1;
    s.nobles_size -= 1;
    s.player_points[static_cast<size_t>(player)] += 3;
    s.player_nobles_count[static_cast<size_t>(player)] += 1;
    clear_pending_nobles();
  };
  auto pay_card = [&](const board_ai::splendor::SplendorCard& card) {
    for (int c = 0; c < 5; ++c) {
      int need = std::max(0, static_cast<int>(card.cost[static_cast<size_t>(c)]) - static_cast<int>(s.player_bonuses[player][static_cast<size_t>(c)]));
      const int use_color = std::min(need, static_cast<int>(s.player_gems[player][static_cast<size_t>(c)]));
      s.player_gems[player][static_cast<size_t>(c)] -= static_cast<std::int8_t>(use_color);
      s.bank[static_cast<size_t>(c)] += static_cast<std::int8_t>(use_color);
      need -= use_color;
      if (need > 0) {
        s.player_gems[player][5] -= static_cast<std::int8_t>(need);
        s.bank[5] += static_cast<std::int8_t>(need);
      }
    }
  };
  auto update_terminal = [&]() {
    if (s.final_round_remaining < 0 && s.player_points[static_cast<size_t>(player)] >= board_ai::splendor::kTargetPoints) {
      s.final_round_remaining = (player == 0) ? 1 : 0;
    } else if (s.final_round_remaining >= 0) {
      s.final_round_remaining -= 1;
    }
    if (s.plies >= board_ai::splendor::kMaxPlies || s.final_round_remaining == 0) {
      s.terminal = true;
      int w = -1;
      if (s.player_points[0] > s.player_points[1]) w = 0;
      else if (s.player_points[1] > s.player_points[0]) w = 1;
      else if (s.player_cards_count[0] < s.player_cards_count[1]) w = 0;
      else if (s.player_cards_count[1] < s.player_cards_count[0]) w = 1;
      s.winner = w;
      s.shared_victory = w < 0;
    } else {
      s.terminal = false;
      s.winner = -1;
      s.shared_victory = false;
    }
  };
  auto finalize_turn = [&]() {
    s.stage = static_cast<std::int8_t>(board_ai::splendor::SplendorTurnStage::kNormal);
    s.pending_returns = 0;
    clear_pending_nobles();
    update_terminal();
    s.current_player = 1 - player;
  };

  if (stage == board_ai::splendor::SplendorTurnStage::kReturnTokens || s.pending_returns > 0) {
    if (!(action >= board_ai::splendor::kReturnTokenOffset &&
          action < (board_ai::splendor::kReturnTokenOffset + board_ai::splendor::kReturnTokenCount))) {
      return false;
    }
    const int token = action - board_ai::splendor::kReturnTokenOffset;
    if (token < 0 || token >= board_ai::splendor::kTokenTypes) return false;
    if (s.player_gems[player][static_cast<size_t>(token)] <= 0) return false;
    s.player_gems[player][static_cast<size_t>(token)] -= 1;
    s.bank[static_cast<size_t>(token)] += 1;
    s.pending_returns -= 1;
    s.plies += 1;
    if (s.pending_returns <= 0) finalize_turn();
    return true;
  }

  if (stage == board_ai::splendor::SplendorTurnStage::kChooseNoble) {
    if (!(action >= kChooseNobleOffset && action < (kChooseNobleOffset + kChooseNobleCount))) return false;
    int chosen = action - kChooseNobleOffset;
    bool ok = false;
    for (int i = 0; i < s.pending_nobles_size; ++i) {
      if (s.pending_noble_slots[static_cast<size_t>(i)] == chosen) ok = true;
    }
    if (!ok) return false;
    claim_noble(chosen);
    s.plies += 1;
    finalize_turn();
    return true;
  }

  if (action >= kBuyFaceupOffset && action < (kBuyFaceupOffset + kBuyFaceupCount)) {
    const int idx = action - kBuyFaceupOffset;
    const int tier = idx / 4;
    const int slot = idx % 4;
    if (tier < 0 || tier >= 3 || slot < 0 || slot >= s.tableau_size[static_cast<size_t>(tier)]) return false;
    const int cid = s.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) return false;
    if (!tail_can_afford(s, player, cards[static_cast<size_t>(cid)])) return false;
    for (int i = slot + 1; i < s.tableau_size[static_cast<size_t>(tier)]; ++i) {
      s.tableau[static_cast<size_t>(tier)][static_cast<size_t>(i - 1)] = s.tableau[static_cast<size_t>(tier)][static_cast<size_t>(i)];
    }
    s.tableau[static_cast<size_t>(tier)][static_cast<size_t>(s.tableau_size[static_cast<size_t>(tier)] - 1)] = -1;
    s.tableau_size[static_cast<size_t>(tier)] -= 1;
    const auto& card = cards[static_cast<size_t>(cid)];
    pay_card(card);
    s.player_bonuses[player][static_cast<size_t>(card.bonus)] += 1;
    s.player_points[static_cast<size_t>(player)] += card.points;
    s.player_cards_count[static_cast<size_t>(player)] += 1;
    bought_card = true;
  } else if (action >= kBuyReservedOffset && action < (kBuyReservedOffset + kBuyReservedCount)) {
    const int slot = action - kBuyReservedOffset;
    if (slot < 0 || slot >= s.reserved_size[static_cast<size_t>(player)]) return false;
    const int cid = s.reserved[static_cast<size_t>(player)][static_cast<size_t>(slot)];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) return false;
    if (!tail_can_afford(s, player, cards[static_cast<size_t>(cid)])) return false;
    for (int i = slot + 1; i < s.reserved_size[static_cast<size_t>(player)]; ++i) {
      s.reserved[static_cast<size_t>(player)][static_cast<size_t>(i - 1)] = s.reserved[static_cast<size_t>(player)][static_cast<size_t>(i)];
      s.reserved_visible[static_cast<size_t>(player)][static_cast<size_t>(i - 1)] = s.reserved_visible[static_cast<size_t>(player)][static_cast<size_t>(i)];
    }
    const int last = s.reserved_size[static_cast<size_t>(player)] - 1;
    s.reserved[static_cast<size_t>(player)][static_cast<size_t>(last)] = -1;
    s.reserved_visible[static_cast<size_t>(player)][static_cast<size_t>(last)] = 0;
    s.reserved_size[static_cast<size_t>(player)] -= 1;
    const auto& card = cards[static_cast<size_t>(cid)];
    pay_card(card);
    s.player_bonuses[player][static_cast<size_t>(card.bonus)] += 1;
    s.player_points[static_cast<size_t>(player)] += card.points;
    s.player_cards_count[static_cast<size_t>(player)] += 1;
    bought_card = true;
  } else if (action >= kReserveFaceupOffset && action < (kReserveFaceupOffset + kReserveFaceupCount)) {
    const int idx = action - kReserveFaceupOffset;
    const int tier = idx / 4;
    const int slot = idx % 4;
    if (s.reserved_size[static_cast<size_t>(player)] >= 3) return false;
    if (tier < 0 || tier >= 3 || slot < 0 || slot >= s.tableau_size[static_cast<size_t>(tier)]) return false;
    const int cid = s.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
    if (cid < 0) return false;
    for (int i = slot + 1; i < s.tableau_size[static_cast<size_t>(tier)]; ++i) {
      s.tableau[static_cast<size_t>(tier)][static_cast<size_t>(i - 1)] = s.tableau[static_cast<size_t>(tier)][static_cast<size_t>(i)];
    }
    s.tableau[static_cast<size_t>(tier)][static_cast<size_t>(s.tableau_size[static_cast<size_t>(tier)] - 1)] = -1;
    s.tableau_size[static_cast<size_t>(tier)] -= 1;
    const int pos = s.reserved_size[static_cast<size_t>(player)];
    s.reserved[static_cast<size_t>(player)][static_cast<size_t>(pos)] = static_cast<std::int16_t>(cid);
    s.reserved_visible[static_cast<size_t>(player)][static_cast<size_t>(pos)] = 1;
    s.reserved_size[static_cast<size_t>(player)] += 1;
    if (s.bank[5] > 0) {
      s.bank[5] -= 1;
      s.player_gems[player][5] += 1;
    }
  } else if (action >= kReserveDeckOffset && action < (kReserveDeckOffset + kReserveDeckCount)) {
    const int tier = action - kReserveDeckOffset;
    if (tier < 0 || tier >= 3) return false;
    if (s.reserved_size[static_cast<size_t>(player)] >= 3) return false;
    if (s.deck_counts[static_cast<size_t>(tier)] <= 0) return false;
    s.deck_counts[static_cast<size_t>(tier)] -= 1;
    const int pos = s.reserved_size[static_cast<size_t>(player)];
    s.reserved[static_cast<size_t>(player)][static_cast<size_t>(pos)] = -1;
    s.reserved_visible[static_cast<size_t>(player)][static_cast<size_t>(pos)] = 0;
    s.reserved_size[static_cast<size_t>(player)] += 1;
    if (s.bank[5] > 0) {
      s.bank[5] -= 1;
      s.player_gems[player][5] += 1;
    }
  } else if (action >= kTakeThreeOffset && action < (kTakeThreeOffset + kTakeThreeCount)) {
    const auto& comb = kTailTakeThreeCombos[static_cast<size_t>(action - kTakeThreeOffset)];
    if (s.bank[static_cast<size_t>(comb[0])] <= 0 || s.bank[static_cast<size_t>(comb[1])] <= 0 || s.bank[static_cast<size_t>(comb[2])] <= 0) {
      return false;
    }
    for (int j = 0; j < 3; ++j) {
      const int c = comb[static_cast<size_t>(j)];
      s.bank[static_cast<size_t>(c)] -= 1;
      s.player_gems[player][static_cast<size_t>(c)] += 1;
    }
  } else if (action >= kTakeTwoDifferentOffset && action < (kTakeTwoDifferentOffset + kTakeTwoDifferentCount)) {
    const auto& comb = kTailTakeTwoDifferentCombos[static_cast<size_t>(action - kTakeTwoDifferentOffset)];
    if (s.bank[static_cast<size_t>(comb[0])] <= 0 || s.bank[static_cast<size_t>(comb[1])] <= 0) return false;
    for (int j = 0; j < 2; ++j) {
      const int c = comb[static_cast<size_t>(j)];
      s.bank[static_cast<size_t>(c)] -= 1;
      s.player_gems[player][static_cast<size_t>(c)] += 1;
    }
  } else if (action >= kTakeOneOffset && action < (kTakeOneOffset + kTakeOneCount)) {
    const int c = kTailTakeOneColors[static_cast<size_t>(action - kTakeOneOffset)];
    if (s.bank[static_cast<size_t>(c)] <= 0) return false;
    s.bank[static_cast<size_t>(c)] -= 1;
    s.player_gems[player][static_cast<size_t>(c)] += 1;
  } else if (action >= kTakeTwoSameOffset && action < (kTakeTwoSameOffset + kTakeTwoSameCount)) {
    const int c = action - kTakeTwoSameOffset;
    if (s.bank[static_cast<size_t>(c)] < 4) return false;
    s.bank[static_cast<size_t>(c)] -= 2;
    s.player_gems[player][static_cast<size_t>(c)] += 2;
  } else if (action == kPassAction) {
  } else {
    return false;
  }

  s.plies += 1;
  int total = 0;
  for (int i = 0; i < board_ai::splendor::kTokenTypes; ++i) total += s.player_gems[player][static_cast<size_t>(i)];
  s.pending_returns = std::max(0, total - 10);
  if (s.pending_returns > 0) {
    s.stage = static_cast<std::int8_t>(board_ai::splendor::SplendorTurnStage::kReturnTokens);
    clear_pending_nobles();
    return true;
  }
  if (bought_card) {
    clear_pending_nobles();
    for (int slot = 0; slot < s.nobles_size && slot < 3; ++slot) {
      if (!noble_met(slot)) continue;
      s.pending_noble_slots[static_cast<size_t>(s.pending_nobles_size)] = static_cast<std::int8_t>(slot);
      s.pending_nobles_size += 1;
    }
    if (s.pending_nobles_size > 0) {
      s.stage = static_cast<std::int8_t>(board_ai::splendor::SplendorTurnStage::kChooseNoble);
      return true;
    }
  }
  finalize_turn();
  return true;
}

SplendorTailSolveResult solve_tail_deterministic(
    const SplendorState& input_state,
    const SplendorRules&,
    int start_ply,
    int node_budget,
    int time_budget_ms,
    int depth_limit,
    double score_diff_weight) {
  SplendorTailSolveResult out{};
  const auto t0 = std::chrono::steady_clock::now();
  TailCompactState root = make_tail_compact_state(input_state.persistent.data());
  const int root_player = root.current_player;
  if (root.plies < start_ply) {
    return out;
  }
  std::unordered_map<std::uint64_t, double> tt{};
  std::unordered_map<std::uint64_t, std::vector<ActionId>> action_cache{};
  bool fully_solved = true;

  struct BudgetExceeded final {};

  auto check_budget = [&]() {
    out.nodes += 1.0;
    if (out.nodes > static_cast<double>(std::max(1, node_budget))) {
      out.budget_hit = true;
      throw BudgetExceeded{};
    }
    if (time_budget_ms > 0) {
      const auto now = std::chrono::steady_clock::now();
      const double elapsed_ms = std::chrono::duration<double, std::milli>(now - t0).count();
      if (elapsed_ms > static_cast<double>(time_budget_ms)) {
        out.budget_hit = true;
        throw BudgetExceeded{};
      }
    }
  };

  std::function<double(TailCompactState&, double, double, int)> ab =
      [&](TailCompactState& st, double alpha, double beta, int depth_left) -> double {
    check_budget();
    if (st.terminal) {
      const double outcome = (st.shared_victory || st.winner < 0) ? 0.0 : ((st.winner == root_player) ? 1.0 : -1.0);
      const int me = st.player_points[static_cast<size_t>(root_player)];
      const int opp = st.player_points[static_cast<size_t>(1 - root_player)];
      return outcome + std::max(0.0, score_diff_weight) * static_cast<double>(me - opp);
    }
    if (depth_limit > 0 && depth_left <= 0) {
      // Depth-limited tail search: unresolved non-terminal leaves are neutral.
      return 0.0;
    }
    const std::uint64_t key = tail_state_hash(st);
    auto it = tt.find(key);
    if (it != tt.end()) {
      return it->second;
    }
    auto ac_it = action_cache.find(key);
    if (ac_it == action_cache.end()) {
      std::vector<ActionId> built{};
      tail_legal_actions(st, built);
      std::sort(built.begin(), built.end(), [&](ActionId a, ActionId b) {
        const bool a_deck = is_tail_reserve_deck_action(a);
        const bool b_deck = is_tail_reserve_deck_action(b);
        if (a_deck != b_deck) return a_deck;
        const double sa = tail_action_order_score_compact(st, a, root_player);
        const double sb = tail_action_order_score_compact(st, b, root_player);
        if (sa == sb) return a < b;
        return sa > sb;
      });
      ac_it = action_cache.emplace(key, std::move(built)).first;
    }
    const auto& actions = ac_it->second;
    if (actions.empty()) {
      fully_solved = false;
      return 0.0;
    }
    const bool maximize = st.current_player == root_player;
    double best = maximize ? -1e18 : 1e18;
    for (ActionId aid : actions) {
      double score = 0.0;
      TailCompactState before = st;
      if (!tail_apply_action(st, aid)) {
        st = before;
        continue;
      }
      score = ab(st, alpha, beta, depth_left - 1);
      st = before;
      if (maximize) {
        best = std::max(best, score);
        alpha = std::max(alpha, best);
      } else {
        best = std::min(best, score);
        beta = std::min(beta, best);
      }
      if (alpha >= beta) {
        out.cutoffs += 1.0;
        break;
      }
    }
    if (best <= -1e17 || best >= 1e17) {
      fully_solved = false;
      return 0.0;
    }
    tt[key] = best;
    return best;
  };

  try {
    const std::uint64_t root_key = tail_state_hash(root);
    auto root_it = action_cache.find(root_key);
    if (root_it == action_cache.end()) {
      std::vector<ActionId> built{};
      tail_legal_actions(root, built);
      std::sort(built.begin(), built.end(), [&](ActionId a, ActionId b) {
        const bool a_deck = is_tail_reserve_deck_action(a);
        const bool b_deck = is_tail_reserve_deck_action(b);
        if (a_deck != b_deck) return a_deck;
        const double sa = tail_action_order_score_compact(root, a, root_player);
        const double sb = tail_action_order_score_compact(root, b, root_player);
        if (sa == sb) return a < b;
        return sa > sb;
      });
      root_it = action_cache.emplace(root_key, std::move(built)).first;
    }
    const auto& root_actions = root_it->second;
    if (root_actions.empty()) {
      fully_solved = false;
    } else {
      double alpha = -1e18;
      double beta = 1e18;
      double best = -1e18;
      ActionId best_aid = -1;
      for (ActionId aid : root_actions) {
        double v = 0.0;
        TailCompactState before = root;
        if (!tail_apply_action(root, aid)) {
          root = before;
          continue;
        }
        v = ab(root, alpha, beta, depth_limit);
        root = before;
        if (best_aid < 0 || v > best) {
          best = v;
          best_aid = aid;
        }
        alpha = std::max(alpha, best);
      }
      out.chosen_action = best_aid;
      out.chosen_value = (best_aid >= 0) ? best : 0.0;
    }
  } catch (const BudgetExceeded&) {
    fully_solved = false;
  }
  const auto t1 = std::chrono::steady_clock::now();
  out.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  out.solved = (!out.budget_hit) && fully_solved && (out.chosen_action >= 0);
  return out;
}

struct AiDecision {
  ActionId action = -1;
  bool has_mcts_best_action_value = false;
  double mcts_best_action_value = 0.0;
  bool tail_solved = false;
  std::vector<ActionId> root_actions{};
  std::vector<int> root_visits{};
};

AiDecision choose_action_for_state(
    SplendorState& state,
    SplendorRules& rules,
    const std::string& engine,
    int simulations,
    double temperature,
    double dirichlet_alpha,
    double dirichlet_epsilon,
    const char* model_path,
    const SplendorSearchOptions& search_options,
    int ply_index = 0) {
  AiDecision d;
  if (engine == "heuristic") {
    d.action = choose_heuristic_action(state, rules);
    return d;
  }
  if (engine == "netmcts") {
    if (search_options.enable_tail_solve) {
      const auto tail = solve_tail_deterministic(
          state,
          rules,
          std::max(0, search_options.tail_solve_start_ply),
          std::max(1, search_options.tail_solve_node_budget),
          std::max(0, search_options.tail_solve_time_ms),
          std::max(0, search_options.tail_solve_depth_limit),
          std::max(0.0, search_options.tail_solve_score_diff_weight));
      const bool tail_value_indicates_terminal = (tail.chosen_value <= -1.0) || (tail.chosen_value >= 1.0);
      if (tail.solved && tail_value_indicates_terminal && tail.chosen_action >= 0 &&
          rules.validate_action(state, tail.chosen_action)) {
        d.action = tail.chosen_action;
        d.has_mcts_best_action_value = true;
        d.mcts_best_action_value = tail.chosen_value;
        d.tail_solved = true;
        return d;
      }
    }
    // Important: never fallback silently to classic MCTS for netmcts requests.
    // Callers should surface configuration/model errors explicitly.
    if (!(model_path && model_path[0] != '\0')) {
      return d;
    }
    SplendorFeatureEncoder encoder;
    OnnxPolicyValueEvaluator evaluator(std::string(model_path), &encoder, {});
    if (!evaluator.is_ready()) {
      return d;
    }
    NetMctsConfig cfg;
    cfg.simulations = std::max(1, simulations);
    cfg.max_depth = std::max(1, search_options.common.max_search_depth);
    cfg.c_puct = 1.4f;
    if (search_options.common.dirichlet_on_first_n_plies > 0) {
      const auto noise = board_ai::search::resolve_root_dirichlet_noise(
          search_options.common.dirichlet_alpha,
          search_options.common.dirichlet_epsilon,
          search_options.common.dirichlet_on_first_n_plies,
          ply_index);
      cfg.root_dirichlet_alpha = noise.alpha;
      cfg.root_dirichlet_epsilon = noise.epsilon;
    } else {
      const auto noise = board_ai::search::resolve_root_dirichlet_noise(dirichlet_alpha, dirichlet_epsilon, 0, ply_index);
      cfg.root_dirichlet_alpha = noise.alpha;
      cfg.root_dirichlet_epsilon = noise.epsilon;
    }
    SplendorNetMctsTraversalLimiter limiter(search_options);
    cfg.traversal_limiter = &limiter;
    SplendorStateValueModel value_model;
    NetMcts netmcts(cfg);
    NetMctsStats net_stats;
    d.action = netmcts.search_root(state, rules, value_model, evaluator, &net_stats);
    if (d.action >= 0) {
      d.has_mcts_best_action_value = true;
      d.mcts_best_action_value = net_stats.best_action_value;
      d.root_actions = std::move(net_stats.root_actions);
      d.root_visits = std::move(net_stats.root_action_visits);
      if (!d.root_actions.empty() && d.root_actions.size() == d.root_visits.size()) {
        const auto& sd = state.persistent.data();
        const std::uint64_t sample_seed =
            static_cast<std::uint64_t>(state.rng_salt) ^
            static_cast<std::uint64_t>(sd.draw_nonce) ^
            (static_cast<std::uint64_t>(ply_index + 1) * 0x9e3779b97f4a7c15ULL);
        d.action = select_action_from_visits(d.root_actions, d.root_visits, temperature, sample_seed, d.action);
      }
    }
    return d;
  }
  return d;
}

int resolve_bought_bonus_color_from_pre_state(const SplendorState& state, int actor, int action_id) {
  const auto& d = state.persistent.data();
  const auto& cards = board_ai::splendor::splendor_card_pool();
  int cid = -1;
  if (action_id >= kBuyFaceupOffset && action_id < (kBuyFaceupOffset + kBuyFaceupCount)) {
    const int idx = action_id - kBuyFaceupOffset;
    const int tier = idx / 4;
    const int slot = idx % 4;
    if (tier >= 0 && tier < 3 && slot >= 0 && slot < d.tableau_size[static_cast<size_t>(tier)]) {
      cid = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
    }
  } else if (action_id >= kBuyReservedOffset && action_id < (kBuyReservedOffset + kBuyReservedCount)) {
    const int slot = action_id - kBuyReservedOffset;
    if (actor >= 0 && actor < 2 && slot >= 0 && slot < d.reserved_size[static_cast<size_t>(actor)]) {
      cid = d.reserved[static_cast<size_t>(actor)][static_cast<size_t>(slot)];
    }
  }
  if (cid < 0 || cid >= static_cast<int>(cards.size())) return -1;
  return static_cast<int>(cards[static_cast<size_t>(cid)].bonus);
}

PyObject* build_public_state(const Session& s) {
  const auto& d = s.state.persistent.data();
  const auto& cards = board_ai::splendor::splendor_card_pool();
  const auto& nobles = board_ai::splendor::splendor_nobles();

  auto build_card_dict = [&cards](int cid) -> PyObject* {
    PyObject* cd = PyDict_New();
    if (!cd) return nullptr;
    if (cid < 0 || cid >= static_cast<int>(cards.size())) {
      Py_INCREF(Py_None);
      PyDict_SetItemString(cd, "id", Py_None);
      return cd;
    }
    const auto& c = cards[static_cast<size_t>(cid)];
    PyDict_SetItemString(cd, "id", PyLong_FromLong(cid));
    PyDict_SetItemString(cd, "tier", PyLong_FromLong(static_cast<long>(c.tier)));
    PyDict_SetItemString(cd, "bonus", PyLong_FromLong(static_cast<long>(c.bonus)));
    PyDict_SetItemString(cd, "points", PyLong_FromLong(static_cast<long>(c.points)));
    PyObject* cost = PyList_New(5);
    for (int i = 0; i < 5; ++i) {
      PyList_SET_ITEM(cost, i, PyLong_FromLong(static_cast<long>(c.cost[static_cast<size_t>(i)])));
    }
    PyDict_SetItemString(cd, "cost", cost);
    Py_DECREF(cost);
    return cd;
  };

  PyObject* root = PyDict_New();
  PyObject* common = PyDict_New();
  PyObject* game = PyDict_New();
  if (!root || !common || !game) {
    Py_XDECREF(root);
    Py_XDECREF(common);
    Py_XDECREF(game);
    return nullptr;
  }

  PyDict_SetItemString(common, "current_player", PyLong_FromLong(d.current_player));
  PyDict_SetItemString(common, "round_index", PyLong_FromLong(d.plies));
  PyObject* scores = PyList_New(2);
  PyList_SET_ITEM(scores, 0, PyLong_FromLong(d.scores[0]));
  PyList_SET_ITEM(scores, 1, PyLong_FromLong(d.scores[1]));
  PyDict_SetItemString(common, "scores", scores);
  Py_DECREF(scores);
  PyDict_SetItemString(common, "is_terminal", d.terminal ? Py_True : Py_False);
  if (d.terminal && d.winner >= 0) {
    PyDict_SetItemString(common, "winner", PyLong_FromLong(d.winner));
  } else {
    Py_INCREF(Py_None);
    PyDict_SetItemString(common, "winner", Py_None);
  }
  PyDict_SetItemString(common, "shared_victory", d.shared_victory ? Py_True : Py_False);

  const char* stage_name = "normal";
  if (d.stage == static_cast<std::int8_t>(board_ai::splendor::SplendorTurnStage::kReturnTokens)) {
    stage_name = "return_tokens";
  } else if (d.stage == static_cast<std::int8_t>(board_ai::splendor::SplendorTurnStage::kChooseNoble)) {
    stage_name = "choose_noble";
  }
  PyDict_SetItemString(game, "stage", PyUnicode_FromString(stage_name));
  PyDict_SetItemString(game, "stage_id", PyLong_FromLong(static_cast<long>(d.stage)));
  PyDict_SetItemString(game, "pending_returns", PyLong_FromLong(d.pending_returns));
  PyDict_SetItemString(game, "final_round_remaining", PyLong_FromLong(d.final_round_remaining));
  PyObject* pending_noble_slots = PyList_New(0);
  for (int i = 0; i < d.pending_nobles_size; ++i) {
    PyList_Append(pending_noble_slots, PyLong_FromLong(d.pending_noble_slots[static_cast<size_t>(i)]));
  }
  PyDict_SetItemString(game, "pending_noble_slots", pending_noble_slots);
  Py_DECREF(pending_noble_slots);

  PyObject* bank = PyList_New(6);
  for (int i = 0; i < 6; ++i) PyList_SET_ITEM(bank, i, PyLong_FromLong(d.bank[i]));
  PyDict_SetItemString(game, "bank", bank);
  Py_DECREF(bank);

  PyObject* players = PyList_New(2);
  for (int p = 0; p < 2; ++p) {
    PyObject* pd = PyDict_New();
    PyObject* gems = PyList_New(6);
    for (int i = 0; i < 6; ++i) PyList_SET_ITEM(gems, i, PyLong_FromLong(d.player_gems[p][i]));
    PyDict_SetItemString(pd, "gems", gems);
    Py_DECREF(gems);
    PyObject* bonuses = PyList_New(5);
    for (int i = 0; i < 5; ++i) PyList_SET_ITEM(bonuses, i, PyLong_FromLong(d.player_bonuses[p][i]));
    PyDict_SetItemString(pd, "bonuses", bonuses);
    Py_DECREF(bonuses);
    PyDict_SetItemString(pd, "points", PyLong_FromLong(d.player_points[p]));
    PyDict_SetItemString(pd, "cards_count", PyLong_FromLong(d.player_cards_count[p]));
    PyDict_SetItemString(pd, "nobles_count", PyLong_FromLong(d.player_nobles_count[p]));
    PyObject* reserved = PyList_New(0);
    for (int i = 0; i < d.reserved_size[p] && i < 3; ++i) {
      const int cid = d.reserved[p][i];
      const bool visible = d.reserved_visible[p][i] != 0;
      PyObject* rd = PyDict_New();
      PyDict_SetItemString(rd, "slot", PyLong_FromLong(i));
      PyDict_SetItemString(rd, "visible_to_opponent", visible ? Py_True : Py_False);
      PyObject* cd = build_card_dict(cid);
      PyDict_SetItemString(rd, "card", cd);
      Py_DECREF(cd);
      PyList_Append(reserved, rd);
      Py_DECREF(rd);
    }
    PyDict_SetItemString(pd, "reserved", reserved);
    Py_DECREF(reserved);
    PyList_SET_ITEM(players, p, pd);
  }
  PyDict_SetItemString(game, "players", players);
  Py_DECREF(players);

  PyObject* decks_remaining = PyList_New(3);
  for (int t = 0; t < 3; ++t) {
    PyList_SET_ITEM(
        decks_remaining, t, PyLong_FromLong(static_cast<long>(d.decks[static_cast<size_t>(t)].size()))
    );
  }
  PyDict_SetItemString(game, "decks_remaining", decks_remaining);
  Py_DECREF(decks_remaining);

  PyObject* tableau = PyList_New(3);
  for (int t = 0; t < 3; ++t) {
    PyObject* row = PyList_New(0);
    const int n = d.tableau_size[static_cast<size_t>(t)];
    for (int i = 0; i < n && i < 4; ++i) {
      const int cid = d.tableau[static_cast<size_t>(t)][static_cast<size_t>(i)];
      PyObject* cd = build_card_dict(cid);
      PyList_Append(row, cd);
      Py_DECREF(cd);
    }
    PyList_SET_ITEM(tableau, t, row);
  }
  PyDict_SetItemString(game, "tableau", tableau);
  Py_DECREF(tableau);

  PyObject* nobles_list = PyList_New(0);
  for (int i = 0; i < d.nobles_size && i < 3; ++i) {
    const int nid = d.nobles[static_cast<size_t>(i)];
    if (nid < 0 || nid >= static_cast<int>(nobles.size())) continue;
    PyObject* nd = PyDict_New();
    const bool selectable = [&d, i]() {
      for (int j = 0; j < d.pending_nobles_size; ++j) {
        if (d.pending_noble_slots[static_cast<size_t>(j)] == i) return true;
      }
      return false;
    }();
    PyDict_SetItemString(nd, "slot", PyLong_FromLong(i));
    PyDict_SetItemString(nd, "selectable", selectable ? Py_True : Py_False);
    PyDict_SetItemString(nd, "id", PyLong_FromLong(nid));
    PyDict_SetItemString(nd, "points", PyLong_FromLong(3));
    PyObject* req = PyList_New(5);
    for (int c = 0; c < 5; ++c) {
      PyList_SET_ITEM(req, c, PyLong_FromLong(nobles[static_cast<size_t>(nid)][static_cast<size_t>(c)]));
    }
    PyDict_SetItemString(nd, "requirements", req);
    Py_DECREF(req);
    PyList_Append(nobles_list, nd);
    Py_DECREF(nd);
  }
  PyDict_SetItemString(game, "nobles", nobles_list);
  Py_DECREF(nobles_list);

  PyDict_SetItemString(root, "common", common);
  PyDict_SetItemString(root, "game", game);
  Py_DECREF(common);
  Py_DECREF(game);
  return root;
}

PyObject* build_payload(Session& s) {
  PyObject* d = PyDict_New();
  PyDict_SetItemString(d, "state_version", PyLong_FromLong(s.cursor));
  PyObject* pub = build_public_state(s);
  PyDict_SetItemString(d, "public_state", pub);
  Py_DECREF(pub);
  return d;
}

PyObject* build_encoded_features_payload(const Session& s, int perspective_player) {
  if (perspective_player < 0 || perspective_player >= 2) {
    PyErr_SetString(PyExc_ValueError, "perspective_player must be 0 or 1");
    return nullptr;
  }
  board_ai::splendor::SplendorFeatureEncoder encoder;
  const auto legal = s.rules.legal_actions(s.state);
  std::vector<float> features;
  std::vector<float> legal_mask;
  if (!encoder.encode(s.state, perspective_player, legal, &features, &legal_mask)) {
    PyErr_SetString(PyExc_RuntimeError, "failed to encode splendor features");
    return nullptr;
  }
  PyObject* out = PyDict_New();
  if (!out) return nullptr;
  PyDict_SetItemString(out, "perspective_player", PyLong_FromLong(perspective_player));
  PyDict_SetItemString(out, "current_player", PyLong_FromLong(s.state.persistent.data().current_player));
  PyObject* feat = PyList_New(static_cast<Py_ssize_t>(features.size()));
  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(features.size()); ++i) {
    PyList_SET_ITEM(feat, i, PyFloat_FromDouble(static_cast<double>(features[static_cast<size_t>(i)])));
  }
  PyDict_SetItemString(out, "features", feat);
  Py_DECREF(feat);
  PyObject* mask = PyList_New(static_cast<Py_ssize_t>(legal_mask.size()));
  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(legal_mask.size()); ++i) {
    PyList_SET_ITEM(mask, i, PyFloat_FromDouble(static_cast<double>(legal_mask[static_cast<size_t>(i)])));
  }
  PyDict_SetItemString(out, "legal_mask", mask);
  Py_DECREF(mask);
  return out;
}

void truncate_timeline_to_cursor(Session& s) {
  if (s.cursor < static_cast<int>(s.timeline_events.size())) s.timeline_events.resize(static_cast<size_t>(s.cursor));
  if (s.cursor + 1 < static_cast<int>(s.timeline_states.size())) {
    s.timeline_states.resize(static_cast<size_t>(s.cursor + 1));
  }
}

PyObject* apply_action_common(Session& s, int action_id, bool forced) {
  truncate_timeline_to_cursor(s);
  const int actor = s.state.persistent.data().current_player;
  const int bought_bonus = resolve_bought_bonus_color_from_pre_state(s.state, actor, action_id);
  if (!s.rules.validate_action(s.state, action_id)) Py_RETURN_NONE;
  s.rules.do_action_fast(s.state, action_id);
  s.timeline_states.push_back(s.state);
  s.cursor += 1;
  EventRec ev;
  ev.ply = s.cursor;
  ev.actor = actor;
  ev.action_id = action_id;
  ev.forced = forced;
  s.timeline_events.push_back(ev);

  PyObject* out = PyDict_New();
  PyDict_SetItemString(out, "ply", PyLong_FromLong(ev.ply));
  PyDict_SetItemString(out, "actor", PyLong_FromLong(ev.actor));
  PyDict_SetItemString(out, "action_id", PyLong_FromLong(ev.action_id));
  PyDict_SetItemString(out, "forced", ev.forced ? Py_True : Py_False);
  if (bought_bonus >= 0 && bought_bonus < 5) {
    PyDict_SetItemString(out, "bought_bonus", PyLong_FromLong(bought_bonus));
  } else {
    Py_INCREF(Py_None);
    PyDict_SetItemString(out, "bought_bonus", Py_None);
  }
  return out;
}

PyObject* py_session_new(PyObject*, PyObject* args) {
  long long seed = 0;
  int human_player = 0;
  if (!PyArg_ParseTuple(args, "Li", &seed, &human_player)) return nullptr;
  const std::int64_t handle = g_next_handle++;
  g_sessions.emplace(handle, std::make_unique<Session>(static_cast<std::uint64_t>(seed), human_player));
  return PyLong_FromLongLong(handle);
}

PyObject* py_session_delete(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  if (!PyArg_ParseTuple(args, "O", &h)) return nullptr;
  const long long handle = PyLong_AsLongLong(h);
  if (PyErr_Occurred()) return nullptr;
  g_sessions.erase(static_cast<std::int64_t>(handle));
  Py_RETURN_NONE;
}

PyObject* py_session_payload(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  if (!PyArg_ParseTuple(args, "O", &h)) return nullptr;
  Session* s = get_session_from_handle(h);
  if (!s) return nullptr;
  return build_payload(*s);
}

PyObject* py_session_encode_features(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  int perspective_player = 0;
  if (!PyArg_ParseTuple(args, "Oi", &h, &perspective_player)) return nullptr;
  Session* s = get_session_from_handle(h);
  if (!s) return nullptr;
  return build_encoded_features_payload(*s, perspective_player);
}

PyObject* py_session_legal_actions(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  if (!PyArg_ParseTuple(args, "O", &h)) return nullptr;
  Session* s = get_session_from_handle(h);
  if (!s) return nullptr;
  const auto legal = s->rules.legal_actions(s->state);
  PyObject* arr = PyList_New(0);
  for (ActionId a : legal) {
    PyObject* item = PyDict_New();
    PyDict_SetItemString(item, "action_id", PyLong_FromLong(a));
    PyList_Append(arr, item);
    Py_DECREF(item);
  }
  return arr;
}

PyObject* py_session_apply_action(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  int action_id = 0;
  int forced = 0;
  if (!PyArg_ParseTuple(args, "Oii", &h, &action_id, &forced)) return nullptr;
  Session* s = get_session_from_handle(h);
  if (!s) return nullptr;
  return apply_action_common(*s, action_id, forced != 0);
}

PyObject* py_session_ai_move(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  const char* engine = "netmcts";
  int simulations = 200;
  double temperature = 0.0;
  int time_budget_ms = 0;
  const char* model_path = nullptr;
  PyObject* search_options = nullptr;
  if (!PyArg_ParseTuple(args, "Osidi|zO", &h, &engine, &simulations, &temperature, &time_budget_ms, &model_path,
                        &search_options)) {
    return nullptr;
  }
  SplendorSearchOptions parsed_options{};
  if (!parse_splendor_search_options(search_options, &parsed_options)) return nullptr;
  Session* s = get_session_from_handle(h);
  if (!s) return nullptr;
  (void)time_budget_ms;
  const AiDecision decision = choose_action_for_state(
      s->state, s->rules, std::string(engine), simulations, temperature, 0.0, 0.0, model_path, parsed_options, 0);
  if (decision.action < 0) {
    if (std::string(engine) == "netmcts") {
      PyErr_SetString(
          PyExc_RuntimeError,
          "netmcts requested but no valid action produced (model path missing/invalid or evaluator unavailable)");
      return nullptr;
    }
    Py_RETURN_NONE;
  }
  PyObject* out = apply_action_common(*s, decision.action, false);
  if (!out) return nullptr;
  if (decision.has_mcts_best_action_value) {
    PyDict_SetItemString(out, "best_action_value", PyFloat_FromDouble(decision.mcts_best_action_value));
  } else {
    Py_INCREF(Py_None);
    PyDict_SetItemString(out, "best_action_value", Py_None);
  }
  PyDict_SetItemString(out, "tail_solved", decision.tail_solved ? Py_True : Py_False);
  return out;
}

PyObject* py_run_selfplay_episode_fast(PyObject*, PyObject* args) {
  long long seed = 0;
  const char* engine = "netmcts";
  int simulations = 200;
  double temperature = 0.0;
  const char* model_path = nullptr;
  double dirichlet_alpha = 0.0;
  double dirichlet_epsilon = 0.0;
  int dirichlet_on_first_n_plies = 0;
  PyObject* search_options = nullptr;
  if (!PyArg_ParseTuple(
          args,
          "Lsid|zddiO",
          &seed,
          &engine,
          &simulations,
          &temperature,
          &model_path,
          &dirichlet_alpha,
          &dirichlet_epsilon,
          &dirichlet_on_first_n_plies,
          &search_options
      )) {
    return nullptr;
  }
  SplendorSearchOptions parsed_options{};
  if (!parse_splendor_search_options(search_options, &parsed_options)) return nullptr;
  parsed_options.common.dirichlet_alpha = static_cast<float>(std::max(0.0, dirichlet_alpha));
  parsed_options.common.dirichlet_epsilon = static_cast<float>(std::max(0.0, std::min(1.0, dirichlet_epsilon)));
  parsed_options.common.dirichlet_on_first_n_plies = std::max(0, dirichlet_on_first_n_plies);
  SplendorState state;
  SplendorRules rules;
  state.reset_with_seed(static_cast<std::uint64_t>(seed));

  int ply = 0;
  PyObject* samples = PyList_New(0);
  while (!state.persistent.data().terminal && ply < std::max(1, parsed_options.common.max_episode_plies)) {
    const double runtime_temperature =
        board_ai::search::resolve_linear_temperature(parsed_options.common.temperature_schedule, temperature, ply);
    const bool allow_noise =
        state.persistent.data().plies < std::max(0, parsed_options.common.dirichlet_on_first_n_plies);
    const AiDecision d = choose_action_for_state(
        state,
        rules,
        std::string(engine),
        simulations,
        runtime_temperature,
        allow_noise ? parsed_options.common.dirichlet_alpha : 0.0,
        allow_noise ? parsed_options.common.dirichlet_epsilon : 0.0,
        model_path,
        parsed_options,
        ply
    );
    if (d.action < 0) {
      if (std::string(engine) == "netmcts") {
        PyErr_SetString(
            PyExc_RuntimeError,
            "netmcts selfplay failed: no valid action produced (model path missing/invalid or evaluator unavailable)");
        return nullptr;
      }
      break;
    }
    const int actor = state.persistent.data().current_player;

    std::vector<float> features;
    std::vector<float> legal_mask;
    board_ai::splendor::SplendorFeatureEncoder enc;
    const auto legal = rules.legal_actions(state);
    enc.encode(state, actor, legal, &features, &legal_mask);

    rules.do_action_fast(state, d.action);
    ply += 1;

    PyObject* rec = PyDict_New();
    PyDict_SetItemString(rec, "ply", PyLong_FromLong(ply));
    PyDict_SetItemString(rec, "player", PyLong_FromLong(actor));
    PyDict_SetItemString(rec, "action_id", PyLong_FromLong(d.action));
    PyDict_SetItemString(rec, "state_version", PyLong_FromLong(ply - 1));

    PyObject* feat_list = PyList_New(static_cast<Py_ssize_t>(features.size()));
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(features.size()); ++i) {
      PyList_SET_ITEM(feat_list, i, PyFloat_FromDouble(features[static_cast<size_t>(i)]));
    }
    PyDict_SetItemString(rec, "features", feat_list);
    Py_DECREF(feat_list);

    PyObject* ids = PyList_New(0);
    PyObject* probs = PyList_New(0);
    if (!d.root_actions.empty() && d.root_actions.size() == d.root_visits.size()) {
      auto p = visit_probs(d.root_actions, d.root_visits);
      for (size_t i = 0; i < d.root_actions.size(); ++i) {
        PyList_Append(ids, PyLong_FromLong(d.root_actions[i]));
        PyList_Append(probs, PyFloat_FromDouble(p[i]));
      }
    } else {
      PyList_Append(ids, PyLong_FromLong(d.action));
      PyList_Append(probs, PyFloat_FromDouble(1.0));
    }
    PyDict_SetItemString(rec, "policy_action_ids", ids);
    PyDict_SetItemString(rec, "policy_probs", probs);
    Py_DECREF(ids);
    Py_DECREF(probs);

    PyList_Append(samples, rec);
    Py_DECREF(rec);
  }

  const auto& end = state.persistent.data();
  PyObject* out = PyDict_New();
  PyDict_SetItemString(out, "samples", samples);
  Py_DECREF(samples);
  PyDict_SetItemString(out, "plies", PyLong_FromLong(ply));
  if (end.winner < 0) {
    Py_INCREF(Py_None);
    PyDict_SetItemString(out, "winner", Py_None);
  } else {
    PyDict_SetItemString(out, "winner", PyLong_FromLong(end.winner));
  }
  PyDict_SetItemString(out, "shared_victory", end.shared_victory ? Py_True : Py_False);
  PyObject* scores = PyList_New(2);
  PyList_SET_ITEM(scores, 0, PyLong_FromLong(end.player_points[0]));
  PyList_SET_ITEM(scores, 1, PyLong_FromLong(end.player_points[1]));
  PyDict_SetItemString(out, "scores", scores);
  Py_DECREF(scores);
  if (!end.terminal && ply >= std::max(1, parsed_options.common.max_episode_plies)) {
    PyDict_SetItemString(out, "shared_victory", Py_True);
    Py_INCREF(Py_None);
    PyDict_SetItemString(out, "winner", Py_None);
  }
  return out;
}

PyObject* py_run_arena_match_fast(PyObject*, PyObject* args) {
  long long seed = 0;
  const char* p0_engine = "netmcts";
  int p0_simulations = 200;
  double p0_temperature = 0.0;
  const char* p0_model_path = nullptr;
  const char* p1_engine = "netmcts";
  int p1_simulations = 200;
  double p1_temperature = 0.0;
  const char* p1_model_path = nullptr;
  PyObject* p0_search_options = nullptr;
  PyObject* p1_search_options = nullptr;
  if (!PyArg_ParseTuple(args, "Lsidzsidz|OO", &seed, &p0_engine, &p0_simulations, &p0_temperature, &p0_model_path,
                        &p1_engine, &p1_simulations, &p1_temperature, &p1_model_path, &p0_search_options,
                        &p1_search_options)) {
    return nullptr;
  }
  SplendorSearchOptions parsed_p0_options{};
  SplendorSearchOptions parsed_p1_options{};
  if (!parse_splendor_search_options(p0_search_options, &parsed_p0_options)) return nullptr;
  if (!parse_splendor_search_options(p1_search_options, &parsed_p1_options)) return nullptr;
  SplendorState state;
  SplendorRules rules;
  state.reset_with_seed(static_cast<std::uint64_t>(seed));
  int cursor = 0;
  const int max_episode_plies = std::max(
      1,
      std::min(parsed_p0_options.common.max_episode_plies, parsed_p1_options.common.max_episode_plies));
  while (!state.persistent.data().terminal && cursor < max_episode_plies) {
    const bool p0_turn = state.persistent.data().current_player == 0;
    const auto& options = p0_turn ? parsed_p0_options : parsed_p1_options;
    const AiDecision d = choose_action_for_state(
        state,
        rules,
        p0_turn ? std::string(p0_engine) : std::string(p1_engine),
        p0_turn ? p0_simulations : p1_simulations,
        p0_turn ? p0_temperature : p1_temperature,
        0.0,
        0.0,
        p0_turn ? p0_model_path : p1_model_path,
        options,
        cursor
    );
    if (d.action < 0) {
      if ((p0_turn ? std::string(p0_engine) : std::string(p1_engine)) == "netmcts") {
        PyErr_SetString(
            PyExc_RuntimeError,
            "netmcts arena failed: no valid action produced (model path missing/invalid or evaluator unavailable)");
        return nullptr;
      }
      break;
    }
    rules.do_action_fast(state, d.action);
    cursor += 1;
  }
  const auto& end = state.persistent.data();
  PyObject* out = PyDict_New();
  if (end.winner < 0) {
    Py_INCREF(Py_None);
    PyDict_SetItemString(out, "winner", Py_None);
  } else {
    PyDict_SetItemString(out, "winner", PyLong_FromLong(end.winner));
  }
  PyDict_SetItemString(out, "shared_victory", end.shared_victory ? Py_True : Py_False);
  PyObject* scores = PyList_New(2);
  PyList_SET_ITEM(scores, 0, PyLong_FromLong(end.player_points[0]));
  PyList_SET_ITEM(scores, 1, PyLong_FromLong(end.player_points[1]));
  PyDict_SetItemString(out, "scores", scores);
  Py_DECREF(scores);
  return out;
}

PyObject* py_onnx_enabled(PyObject*, PyObject*) {
#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

PyObject* py_feature_dim(PyObject*, PyObject*) { return PyLong_FromLong(kFeatureDim); }

PyObject* py_action_space(PyObject*, PyObject*) { return PyLong_FromLong(kActionSpace); }

#include "cpp_splendor_engine_replay.inc"

PyMethodDef kMethods[] = {
    {"session_new", py_session_new, METH_VARARGS, "Create session"},
    {"session_delete", py_session_delete, METH_VARARGS, "Delete session"},
    {"session_payload", py_session_payload, METH_VARARGS, "Get state payload"},
    {"session_encode_features", py_session_encode_features, METH_VARARGS, "Get encoded features for perspective"},
    {"session_legal_actions", py_session_legal_actions, METH_VARARGS, "Get legal actions"},
    {"session_apply_action", py_session_apply_action, METH_VARARGS, "Apply action"},
    {"session_ai_move", py_session_ai_move, METH_VARARGS, "Apply AI move"},
    {"session_step_back", py_session_step_back, METH_VARARGS, "Step back"},
    {"session_can_step_back", py_session_can_step_back, METH_VARARGS, "Can step back"},
    {"session_seek", py_session_seek, METH_VARARGS, "Seek ply"},
    {"session_replay_events", py_session_replay_events, METH_VARARGS, "Replay events"},
    {"session_frames_range", py_session_frames_range, METH_VARARGS, "Frames range"},
    {"session_frame_at", py_session_frame_at, METH_VARARGS, "Frame at"},
    {"session_rebuild_frames", py_session_rebuild_frames, METH_VARARGS, "Rebuild frames"},
    {"onnx_enabled", py_onnx_enabled, METH_NOARGS, "Return whether ONNX runtime is enabled in this build"},
    {"feature_dim", py_feature_dim, METH_NOARGS, "Return encoded feature dimension"},
    {"action_space", py_action_space, METH_NOARGS, "Return action space size"},
    {"run_selfplay_episode_fast", py_run_selfplay_episode_fast, METH_VARARGS, "Run one full self-play episode in C++"},
    {"run_arena_match_fast", py_run_arena_match_fast, METH_VARARGS, "Run one full arena match in C++"},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModuleDef = {
    PyModuleDef_HEAD_INIT,
    "cpp_splendor_engine_v1",
    "DinoBoard C++ Splendor engine",
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit_cpp_splendor_engine_v1(void) { return PyModule_Create(&kModuleDef); }

