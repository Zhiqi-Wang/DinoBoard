#include "splendor_rules.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace board_ai::splendor {

namespace {

std::uint64_t splitmix64(std::uint64_t& x) {
  x += 0x9e3779b97f4a7c15ULL;
  std::uint64_t z = x;
  z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31U);
}

std::int16_t draw_random_from_deck(SplendorData& d, int tier_index) {
  auto& deck = d.decks[static_cast<size_t>(tier_index)];
  if (deck.empty()) return -1;
  const std::uint64_t r = splitmix64(d.draw_nonce);
  const size_t idx = static_cast<size_t>(r % static_cast<std::uint64_t>(deck.size()));
  const std::int16_t picked = deck[idx];
  if (idx + 1 < deck.size()) {
    deck[idx] = deck.back();
  }
  deck.pop_back();
  return picked;
}

constexpr std::array<std::array<int, 3>, 10> kTakeThreeCombos{{
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

constexpr std::array<std::array<int, 2>, 10> kTakeTwoDifferentCombos{{
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

constexpr std::array<int, 5> kTakeOneColors{{0, 1, 2, 3, 4}};

SplendorTurnStage stage_of(const SplendorData& d) {
  return static_cast<SplendorTurnStage>(d.stage);
}

void set_stage(SplendorData& d, SplendorTurnStage stage) {
  d.stage = static_cast<std::int8_t>(stage);
}

void clear_pending_nobles(SplendorData& d) {
  d.pending_noble_slots = {{-1, -1, -1}};
  d.pending_nobles_size = 0;
}

bool can_afford(const SplendorData& d, int player, const SplendorCard& card) {
  int need_gold = 0;
  for (int c = 0; c < kColorCount; ++c) {
    const int remaining =
        std::max(0, static_cast<int>(card.cost[static_cast<size_t>(c)]) - static_cast<int>(d.player_bonuses[player][c]) -
                        static_cast<int>(d.player_gems[player][c]));
    need_gold += remaining;
  }
  return need_gold <= static_cast<int>(d.player_gems[player][5]);
}

bool noble_requirements_met(const SplendorData& d, int player, int noble_slot) {
  if (noble_slot < 0 || noble_slot >= d.nobles_size) return false;
  const auto& nobles = splendor_nobles();
  const int nid = d.nobles[static_cast<size_t>(noble_slot)];
  if (nid < 0 || nid >= static_cast<int>(nobles.size())) return false;
  for (int c = 0; c < kColorCount; ++c) {
    if (d.player_bonuses[player][c] < nobles[static_cast<size_t>(nid)][static_cast<size_t>(c)]) {
      return false;
    }
  }
  return true;
}

bool pending_noble_slot_enabled(const SplendorData& d, int noble_slot) {
  for (int i = 0; i < d.pending_nobles_size; ++i) {
    if (d.pending_noble_slots[static_cast<size_t>(i)] == noble_slot) return true;
  }
  return false;
}

bool populate_pending_nobles(SplendorData& d, int player) {
  clear_pending_nobles(d);
  for (int slot = 0; slot < d.nobles_size && slot < 3; ++slot) {
    if (!noble_requirements_met(d, player, slot)) continue;
    d.pending_noble_slots[static_cast<size_t>(d.pending_nobles_size)] = static_cast<std::int8_t>(slot);
    d.pending_nobles_size += 1;
  }
  if (d.pending_nobles_size > 0) {
    set_stage(d, SplendorTurnStage::kChooseNoble);
    return true;
  }
  return false;
}

void claim_noble_at_slot(SplendorData& d, int player, int noble_slot) {
  if (noble_slot < 0 || noble_slot >= d.nobles_size) return;
  if (!noble_requirements_met(d, player, noble_slot)) return;
  for (int j = noble_slot + 1; j < d.nobles_size; ++j) {
    d.nobles[static_cast<size_t>(j - 1)] = d.nobles[static_cast<size_t>(j)];
  }
  d.nobles[static_cast<size_t>(d.nobles_size - 1)] = -1;
  d.nobles_size -= 1;
  d.player_points[player] += 3;
  d.player_nobles_count[player] += 1;
  clear_pending_nobles(d);
}

int total_tokens_of_player(const SplendorData& d, int player) {
  int total = 0;
  for (int i = 0; i < kTokenTypes; ++i) total += d.player_gems[player][static_cast<size_t>(i)];
  return total;
}

void update_pending_returns_state(SplendorData& d, int player) {
  d.pending_returns = std::max(0, total_tokens_of_player(d, player) - 10);
  if (d.pending_returns > 0) {
    clear_pending_nobles(d);
    set_stage(d, SplendorTurnStage::kReturnTokens);
  } else if (stage_of(d) == SplendorTurnStage::kReturnTokens) {
    set_stage(d, SplendorTurnStage::kNormal);
  }
}

void trim_tokens_to_limit(SplendorData& d, int player) {
  while (true) {
    int total = 0;
    for (int i = 0; i < kTokenTypes; ++i) total += d.player_gems[player][i];
    if (total <= 10) break;
    int pick = 0;
    int best = -1;
    for (int i = 0; i < kTokenTypes; ++i) {
      const int v = d.player_gems[player][i];
      if (v > best) {
        best = v;
        pick = i;
      }
    }
    if (best <= 0) break;
    d.player_gems[player][pick] -= 1;
    d.bank[static_cast<size_t>(pick)] += 1;
  }
}

void draw_tableau(SplendorData& d, int tier_index) {
  while (d.tableau_size[static_cast<size_t>(tier_index)] < 4 && !d.decks[static_cast<size_t>(tier_index)].empty()) {
    const auto id = draw_random_from_deck(d, tier_index);
    d.tableau[static_cast<size_t>(tier_index)][static_cast<size_t>(d.tableau_size[static_cast<size_t>(tier_index)])] = id;
    d.tableau_size[static_cast<size_t>(tier_index)] += 1;
  }
}

void erase_tableau_slot(SplendorData& d, int tier_index, int slot) {
  auto& row = d.tableau[static_cast<size_t>(tier_index)];
  const int n = d.tableau_size[static_cast<size_t>(tier_index)];
  for (int i = slot + 1; i < n; ++i) {
    row[static_cast<size_t>(i - 1)] = row[static_cast<size_t>(i)];
  }
  row[static_cast<size_t>(n - 1)] = -1;
  d.tableau_size[static_cast<size_t>(tier_index)] -= 1;
}

void remove_reserved_at(SplendorData& d, int player, int idx) {
  auto& r = d.reserved[static_cast<size_t>(player)];
  auto& v = d.reserved_visible[static_cast<size_t>(player)];
  const int n = d.reserved_size[static_cast<size_t>(player)];
  for (int i = idx + 1; i < n; ++i) {
    r[static_cast<size_t>(i - 1)] = r[static_cast<size_t>(i)];
    v[static_cast<size_t>(i - 1)] = v[static_cast<size_t>(i)];
  }
  r[static_cast<size_t>(n - 1)] = -1;
  v[static_cast<size_t>(n - 1)] = 0;
  d.reserved_size[static_cast<size_t>(player)] -= 1;
}

void pay_for_card(SplendorData& d, int player, const SplendorCard& card) {
  for (int c = 0; c < kColorCount; ++c) {
    int need = std::max(0, static_cast<int>(card.cost[static_cast<size_t>(c)]) - static_cast<int>(d.player_bonuses[player][c]));
    const int use_color = std::min(need, static_cast<int>(d.player_gems[player][c]));
    d.player_gems[player][c] -= static_cast<std::int8_t>(use_color);
    d.bank[static_cast<size_t>(c)] += static_cast<std::int8_t>(use_color);
    need -= use_color;
    if (need > 0) {
      d.player_gems[player][5] -= static_cast<std::int8_t>(need);
      d.bank[5] += static_cast<std::int8_t>(need);
    }
  }
}

void update_terminal(SplendorData& d, int actor) {
  if (d.final_round_remaining < 0 && d.player_points[actor] >= kTargetPoints) {
    // Splendor ends at the end of the round in which 15+ is reached.
    // With fixed turn order (player 0 then player 1), only player 0
    // reaching the target grants one final turn to player 1.
    d.final_round_remaining = (actor == 0) ? 1 : 0;
  } else if (d.final_round_remaining >= 0) {
    d.final_round_remaining -= 1;
  }
  if (d.plies >= kMaxPlies || d.final_round_remaining == 0) {
    d.terminal = true;
    int winner = -1;
    if (d.player_points[0] > d.player_points[1]) {
      winner = 0;
    } else if (d.player_points[1] > d.player_points[0]) {
      winner = 1;
    } else if (d.player_cards_count[0] < d.player_cards_count[1]) {
      winner = 0;
    } else if (d.player_cards_count[1] < d.player_cards_count[0]) {
      winner = 1;
    }
    d.winner = winner;
    d.shared_victory = winner < 0;
    if (winner < 0) {
      d.scores = {0, 0};
    } else {
      d.scores = {winner == 0 ? 1 : -1, winner == 1 ? 1 : -1};
    }
  } else {
    d.terminal = false;
    d.winner = -1;
    d.shared_victory = false;
    d.scores = {0, 0};
  }
}

void finalize_turn(SplendorData& d, int actor) {
  set_stage(d, SplendorTurnStage::kNormal);
  d.pending_returns = 0;
  clear_pending_nobles(d);
  update_terminal(d, actor);
  d.current_player = 1 - actor;
}

}  // namespace

const SplendorState* SplendorRules::as_state(const IGameState& state) {
  auto p = dynamic_cast<const SplendorState*>(&state);
  if (!p) {
    throw std::invalid_argument("SplendorRules expects SplendorState");
  }
  return p;
}

SplendorState* SplendorRules::as_state(IGameState& state) {
  auto p = dynamic_cast<SplendorState*>(&state);
  if (!p) {
    throw std::invalid_argument("SplendorRules expects SplendorState");
  }
  return p;
}

bool SplendorRules::is_terminal_data(const SplendorData& data) { return data.terminal; }

std::vector<ActionId> SplendorRules::legal_actions_data(const SplendorData& d) {
  std::vector<ActionId> out;
  if (d.terminal) return out;
  const int player = d.current_player;
  if (stage_of(d) == SplendorTurnStage::kReturnTokens || d.pending_returns > 0) {
    for (int c = 0; c < kReturnTokenCount; ++c) {
      if (d.player_gems[player][static_cast<size_t>(c)] > 0) {
        out.push_back(kReturnTokenOffset + c);
      }
    }
    if (out.empty()) out.push_back(kPassAction);
    return out;
  }
  if (stage_of(d) == SplendorTurnStage::kChooseNoble) {
    for (int slot = 0; slot < kChooseNobleCount; ++slot) {
      if (slot < d.nobles_size && pending_noble_slot_enabled(d, slot)) {
        out.push_back(kChooseNobleOffset + slot);
      }
    }
    if (out.empty()) out.push_back(kPassAction);
    return out;
  }
  const auto& cards = splendor_card_pool();

  // Buy faceup.
  for (int tier = 0; tier < 3; ++tier) {
    for (int slot = 0; slot < d.tableau_size[static_cast<size_t>(tier)]; ++slot) {
      const int id = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      if (id < 0 || id >= static_cast<int>(cards.size())) continue;
      if (can_afford(d, player, cards[static_cast<size_t>(id)])) {
        out.push_back(kBuyFaceupOffset + tier * 4 + slot);
      }
    }
  }
  // Buy reserved.
  for (int i = 0; i < d.reserved_size[static_cast<size_t>(player)] && i < kBuyReservedCount; ++i) {
    const int id = d.reserved[static_cast<size_t>(player)][static_cast<size_t>(i)];
    if (id < 0 || id >= static_cast<int>(cards.size())) continue;
    if (can_afford(d, player, cards[static_cast<size_t>(id)])) {
      out.push_back(kBuyReservedOffset + i);
    }
  }
  // Reserve faceup / deck.
  if (d.reserved_size[static_cast<size_t>(player)] < 3) {
    for (int tier = 0; tier < 3; ++tier) {
      for (int slot = 0; slot < d.tableau_size[static_cast<size_t>(tier)] && slot < 4; ++slot) {
        out.push_back(kReserveFaceupOffset + tier * 4 + slot);
      }
    }
    for (int tier = 0; tier < 3; ++tier) {
      if (!d.decks[static_cast<size_t>(tier)].empty()) {
        out.push_back(kReserveDeckOffset + tier);
      }
    }
  }
  int available_token_colors = 0;
  for (int c = 0; c < kColorCount; ++c) {
    if (d.bank[static_cast<size_t>(c)] > 0) {
      available_token_colors += 1;
    }
  }
  // Take three different.
  for (int i = 0; i < kTakeThreeCount; ++i) {
    const auto& comb = kTakeThreeCombos[static_cast<size_t>(i)];
    if (d.bank[static_cast<size_t>(comb[0])] > 0 && d.bank[static_cast<size_t>(comb[1])] > 0 &&
        d.bank[static_cast<size_t>(comb[2])] > 0) {
      out.push_back(kTakeThreeOffset + i);
    }
  }
  if (available_token_colors < 3) {
    // Only allow reduced different-color takes when the bank cannot provide 3 colors.
    for (int i = 0; i < kTakeTwoDifferentCount; ++i) {
      const auto& comb = kTakeTwoDifferentCombos[static_cast<size_t>(i)];
      if (d.bank[static_cast<size_t>(comb[0])] > 0 && d.bank[static_cast<size_t>(comb[1])] > 0) {
        out.push_back(kTakeTwoDifferentOffset + i);
      }
    }
    for (int i = 0; i < kTakeOneCount; ++i) {
      const int c = kTakeOneColors[static_cast<size_t>(i)];
      if (d.bank[static_cast<size_t>(c)] > 0) {
        out.push_back(kTakeOneOffset + i);
      }
    }
  }
  // Take two same.
  for (int c = 0; c < kTakeTwoSameCount; ++c) {
    if (d.bank[static_cast<size_t>(c)] >= 4) out.push_back(kTakeTwoSameOffset + c);
  }
  if (out.empty()) out.push_back(kPassAction);
  return out;
}

SplendorData SplendorRules::apply_action_copy(const SplendorData& src, ActionId action) {
  SplendorData d = src;
  if (d.terminal) return d;
  const auto legal = legal_actions_data(d);
  if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
    action = kPassAction;
  }

  const auto& cards = splendor_card_pool();
  const int player = d.current_player;
  bool bought_card = false;

  if (stage_of(d) == SplendorTurnStage::kReturnTokens || d.pending_returns > 0) {
    if (action >= kReturnTokenOffset && action < kReturnTokenOffset + kReturnTokenCount) {
      const int token = action - kReturnTokenOffset;
      if (d.player_gems[player][static_cast<size_t>(token)] > 0) {
        d.player_gems[player][static_cast<size_t>(token)] -= 1;
        d.bank[static_cast<size_t>(token)] += 1;
        d.pending_returns -= 1;
      }
    } else {
      // fallback: auto-return highest pile to avoid deadlock
      int pick = 0;
      int best = -1;
      for (int c = 0; c < kTokenTypes; ++c) {
        if (d.player_gems[player][static_cast<size_t>(c)] > best) {
          best = d.player_gems[player][static_cast<size_t>(c)];
          pick = c;
        }
      }
      if (best > 0) {
        d.player_gems[player][static_cast<size_t>(pick)] -= 1;
        d.bank[static_cast<size_t>(pick)] += 1;
        d.pending_returns -= 1;
      } else {
        d.pending_returns = 0;
      }
    }
    d.plies += 1;
    if (d.pending_returns <= 0) {
      d.pending_returns = 0;
      set_stage(d, SplendorTurnStage::kNormal);
      finalize_turn(d, player);
    }
    return d;
  }
  if (stage_of(d) == SplendorTurnStage::kChooseNoble) {
    int chosen_slot = -1;
    if (action >= kChooseNobleOffset && action < kChooseNobleOffset + kChooseNobleCount) {
      const int slot = action - kChooseNobleOffset;
      if (pending_noble_slot_enabled(d, slot)) chosen_slot = slot;
    }
    if (chosen_slot < 0 && d.pending_nobles_size > 0) {
      chosen_slot = d.pending_noble_slots[0];
    }
    if (chosen_slot >= 0) {
      claim_noble_at_slot(d, player, chosen_slot);
    } else {
      clear_pending_nobles(d);
    }
    d.plies += 1;
    finalize_turn(d, player);
    return d;
  }

  if (action >= kBuyFaceupOffset && action < kBuyFaceupOffset + kBuyFaceupCount) {
    const int local = action - kBuyFaceupOffset;
    const int tier = local / 4;
    const int slot = local % 4;
    if (slot < d.tableau_size[static_cast<size_t>(tier)]) {
      const int cid = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      erase_tableau_slot(d, tier, slot);
      const auto& card = cards[static_cast<size_t>(cid)];
      pay_for_card(d, player, card);
      d.player_bonuses[player][static_cast<size_t>(card.bonus)] += 1;
      d.player_points[player] += card.points;
      d.player_cards_count[player] += 1;
      bought_card = true;
      draw_tableau(d, tier);
    }
  } else if (action >= kBuyReservedOffset && action < kBuyReservedOffset + kBuyReservedCount) {
    const int idx = action - kBuyReservedOffset;
    if (idx < d.reserved_size[static_cast<size_t>(player)]) {
      const int cid = d.reserved[static_cast<size_t>(player)][static_cast<size_t>(idx)];
      remove_reserved_at(d, player, idx);
      const auto& card = cards[static_cast<size_t>(cid)];
      pay_for_card(d, player, card);
      d.player_bonuses[player][static_cast<size_t>(card.bonus)] += 1;
      d.player_points[player] += card.points;
      d.player_cards_count[player] += 1;
      bought_card = true;
    }
  } else if (action >= kReserveFaceupOffset && action < kReserveFaceupOffset + kReserveFaceupCount) {
    const int local = action - kReserveFaceupOffset;
    const int tier = local / 4;
    const int slot = local % 4;
    if (d.reserved_size[static_cast<size_t>(player)] < 3 && slot < d.tableau_size[static_cast<size_t>(tier)]) {
      const int cid = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      erase_tableau_slot(d, tier, slot);
      const int idx = d.reserved_size[static_cast<size_t>(player)];
      d.reserved[static_cast<size_t>(player)][static_cast<size_t>(idx)] = static_cast<std::int16_t>(cid);
      d.reserved_visible[static_cast<size_t>(player)][static_cast<size_t>(idx)] = 1;
      d.reserved_size[static_cast<size_t>(player)] += 1;
      draw_tableau(d, tier);
      if (d.bank[5] > 0) {
        d.bank[5] -= 1;
        d.player_gems[player][5] += 1;
      }
    }
  } else if (action >= kReserveDeckOffset && action < kReserveDeckOffset + kReserveDeckCount) {
    const int tier = action - kReserveDeckOffset;
    if (d.reserved_size[static_cast<size_t>(player)] < 3 && !d.decks[static_cast<size_t>(tier)].empty()) {
      const int cid = draw_random_from_deck(d, tier);
      const int idx = d.reserved_size[static_cast<size_t>(player)];
      d.reserved[static_cast<size_t>(player)][static_cast<size_t>(idx)] = static_cast<std::int16_t>(cid);
      d.reserved_visible[static_cast<size_t>(player)][static_cast<size_t>(idx)] = 0;
      d.reserved_size[static_cast<size_t>(player)] += 1;
      if (d.bank[5] > 0) {
        d.bank[5] -= 1;
        d.player_gems[player][5] += 1;
      }
    }
  } else if (action >= kTakeThreeOffset && action < kTakeThreeOffset + kTakeThreeCount) {
    const int idx = action - kTakeThreeOffset;
    const auto& comb = kTakeThreeCombos[static_cast<size_t>(idx)];
    if (d.bank[static_cast<size_t>(comb[0])] > 0 && d.bank[static_cast<size_t>(comb[1])] > 0 &&
        d.bank[static_cast<size_t>(comb[2])] > 0) {
      for (int j = 0; j < 3; ++j) {
        const int c = comb[static_cast<size_t>(j)];
        d.bank[static_cast<size_t>(c)] -= 1;
        d.player_gems[player][static_cast<size_t>(c)] += 1;
      }
    }
  } else if (action >= kTakeTwoDifferentOffset && action < kTakeTwoDifferentOffset + kTakeTwoDifferentCount) {
    const int idx = action - kTakeTwoDifferentOffset;
    const auto& comb = kTakeTwoDifferentCombos[static_cast<size_t>(idx)];
    if (d.bank[static_cast<size_t>(comb[0])] > 0 && d.bank[static_cast<size_t>(comb[1])] > 0) {
      for (int j = 0; j < 2; ++j) {
        const int c = comb[static_cast<size_t>(j)];
        d.bank[static_cast<size_t>(c)] -= 1;
        d.player_gems[player][static_cast<size_t>(c)] += 1;
      }
    }
  } else if (action >= kTakeOneOffset && action < kTakeOneOffset + kTakeOneCount) {
    const int c = kTakeOneColors[static_cast<size_t>(action - kTakeOneOffset)];
    if (d.bank[static_cast<size_t>(c)] > 0) {
      d.bank[static_cast<size_t>(c)] -= 1;
      d.player_gems[player][static_cast<size_t>(c)] += 1;
    }
  } else if (action >= kTakeTwoSameOffset && action < kTakeTwoSameOffset + kTakeTwoSameCount) {
    const int c = action - kTakeTwoSameOffset;
    if (d.bank[static_cast<size_t>(c)] >= 4) {
      d.bank[static_cast<size_t>(c)] -= 2;
      d.player_gems[player][static_cast<size_t>(c)] += 2;
    }
  }

  d.plies += 1;
  update_pending_returns_state(d, player);
  if (d.pending_returns > 0) {
    return d;
  }
  if (bought_card && populate_pending_nobles(d, player)) {
    return d;
  }
  finalize_turn(d, player);
  return d;
}

bool SplendorRules::validate_action(const IGameState& state, ActionId action) const {
  const SplendorState* s = as_state(state);
  const auto legal = legal_actions_data(s->persistent.data());
  return std::find(legal.begin(), legal.end(), action) != legal.end();
}

std::vector<ActionId> SplendorRules::legal_actions(const IGameState& state) const {
  const SplendorState* s = as_state(state);
  return legal_actions_data(s->persistent.data());
}

UndoToken SplendorRules::do_action_fast(IGameState& state, ActionId action) const {
  SplendorState* s = as_state(state);
  UndoToken t{};
  t.actor = static_cast<std::uint32_t>(s->persistent.data().current_player);
  t.undo_depth = static_cast<std::uint32_t>(s->undo_stack.size());
  s->undo_stack.push_back(s->persistent);
  if (validate_action(*s, action)) {
    s->persistent = s->persistent.advance(action);
  }
  return t;
}

void SplendorRules::undo_action(IGameState& state, const UndoToken& token) const {
  SplendorState* s = as_state(state);
  if (s->undo_stack.empty()) return;
  s->persistent = s->undo_stack.back();
  s->undo_stack.pop_back();
  (void)token;
}

}  // namespace board_ai::splendor

