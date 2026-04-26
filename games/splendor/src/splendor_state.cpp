#include "splendor_state.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>

#include "splendor_rules.h"

namespace board_ai::splendor {

namespace {

std::uint64_t splitmix64(std::uint64_t& x) {
  x += 0x9e3779b97f4a7c15ULL;
  std::uint64_t z = x;
  z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31U);
}

template <typename T>
T take_random_from_vector(std::vector<T>& deck, std::uint64_t& nonce) {
  if (deck.empty()) return T{};
  const std::uint64_t r = splitmix64(nonce);
  const size_t idx = static_cast<size_t>(r % static_cast<std::uint64_t>(deck.size()));
  const T picked = deck[idx];
  if (idx + 1 < deck.size()) {
    deck[idx] = deck.back();
  }
  deck.pop_back();
  return picked;
}

std::vector<SplendorCard> build_card_pool() {
  // Official 90 development cards (from Splendor base game).
  struct CardDef {
    std::int8_t tier;
    std::int8_t bonus;
    std::int8_t points;
    std::array<std::int8_t, 5> cost;
  };
  static const std::array<CardDef, 90> defs{{
      {1, 1, 0, {0, 0, 0, 0, 3}}, {1, 1, 0, {1, 0, 0, 0, 2}}, {1, 1, 0, {0, 0, 2, 0, 2}},
      {1, 1, 0, {1, 0, 2, 2, 0}}, {1, 1, 0, {0, 1, 3, 1, 0}}, {1, 1, 0, {1, 0, 1, 1, 1}},
      {1, 1, 0, {1, 0, 1, 2, 1}}, {1, 1, 1, {0, 0, 0, 4, 0}}, {1, 3, 0, {3, 0, 0, 0, 0}},
      {1, 3, 0, {0, 2, 1, 0, 0}}, {1, 3, 0, {2, 0, 0, 2, 0}}, {1, 3, 0, {2, 0, 1, 0, 2}},
      {1, 3, 0, {1, 0, 0, 1, 3}}, {1, 3, 0, {1, 1, 1, 0, 1}}, {1, 3, 0, {2, 1, 1, 0, 1}},
      {1, 3, 1, {4, 0, 0, 0, 0}}, {1, 4, 0, {0, 0, 3, 0, 0}}, {1, 4, 0, {0, 0, 2, 1, 0}},
      {1, 4, 0, {2, 0, 2, 0, 0}}, {1, 4, 0, {2, 2, 0, 1, 0}}, {1, 4, 0, {0, 0, 1, 3, 1}},
      {1, 4, 0, {1, 1, 1, 1, 0}}, {1, 4, 0, {1, 2, 1, 1, 0}}, {1, 4, 1, {0, 4, 0, 0, 0}},
      {1, 0, 0, {0, 3, 0, 0, 0}}, {1, 0, 0, {0, 0, 0, 2, 1}}, {1, 0, 0, {0, 2, 0, 0, 2}},
      {1, 0, 0, {0, 2, 2, 0, 1}}, {1, 0, 0, {3, 1, 0, 0, 1}}, {1, 0, 0, {0, 1, 1, 1, 1}},
      {1, 0, 0, {0, 1, 2, 1, 1}}, {1, 0, 1, {0, 0, 4, 0, 0}}, {1, 2, 0, {0, 0, 0, 3, 0}},
      {1, 2, 0, {2, 1, 0, 0, 0}}, {1, 2, 0, {0, 2, 0, 2, 0}}, {1, 2, 0, {0, 1, 0, 2, 2}},
      {1, 2, 0, {1, 3, 1, 0, 0}}, {1, 2, 0, {1, 1, 0, 1, 1}}, {1, 2, 0, {1, 1, 0, 1, 2}},
      {1, 2, 1, {0, 0, 0, 0, 4}}, {2, 1, 1, {0, 2, 2, 3, 0}}, {2, 1, 1, {0, 2, 3, 0, 3}},
      {2, 1, 2, {0, 5, 0, 0, 0}}, {2, 1, 2, {5, 3, 0, 0, 0}}, {2, 1, 2, {2, 0, 0, 1, 4}},
      {2, 1, 3, {0, 6, 0, 0, 0}}, {2, 3, 1, {2, 0, 0, 2, 3}}, {2, 3, 1, {0, 3, 0, 2, 3}},
      {2, 3, 2, {0, 0, 0, 0, 5}}, {2, 3, 2, {3, 0, 0, 0, 5}}, {2, 3, 2, {1, 4, 2, 0, 0}},
      {2, 3, 3, {0, 0, 0, 6, 0}}, {2, 4, 1, {3, 2, 2, 0, 0}}, {2, 4, 1, {3, 0, 3, 0, 2}},
      {2, 4, 2, {5, 0, 0, 0, 0}}, {2, 4, 2, {0, 0, 5, 3, 0}}, {2, 4, 2, {0, 1, 4, 2, 0}},
      {2, 4, 3, {0, 0, 0, 0, 6}}, {2, 0, 1, {0, 0, 3, 2, 2}}, {2, 0, 1, {2, 3, 0, 3, 0}},
      {2, 0, 2, {0, 0, 0, 5, 0}}, {2, 0, 2, {0, 0, 0, 5, 3}}, {2, 0, 2, {0, 0, 1, 4, 2}},
      {2, 0, 3, {6, 0, 0, 0, 0}}, {2, 2, 1, {2, 3, 0, 0, 2}}, {2, 2, 1, {3, 0, 2, 3, 0}},
      {2, 2, 2, {0, 0, 5, 0, 0}}, {2, 2, 2, {0, 5, 3, 0, 0}}, {2, 2, 2, {4, 2, 0, 0, 1}},
      {2, 2, 3, {0, 0, 6, 0, 0}}, {3, 1, 3, {3, 0, 3, 3, 5}}, {3, 1, 4, {7, 0, 0, 0, 0}},
      {3, 1, 4, {6, 3, 0, 0, 3}}, {3, 1, 5, {7, 3, 0, 0, 0}}, {3, 3, 3, {3, 5, 3, 0, 3}},
      {3, 3, 4, {0, 0, 7, 0, 0}}, {3, 3, 4, {0, 3, 6, 3, 0}}, {3, 3, 5, {0, 0, 7, 3, 0}},
      {3, 4, 3, {3, 3, 5, 3, 0}}, {3, 4, 4, {0, 0, 0, 7, 0}}, {3, 4, 4, {0, 0, 3, 6, 3}},
      {3, 4, 5, {0, 0, 0, 7, 3}}, {3, 0, 3, {0, 3, 3, 5, 3}}, {3, 0, 4, {0, 0, 0, 0, 7}},
      {3, 0, 4, {3, 0, 0, 3, 6}}, {3, 0, 5, {3, 0, 0, 0, 7}}, {3, 2, 3, {5, 3, 0, 3, 3}},
      {3, 2, 4, {0, 7, 0, 0, 0}}, {3, 2, 4, {3, 6, 3, 0, 0}}, {3, 2, 5, {0, 7, 3, 0, 0}},
  }};
  std::vector<SplendorCard> cards;
  cards.reserve(defs.size());
  for (const auto& def : defs) {
    SplendorCard c;
    c.tier = def.tier;
    c.bonus = def.bonus;
    c.points = def.points;
    for (int i = 0; i < kColorCount; ++i) {
      c.cost[static_cast<size_t>(i)] = def.cost[static_cast<size_t>(i)];
    }
    cards.push_back(c);
  }
  return cards;
}

}  // namespace

const std::vector<SplendorCard>& splendor_card_pool() {
  static const std::vector<SplendorCard> cards = build_card_pool();
  return cards;
}

const std::array<std::array<std::int8_t, kColorCount>, 12>& splendor_nobles() {
  static const std::array<std::array<std::int8_t, kColorCount>, 12> nobles{{
      {{0, 0, 4, 4, 0}},
      {{0, 0, 0, 4, 4}},
      {{0, 4, 4, 0, 0}},
      {{4, 0, 0, 0, 4}},
      {{4, 4, 0, 0, 0}},
      {{3, 0, 0, 3, 3}},
      {{3, 3, 3, 0, 0}},
      {{0, 0, 3, 3, 3}},
      {{0, 3, 3, 3, 0}},
      {{3, 3, 0, 0, 3}},
      {{4, 0, 0, 4, 0}},
      {{0, 3, 3, 0, 3}},
  }};
  return nobles;
}

SplendorPersistentState::SplendorPersistentState(std::shared_ptr<const SplendorPersistentNode> node) : node_(std::move(node)) {}

SplendorPersistentState SplendorPersistentState::root_from_seed(std::uint64_t seed) {
  auto data = std::make_shared<SplendorData>();
  data->current_player = 0;
  data->plies = 0;
  data->final_round_remaining = -1;
  data->stage = static_cast<std::int8_t>(SplendorTurnStage::kNormal);
  data->pending_returns = 0;
  data->pending_noble_slots = {{-1, -1, -1}};
  data->pending_nobles_size = 0;
  data->draw_nonce = seed == 0 ? 0x9e3779b97f4a7c15ULL : seed;
  data->winner = -1;
  data->terminal = false;
  data->shared_victory = false;
  data->scores = {0, 0};
  data->bank = {4, 4, 4, 4, 4, 5};
  for (int p = 0; p < kPlayers; ++p) {
    data->player_gems[p].fill(0);
    data->player_bonuses[p].fill(0);
    data->player_points[p] = 0;
    data->player_cards_count[p] = 0;
    data->player_nobles_count[p] = 0;
    data->reserved[p] = {{-1, -1, -1}};
    data->reserved_visible[p] = {{0, 0, 0}};
    data->reserved_size[p] = 0;
  }
  for (int t = 0; t < 3; ++t) {
    data->tableau[t] = {{-1, -1, -1, -1}};
    data->tableau_size[t] = 0;
  }

  const auto& cards = splendor_card_pool();
  for (int id = 0; id < static_cast<int>(cards.size()); ++id) {
    const int tier = static_cast<int>(cards[static_cast<size_t>(id)].tier);
    if (tier >= 1 && tier <= 3) {
      data->decks[static_cast<size_t>(tier - 1)].push_back(static_cast<std::int16_t>(id));
    }
  }
  for (int t = 0; t < 3; ++t) {
    auto& deck = data->decks[static_cast<size_t>(t)];
    for (int k = 0; k < 4 && !deck.empty(); ++k) {
      data->tableau[static_cast<size_t>(t)][static_cast<size_t>(k)] = take_random_from_vector(deck, data->draw_nonce);
      data->tableau_size[static_cast<size_t>(t)] += 1;
    }
  }

  std::vector<int> noble_ids(12);
  for (int i = 0; i < 12; ++i) noble_ids[static_cast<size_t>(i)] = i;
  data->nobles_size = 3;
  for (int i = 0; i < 3; ++i) {
    data->nobles[static_cast<size_t>(i)] =
        static_cast<std::int16_t>(take_random_from_vector(noble_ids, data->draw_nonce));
  }

  auto node = std::make_shared<SplendorPersistentNode>();
  node->action_from_parent = -1;
  node->materialized = std::move(data);
  return SplendorPersistentState(std::move(node));
}

const SplendorData& SplendorPersistentState::data() const {
  if (!node_) {
    throw std::runtime_error("SplendorPersistentState is not initialized");
  }
  if (node_->materialized) {
    return *node_->materialized;
  }
  if (!node_->parent) {
    throw std::runtime_error("SplendorPersistentState root is missing materialized data");
  }
  const SplendorData& parent_data = SplendorPersistentState(node_->parent).data();
  auto next = std::make_shared<SplendorData>(SplendorRules::apply_action_copy(parent_data, node_->action_from_parent));
  node_->materialized = next;
  return *node_->materialized;
}

SplendorPersistentState SplendorPersistentState::advance(ActionId action) const {
  auto node = std::make_shared<SplendorPersistentNode>();
  node->parent = node_;
  node->action_from_parent = action;
  return SplendorPersistentState(std::move(node));
}

StateHash64 SplendorPersistentState::state_hash(bool include_hidden_rng, std::uint64_t rng_salt) const {
  const SplendorData& d = data();
  std::size_t h = 0;
  auto mix = [&h](std::size_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
  };
  mix(static_cast<std::size_t>(d.current_player + 3));
  mix(static_cast<std::size_t>(d.plies + 17));
  mix(static_cast<std::size_t>(d.final_round_remaining + 9));
  mix(static_cast<std::size_t>(d.stage + 21));
  mix(static_cast<std::size_t>(d.pending_returns + 25));
  mix(static_cast<std::size_t>(d.pending_nobles_size + 27));
  for (auto slot : d.pending_noble_slots) mix(static_cast<std::size_t>(slot + 29));
  mix(static_cast<std::size_t>(d.winner + 11));
  mix(static_cast<std::size_t>(d.terminal ? 1 : 0));
  if (include_hidden_rng) {
    mix(static_cast<std::size_t>(d.draw_nonce));
  }
  for (int v : d.scores) mix(static_cast<std::size_t>(v + 101));
  for (auto v : d.bank) mix(static_cast<std::size_t>(v + 7));
  const int actor = d.current_player;
  for (int p = 0; p < kPlayers; ++p) {
    for (auto v : d.player_gems[p]) mix(static_cast<std::size_t>(v + 13));
    for (auto v : d.player_bonuses[p]) mix(static_cast<std::size_t>(v + 19));
    mix(static_cast<std::size_t>(d.player_points[p] + 23));
    mix(static_cast<std::size_t>(d.player_cards_count[p] + 29));
    mix(static_cast<std::size_t>(d.player_nobles_count[p] + 31));
    mix(static_cast<std::size_t>(d.reserved_size[p] + 37));
    for (int i = 0; i < 3; ++i) {
      const std::int16_t cid = d.reserved[p][static_cast<size_t>(i)];
      const bool visible_to_opponent = d.reserved_visible[p][static_cast<size_t>(i)] != 0;
      // include_hidden_rng=false is used for chance grouping; avoid mixing hidden info
      // that the current actor cannot observe.
      if (include_hidden_rng || p == actor || visible_to_opponent) {
        mix(static_cast<std::size_t>(cid + 41));
      } else {
        mix(static_cast<std::size_t>(-1 + 41));
      }
      mix(static_cast<std::size_t>((visible_to_opponent ? 1 : 0) + 43));
    }
  }
  for (int t = 0; t < 3; ++t) {
    mix(static_cast<std::size_t>(d.tableau_size[t] + 47));
    for (auto cid : d.tableau[t]) mix(static_cast<std::size_t>(cid + 53));
    mix(static_cast<std::size_t>(d.decks[t].size() + 59));
    if (include_hidden_rng) {
      const int tail = std::min<int>(3, static_cast<int>(d.decks[t].size()));
      for (int i = 0; i < tail; ++i) {
        const auto cid = d.decks[t][d.decks[t].size() - 1U - static_cast<size_t>(i)];
        mix(static_cast<std::size_t>(cid + 61 + i));
      }
    }
  }
  mix(static_cast<std::size_t>(d.nobles_size + 67));
  for (int i = 0; i < 3; ++i) mix(static_cast<std::size_t>(d.nobles[static_cast<size_t>(i)] + 71));
  if (include_hidden_rng) {
    mix(static_cast<std::size_t>(rng_salt));
  }
  return static_cast<StateHash64>(h);
}

SplendorState::SplendorState() { reset_with_seed(0xC0FFEEu); }

void SplendorState::reset_with_seed(std::uint64_t seed) {
  rng_salt = seed == 0 ? 0x9e3779b97f4a7c15ULL : seed;
  persistent = SplendorPersistentState::root_from_seed(rng_salt);
  undo_stack.clear();
}

std::unique_ptr<IGameState> SplendorState::clone_state() const {
  return std::make_unique<SplendorState>(*this);
}

StateHash64 SplendorState::state_hash(bool include_hidden_rng) const {
  return persistent.state_hash(include_hidden_rng, rng_salt);
}

}  // namespace board_ai::splendor

