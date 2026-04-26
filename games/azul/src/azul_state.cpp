#include "azul_state.h"

#include <algorithm>
#include <functional>

namespace board_ai::azul {

AzulState::AzulState() {
  reset_with_seed(0xC0FFEEu);
}

std::unique_ptr<IGameState> AzulState::clone_state() const {
  return std::make_unique<AzulState>(*this);
}

void AzulState::reset_with_seed(std::uint64_t seed) {
  current_player = 0;
  first_player_next_round = 0;
  winner = -1;
  round_index = 0;
  terminal = false;
  first_player_token_in_center = true;
  shared_victory = false;
  scores = {0, 0};
  factories = {};
  center = {};
  bag.clear();
  box_lid.clear();
  players = {};
  undo_stack.clear();
  persistent_tree_cache.tree.clear();
  persistent_tree_cache.chance_buckets.clear();
  persistent_tree_cache.sig_to_node.clear();
  rng_salt = seed == 0 ? 0x9e3779b97f4a7c15ULL : seed;
  bag.reserve(100);
  for (int c = 0; c < kColors; ++c) {
    for (int i = 0; i < 20; ++i) {
      bag.push_back(static_cast<std::int8_t>(c));
    }
  }
  for (int i = static_cast<int>(bag.size()) - 1; i > 0; --i) {
    const int j = static_cast<int>(next_rand_u32() % static_cast<std::uint32_t>(i + 1));
    std::swap(bag[static_cast<size_t>(i)], bag[static_cast<size_t>(j)]);
  }
  refill_factories_from_rng();
}

bool AzulState::all_sources_empty() const {
  for (const auto& fac : factories) {
    for (std::uint8_t c : fac) {
      if (c > 0) {
        return false;
      }
    }
  }
  for (std::uint8_t c : center) {
    if (c > 0) {
      return false;
    }
  }
  return !first_player_token_in_center;
}

std::uint32_t AzulState::next_rand_u32() {
  // xorshift64*
  std::uint64_t x = rng_salt;
  x ^= x >> 12U;
  x ^= x << 25U;
  x ^= x >> 27U;
  rng_salt = x;
  return static_cast<std::uint32_t>((x * 2685821657736338717ULL) >> 32U);
}

int AzulState::draw_one_tile() {
  if (bag.empty()) {
    if (box_lid.empty()) {
      return -1;
    }
    bag.assign(box_lid.begin(), box_lid.end());
    box_lid.clear();
    for (int i = static_cast<int>(bag.size()) - 1; i > 0; --i) {
      const int j = static_cast<int>(next_rand_u32() % static_cast<std::uint32_t>(i + 1));
      std::swap(bag[static_cast<size_t>(i)], bag[static_cast<size_t>(j)]);
    }
  }
  const int t = bag.back();
  bag.pop_back();
  return t;
}

void AzulState::refill_factories_from_rng() {
  factories = {};
  center = {};
  for (int f = 0; f < kFactories; ++f) {
    for (int i = 0; i < 4; ++i) {
      const int color = draw_one_tile();
      if (color < 0 || color >= kColors) {
        continue;
      }
      factories[f][color] = static_cast<std::uint8_t>(factories[f][color] + 1);
    }
  }
}

bool AzulState::is_tree_cache_consistent() const {
  if (persistent_tree_cache.tree.size() < persistent_tree_cache.chance_buckets.size()) {
    return false;
  }
  for (const auto& kv : persistent_tree_cache.sig_to_node) {
    const int idx = kv.second;
    if (idx < 0 || idx >= static_cast<int>(persistent_tree_cache.tree.size())) {
      return false;
    }
  }
  return true;
}

StateHash64 AzulState::state_hash(bool include_hidden_rng) const {
  std::size_t seed = 0;
  auto mix = [&seed](std::size_t v) {
    seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
  };

  mix(static_cast<std::size_t>(current_player));
  mix(static_cast<std::size_t>(first_player_next_round));
  mix(static_cast<std::size_t>(winner + 1));
  mix(static_cast<std::size_t>(round_index));
  mix(static_cast<std::size_t>(terminal ? 1 : 0));
  mix(static_cast<std::size_t>(first_player_token_in_center ? 1 : 0));
  mix(static_cast<std::size_t>(shared_victory ? 1 : 0));
  mix(static_cast<std::size_t>(scores[0]));
  mix(static_cast<std::size_t>(scores[1]));
  for (const auto& fac : factories) {
    for (std::uint8_t c : fac) {
      mix(static_cast<std::size_t>(c));
    }
  }
  for (std::uint8_t c : center) {
    mix(static_cast<std::size_t>(c));
  }
  mix(static_cast<std::size_t>(bag.size()));
  for (std::int8_t t : bag) {
    mix(static_cast<std::size_t>(t + 1));
  }
  mix(static_cast<std::size_t>(box_lid.size()));
  for (std::int8_t t : box_lid) {
    mix(static_cast<std::size_t>(t + 1));
  }
  for (const auto& p : players) {
    for (std::uint8_t len : p.line_len) {
      mix(static_cast<std::size_t>(len));
    }
    for (std::int8_t color : p.line_color) {
      mix(static_cast<std::size_t>(color + 1));
    }
    for (std::uint8_t m : p.wall_mask) {
      mix(static_cast<std::size_t>(m));
    }
    mix(static_cast<std::size_t>(p.floor_count));
    for (std::int8_t f : p.floor) {
      mix(static_cast<std::size_t>(f + 1));
    }
    mix(static_cast<std::size_t>(p.score));
  }
  if (include_hidden_rng) {
    mix(static_cast<std::size_t>(rng_salt));
  }
  return static_cast<StateHash64>(seed);
}

}  // namespace board_ai::azul
