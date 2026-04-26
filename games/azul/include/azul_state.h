#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/game_interfaces.h"

namespace board_ai::azul {

constexpr int kPlayers = 2;
constexpr int kRows = 5;
constexpr int kColors = 5;
constexpr int kFactories = 5;
constexpr int kCenterSource = 5;
constexpr int kTargetsPerColor = 6;  // line[0..4] + floor
constexpr int kActionSpace = (kFactories + 1) * kColors * kTargetsPerColor;

struct PlayerState {
  std::array<std::uint8_t, kRows> line_len{};
  std::array<std::int8_t, kRows> line_color{{-1, -1, -1, -1, -1}};
  std::array<std::uint8_t, kRows> wall_mask{};
  std::array<std::int8_t, 7> floor{{-1, -1, -1, -1, -1, -1, -1}};
  std::uint8_t floor_count = 0;
  int score = 0;
};

struct AzulSnapshot {
  int current_player = 0;
  int first_player_next_round = 0;
  int winner = -1;
  int round_index = 0;
  bool terminal = false;
  bool first_player_token_in_center = true;
  bool shared_victory = false;
  std::array<int, kPlayers> scores{0, 0};
  std::array<std::array<std::uint8_t, kColors>, kFactories> factories{};
  std::array<std::uint8_t, kColors> center{};
  std::vector<std::int8_t> bag{};
  std::vector<std::int8_t> box_lid{};
  std::array<PlayerState, kPlayers> players{};
  std::uint64_t rng_salt = 0;
};

struct UndoRecord {
  bool use_full_snapshot = false;
  AzulSnapshot full_before{};
  int prev_current_player = 0;
  int prev_first_player_next_round = 0;
  int prev_winner = -1;
  int prev_round_index = 0;
  bool prev_terminal = false;
  bool prev_first_player_token_in_center = true;
  bool prev_shared_victory = false;
  std::array<int, kPlayers> prev_scores{0, 0};
  std::array<std::uint8_t, kColors> prev_center{};
  bool has_factory_source = false;
  int source_factory_idx = -1;
  std::array<std::uint8_t, kColors> prev_factory_source{};
  PlayerState prev_player{};
  std::uint64_t prev_rng_salt = 0;
};

struct PersistentTreeCache {
  std::vector<int> tree{};
  std::vector<int> chance_buckets{};
  std::unordered_map<StateHash64, int> sig_to_node{};
};

class AzulState final : public IGameState {
 public:
  AzulState();

  std::unique_ptr<IGameState> clone_state() const override;
  StateHash64 state_hash(bool include_hidden_rng) const override;

  int current_player = 0;
  int first_player_next_round = 0;
  int winner = -1;  // -1 means no single winner
  int round_index = 0;
  bool terminal = false;
  bool first_player_token_in_center = true;
  bool shared_victory = false;
  std::array<int, kPlayers> scores{0, 0};

  std::array<std::array<std::uint8_t, kColors>, kFactories> factories{};
  std::array<std::uint8_t, kColors> center{};
  std::vector<std::int8_t> bag{};
  std::vector<std::int8_t> box_lid{};
  std::array<PlayerState, kPlayers> players{};

  std::vector<UndoRecord> undo_stack{};
  PersistentTreeCache persistent_tree_cache{};
  std::uint64_t rng_salt = 0;

  void reset_with_seed(std::uint64_t seed);
  bool all_sources_empty() const;
  void refill_factories_from_rng();
  int draw_one_tile();
  std::uint32_t next_rand_u32();
  StateHash64 state_signature() const { return state_hash(false); }
  bool is_tree_cache_consistent() const;
};

}  // namespace board_ai::azul
