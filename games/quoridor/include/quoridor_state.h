#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/game_interfaces.h"

namespace board_ai::quoridor {

constexpr int kBoardSize = 9;
constexpr int kPlayers = 2;
constexpr int kWallGrid = 8;
constexpr int kCellCount = kBoardSize * kBoardSize;
constexpr int kWallSlots = kWallGrid * kWallGrid;
constexpr int kMaxWallsPerPlayer = 10;

constexpr int kActionMoveStart = 0;
constexpr int kActionMoveCount = kCellCount;
constexpr int kActionHWallStart = kActionMoveStart + kActionMoveCount;
constexpr int kActionHWallCount = kWallSlots;
constexpr int kActionVWallStart = kActionHWallStart + kActionHWallCount;
constexpr int kActionVWallCount = kWallSlots;
constexpr int kActionSpace = kActionVWallStart + kActionVWallCount;

inline bool is_move_action(ActionId action) {
  return action >= kActionMoveStart && action < (kActionMoveStart + kActionMoveCount);
}

inline bool is_hwall_action(ActionId action) {
  return action >= kActionHWallStart && action < (kActionHWallStart + kActionHWallCount);
}

inline bool is_vwall_action(ActionId action) {
  return action >= kActionVWallStart && action < (kActionVWallStart + kActionVWallCount);
}

inline ActionId encode_move_action(int row, int col) {
  return static_cast<ActionId>(row * kBoardSize + col);
}

inline int decode_move_row(ActionId action) {
  return static_cast<int>(action) / kBoardSize;
}

inline int decode_move_col(ActionId action) {
  return static_cast<int>(action) % kBoardSize;
}

inline ActionId encode_hwall_action(int row, int col) {
  return static_cast<ActionId>(kActionHWallStart + row * kWallGrid + col);
}

inline ActionId encode_vwall_action(int row, int col) {
  return static_cast<ActionId>(kActionVWallStart + row * kWallGrid + col);
}

inline int decode_hwall_row(ActionId action) {
  return (static_cast<int>(action) - kActionHWallStart) / kWallGrid;
}

inline int decode_hwall_col(ActionId action) {
  return (static_cast<int>(action) - kActionHWallStart) % kWallGrid;
}

inline int decode_vwall_row(ActionId action) {
  return (static_cast<int>(action) - kActionVWallStart) / kWallGrid;
}

inline int decode_vwall_col(ActionId action) {
  return (static_cast<int>(action) - kActionVWallStart) % kWallGrid;
}

inline int wall_index(int row, int col) {
  return row * kWallGrid + col;
}

inline int goal_row_for_player(int player) {
  return player == 0 ? (kBoardSize - 1) : 0;
}

struct UndoRecord {
  int prev_player = 0;
  int prev_winner = -1;
  bool prev_terminal = false;
  int prev_move_count = 0;
  std::array<int, kPlayers> prev_scores{};
  std::array<std::int8_t, kPlayers> prev_pawn_row{};
  std::array<std::int8_t, kPlayers> prev_pawn_col{};
  std::array<std::int8_t, kPlayers> prev_walls_remaining{};
  int placed_wall_kind = 0;  // 0 none, 1 horizontal, 2 vertical
  int placed_wall_index = -1;
};

struct QuoridorState final : public IGameState {
  int current_player = 0;
  int winner = -1;
  bool terminal = false;
  int move_count = 0;
  std::array<int, kPlayers> scores{0, 0};
  std::array<std::int8_t, kPlayers> pawn_row{};
  std::array<std::int8_t, kPlayers> pawn_col{};
  std::array<std::int8_t, kPlayers> walls_remaining{};
  std::array<std::uint8_t, kWallSlots> h_walls{};
  std::array<std::uint8_t, kWallSlots> v_walls{};
  std::vector<UndoRecord> undo_stack{};
  std::uint64_t rng_salt = 0;

  QuoridorState();

  std::unique_ptr<IGameState> clone_state() const override;
  StateHash64 state_hash(bool include_hidden_rng) const override;

  void reset_with_seed(std::uint64_t seed);
};

}  // namespace board_ai::quoridor

