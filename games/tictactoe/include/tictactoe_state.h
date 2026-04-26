#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/game_interfaces.h"

namespace board_ai::tictactoe {

constexpr int kBoardSize = 9;
constexpr int kPlayers = 2;
constexpr int kEmptyCell = -1;

struct UndoRecord {
  int action = -1;
  int prev_player = 0;
  int prev_winner = -1;
  bool prev_terminal = false;
  int prev_move_count = 0;
  std::array<int, kPlayers> prev_scores{};
};

struct TicTacToeState final : public IGameState {
  int current_player = 0;
  int winner = -1;
  bool terminal = false;
  int move_count = 0;
  std::array<int, kPlayers> scores{0, 0};
  std::array<std::int8_t, kBoardSize> board{};
  std::vector<UndoRecord> undo_stack{};
  std::uint64_t rng_salt = 0;

  TicTacToeState();

  std::unique_ptr<IGameState> clone_state() const override;
  StateHash64 state_hash(bool include_hidden_rng) const override;

  void reset_with_seed(std::uint64_t seed);
};

}  // namespace board_ai::tictactoe
