#include "tictactoe_state.h"

#include <functional>

namespace board_ai::tictactoe {

TicTacToeState::TicTacToeState() {
  reset_with_seed(0xC0FFEEu);
}

std::unique_ptr<IGameState> TicTacToeState::clone_state() const {
  return std::make_unique<TicTacToeState>(*this);
}

void TicTacToeState::reset_with_seed(std::uint64_t seed) {
  current_player = 0;
  winner = -1;
  terminal = false;
  move_count = 0;
  scores = {0, 0};
  board.fill(static_cast<std::int8_t>(kEmptyCell));
  undo_stack.clear();
  rng_salt = seed == 0 ? 0x9e3779b97f4a7c15ULL : seed;
}

StateHash64 TicTacToeState::state_hash(bool include_hidden_rng) const {
  std::size_t seed = 0;
  auto mix = [&seed](std::size_t v) {
    seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
  };
  mix(static_cast<std::size_t>(current_player));
  mix(static_cast<std::size_t>(winner + 1));
  mix(static_cast<std::size_t>(terminal ? 1 : 0));
  mix(static_cast<std::size_t>(move_count));
  mix(static_cast<std::size_t>(scores[0] + 2));
  mix(static_cast<std::size_t>(scores[1] + 2));
  for (std::int8_t c : board) {
    mix(static_cast<std::size_t>(c + 2));
  }
  if (include_hidden_rng) {
    mix(static_cast<std::size_t>(rng_salt));
  }
  return static_cast<StateHash64>(seed);
}

}  // namespace board_ai::tictactoe

