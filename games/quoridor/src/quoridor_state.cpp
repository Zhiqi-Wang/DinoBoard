#include "quoridor_state.h"

#include <functional>

namespace board_ai::quoridor {

namespace {

inline void mix_hash_value(std::size_t* seed, std::size_t value) {
  if (!seed) {
    return;
  }
  *seed ^= value + 0x9e3779b97f4a7c15ULL + (*seed << 6U) + (*seed >> 2U);
}

}  // namespace

QuoridorState::QuoridorState() {
  reset_with_seed(0xC0FFEE1234ULL);
}

std::unique_ptr<IGameState> QuoridorState::clone_state() const {
  return std::make_unique<QuoridorState>(*this);
}

void QuoridorState::reset_with_seed(std::uint64_t seed) {
  current_player = 0;
  winner = -1;
  terminal = false;
  move_count = 0;
  scores = {0, 0};
  pawn_row = {0, static_cast<std::int8_t>(kBoardSize - 1)};
  pawn_col = {4, 4};
  walls_remaining = {kMaxWallsPerPlayer, kMaxWallsPerPlayer};
  h_walls.fill(0);
  v_walls.fill(0);
  undo_stack.clear();
  rng_salt = seed == 0 ? 0x9e3779b97f4a7c15ULL : seed;
}

StateHash64 QuoridorState::state_hash(bool include_hidden_rng) const {
  std::size_t seed = 0;

  mix_hash_value(&seed, static_cast<std::size_t>(current_player));
  mix_hash_value(&seed, static_cast<std::size_t>(winner + 1));
  mix_hash_value(&seed, static_cast<std::size_t>(terminal ? 1 : 0));
  mix_hash_value(&seed, static_cast<std::size_t>(move_count));
  mix_hash_value(&seed, static_cast<std::size_t>(scores[0] + 2));
  mix_hash_value(&seed, static_cast<std::size_t>(scores[1] + 2));
  for (int p = 0; p < kPlayers; ++p) {
    mix_hash_value(&seed, static_cast<std::size_t>(pawn_row[static_cast<size_t>(p)] + 1));
    mix_hash_value(&seed, static_cast<std::size_t>(pawn_col[static_cast<size_t>(p)] + 1));
    mix_hash_value(&seed, static_cast<std::size_t>(walls_remaining[static_cast<size_t>(p)] + 1));
  }
  for (std::uint8_t w : h_walls) mix_hash_value(&seed, static_cast<std::size_t>(w));
  for (std::uint8_t w : v_walls) mix_hash_value(&seed, static_cast<std::size_t>(w));
  if (include_hidden_rng) {
    mix_hash_value(&seed, static_cast<std::size_t>(rng_salt));
  }
  return static_cast<StateHash64>(seed);
}

}  // namespace board_ai::quoridor

