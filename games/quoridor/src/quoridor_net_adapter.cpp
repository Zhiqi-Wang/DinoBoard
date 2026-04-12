#include "quoridor_net_adapter.h"

namespace board_ai::quoridor {

namespace {

const QuoridorState* as_quoridor(const IGameState& state) {
  return dynamic_cast<const QuoridorState*>(&state);
}

inline int orient_cell_row(int row, int perspective_player) {
  return perspective_player == 0 ? row : (kBoardSize - 1 - row);
}

inline int orient_cell_col(int col, int perspective_player) {
  return perspective_player == 0 ? col : (kBoardSize - 1 - col);
}

inline int orient_wall_row(int row, int perspective_player) {
  return perspective_player == 0 ? row : (kWallGrid - 1 - row);
}

inline int orient_wall_col(int col, int perspective_player) {
  return perspective_player == 0 ? col : (kWallGrid - 1 - col);
}

ActionId canonicalize_action_for_perspective(ActionId action, int perspective_player) {
  if (perspective_player == 0) return action;
  if (is_move_action(action)) {
    const int r = orient_cell_row(decode_move_row(action), perspective_player);
    const int c = orient_cell_col(decode_move_col(action), perspective_player);
    return encode_move_action(r, c);
  }
  if (is_hwall_action(action)) {
    const int r = orient_wall_row(decode_hwall_row(action), perspective_player);
    const int c = orient_wall_col(decode_hwall_col(action), perspective_player);
    return encode_hwall_action(r, c);
  }
  if (is_vwall_action(action)) {
    const int r = orient_wall_row(decode_vwall_row(action), perspective_player);
    const int c = orient_wall_col(decode_vwall_col(action), perspective_player);
    return encode_vwall_action(r, c);
  }
  return action;
}

}  // namespace

bool QuoridorFeatureEncoder::encode(
    const IGameState& state,
    int perspective_player,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* features,
    std::vector<float>* legal_mask) const {
  const QuoridorState* s = as_quoridor(state);
  if (!s || !features || !legal_mask || perspective_player < 0 || perspective_player >= kPlayers) {
    return false;
  }
  const int opp = 1 - perspective_player;
  const int me_r = orient_cell_row(s->pawn_row[static_cast<size_t>(perspective_player)], perspective_player);
  const int me_c = orient_cell_col(s->pawn_col[static_cast<size_t>(perspective_player)], perspective_player);
  const int opp_r = orient_cell_row(s->pawn_row[static_cast<size_t>(opp)], perspective_player);
  const int opp_c = orient_cell_col(s->pawn_col[static_cast<size_t>(opp)], perspective_player);
  const int me_idx = me_r * kBoardSize + me_c;
  const int opp_idx = opp_r * kBoardSize + opp_c;

  features->clear();
  features->reserve(static_cast<size_t>(feature_dim()));

  for (int i = 0; i < kCellCount; ++i) {
    features->push_back(i == me_idx ? 1.0f : 0.0f);
  }
  for (int i = 0; i < kCellCount; ++i) {
    features->push_back(i == opp_idx ? 1.0f : 0.0f);
  }
  for (int r = 0; r < kWallGrid; ++r) {
    for (int c = 0; c < kWallGrid; ++c) {
      const int src_r = orient_wall_row(r, perspective_player);
      const int src_c = orient_wall_col(c, perspective_player);
      const int src_idx = wall_index(src_r, src_c);
      features->push_back(s->h_walls[static_cast<size_t>(src_idx)] ? 1.0f : 0.0f);
    }
  }
  for (int r = 0; r < kWallGrid; ++r) {
    for (int c = 0; c < kWallGrid; ++c) {
      const int src_r = orient_wall_row(r, perspective_player);
      const int src_c = orient_wall_col(c, perspective_player);
      const int src_idx = wall_index(src_r, src_c);
      features->push_back(s->v_walls[static_cast<size_t>(src_idx)] ? 1.0f : 0.0f);
    }
  }
  features->push_back(static_cast<float>(s->walls_remaining[static_cast<size_t>(perspective_player)]) /
                      static_cast<float>(kMaxWallsPerPlayer));
  features->push_back(static_cast<float>(s->walls_remaining[static_cast<size_t>(opp)]) /
                      static_cast<float>(kMaxWallsPerPlayer));
  // Perspective-normalized encoding keeps the side identity implicit.
  // Keep this slot as a constant bias term to preserve feature dim.
  features->push_back(1.0f);
  features->push_back(static_cast<float>(s->move_count) / 128.0f);

  legal_mask->assign(static_cast<size_t>(action_space()), 0.0f);
  for (ActionId a : legal_actions) {
    const ActionId canonical_a = canonicalize_action_for_perspective(a, perspective_player);
    if (canonical_a >= 0 && canonical_a < action_space()) {
      (*legal_mask)[static_cast<size_t>(canonical_a)] = 1.0f;
    }
  }

  return static_cast<int>(features->size()) == feature_dim();
}

ActionId QuoridorFeatureEncoder::canonicalize_action(ActionId action, int perspective_player) const {
  return canonicalize_action_for_perspective(action, perspective_player);
}

ActionId QuoridorFeatureEncoder::decanonicalize_action(ActionId canonical_action, int perspective_player) const {
  // 180-degree transform is self-inverse.
  return canonicalize_action_for_perspective(canonical_action, perspective_player);
}

int QuoridorStateValueModel::current_player(const IGameState& state) const {
  const QuoridorState* s = as_quoridor(state);
  return s ? s->current_player : 0;
}

bool QuoridorStateValueModel::is_terminal(const IGameState& state) const {
  const QuoridorState* s = as_quoridor(state);
  return s ? s->terminal : true;
}

float QuoridorStateValueModel::terminal_value_for_player(const IGameState& state, int perspective_player) const {
  const QuoridorState* s = as_quoridor(state);
  if (!s || !s->terminal) return 0.0f;
  if (s->winner < 0) return 0.0f;
  return s->winner == perspective_player ? 1.0f : -1.0f;
}

}  // namespace board_ai::quoridor

