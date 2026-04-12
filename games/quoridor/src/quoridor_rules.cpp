#include "quoridor_rules.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace board_ai::quoridor {

namespace {

constexpr std::array<int, 4> kDirR = {-1, 1, 0, 0};
constexpr std::array<int, 4> kDirC = {0, 0, -1, 1};

bool in_bounds(int row, int col) {
  return row >= 0 && row < kBoardSize && col >= 0 && col < kBoardSize;
}

bool has_h_wall(const QuoridorState& state, int row, int col) {
  if (row < 0 || row >= kWallGrid || col < 0 || col >= kWallGrid) return false;
  return state.h_walls[static_cast<size_t>(wall_index(row, col))] != 0;
}

bool has_v_wall(const QuoridorState& state, int row, int col) {
  if (row < 0 || row >= kWallGrid || col < 0 || col >= kWallGrid) return false;
  return state.v_walls[static_cast<size_t>(wall_index(row, col))] != 0;
}

// Vertical step: between (row,col) and (row+1,col), row in [0,7].
bool blocked_vertical_step(const QuoridorState& state, int row, int col) {
  if (row < 0 || row >= kWallGrid || col < 0 || col >= kBoardSize) return true;
  return has_h_wall(state, row, col) || has_h_wall(state, row, col - 1);
}

// Horizontal step: between (row,col) and (row,col+1), col in [0,7].
bool blocked_horizontal_step(const QuoridorState& state, int row, int col) {
  if (col < 0 || col >= kWallGrid || row < 0 || row >= kBoardSize) return true;
  return has_v_wall(state, row, col) || has_v_wall(state, row - 1, col);
}

bool is_blocked_between(const QuoridorState& state, int r0, int c0, int r1, int c1) {
  if (!in_bounds(r0, c0) || !in_bounds(r1, c1)) return true;
  if (r0 == r1) {
    if (c1 == c0 + 1) return blocked_horizontal_step(state, r0, c0);
    if (c1 == c0 - 1) return blocked_horizontal_step(state, r0, c1);
    return true;
  }
  if (c0 == c1) {
    if (r1 == r0 + 1) return blocked_vertical_step(state, r0, c0);
    if (r1 == r0 - 1) return blocked_vertical_step(state, r1, c0);
    return true;
  }
  return true;
}

inline int cell_index(int row, int col) {
  return row * kBoardSize + col;
}

void compute_goal_distance_map(const QuoridorState& state, int player, std::array<int, kCellCount>* out_dist) {
  if (!out_dist) return;
  out_dist->fill(-1);
  if (player < 0 || player >= kPlayers) return;

  const int target_r = goal_row_for_player(player);
  std::array<int, kCellCount> q{};
  int q_head = 0;
  int q_tail = 0;

  for (int c = 0; c < kBoardSize; ++c) {
    const int idx = cell_index(target_r, c);
    (*out_dist)[static_cast<size_t>(idx)] = 0;
    q[static_cast<size_t>(q_tail++)] = idx;
  }

  while (q_head < q_tail) {
    const int idx = q[static_cast<size_t>(q_head++)];
    const int r = idx / kBoardSize;
    const int c = idx % kBoardSize;
    const int base_dist = (*out_dist)[static_cast<size_t>(idx)];

    for (size_t i = 0; i < kDirR.size(); ++i) {
      const int nr = r + kDirR[i];
      const int nc = c + kDirC[i];
      if (!in_bounds(nr, nc)) continue;
      if (is_blocked_between(state, r, c, nr, nc)) continue;
      const int nidx = cell_index(nr, nc);
      if ((*out_dist)[static_cast<size_t>(nidx)] >= 0) continue;
      (*out_dist)[static_cast<size_t>(nidx)] = base_dist + 1;
      q[static_cast<size_t>(q_tail++)] = nidx;
    }
  }
}

inline int distance_or_inf(int dist) {
  return dist >= 0 ? dist : 9999;
}

constexpr int kVerticalStepCount = kWallGrid * kBoardSize;    // 8 * 9 = 72
constexpr int kHorizontalStepCount = kBoardSize * kWallGrid;  // 9 * 8 = 72

inline int v_step_index(int row, int col) {
  return row * kBoardSize + col;
}

inline int h_step_index(int row, int col) {
  return row * kWallGrid + col;
}

void build_base_step_passability(
    const QuoridorState& state,
    std::array<std::uint8_t, kVerticalStepCount>* out_v_open,
    std::array<std::uint8_t, kHorizontalStepCount>* out_h_open) {
  if (!out_v_open || !out_h_open) return;
  for (int r = 0; r < kWallGrid; ++r) {
    for (int c = 0; c < kBoardSize; ++c) {
      (*out_v_open)[static_cast<size_t>(v_step_index(r, c))] = blocked_vertical_step(state, r, c) ? 0 : 1;
    }
  }
  for (int r = 0; r < kBoardSize; ++r) {
    for (int c = 0; c < kWallGrid; ++c) {
      (*out_h_open)[static_cast<size_t>(h_step_index(r, c))] = blocked_horizontal_step(state, r, c) ? 0 : 1;
    }
  }
}

bool both_players_have_goal_path_with_overlay(
    const QuoridorState& state,
    int overlay_kind,  // 0 none, 1 hwall, 2 vwall
    int overlay_row,
    int overlay_col,
    const std::array<std::uint8_t, kVerticalStepCount>& v_open,
    const std::array<std::uint8_t, kHorizontalStepCount>& h_open) {
  auto vertical_open = [&](int row, int col) -> bool {
    if (row < 0 || row >= kWallGrid || col < 0 || col >= kBoardSize) return false;
    if (v_open[static_cast<size_t>(v_step_index(row, col))] == 0) return false;
    if (overlay_kind == 1 && overlay_row == row && (overlay_col == col || overlay_col + 1 == col)) {
      return false;
    }
    return true;
  };
  auto horizontal_open = [&](int row, int col) -> bool {
    if (row < 0 || row >= kBoardSize || col < 0 || col >= kWallGrid) return false;
    if (h_open[static_cast<size_t>(h_step_index(row, col))] == 0) return false;
    if (overlay_kind == 2 && overlay_col == col && (overlay_row == row || overlay_row + 1 == row)) {
      return false;
    }
    return true;
  };

  std::array<int, kCellCount> comp{};
  comp.fill(-1);
  std::array<std::uint8_t, kCellCount> comp_goal_mask{};
  comp_goal_mask.fill(0);
  std::array<int, kCellCount> q{};

  int comp_count = 0;
  for (int start = 0; start < kCellCount; ++start) {
    if (comp[static_cast<size_t>(start)] >= 0) continue;

    int q_head = 0;
    int q_tail = 0;
    q[static_cast<size_t>(q_tail++)] = start;
    comp[static_cast<size_t>(start)] = comp_count;

    std::uint8_t goal_mask = 0;
    while (q_head < q_tail) {
      const int idx = q[static_cast<size_t>(q_head++)];
      const int r = idx / kBoardSize;
      const int c = idx % kBoardSize;
      if (r == goal_row_for_player(0)) goal_mask |= 0x1;
      if (r == goal_row_for_player(1)) goal_mask |= 0x2;

      if (r > 0 && vertical_open(r - 1, c)) {
        const int nidx = cell_index(r - 1, c);
        if (comp[static_cast<size_t>(nidx)] < 0) {
          comp[static_cast<size_t>(nidx)] = comp_count;
          q[static_cast<size_t>(q_tail++)] = nidx;
        }
      }
      if (r + 1 < kBoardSize && vertical_open(r, c)) {
        const int nidx = cell_index(r + 1, c);
        if (comp[static_cast<size_t>(nidx)] < 0) {
          comp[static_cast<size_t>(nidx)] = comp_count;
          q[static_cast<size_t>(q_tail++)] = nidx;
        }
      }
      if (c > 0 && horizontal_open(r, c - 1)) {
        const int nidx = cell_index(r, c - 1);
        if (comp[static_cast<size_t>(nidx)] < 0) {
          comp[static_cast<size_t>(nidx)] = comp_count;
          q[static_cast<size_t>(q_tail++)] = nidx;
        }
      }
      if (c + 1 < kBoardSize && horizontal_open(r, c)) {
        const int nidx = cell_index(r, c + 1);
        if (comp[static_cast<size_t>(nidx)] < 0) {
          comp[static_cast<size_t>(nidx)] = comp_count;
          q[static_cast<size_t>(q_tail++)] = nidx;
        }
      }
    }

    comp_goal_mask[static_cast<size_t>(comp_count)] = goal_mask;
    ++comp_count;
  }

  const int p0_idx = cell_index(
      state.pawn_row[static_cast<size_t>(0)],
      state.pawn_col[static_cast<size_t>(0)]);
  const int p1_idx = cell_index(
      state.pawn_row[static_cast<size_t>(1)],
      state.pawn_col[static_cast<size_t>(1)]);
  const int p0_comp = comp[static_cast<size_t>(p0_idx)];
  const int p1_comp = comp[static_cast<size_t>(p1_idx)];
  if (p0_comp < 0 || p1_comp < 0) return false;
  const bool p0_ok = (comp_goal_mask[static_cast<size_t>(p0_comp)] & 0x1) != 0;
  const bool p1_ok = (comp_goal_mask[static_cast<size_t>(p1_comp)] & 0x2) != 0;
  return p0_ok && p1_ok;
}

bool both_players_have_goal_path(const QuoridorState& state) {
  std::array<int, kCellCount> comp{};
  comp.fill(-1);
  std::array<std::uint8_t, kCellCount> comp_goal_mask{};
  comp_goal_mask.fill(0);
  std::array<int, kCellCount> q{};

  int comp_count = 0;
  for (int start = 0; start < kCellCount; ++start) {
    if (comp[static_cast<size_t>(start)] >= 0) continue;

    int q_head = 0;
    int q_tail = 0;
    q[static_cast<size_t>(q_tail++)] = start;
    comp[static_cast<size_t>(start)] = comp_count;

    std::uint8_t goal_mask = 0;
    while (q_head < q_tail) {
      const int idx = q[static_cast<size_t>(q_head++)];
      const int r = idx / kBoardSize;
      const int c = idx % kBoardSize;
      if (r == goal_row_for_player(0)) goal_mask |= 0x1;
      if (r == goal_row_for_player(1)) goal_mask |= 0x2;

      for (size_t i = 0; i < kDirR.size(); ++i) {
        const int nr = r + kDirR[i];
        const int nc = c + kDirC[i];
        if (!in_bounds(nr, nc)) continue;
        if (is_blocked_between(state, r, c, nr, nc)) continue;
        const int nidx = cell_index(nr, nc);
        if (comp[static_cast<size_t>(nidx)] >= 0) continue;
        comp[static_cast<size_t>(nidx)] = comp_count;
        q[static_cast<size_t>(q_tail++)] = nidx;
      }
    }

    comp_goal_mask[static_cast<size_t>(comp_count)] = goal_mask;
    ++comp_count;
  }

  const int p0_idx = cell_index(
      state.pawn_row[static_cast<size_t>(0)],
      state.pawn_col[static_cast<size_t>(0)]);
  const int p1_idx = cell_index(
      state.pawn_row[static_cast<size_t>(1)],
      state.pawn_col[static_cast<size_t>(1)]);
  const int p0_comp = comp[static_cast<size_t>(p0_idx)];
  const int p1_comp = comp[static_cast<size_t>(p1_idx)];
  if (p0_comp < 0 || p1_comp < 0) return false;
  const bool p0_ok = (comp_goal_mask[static_cast<size_t>(p0_comp)] & 0x1) != 0;
  const bool p1_ok = (comp_goal_mask[static_cast<size_t>(p1_comp)] & 0x2) != 0;
  return p0_ok && p1_ok;
}

bool can_place_hwall_with_base(
    const QuoridorState& state,
    int row,
    int col,
    const std::array<std::uint8_t, kVerticalStepCount>& v_open,
    const std::array<std::uint8_t, kHorizontalStepCount>& h_open) {
  if (row < 0 || row >= kWallGrid || col < 0 || col >= kWallGrid) return false;
  const int idx = wall_index(row, col);
  if (state.h_walls[static_cast<size_t>(idx)] != 0) return false;
  if (has_h_wall(state, row, col - 1) || has_h_wall(state, row, col + 1)) return false;
  if (state.v_walls[static_cast<size_t>(idx)] != 0) return false;
  return both_players_have_goal_path_with_overlay(state, 1, row, col, v_open, h_open);
}

bool can_place_vwall_with_base(
    const QuoridorState& state,
    int row,
    int col,
    const std::array<std::uint8_t, kVerticalStepCount>& v_open,
    const std::array<std::uint8_t, kHorizontalStepCount>& h_open) {
  if (row < 0 || row >= kWallGrid || col < 0 || col >= kWallGrid) return false;
  const int idx = wall_index(row, col);
  if (state.v_walls[static_cast<size_t>(idx)] != 0) return false;
  if (has_v_wall(state, row - 1, col) || has_v_wall(state, row + 1, col)) return false;
  if (state.h_walls[static_cast<size_t>(idx)] != 0) return false;
  return both_players_have_goal_path_with_overlay(state, 2, row, col, v_open, h_open);
}

void collect_pawn_destinations(const QuoridorState& state, int player, std::vector<ActionId>* out_actions) {
  if (!out_actions) return;
  out_actions->clear();
  const int me_r = state.pawn_row[static_cast<size_t>(player)];
  const int me_c = state.pawn_col[static_cast<size_t>(player)];
  const int opp = 1 - player;
  const int opp_r = state.pawn_row[static_cast<size_t>(opp)];
  const int opp_c = state.pawn_col[static_cast<size_t>(opp)];

  std::array<std::uint8_t, kCellCount> marked{};
  auto push_cell = [&](int row, int col) {
    if (!in_bounds(row, col)) return;
    const int idx = row * kBoardSize + col;
    if (marked[static_cast<size_t>(idx)] != 0) return;
    marked[static_cast<size_t>(idx)] = 1;
    out_actions->push_back(encode_move_action(row, col));
  };

  for (size_t i = 0; i < kDirR.size(); ++i) {
    const int dr = kDirR[i];
    const int dc = kDirC[i];
    const int nr = me_r + dr;
    const int nc = me_c + dc;
    if (!in_bounds(nr, nc)) continue;
    if (is_blocked_between(state, me_r, me_c, nr, nc)) continue;

    if (!(nr == opp_r && nc == opp_c)) {
      push_cell(nr, nc);
      continue;
    }

    const int jr = opp_r + dr;
    const int jc = opp_c + dc;
    if (in_bounds(jr, jc) && !is_blocked_between(state, opp_r, opp_c, jr, jc)) {
      push_cell(jr, jc);
      continue;
    }

    if (dr != 0) {
      for (int side_dc : {-1, 1}) {
        const int sr = opp_r;
        const int sc = opp_c + side_dc;
        if (!in_bounds(sr, sc)) continue;
        if (is_blocked_between(state, opp_r, opp_c, sr, sc)) continue;
        push_cell(sr, sc);
      }
    } else {
      for (int side_dr : {-1, 1}) {
        const int sr = opp_r + side_dr;
        const int sc = opp_c;
        if (!in_bounds(sr, sc)) continue;
        if (is_blocked_between(state, opp_r, opp_c, sr, sc)) continue;
        push_cell(sr, sc);
      }
    }
  }
}

bool has_path_internal(const QuoridorState& state, int player) {
  if (player < 0 || player >= kPlayers) return false;
  const int start_r = state.pawn_row[static_cast<size_t>(player)];
  const int start_c = state.pawn_col[static_cast<size_t>(player)];
  std::array<int, kCellCount> dist{};
  compute_goal_distance_map(state, player, &dist);
  return dist[static_cast<size_t>(cell_index(start_r, start_c))] >= 0;
}

int shortest_path_internal(const QuoridorState& state, int player) {
  if (player < 0 || player >= kPlayers) return 9999;
  const int start_r = state.pawn_row[static_cast<size_t>(player)];
  const int start_c = state.pawn_col[static_cast<size_t>(player)];
  std::array<int, kCellCount> dist{};
  compute_goal_distance_map(state, player, &dist);
  return distance_or_inf(dist[static_cast<size_t>(cell_index(start_r, start_c))]);
}

bool can_place_hwall(QuoridorState& state, int row, int col) {
  std::array<std::uint8_t, kVerticalStepCount> v_open{};
  std::array<std::uint8_t, kHorizontalStepCount> h_open{};
  build_base_step_passability(state, &v_open, &h_open);
  return can_place_hwall_with_base(state, row, col, v_open, h_open);
}

bool can_place_vwall(QuoridorState& state, int row, int col) {
  std::array<std::uint8_t, kVerticalStepCount> v_open{};
  std::array<std::uint8_t, kHorizontalStepCount> h_open{};
  build_base_step_passability(state, &v_open, &h_open);
  return can_place_vwall_with_base(state, row, col, v_open, h_open);
}

}  // namespace

const QuoridorState* QuoridorRules::as_state(const IGameState& state) {
  auto p = dynamic_cast<const QuoridorState*>(&state);
  if (!p) {
    throw std::invalid_argument("QuoridorRules expects QuoridorState");
  }
  return p;
}

QuoridorState* QuoridorRules::as_state(IGameState& state) {
  auto p = dynamic_cast<QuoridorState*>(&state);
  if (!p) {
    throw std::invalid_argument("QuoridorRules expects QuoridorState");
  }
  return p;
}

bool QuoridorRules::has_path_to_goal(const QuoridorState& state, int player) {
  return has_path_internal(state, player);
}

int QuoridorRules::shortest_path_distance(const QuoridorState& state, int player) {
  return shortest_path_internal(state, player);
}

bool QuoridorRules::validate_action(const IGameState& state, ActionId action) const {
  const QuoridorState* s = as_state(state);
  if (s->terminal) return false;
  if (action < 0 || action >= kActionSpace) return false;

  if (is_move_action(action)) {
    std::vector<ActionId> pawn_moves;
    collect_pawn_destinations(*s, s->current_player, &pawn_moves);
    const bool in_pawn_moves = std::find(pawn_moves.begin(), pawn_moves.end(), action) != pawn_moves.end();
    if (!in_pawn_moves) {
      return false;
    }
    // Additional rule: a pawn move is only legal when it does not worsen
    // current player's shortest path length to the goal row.
    const int player = s->current_player;
    std::array<int, kCellCount> dist{};
    compute_goal_distance_map(*s, player, &dist);
    const int before_idx = cell_index(
        s->pawn_row[static_cast<size_t>(player)],
        s->pawn_col[static_cast<size_t>(player)]);
    const int after_idx = cell_index(decode_move_row(action), decode_move_col(action));
    const int before_dist = distance_or_inf(dist[static_cast<size_t>(before_idx)]);
    const int after_dist = distance_or_inf(dist[static_cast<size_t>(after_idx)]);
    return after_dist <= before_dist;
  }

  if (s->walls_remaining[static_cast<size_t>(s->current_player)] <= 0) {
    return false;
  }
  std::array<std::uint8_t, kVerticalStepCount> v_open{};
  std::array<std::uint8_t, kHorizontalStepCount> h_open{};
  build_base_step_passability(*s, &v_open, &h_open);
  QuoridorState tmp = *s;
  if (is_hwall_action(action)) {
    return can_place_hwall_with_base(tmp, decode_hwall_row(action), decode_hwall_col(action), v_open, h_open);
  }
  if (is_vwall_action(action)) {
    return can_place_vwall_with_base(tmp, decode_vwall_row(action), decode_vwall_col(action), v_open, h_open);
  }
  return false;
}

std::vector<ActionId> QuoridorRules::legal_actions(const IGameState& state) const {
  const QuoridorState* s = as_state(state);
  std::vector<ActionId> out;
  if (s->terminal) return out;

  std::vector<ActionId> pawn_moves;
  collect_pawn_destinations(*s, s->current_player, &pawn_moves);
  std::array<int, kCellCount> dist{};
  compute_goal_distance_map(*s, s->current_player, &dist);
  const int before_idx = cell_index(
      s->pawn_row[static_cast<size_t>(s->current_player)],
      s->pawn_col[static_cast<size_t>(s->current_player)]);
  const int before_dist = distance_or_inf(dist[static_cast<size_t>(before_idx)]);
  for (ActionId a : pawn_moves) {
    const int after_idx = cell_index(decode_move_row(a), decode_move_col(a));
    const int after_dist = distance_or_inf(dist[static_cast<size_t>(after_idx)]);
    if (after_dist <= before_dist) {
      out.push_back(a);
    }
  }
  if (s->walls_remaining[static_cast<size_t>(s->current_player)] <= 0) {
    return out;
  }

  std::array<std::uint8_t, kVerticalStepCount> v_open{};
  std::array<std::uint8_t, kHorizontalStepCount> h_open{};
  build_base_step_passability(*s, &v_open, &h_open);
  QuoridorState tmp = *s;
  for (int r = 0; r < kWallGrid; ++r) {
    for (int c = 0; c < kWallGrid; ++c) {
      if (can_place_hwall_with_base(tmp, r, c, v_open, h_open)) {
        out.push_back(encode_hwall_action(r, c));
      }
      if (can_place_vwall_with_base(tmp, r, c, v_open, h_open)) {
        out.push_back(encode_vwall_action(r, c));
      }
    }
  }
  return out;
}

UndoToken QuoridorRules::do_action_fast(IGameState& state, ActionId action) const {
  QuoridorState* s = as_state(state);
  UndoToken token{};
  token.actor = static_cast<std::uint32_t>(s->current_player);
  token.undo_depth = static_cast<std::uint32_t>(s->undo_stack.size());

  if (!validate_action(*s, action)) {
    return token;
  }

  UndoRecord rec{};
  rec.prev_player = s->current_player;
  rec.prev_winner = s->winner;
  rec.prev_terminal = s->terminal;
  rec.prev_move_count = s->move_count;
  rec.prev_scores = s->scores;
  rec.prev_pawn_row = s->pawn_row;
  rec.prev_pawn_col = s->pawn_col;
  rec.prev_walls_remaining = s->walls_remaining;
  s->undo_stack.push_back(rec);
  UndoRecord& back = s->undo_stack.back();

  if (is_move_action(action)) {
    s->pawn_row[static_cast<size_t>(s->current_player)] = static_cast<std::int8_t>(decode_move_row(action));
    s->pawn_col[static_cast<size_t>(s->current_player)] = static_cast<std::int8_t>(decode_move_col(action));
  } else if (is_hwall_action(action)) {
    const int idx = wall_index(decode_hwall_row(action), decode_hwall_col(action));
    s->h_walls[static_cast<size_t>(idx)] = 1;
    s->walls_remaining[static_cast<size_t>(s->current_player)] -= 1;
    back.placed_wall_kind = 1;
    back.placed_wall_index = idx;
  } else if (is_vwall_action(action)) {
    const int idx = wall_index(decode_vwall_row(action), decode_vwall_col(action));
    s->v_walls[static_cast<size_t>(idx)] = 1;
    s->walls_remaining[static_cast<size_t>(s->current_player)] -= 1;
    back.placed_wall_kind = 2;
    back.placed_wall_index = idx;
  }

  s->move_count += 1;
  if (s->pawn_row[0] == goal_row_for_player(0)) {
    s->terminal = true;
    s->winner = 0;
    s->scores = {1, -1};
  } else if (s->pawn_row[1] == goal_row_for_player(1)) {
    s->terminal = true;
    s->winner = 1;
    s->scores = {-1, 1};
  } else {
    s->current_player = 1 - s->current_player;
  }
  return token;
}

void QuoridorRules::undo_action(IGameState& state, const UndoToken& token) const {
  QuoridorState* s = as_state(state);
  if (s->undo_stack.empty()) return;
  const UndoRecord rec = s->undo_stack.back();
  s->undo_stack.pop_back();

  if (rec.placed_wall_kind == 1 && rec.placed_wall_index >= 0 && rec.placed_wall_index < kWallSlots) {
    s->h_walls[static_cast<size_t>(rec.placed_wall_index)] = 0;
  } else if (rec.placed_wall_kind == 2 && rec.placed_wall_index >= 0 && rec.placed_wall_index < kWallSlots) {
    s->v_walls[static_cast<size_t>(rec.placed_wall_index)] = 0;
  }
  s->current_player = rec.prev_player;
  s->winner = rec.prev_winner;
  s->terminal = rec.prev_terminal;
  s->move_count = rec.prev_move_count;
  s->scores = rec.prev_scores;
  s->pawn_row = rec.prev_pawn_row;
  s->pawn_col = rec.prev_pawn_col;
  s->walls_remaining = rec.prev_walls_remaining;
  (void)token;
}

}  // namespace board_ai::quoridor

