#include "tictactoe_rules.h"

#include <stdexcept>

namespace board_ai::tictactoe {

namespace {

constexpr int kLines[8][3] = {
    {0, 1, 2},
    {3, 4, 5},
    {6, 7, 8},
    {0, 3, 6},
    {1, 4, 7},
    {2, 5, 8},
    {0, 4, 8},
    {2, 4, 6},
};

}  // namespace

const TicTacToeState* TicTacToeRules::as_state(const IGameState& state) {
  auto p = dynamic_cast<const TicTacToeState*>(&state);
  if (!p) {
    throw std::invalid_argument("TicTacToeRules expects TicTacToeState");
  }
  return p;
}

TicTacToeState* TicTacToeRules::as_state(IGameState& state) {
  auto p = dynamic_cast<TicTacToeState*>(&state);
  if (!p) {
    throw std::invalid_argument("TicTacToeRules expects TicTacToeState");
  }
  return p;
}

int TicTacToeRules::evaluate_winner(const TicTacToeState& state) {
  for (const auto& line : kLines) {
    const int a = state.board[static_cast<size_t>(line[0])];
    if (a == kEmptyCell) {
      continue;
    }
    if (a == state.board[static_cast<size_t>(line[1])] && a == state.board[static_cast<size_t>(line[2])]) {
      return a;
    }
  }
  return -1;
}

bool TicTacToeRules::validate_action(const IGameState& state, ActionId action) const {
  const TicTacToeState* s = as_state(state);
  if (s->terminal) {
    return false;
  }
  if (action < 0 || action >= kBoardSize) {
    return false;
  }
  return s->board[static_cast<size_t>(action)] == kEmptyCell;
}

std::vector<ActionId> TicTacToeRules::legal_actions(const IGameState& state) const {
  const TicTacToeState* s = as_state(state);
  std::vector<ActionId> out;
  if (s->terminal) {
    return out;
  }
  out.reserve(kBoardSize);
  for (int i = 0; i < kBoardSize; ++i) {
    if (s->board[static_cast<size_t>(i)] == kEmptyCell) {
      out.push_back(i);
    }
  }
  return out;
}

UndoToken TicTacToeRules::do_action_fast(IGameState& state, ActionId action) const {
  TicTacToeState* s = as_state(state);
  UndoToken token{};
  token.actor = static_cast<std::uint32_t>(s->current_player);
  token.undo_depth = static_cast<std::uint32_t>(s->undo_stack.size());

  UndoRecord rec{};
  rec.action = action;
  rec.prev_player = s->current_player;
  rec.prev_winner = s->winner;
  rec.prev_terminal = s->terminal;
  rec.prev_move_count = s->move_count;
  rec.prev_scores = s->scores;
  s->undo_stack.push_back(rec);

  if (!validate_action(*s, action)) {
    return token;
  }
  s->board[static_cast<size_t>(action)] = static_cast<std::int8_t>(s->current_player);
  s->move_count += 1;

  const int w = evaluate_winner(*s);
  if (w >= 0) {
    s->winner = w;
    s->terminal = true;
    s->scores[w] = 1;
    s->scores[1 - w] = -1;
  } else if (s->move_count >= kBoardSize) {
    s->winner = -1;
    s->terminal = true;
    s->scores = {0, 0};
  } else {
    s->current_player = 1 - s->current_player;
  }
  return token;
}

void TicTacToeRules::undo_action(IGameState& state, const UndoToken& token) const {
  TicTacToeState* s = as_state(state);
  if (s->undo_stack.empty()) {
    return;
  }
  UndoRecord rec = s->undo_stack.back();
  s->undo_stack.pop_back();
  if (rec.action >= 0 && rec.action < kBoardSize) {
    s->board[static_cast<size_t>(rec.action)] = static_cast<std::int8_t>(kEmptyCell);
  }
  s->current_player = rec.prev_player;
  s->winner = rec.prev_winner;
  s->terminal = rec.prev_terminal;
  s->move_count = rec.prev_move_count;
  s->scores = rec.prev_scores;
  (void)token;
}

}  // namespace board_ai::tictactoe

