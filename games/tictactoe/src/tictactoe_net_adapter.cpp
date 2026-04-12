#include "tictactoe_net_adapter.h"

namespace board_ai::tictactoe {

namespace {

const TicTacToeState* as_ttt(const IGameState& s) {
  return dynamic_cast<const TicTacToeState*>(&s);
}

}  // namespace

bool TicTacToeFeatureEncoder::encode(
    const IGameState& state,
    int perspective_player,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* features,
    std::vector<float>* legal_mask) const {
  const TicTacToeState* s = as_ttt(state);
  if (!s || !features || !legal_mask || perspective_player < 0 || perspective_player >= kPlayers) {
    return false;
  }
  const int opp = 1 - perspective_player;

  features->clear();
  features->reserve(static_cast<size_t>(feature_dim()));
  for (int i = 0; i < kBoardSize; ++i) {
    const int v = static_cast<int>(s->board[static_cast<size_t>(i)]);
    features->push_back(v == perspective_player ? 1.0f : 0.0f);
    features->push_back(v == opp ? 1.0f : 0.0f);
    features->push_back(v == kEmptyCell ? 1.0f : 0.0f);
  }

  legal_mask->assign(static_cast<size_t>(action_space()), 0.0f);
  for (ActionId a : legal_actions) {
    if (a >= 0 && a < action_space()) {
      (*legal_mask)[static_cast<size_t>(a)] = 1.0f;
    }
  }

  return static_cast<int>(features->size()) == feature_dim();
}

int TicTacToeStateValueModel::current_player(const IGameState& state) const {
  const TicTacToeState* s = as_ttt(state);
  return s ? s->current_player : 0;
}

bool TicTacToeStateValueModel::is_terminal(const IGameState& state) const {
  const TicTacToeState* s = as_ttt(state);
  return s ? s->terminal : true;
}

float TicTacToeStateValueModel::terminal_value_for_player(const IGameState& state, int perspective_player) const {
  const TicTacToeState* s = as_ttt(state);
  if (!s || !s->terminal) return 0.0f;
  if (s->winner < 0) return 0.0f;
  return s->winner == perspective_player ? 1.0f : -1.0f;
}

}  // namespace board_ai::tictactoe

