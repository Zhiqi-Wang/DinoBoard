#include "azul_net_adapter.h"

#include <algorithm>
#include <array>

namespace board_ai::azul {

namespace {
constexpr int kFirstPlayerToken = -2;

const AzulState* as_azul(const IGameState& s) {
  return dynamic_cast<const AzulState*>(&s);
}

void append_wall_features(const PlayerState& p, std::vector<float>* out) {
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kColors; ++c) {
      out->push_back(((p.wall_mask[r] >> c) & 1U) ? 1.0f : 0.0f);
    }
  }
}

void append_pattern_features(const PlayerState& p, std::vector<float>* out) {
  for (int row = 0; row < kRows; ++row) {
    std::array<float, kColors> counts{};
    if (p.line_color[row] >= 0) {
      counts[static_cast<size_t>(p.line_color[row])] = static_cast<float>(p.line_len[row]);
    }
    const float cap = static_cast<float>(row + 1);
    for (float v : counts) {
      out->push_back(v / cap);
    }
    out->push_back(static_cast<float>(p.line_len[row]) / cap);
  }
}

void append_floor_features(const PlayerState& p, std::vector<float>* out) {
  std::array<float, 6> counts{};
  for (int i = 0; i < p.floor_count && i < 7; ++i) {
    const int t = p.floor[i];
    if (t >= 0 && t < kColors) {
      counts[static_cast<size_t>(t)] += 1.0f;
    } else if (t == kFirstPlayerToken) {
      counts[5] = 1.0f;
    }
  }
  for (float& v : counts) {
    v /= 7.0f;
    out->push_back(v);
  }
}

}  // namespace

bool AzulFeatureEncoder::encode(
    const IGameState& state,
    int perspective_player,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* features,
    std::vector<float>* legal_mask) const {
  const AzulState* s = as_azul(state);
  if (!s || !features || !legal_mask || perspective_player < 0 || perspective_player >= kPlayers) {
    return false;
  }
  const int opp = 1 - perspective_player;
  const auto& me = s->players[perspective_player];
  const auto& enemy = s->players[opp];

  features->clear();
  features->reserve(static_cast<size_t>(feature_dim()));

  append_wall_features(me, features);
  append_wall_features(enemy, features);
  append_pattern_features(me, features);
  append_pattern_features(enemy, features);
  append_floor_features(me, features);
  append_floor_features(enemy, features);

  features->push_back(std::min(me.score, 200) / 200.0f);
  features->push_back(std::min(enemy.score, 200) / 200.0f);

  for (int f = 0; f < kFactories; ++f) {
    for (int c = 0; c < kColors; ++c) {
      features->push_back(static_cast<float>(s->factories[f][c]) / 4.0f);
    }
  }
  for (int c = 0; c < kColors; ++c) {
    features->push_back(static_cast<float>(s->center[c]) / 20.0f);
  }

  std::array<int, kColors> bag_counts{};
  bag_counts.fill(0);
  for (std::int8_t t : s->bag) {
    if (t >= 0 && t < kColors) {
      bag_counts[static_cast<size_t>(t)] += 1;
    }
  }
  for (int c = 0; c < kColors; ++c) {
    features->push_back(static_cast<float>(bag_counts[static_cast<size_t>(c)]) / 20.0f);
  }
  features->push_back(s->first_player_token_in_center ? 1.0f : 0.0f);
  features->push_back(s->current_player == perspective_player ? 1.0f : 0.0f);
  features->push_back(std::min(s->round_index, 20) / 20.0f);
  features->push_back(static_cast<float>(s->bag.size()) / 100.0f);

  legal_mask->assign(static_cast<size_t>(kActionSpace), 0.0f);
  for (ActionId a : legal_actions) {
    if (a >= 0 && a < kActionSpace) {
      (*legal_mask)[static_cast<size_t>(a)] = 1.0f;
    }
  }

  return static_cast<int>(features->size()) == feature_dim();
}

int AzulStateValueModel::current_player(const IGameState& state) const {
  const AzulState* s = as_azul(state);
  return s ? s->current_player : 0;
}

bool AzulStateValueModel::is_terminal(const IGameState& state) const {
  const AzulState* s = as_azul(state);
  return s ? s->terminal : true;
}

float AzulStateValueModel::terminal_value_for_player(const IGameState& state, int perspective_player) const {
  const AzulState* s = as_azul(state);
  if (!s || !s->terminal) {
    return 0.0f;
  }
  if (s->shared_victory) {
    return 0.0f;
  }
  if (s->winner >= 0) {
    return s->winner == perspective_player ? 1.0f : -1.0f;
  }
  if (s->scores[0] == s->scores[1]) {
    return 0.0f;
  }
  const int best = (s->scores[0] > s->scores[1]) ? 0 : 1;
  return best == perspective_player ? 1.0f : -1.0f;
}

}  // namespace board_ai::azul
