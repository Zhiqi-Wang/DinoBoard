#include "splendor_net_adapter.h"

#include <algorithm>

namespace board_ai::splendor {

namespace {

const SplendorState* as_state(const IGameState& s) { return dynamic_cast<const SplendorState*>(&s); }

void encode_card(const SplendorCard* card, std::vector<float>* out) {
  if (!card) {
    out->insert(out->end(), 13, 0.0f);
    return;
  }
  out->push_back(1.0f);
  out->push_back(static_cast<float>(card->tier) / 3.0f);
  out->push_back(static_cast<float>(card->points) / 5.0f);
  for (int c = 0; c < kColorCount; ++c) {
    out->push_back(card->bonus == c ? 1.0f : 0.0f);
  }
  for (int c = 0; c < kColorCount; ++c) {
    out->push_back(static_cast<float>(card->cost[static_cast<size_t>(c)]) / 7.0f);
  }
}

void encode_hidden_reserved_placeholder(std::vector<float>* out) {
  // "occupied but hidden": exists=1 and remaining fields zeros.
  out->push_back(1.0f);
  out->insert(out->end(), 12, 0.0f);
}

}  // namespace

bool SplendorFeatureEncoder::encode(
    const IGameState& state,
    int perspective_player,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* features,
    std::vector<float>* legal_mask) const {
  const SplendorState* s = as_state(state);
  if (!s || !features || !legal_mask || perspective_player < 0 || perspective_player >= kPlayers) return false;
  const SplendorData& d = s->persistent.data();
  const int me = perspective_player;
  const int op = 1 - me;
  const auto& cards = splendor_card_pool();
  const auto& nobles = splendor_nobles();

  features->clear();
  features->reserve(kFeatureDim);

  for (int i = 0; i < kTokenTypes; ++i) {
    const float denom = (i == 5) ? 5.0f : 4.0f;
    features->push_back(static_cast<float>(d.bank[static_cast<size_t>(i)]) / denom);
  }

  for (int pid : {me, op}) {
    for (int i = 0; i < kTokenTypes; ++i) {
      features->push_back(static_cast<float>(d.player_gems[static_cast<size_t>(pid)][static_cast<size_t>(i)]) / 10.0f);
    }
    for (int i = 0; i < kColorCount; ++i) {
      features->push_back(static_cast<float>(d.player_bonuses[static_cast<size_t>(pid)][static_cast<size_t>(i)]) / 7.0f);
    }
    features->push_back(static_cast<float>(d.player_points[static_cast<size_t>(pid)]) / 20.0f);
    features->push_back(static_cast<float>(d.reserved_size[static_cast<size_t>(pid)]) / 3.0f);
    features->push_back(static_cast<float>(d.player_cards_count[static_cast<size_t>(pid)]) / 20.0f);
    features->push_back(static_cast<float>(d.player_nobles_count[static_cast<size_t>(pid)]) / 3.0f);
  }

  for (int i = 0; i < 3; ++i) {
    if (i < d.nobles_size) {
      const int nid = d.nobles[static_cast<size_t>(i)];
      if (nid >= 0 && nid < static_cast<int>(nobles.size())) {
        for (int c = 0; c < kColorCount; ++c) {
          features->push_back(static_cast<float>(nobles[static_cast<size_t>(nid)][static_cast<size_t>(c)]) / 4.0f);
        }
        features->push_back(1.0f);
      } else {
        features->insert(features->end(), 6, 0.0f);
      }
    } else {
      features->insert(features->end(), 6, 0.0f);
    }
  }

  for (int tier = 0; tier < 3; ++tier) {
    for (int slot = 0; slot < 4; ++slot) {
      const bool exists = slot < d.tableau_size[static_cast<size_t>(tier)];
      if (!exists) {
        features->insert(features->end(), 13, 0.0f);
        continue;
      }
      const int cid = d.tableau[static_cast<size_t>(tier)][static_cast<size_t>(slot)];
      if (cid < 0 || cid >= static_cast<int>(cards.size())) {
        features->insert(features->end(), 13, 0.0f);
      } else {
        encode_card(&cards[static_cast<size_t>(cid)], features);
      }
    }
  }

  // my reserved cards: always fully known
  for (int slot = 0; slot < 3; ++slot) {
    const bool exists = slot < d.reserved_size[static_cast<size_t>(me)];
    if (!exists) {
      features->insert(features->end(), 13, 0.0f);
      continue;
    }
    const int cid = d.reserved[static_cast<size_t>(me)][static_cast<size_t>(slot)];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) {
      features->insert(features->end(), 13, 0.0f);
    } else {
      encode_card(&cards[static_cast<size_t>(cid)], features);
    }
  }

  // opponent reserved cards: faceup-reserved => real card; deck-reserved => hidden placeholder
  for (int slot = 0; slot < 3; ++slot) {
    const bool exists = slot < d.reserved_size[static_cast<size_t>(op)];
    if (!exists) {
      features->insert(features->end(), 13, 0.0f);
      continue;
    }
    const bool visible = d.reserved_visible[static_cast<size_t>(op)][static_cast<size_t>(slot)] != 0;
    if (!visible) {
      encode_hidden_reserved_placeholder(features);
      continue;
    }
    const int cid = d.reserved[static_cast<size_t>(op)][static_cast<size_t>(slot)];
    if (cid < 0 || cid >= static_cast<int>(cards.size())) {
      features->insert(features->end(), 13, 0.0f);
    } else {
      encode_card(&cards[static_cast<size_t>(cid)], features);
    }
  }

  features->push_back(static_cast<float>(std::min(d.plies, kMaxPlies)) / static_cast<float>(kMaxPlies));
  features->push_back(d.current_player == me ? 1.0f : 0.0f);
  features->push_back(static_cast<float>(std::max(0, d.pending_returns)) / 5.0f);
  const SplendorTurnStage stage = static_cast<SplendorTurnStage>(d.stage);
  features->push_back(stage == SplendorTurnStage::kNormal ? 1.0f : 0.0f);
  features->push_back(stage == SplendorTurnStage::kReturnTokens ? 1.0f : 0.0f);
  features->push_back(stage == SplendorTurnStage::kChooseNoble ? 1.0f : 0.0f);

  if (features->size() < static_cast<size_t>(kFeatureDim)) {
    features->insert(features->end(), static_cast<size_t>(kFeatureDim) - features->size(), 0.0f);
  } else if (features->size() > static_cast<size_t>(kFeatureDim)) {
    features->resize(static_cast<size_t>(kFeatureDim));
  }

  legal_mask->assign(static_cast<size_t>(kActionSpace), 0.0f);
  for (ActionId a : legal_actions) {
    if (a >= 0 && a < kActionSpace) {
      (*legal_mask)[static_cast<size_t>(a)] = 1.0f;
    }
  }
  return static_cast<int>(features->size()) == kFeatureDim;
}

int SplendorStateValueModel::current_player(const IGameState& state) const {
  const SplendorState* s = as_state(state);
  return s ? s->persistent.data().current_player : 0;
}

bool SplendorStateValueModel::is_terminal(const IGameState& state) const {
  const SplendorState* s = as_state(state);
  return s ? s->persistent.data().terminal : true;
}

float SplendorStateValueModel::terminal_value_for_player(const IGameState& state, int perspective_player) const {
  const SplendorState* s = as_state(state);
  if (!s) return 0.0f;
  const auto& d = s->persistent.data();
  if (!d.terminal || d.winner < 0) return 0.0f;
  return d.winner == perspective_player ? 1.0f : -1.0f;
}

}  // namespace board_ai::splendor

