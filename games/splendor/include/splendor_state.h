#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/game_interfaces.h"

namespace board_ai::splendor {

constexpr int kPlayers = 2;
constexpr int kColorCount = 5;
constexpr int kTokenTypes = 6;
constexpr int kTargetPoints = 15;
constexpr int kMaxPlies = 160;

constexpr int kBuyFaceupOffset = 0;
constexpr int kBuyFaceupCount = 12;
constexpr int kReserveFaceupOffset = kBuyFaceupOffset + kBuyFaceupCount;
constexpr int kReserveFaceupCount = 12;
constexpr int kReserveDeckOffset = kReserveFaceupOffset + kReserveFaceupCount;
constexpr int kReserveDeckCount = 3;
constexpr int kBuyReservedOffset = kReserveDeckOffset + kReserveDeckCount;
constexpr int kBuyReservedCount = 3;
constexpr int kTakeThreeOffset = kBuyReservedOffset + kBuyReservedCount;
constexpr int kTakeThreeCount = 10;
constexpr int kTakeTwoDifferentOffset = kTakeThreeOffset + kTakeThreeCount;
constexpr int kTakeTwoDifferentCount = 10;
constexpr int kTakeOneOffset = kTakeTwoDifferentOffset + kTakeTwoDifferentCount;
constexpr int kTakeOneCount = 5;
constexpr int kTakeTwoSameOffset = kTakeOneOffset + kTakeOneCount;
constexpr int kTakeTwoSameCount = 5;
constexpr int kChooseNobleOffset = kTakeTwoSameOffset + kTakeTwoSameCount;
constexpr int kChooseNobleCount = 3;
constexpr int kReturnTokenOffset = kChooseNobleOffset + kChooseNobleCount;
constexpr int kReturnTokenCount = 6;
constexpr int kPassAction = kReturnTokenOffset + kReturnTokenCount;
constexpr int kActionSpace = kPassAction + 1;

constexpr int kFeatureDim = 294;

enum class SplendorTurnStage : std::int8_t {
  kNormal = 0,
  kReturnTokens = 1,
  kChooseNoble = 2,
};

struct SplendorCard {
  std::int8_t tier = 1;
  std::int8_t bonus = 0;
  std::int8_t points = 0;
  std::array<std::int8_t, kColorCount> cost{};
};

struct SplendorData {
  int current_player = 0;
  int plies = 0;
  int final_round_remaining = -1;  // -1 means not triggered.
  std::int8_t stage = static_cast<std::int8_t>(SplendorTurnStage::kNormal);
  int pending_returns = 0;
  std::array<std::int8_t, 3> pending_noble_slots{{-1, -1, -1}};
  std::int8_t pending_nobles_size = 0;
  std::uint64_t draw_nonce = 0;
  int winner = -1;
  bool terminal = false;
  bool shared_victory = false;
  std::array<int, kPlayers> scores{0, 0};

  std::array<std::int8_t, kTokenTypes> bank{};
  std::array<std::array<std::int8_t, kTokenTypes>, kPlayers> player_gems{};
  std::array<std::array<std::int8_t, kColorCount>, kPlayers> player_bonuses{};
  std::array<std::int16_t, kPlayers> player_points{};
  std::array<std::int16_t, kPlayers> player_cards_count{};
  std::array<std::int16_t, kPlayers> player_nobles_count{};

  // Reserve area: card ids; -1 means empty.
  std::array<std::array<std::int16_t, 3>, kPlayers> reserved{};
  // Visibility from opponent perspective: 1 means faceup-reserved(visible), 0 means deck-top reserved(hidden).
  std::array<std::array<std::int8_t, 3>, kPlayers> reserved_visible{};
  std::array<std::int8_t, kPlayers> reserved_size{};

  // Deck and tableau use card ids from global card pool.
  std::array<std::vector<std::int16_t>, 3> decks{};
  std::array<std::array<std::int16_t, 4>, 3> tableau{};
  std::array<std::int8_t, 3> tableau_size{};

  // Noble ids from global noble table.
  std::array<std::int16_t, 3> nobles{};
  std::int8_t nobles_size = 0;
};

struct SplendorPersistentNode {
  std::shared_ptr<const SplendorPersistentNode> parent{};
  ActionId action_from_parent = -1;
  mutable std::shared_ptr<const SplendorData> materialized{};
};

class SplendorPersistentState {
 public:
  SplendorPersistentState() = default;
  explicit SplendorPersistentState(std::shared_ptr<const SplendorPersistentNode> node);

  static SplendorPersistentState root_from_seed(std::uint64_t seed);

  bool valid() const { return static_cast<bool>(node_); }
  const SplendorData& data() const;
  SplendorPersistentState advance(ActionId action) const;
  StateHash64 state_hash(bool include_hidden_rng, std::uint64_t rng_salt) const;

 private:
  std::shared_ptr<const SplendorPersistentNode> node_{};
};

struct SplendorState final : public IGameState {
  SplendorPersistentState persistent{};
  std::uint64_t rng_salt = 0;
  std::vector<SplendorPersistentState> undo_stack{};

  SplendorState();
  void reset_with_seed(std::uint64_t seed);

  std::unique_ptr<IGameState> clone_state() const override;
  StateHash64 state_hash(bool include_hidden_rng) const override;
};

const std::vector<SplendorCard>& splendor_card_pool();
const std::array<std::array<std::int8_t, kColorCount>, 12>& splendor_nobles();

}  // namespace board_ai::splendor
