#pragma once

#include <cstdint>

namespace board_ai {

using ActionId = std::int32_t;
using StateHash64 = std::uint64_t;

struct UndoToken {
  bool is_snapshot = false;
  std::uint32_t actor = 0;
  std::uint32_t round_before = 0;
  std::uint32_t undo_depth = 0;
};

}  // namespace board_ai
