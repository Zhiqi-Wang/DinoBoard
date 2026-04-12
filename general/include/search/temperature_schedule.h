#pragma once

#include <algorithm>

namespace board_ai {
namespace search {

struct TemperatureSchedule {
  bool enabled = false;
  bool has_initial = false;
  bool has_final = false;
  double initial = 0.0;
  double final = 0.0;
  int decay_plies = 0;
};

inline double resolve_linear_temperature(
    const TemperatureSchedule& schedule,
    double base_temperature,
    int ply_index) {
  const double base = std::max(0.0, base_temperature);
  if (!schedule.enabled) return base;

  const double t0 = schedule.has_initial ? std::max(0.0, schedule.initial) : base;
  const double t1 = schedule.has_final ? std::max(0.0, schedule.final) : base;
  const int decay = std::max(0, schedule.decay_plies);
  if (decay <= 0) return t1;

  const double clipped_ply = static_cast<double>(std::max(0, ply_index));
  const double progress = std::min(1.0, clipped_ply / static_cast<double>(decay));
  return t0 + (t1 - t0) * progress;
}

}  // namespace search
}  // namespace board_ai
