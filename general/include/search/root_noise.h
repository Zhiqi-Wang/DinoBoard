#pragma once

#include <algorithm>

namespace board_ai {
namespace search {

struct ResolvedRootDirichletNoise {
  bool enabled = false;
  float alpha = 0.0f;
  float epsilon = 0.0f;
};

inline ResolvedRootDirichletNoise resolve_root_dirichlet_noise(
    double alpha,
    double epsilon,
    int on_first_n_plies,
    int ply_index) {
  const float a = static_cast<float>(std::max(0.0, alpha));
  const float e = static_cast<float>(std::max(0.0, std::min(1.0, epsilon)));
  const bool in_window = (on_first_n_plies <= 0) || (ply_index < std::max(0, on_first_n_plies));
  ResolvedRootDirichletNoise out{};
  out.enabled = (a > 0.0f) && (e > 0.0f) && in_window;
  out.alpha = out.enabled ? a : 0.0f;
  out.epsilon = out.enabled ? e : 0.0f;
  return out;
}

}  // namespace search
}  // namespace board_ai
