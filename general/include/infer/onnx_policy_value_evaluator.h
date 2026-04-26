#pragma once

#include <string>
#include <vector>

#include "infer/feature_encoder.h"
#include "search/net_mcts.h"

namespace board_ai::infer {

struct OnnxEvaluatorConfig {
  int intra_threads = 1;
  int inter_threads = 1;
};

class OnnxPolicyValueEvaluator final : public search::IPolicyValueEvaluator {
 public:
  OnnxPolicyValueEvaluator(
      std::string model_path,
      const IFeatureEncoder* encoder,
      OnnxEvaluatorConfig cfg = {});
  ~OnnxPolicyValueEvaluator();

  OnnxPolicyValueEvaluator(const OnnxPolicyValueEvaluator&) = delete;
  OnnxPolicyValueEvaluator& operator=(const OnnxPolicyValueEvaluator&) = delete;

  bool is_ready() const { return ready_; }
  const std::string& last_error() const { return last_error_; }

  bool evaluate(
      const IGameState& state,
      int perspective_player,
      const std::vector<ActionId>& legal_actions,
      std::vector<float>* priors,
      float* value) const override;

 private:
  std::string model_path_;
  const IFeatureEncoder* encoder_ = nullptr;
  OnnxEvaluatorConfig cfg_{};
  bool ready_ = false;
  std::string last_error_{};

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  struct Impl;
  Impl* impl_ = nullptr;
#endif
};

}  // namespace board_ai::infer
