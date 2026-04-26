#include "infer/onnx_policy_value_evaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
#include <onnxruntime_c_api.h>
#endif

namespace board_ai::infer {

namespace {

static void masked_softmax(
    const std::vector<float>& logits,
    const std::vector<float>& mask,
    const std::vector<ActionId>& legal_actions,
    const IFeatureEncoder* encoder,
    int perspective_player,
    std::vector<float>* out) {
  out->assign(legal_actions.size(), 0.0f);
  if (legal_actions.empty() || logits.empty() || mask.empty()) {
    return;
  }
  float max_logit = -std::numeric_limits<float>::infinity();
  for (ActionId a : legal_actions) {
    const ActionId canonical = encoder ? encoder->canonicalize_action(a, perspective_player) : a;
    const int idx = static_cast<int>(canonical);
    if (idx >= 0 && idx < static_cast<int>(logits.size()) && idx < static_cast<int>(mask.size()) && mask[idx] > 0.0f) {
      max_logit = std::max(max_logit, logits[idx]);
    }
  }
  if (!std::isfinite(max_logit)) {
    const float u = 1.0f / static_cast<float>(legal_actions.size());
    std::fill(out->begin(), out->end(), u);
    return;
  }
  float z = 0.0f;
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    const ActionId canonical = encoder ? encoder->canonicalize_action(legal_actions[i], perspective_player)
                                       : legal_actions[i];
    const int idx = static_cast<int>(canonical);
    if (idx >= 0 && idx < static_cast<int>(logits.size()) && idx < static_cast<int>(mask.size()) && mask[idx] > 0.0f) {
      const float e = std::exp(logits[idx] - max_logit);
      (*out)[i] = e;
      z += e;
    }
  }
  if (z <= 1e-9f) {
    const float u = 1.0f / static_cast<float>(legal_actions.size());
    std::fill(out->begin(), out->end(), u);
    return;
  }
  for (float& x : *out) {
    x /= z;
  }
}

}  // namespace

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
namespace {
const OrtApi& ort_api() noexcept {
  static const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  return *api;
}

void throw_on_ort_error(OrtStatus* status) {
  if (!status) {
    return;
  }
  const char* message = ort_api().GetErrorMessage(status);
  std::string error = message ? message : "onnxruntime error";
  ort_api().ReleaseStatus(status);
  throw std::runtime_error(error);
}

struct OrtEnvDeleter {
  void operator()(OrtEnv* ptr) const noexcept {
    if (ptr) {
      ort_api().ReleaseEnv(ptr);
    }
  }
};

struct OrtSessionOptionsDeleter {
  void operator()(OrtSessionOptions* ptr) const noexcept {
    if (ptr) {
      ort_api().ReleaseSessionOptions(ptr);
    }
  }
};

struct OrtSessionDeleter {
  void operator()(OrtSession* ptr) const noexcept {
    if (ptr) {
      ort_api().ReleaseSession(ptr);
    }
  }
};

struct OrtMemoryInfoDeleter {
  void operator()(OrtMemoryInfo* ptr) const noexcept {
    if (ptr) {
      ort_api().ReleaseMemoryInfo(ptr);
    }
  }
};

struct OrtValueDeleter {
  void operator()(OrtValue* ptr) const noexcept {
    if (ptr) {
      ort_api().ReleaseValue(ptr);
    }
  }
};

struct OrtTensorShapeDeleter {
  void operator()(OrtTensorTypeAndShapeInfo* ptr) const noexcept {
    if (ptr) {
      ort_api().ReleaseTensorTypeAndShapeInfo(ptr);
    }
  }
};

using OrtEnvPtr = std::unique_ptr<OrtEnv, OrtEnvDeleter>;
using OrtSessionOptionsPtr = std::unique_ptr<OrtSessionOptions, OrtSessionOptionsDeleter>;
using OrtSessionPtr = std::unique_ptr<OrtSession, OrtSessionDeleter>;
using OrtMemoryInfoPtr = std::unique_ptr<OrtMemoryInfo, OrtMemoryInfoDeleter>;
using OrtValuePtr = std::unique_ptr<OrtValue, OrtValueDeleter>;
using OrtTensorShapePtr = std::unique_ptr<OrtTensorTypeAndShapeInfo, OrtTensorShapeDeleter>;

std::string get_session_name(OrtSession* session, size_t index, OrtAllocator* allocator, bool input) {
  char* raw_name = nullptr;
  if (input) {
    throw_on_ort_error(ort_api().SessionGetInputName(session, index, allocator, &raw_name));
  } else {
    throw_on_ort_error(ort_api().SessionGetOutputName(session, index, allocator, &raw_name));
  }
  std::string name = raw_name ? raw_name : "";
  if (raw_name) {
    throw_on_ort_error(ort_api().AllocatorFree(allocator, raw_name));
  }
  return name;
}

struct CachedOrtSessionBundle {
  OrtEnvPtr env{};
  OrtSessionOptionsPtr opts{};
  OrtSessionPtr session{};
  std::string input_name;
  std::string policy_output_name;
  std::string value_output_name;
};
}  // namespace

struct OnnxPolicyValueEvaluator::Impl {
  std::shared_ptr<CachedOrtSessionBundle> bundle;
};

namespace {
std::mutex g_onnx_session_cache_mu;
std::unordered_map<std::string, std::shared_ptr<CachedOrtSessionBundle>> g_onnx_session_cache;
constexpr std::size_t kMaxCachedOrtSessions = 8;

std::string make_session_cache_key(const std::string& model_path, const OnnxEvaluatorConfig& cfg) {
  std::string model_fingerprint = "missing";
  try {
    namespace fs = std::filesystem;
    const fs::path p(model_path);
    if (fs::exists(p)) {
      const auto file_size = fs::file_size(p);
      const auto write_time = fs::last_write_time(p).time_since_epoch().count();
      model_fingerprint = std::to_string(static_cast<unsigned long long>(file_size)) + "@" +
          std::to_string(static_cast<long long>(write_time));
    }
  } catch (...) {
    model_fingerprint = "stat_error";
  }
  return model_path + "|fp=" + model_fingerprint + "|intra=" + std::to_string(std::max(1, cfg.intra_threads)) +
      "|inter=" + std::to_string(std::max(1, cfg.inter_threads));
}
}  // namespace
#endif

OnnxPolicyValueEvaluator::OnnxPolicyValueEvaluator(
    std::string model_path,
    const IFeatureEncoder* encoder,
    OnnxEvaluatorConfig cfg)
    : model_path_(std::move(model_path)), encoder_(encoder), cfg_(cfg), ready_(false) {
  if (!encoder_) {
    last_error_ = "encoder is null";
    return;
  }

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  try {
    const std::string cache_key = make_session_cache_key(model_path_, cfg_);
    std::shared_ptr<CachedOrtSessionBundle> bundle;
    {
      std::lock_guard<std::mutex> lk(g_onnx_session_cache_mu);
      auto it = g_onnx_session_cache.find(cache_key);
      if (it != g_onnx_session_cache.end()) {
        bundle = it->second;
      }
      if (!bundle) {
        bundle = std::make_shared<CachedOrtSessionBundle>();
        OrtEnv* raw_env = nullptr;
        throw_on_ort_error(ort_api().CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dino_onnx_eval", &raw_env));
        bundle->env.reset(raw_env);

        OrtSessionOptions* raw_opts = nullptr;
        throw_on_ort_error(ort_api().CreateSessionOptions(&raw_opts));
        bundle->opts.reset(raw_opts);
        throw_on_ort_error(ort_api().SetIntraOpNumThreads(bundle->opts.get(), std::max(1, cfg_.intra_threads)));
        throw_on_ort_error(ort_api().SetInterOpNumThreads(bundle->opts.get(), std::max(1, cfg_.inter_threads)));
        throw_on_ort_error(
            ort_api().SetSessionGraphOptimizationLevel(bundle->opts.get(), GraphOptimizationLevel::ORT_ENABLE_EXTENDED));

        OrtSession* raw_session = nullptr;
#if defined(_WIN32)
        std::wstring model_path_w(model_path_.begin(), model_path_.end());
        throw_on_ort_error(ort_api().CreateSession(bundle->env.get(), model_path_w.c_str(), bundle->opts.get(), &raw_session));
#else
        throw_on_ort_error(ort_api().CreateSession(bundle->env.get(), model_path_.c_str(), bundle->opts.get(), &raw_session));
#endif
        bundle->session.reset(raw_session);

        OrtAllocator* allocator = nullptr;
        throw_on_ort_error(ort_api().GetAllocatorWithDefaultOptions(&allocator));
        bundle->input_name = get_session_name(bundle->session.get(), 0, allocator, true);
        bundle->policy_output_name = get_session_name(bundle->session.get(), 0, allocator, false);
        bundle->value_output_name = get_session_name(bundle->session.get(), 1, allocator, false);
        g_onnx_session_cache[cache_key] = bundle;
        while (g_onnx_session_cache.size() > kMaxCachedOrtSessions) {
          auto erase_it = g_onnx_session_cache.begin();
          if (erase_it == g_onnx_session_cache.end()) {
            break;
          }
          if (erase_it->first == cache_key && g_onnx_session_cache.size() > 1) {
            ++erase_it;
            if (erase_it == g_onnx_session_cache.end()) {
              erase_it = g_onnx_session_cache.begin();
            }
          }
          g_onnx_session_cache.erase(erase_it);
        }
      }
    }
    impl_ = new Impl();
    impl_->bundle = std::move(bundle);
    ready_ = true;
  } catch (const std::exception& e) {
    last_error_ = e.what();
    ready_ = false;
  }
#else
  (void)model_path_;
  (void)cfg_;
  // Keep evaluator constructible in environments without ORT.
  // It returns uniform policy + 0 value in evaluate().
  ready_ = false;
  last_error_ = "onnx runtime not enabled at build time";
#endif
}

OnnxPolicyValueEvaluator::~OnnxPolicyValueEvaluator() {
#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  delete impl_;
  impl_ = nullptr;
#endif
}

bool OnnxPolicyValueEvaluator::evaluate(
    const IGameState& state,
    int perspective_player,
    const std::vector<ActionId>& legal_actions,
    std::vector<float>* priors,
    float* value) const {
  if (!priors || !value || !encoder_) {
    return false;
  }
  std::vector<float> features;
  std::vector<float> legal_mask;
  if (!encoder_->encode(state, perspective_player, legal_actions, &features, &legal_mask)) {
    return false;
  }

#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  if (!ready_ || !impl_) {
    priors->assign(legal_actions.size(), 1.0f / static_cast<float>(std::max<size_t>(1, legal_actions.size())));
    *value = 0.0f;
    return true;
  }

  try {
    std::array<int64_t, 2> shape = {1, static_cast<int64_t>(features.size())};
    OrtMemoryInfo* raw_memory_info = nullptr;
    throw_on_ort_error(ort_api().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &raw_memory_info));
    OrtMemoryInfoPtr memory_info(raw_memory_info);

    OrtValue* raw_input = nullptr;
    throw_on_ort_error(ort_api().CreateTensorWithDataAsOrtValue(
        memory_info.get(),
        features.data(),
        features.size() * sizeof(float),
        shape.data(),
        shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &raw_input));
    OrtValuePtr input(raw_input);

    const char* in_names[] = {impl_->bundle->input_name.c_str()};
    const char* out_names[] = {impl_->bundle->policy_output_name.c_str(), impl_->bundle->value_output_name.c_str()};
    const OrtValue* input_values[] = {input.get()};
    OrtValue* raw_outputs[2] = {nullptr, nullptr};
    throw_on_ort_error(ort_api().Run(
        impl_->bundle->session.get(), nullptr, in_names, input_values, 1, out_names, 2, raw_outputs));
    OrtValuePtr policy_output(raw_outputs[0]);
    OrtValuePtr value_output(raw_outputs[1]);

    OrtTensorTypeAndShapeInfo* raw_policy_shape = nullptr;
    throw_on_ort_error(ort_api().GetTensorTypeAndShape(policy_output.get(), &raw_policy_shape));
    OrtTensorShapePtr policy_shape(raw_policy_shape);
    size_t dim_count = 0;
    throw_on_ort_error(ort_api().GetDimensionsCount(policy_shape.get(), &dim_count));
    if (dim_count == 0) {
      return false;
    }
    std::vector<int64_t> policy_dims(dim_count, 0);
    throw_on_ort_error(ort_api().GetDimensions(policy_shape.get(), policy_dims.data(), policy_dims.size()));
    const int64_t policy_len64 = policy_dims.back();
    if (policy_len64 <= 0 || policy_len64 > static_cast<int64_t>(std::numeric_limits<int>::max())) {
      return false;
    }
    void* policy_raw = nullptr;
    throw_on_ort_error(ort_api().GetTensorMutableData(policy_output.get(), &policy_raw));
    const float* policy_ptr = static_cast<const float*>(policy_raw);
    const int policy_len = static_cast<int>(policy_len64);
    std::vector<float> logits(static_cast<size_t>(policy_len), 0.0f);
    std::copy(policy_ptr, policy_ptr + policy_len, logits.begin());

    void* value_raw = nullptr;
    throw_on_ort_error(ort_api().GetTensorMutableData(value_output.get(), &value_raw));
    const float* value_ptr = static_cast<const float*>(value_raw);
    *value = std::max(-1.0f, std::min(1.0f, value_ptr[0]));

    masked_softmax(logits, legal_mask, legal_actions, encoder_, perspective_player, priors);
    return true;
  } catch (...) {
    return false;
  }
#else
  priors->assign(legal_actions.size(), 1.0f / static_cast<float>(std::max<size_t>(1, legal_actions.size())));
  *value = 0.0f;
  return true;
#endif
}

}  // namespace board_ai::infer
