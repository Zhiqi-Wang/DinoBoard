#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "infer/onnx_policy_value_evaluator.h"
#include "quoridor_net_adapter.h"
#include "quoridor_rules.h"
#include "quoridor_state.h"
#include "search/net_mcts.h"
#include "search/root_noise.h"
#include "search/search_options_common.h"
#include "search/temperature_schedule.h"

namespace {

class SplitMix64Engine {
 public:
  using result_type = std::uint64_t;

  explicit SplitMix64Engine(std::uint64_t seed) : state_(seed) {}

  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return UINT64_MAX; }

  result_type operator()() {
    std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
  }

 private:
  std::uint64_t state_;
};

using board_ai::ActionId;
using board_ai::quoridor::QuoridorFeatureEncoder;
using board_ai::quoridor::QuoridorRules;
using board_ai::quoridor::QuoridorStateValueModel;
using board_ai::quoridor::QuoridorState;
using board_ai::quoridor::kActionSpace;
using board_ai::quoridor::kBoardSize;
using board_ai::quoridor::kFeatureDim;
using board_ai::quoridor::kPlayers;
using board_ai::quoridor::kWallGrid;
using board_ai::search::select_action_from_visits;

struct QuoridorSearchOptions {
  board_ai::search::CommonSearchOptions common{};
  double heuristic_random_action_prob = 0.0;
  QuoridorSearchOptions() {
    common.max_search_depth = 128;
    common.max_episode_plies = 220;
  }
};

struct EventRec {
  int ply = 0;
  int actor = 0;
  int action_id = 0;
  bool forced = false;
};

struct Session {
  QuoridorState state;
  QuoridorRules rules;
  int human_player = 0;
  std::vector<QuoridorState> timeline_states{};
  std::vector<EventRec> timeline_events{};
  int cursor = 0;

  Session(std::uint64_t seed, int human) : state(), rules(), human_player(human), cursor(0) {
    state.reset_with_seed(seed);
    timeline_states.push_back(state);
  }
};

struct AiDecision {
  ActionId action = -1;
  bool has_mcts_best_action_value = false;
  double mcts_best_action_value = 0.0;
  std::vector<ActionId> root_actions;
  std::vector<int> root_visits;
};

static std::unordered_map<std::int64_t, std::unique_ptr<Session>> g_sessions;
static std::int64_t g_next_handle = 1;

bool parse_quoridor_search_options(PyObject* obj, QuoridorSearchOptions* out) {
  if (!out) return false;
  if (!board_ai::search::parse_common_search_options(obj, &out->common)) return false;
  double prob = out->heuristic_random_action_prob;
  if (!board_ai::search::parse_probability_option(
          obj,
          "heuristic_random_action_prob",
          &prob,
          "heuristic_random_prob")) {
    return false;
  }
  out->heuristic_random_action_prob = prob;
  return true;
}

Session* get_session_from_handle(PyObject* handle_obj) {
  const long long handle = PyLong_AsLongLong(handle_obj);
  if (PyErr_Occurred()) return nullptr;
  auto it = g_sessions.find(static_cast<std::int64_t>(handle));
  if (it == g_sessions.end()) {
    PyErr_SetString(PyExc_KeyError, "session handle not found");
    return nullptr;
  }
  return it->second.get();
}

std::vector<double> visit_probs(const std::vector<ActionId>& actions, const std::vector<int>& visits) {
  std::vector<double> out(actions.size(), 0.0);
  double sum = 0.0;
  for (int v : visits) sum += static_cast<double>(std::max(0, v));
  if (sum <= 1e-12) {
    if (!out.empty()) {
      const double u = 1.0 / static_cast<double>(out.size());
      for (double& p : out) p = u;
    }
    return out;
  }
  for (size_t i = 0; i < out.size() && i < visits.size(); ++i) {
    out[i] = static_cast<double>(std::max(0, visits[i])) / sum;
  }
  return out;
}

void append_action_metadata(PyObject* d, ActionId action_id) {
  using namespace board_ai::quoridor;
  if (is_move_action(action_id)) {
    PyDict_SetItemString(d, "type", PyUnicode_FromString("move"));
    PyDict_SetItemString(d, "row", PyLong_FromLong(decode_move_row(action_id)));
    PyDict_SetItemString(d, "col", PyLong_FromLong(decode_move_col(action_id)));
    return;
  }
  if (is_hwall_action(action_id)) {
    PyDict_SetItemString(d, "type", PyUnicode_FromString("hwall"));
    PyDict_SetItemString(d, "row", PyLong_FromLong(decode_hwall_row(action_id)));
    PyDict_SetItemString(d, "col", PyLong_FromLong(decode_hwall_col(action_id)));
    return;
  }
  if (is_vwall_action(action_id)) {
    PyDict_SetItemString(d, "type", PyUnicode_FromString("vwall"));
    PyDict_SetItemString(d, "row", PyLong_FromLong(decode_vwall_row(action_id)));
    PyDict_SetItemString(d, "col", PyLong_FromLong(decode_vwall_col(action_id)));
    return;
  }
  PyDict_SetItemString(d, "type", PyUnicode_FromString("unknown"));
}

PyObject* build_public_state(const Session& s) {
  PyObject* root = PyDict_New();
  PyObject* common = PyDict_New();
  PyObject* game = PyDict_New();
  if (!root || !common || !game) {
    Py_XDECREF(root);
    Py_XDECREF(common);
    Py_XDECREF(game);
    return nullptr;
  }

  PyDict_SetItemString(common, "current_player", PyLong_FromLong(s.state.current_player));
  PyDict_SetItemString(common, "round_index", PyLong_FromLong(s.state.move_count / 2));
  PyObject* scores = PyList_New(2);
  PyList_SET_ITEM(scores, 0, PyLong_FromLong(s.state.scores[0]));
  PyList_SET_ITEM(scores, 1, PyLong_FromLong(s.state.scores[1]));
  PyDict_SetItemString(common, "scores", scores);
  Py_DECREF(scores);
  PyDict_SetItemString(common, "is_terminal", s.state.terminal ? Py_True : Py_False);
  if (!s.state.terminal || s.state.winner < 0) {
    Py_INCREF(Py_None);
    PyDict_SetItemString(common, "winner", Py_None);
  } else {
    PyDict_SetItemString(common, "winner", PyLong_FromLong(s.state.winner));
  }
  PyDict_SetItemString(common, "shared_victory", Py_False);

  PyDict_SetItemString(game, "board_size", PyLong_FromLong(kBoardSize));
  PyDict_SetItemString(game, "move_count", PyLong_FromLong(s.state.move_count));
  PyObject* goal_rows = PyList_New(2);
  PyList_SET_ITEM(goal_rows, 0, PyLong_FromLong(kBoardSize - 1));
  PyList_SET_ITEM(goal_rows, 1, PyLong_FromLong(0));
  PyDict_SetItemString(game, "goal_rows", goal_rows);
  Py_DECREF(goal_rows);

  PyObject* pawns = PyList_New(kPlayers);
  for (int p = 0; p < kPlayers; ++p) {
    PyObject* item = PyDict_New();
    PyDict_SetItemString(item, "player", PyLong_FromLong(p));
    PyDict_SetItemString(item, "row", PyLong_FromLong(s.state.pawn_row[static_cast<size_t>(p)]));
    PyDict_SetItemString(item, "col", PyLong_FromLong(s.state.pawn_col[static_cast<size_t>(p)]));
    PyList_SET_ITEM(pawns, p, item);
  }
  PyDict_SetItemString(game, "pawns", pawns);
  Py_DECREF(pawns);

  PyObject* walls_remaining = PyList_New(kPlayers);
  for (int p = 0; p < kPlayers; ++p) {
    PyList_SET_ITEM(
        walls_remaining,
        p,
        PyLong_FromLong(static_cast<long>(s.state.walls_remaining[static_cast<size_t>(p)])));
  }
  PyDict_SetItemString(game, "walls_remaining", walls_remaining);
  Py_DECREF(walls_remaining);

  PyObject* h_walls = PyList_New(0);
  PyObject* v_walls = PyList_New(0);
  for (int r = 0; r < kWallGrid; ++r) {
    for (int c = 0; c < kWallGrid; ++c) {
      const int idx = board_ai::quoridor::wall_index(r, c);
      if (s.state.h_walls[static_cast<size_t>(idx)] != 0) {
        PyObject* item = PyDict_New();
        PyDict_SetItemString(item, "row", PyLong_FromLong(r));
        PyDict_SetItemString(item, "col", PyLong_FromLong(c));
        PyDict_SetItemString(item, "action_id", PyLong_FromLong(board_ai::quoridor::encode_hwall_action(r, c)));
        PyList_Append(h_walls, item);
        Py_DECREF(item);
      }
      if (s.state.v_walls[static_cast<size_t>(idx)] != 0) {
        PyObject* item = PyDict_New();
        PyDict_SetItemString(item, "row", PyLong_FromLong(r));
        PyDict_SetItemString(item, "col", PyLong_FromLong(c));
        PyDict_SetItemString(item, "action_id", PyLong_FromLong(board_ai::quoridor::encode_vwall_action(r, c)));
        PyList_Append(v_walls, item);
        Py_DECREF(item);
      }
    }
  }
  PyDict_SetItemString(game, "horizontal_walls", h_walls);
  PyDict_SetItemString(game, "vertical_walls", v_walls);
  Py_DECREF(h_walls);
  Py_DECREF(v_walls);

  PyDict_SetItemString(root, "common", common);
  PyDict_SetItemString(root, "game", game);
  Py_DECREF(common);
  Py_DECREF(game);
  return root;
}

PyObject* build_payload(Session& s) {
  PyObject* d = PyDict_New();
  PyDict_SetItemString(d, "state_version", PyLong_FromLong(s.cursor));
  PyObject* pub = build_public_state(s);
  PyDict_SetItemString(d, "public_state", pub);
  Py_DECREF(pub);
  return d;
}

void truncate_timeline_to_cursor(Session& s) {
  if (s.cursor < static_cast<int>(s.timeline_events.size())) {
    s.timeline_events.resize(static_cast<size_t>(s.cursor));
  }
  if (s.cursor + 1 < static_cast<int>(s.timeline_states.size())) {
    s.timeline_states.resize(static_cast<size_t>(s.cursor + 1));
  }
}

PyObject* apply_action_common(Session& s, int action_id, bool forced) {
  truncate_timeline_to_cursor(s);
  const int actor = s.state.current_player;
  if (!s.rules.validate_action(s.state, action_id)) {
    Py_RETURN_NONE;
  }
  s.rules.do_action_fast(s.state, action_id);
  s.timeline_states.push_back(s.state);
  s.cursor += 1;
  EventRec ev;
  ev.ply = s.cursor;
  ev.actor = actor;
  ev.action_id = action_id;
  ev.forced = forced;
  s.timeline_events.push_back(ev);

  PyObject* out = PyDict_New();
  PyDict_SetItemString(out, "ply", PyLong_FromLong(ev.ply));
  PyDict_SetItemString(out, "actor", PyLong_FromLong(ev.actor));
  PyDict_SetItemString(out, "action_id", PyLong_FromLong(ev.action_id));
  PyDict_SetItemString(out, "forced", ev.forced ? Py_True : Py_False);
  append_action_metadata(out, action_id);
  return out;
}

double heuristic_score_after_action(const QuoridorState& before, const QuoridorState& after, ActionId action) {
  const int me = before.current_player;
  const int opp = 1 - me;
  if (after.terminal && after.winner == me) {
    return 1e6;
  }
  const int before_me = QuoridorRules::shortest_path_distance(before, me);
  const int before_opp = QuoridorRules::shortest_path_distance(before, opp);
  const int after_me = QuoridorRules::shortest_path_distance(after, me);
  const int after_opp = QuoridorRules::shortest_path_distance(after, opp);
  double score = static_cast<double>(after_opp - after_me) * 10.0;
  score += static_cast<double>(after_opp - before_opp) * 6.0;
  score -= static_cast<double>(after_me - before_me) * 4.0;
  if (board_ai::quoridor::is_move_action(action)) {
    score += 0.5;
    const int c = board_ai::quoridor::decode_move_col(action);
    score -= std::abs(c - 4) * 0.1;
  }
  return score;
}

AiDecision choose_action_for_state(
    QuoridorState& state,
    QuoridorRules& rules,
    const std::string& engine,
    int simulations,
    double temperature,
    double dirichlet_alpha,
    double dirichlet_epsilon,
    const char* model_path,
    const QuoridorSearchOptions& options) {
  AiDecision decision;
  const auto legal = rules.legal_actions(state);
  if (legal.empty()) return decision;

  if (engine == "heuristic") {
    std::vector<ActionId> best_actions;
    double best_score = -1e18;
    constexpr double kTieEps = 1e-9;
    for (ActionId a : legal) {
      QuoridorState tmp = state;
      rules.do_action_fast(tmp, a);
      const double score = heuristic_score_after_action(state, tmp, a);
      if (score > best_score + kTieEps) {
        best_score = score;
        best_actions.clear();
        best_actions.push_back(a);
      } else if (std::abs(score - best_score) <= kTieEps) {
        best_actions.push_back(a);
      }
    }
    if (best_actions.empty()) {
      best_actions.push_back(legal.front());
    }
    // Export heuristic "policy targets" as the set of tied-best actions.
    // This keeps warm-start labels aligned with heuristic intent even when
    // action execution injects exploration randomness.
    decision.root_actions = best_actions;
    decision.root_visits.assign(best_actions.size(), 1);
    const std::uint64_t tie_break_seed = static_cast<std::uint64_t>(state.rng_salt) ^
                                         (static_cast<std::uint64_t>(state.move_count + 43) << 32U) ^
                                         (static_cast<std::uint64_t>(state.current_player + 131) << 8U) ^
                                         static_cast<std::uint64_t>(best_actions.size() * 31337);
    SplitMix64Engine tie_rng(tie_break_seed);
    if (best_actions.size() == 1) {
      decision.action = best_actions.front();
    } else {
      std::uniform_int_distribution<size_t> tie_pick(0, best_actions.size() - 1);
      decision.action = best_actions[tie_pick(tie_rng)];
    }
    if (legal.size() > 1 && options.heuristic_random_action_prob > 1e-12) {
      const std::uint64_t random_override_seed = static_cast<std::uint64_t>(state.rng_salt) ^
                                                 (static_cast<std::uint64_t>(state.move_count + 17) << 32U) ^
                                                 (static_cast<std::uint64_t>(state.current_player + 101) << 8U) ^
                                                 static_cast<std::uint64_t>(legal.size() * 911);
      SplitMix64Engine rng(random_override_seed);
      std::uniform_real_distribution<double> bernoulli(0.0, 1.0);
      if (bernoulli(rng) < options.heuristic_random_action_prob) {
        std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
        decision.action = legal[pick(rng)];
      }
    }
    return decision;
  }

  if (engine != "netmcts") return decision;
  if (!(model_path && model_path[0] != '\0')) return decision;

  QuoridorFeatureEncoder encoder;
  board_ai::infer::OnnxPolicyValueEvaluator evaluator(std::string(model_path), &encoder, {});
  if (!evaluator.is_ready()) {
    PyErr_SetString(
        PyExc_RuntimeError,
        ("netmcts evaluator init failed: " + evaluator.last_error()).c_str());
    return decision;
  }

  board_ai::search::NetMctsConfig cfg;
  cfg.simulations = std::max(1, simulations);
  cfg.max_depth = std::max(1, options.common.max_search_depth);
  cfg.c_puct = 1.4f;
  if (options.common.dirichlet_on_first_n_plies > 0) {
    const auto root_noise = board_ai::search::resolve_root_dirichlet_noise(
        options.common.dirichlet_alpha,
        options.common.dirichlet_epsilon,
        options.common.dirichlet_on_first_n_plies,
        state.move_count);
    cfg.root_dirichlet_alpha = root_noise.alpha;
    cfg.root_dirichlet_epsilon = root_noise.epsilon;
  } else {
    const auto root_noise = board_ai::search::resolve_root_dirichlet_noise(
        dirichlet_alpha,
        dirichlet_epsilon,
        0,
        state.move_count);
    cfg.root_dirichlet_alpha = root_noise.alpha;
    cfg.root_dirichlet_epsilon = root_noise.epsilon;
  }

  QuoridorStateValueModel value_model;
  board_ai::search::NetMcts netmcts(cfg);
  board_ai::search::NetMctsStats stats;
  const ActionId best = netmcts.search_root(state, rules, value_model, evaluator, &stats);
  if (best < 0) return decision;

  decision.root_actions = stats.root_actions;
  decision.root_visits = stats.root_action_visits;
  const std::uint64_t sample_seed = static_cast<std::uint64_t>(state.rng_salt) ^
                                    (static_cast<std::uint64_t>(state.move_count + 1) << 32U) ^
                                    static_cast<std::uint64_t>(state.current_player + 31);
  decision.action = select_action_from_visits(
      stats.root_actions,
      stats.root_action_visits,
      temperature,
      sample_seed,
      best);
  decision.mcts_best_action_value = std::max(-1.0, std::min(1.0, stats.best_action_value));
  decision.has_mcts_best_action_value = true;
  return decision;
}

PyObject* py_session_new(PyObject*, PyObject* args) {
  long long seed = 0;
  int human_player = 0;
  if (!PyArg_ParseTuple(args, "Li", &seed, &human_player)) return nullptr;
  const std::int64_t handle = g_next_handle++;
  g_sessions.emplace(handle, std::make_unique<Session>(static_cast<std::uint64_t>(seed), human_player));
  return PyLong_FromLongLong(handle);
}

PyObject* py_session_delete(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  if (!PyArg_ParseTuple(args, "O", &h)) return nullptr;
  const long long handle = PyLong_AsLongLong(h);
  if (PyErr_Occurred()) return nullptr;
  g_sessions.erase(static_cast<std::int64_t>(handle));
  Py_RETURN_NONE;
}

PyObject* py_session_payload(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  if (!PyArg_ParseTuple(args, "O", &h)) return nullptr;
  Session* s = get_session_from_handle(h);
  if (!s) return nullptr;
  return build_payload(*s);
}

PyObject* py_session_legal_actions(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  if (!PyArg_ParseTuple(args, "O", &h)) return nullptr;
  Session* s = get_session_from_handle(h);
  if (!s) return nullptr;
  const auto legal = s->rules.legal_actions(s->state);
  PyObject* arr = PyList_New(0);
  for (ActionId a : legal) {
    PyObject* item = PyDict_New();
    PyDict_SetItemString(item, "action_id", PyLong_FromLong(a));
    append_action_metadata(item, a);
    PyList_Append(arr, item);
    Py_DECREF(item);
  }
  return arr;
}

PyObject* py_session_apply_action(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  int action_id = 0;
  int forced = 0;
  if (!PyArg_ParseTuple(args, "Oii", &h, &action_id, &forced)) return nullptr;
  Session* s = get_session_from_handle(h);
  if (!s) return nullptr;
  return apply_action_common(*s, action_id, forced != 0);
}

PyObject* py_session_ai_move(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  const char* engine = "netmcts";
  int simulations = 200;
  double temperature = 0.0;
  int time_budget_ms = 0;
  const char* model_path = nullptr;
  PyObject* search_options = nullptr;
  if (!PyArg_ParseTuple(
          args,
          "Osidi|zO",
          &h,
          &engine,
          &simulations,
          &temperature,
          &time_budget_ms,
          &model_path,
          &search_options)) {
    return nullptr;
  }
  (void)time_budget_ms;
  QuoridorSearchOptions parsed_options{};
  if (!parse_quoridor_search_options(search_options, &parsed_options)) return nullptr;
  Session* s = get_session_from_handle(h);
  if (!s) return nullptr;
  const AiDecision decision = choose_action_for_state(
      s->state,
      s->rules,
      std::string(engine),
      simulations,
      temperature,
      0.0,
      0.0,
      model_path,
      parsed_options);
  if (decision.action < 0) {
    if (std::string(engine) == "netmcts") {
      PyErr_SetString(
          PyExc_RuntimeError,
          "netmcts requested but no valid action produced (model path missing/invalid or evaluator unavailable)");
      return nullptr;
    }
    Py_RETURN_NONE;
  }
  PyObject* out = apply_action_common(*s, decision.action, false);
  if (!out) return nullptr;
  if (decision.has_mcts_best_action_value) {
    PyDict_SetItemString(out, "best_action_value", PyFloat_FromDouble(decision.mcts_best_action_value));
  } else {
    Py_INCREF(Py_None);
    PyDict_SetItemString(out, "best_action_value", Py_None);
  }
  return out;
}

PyObject* py_run_selfplay_episode_fast(PyObject*, PyObject* args) {
  long long seed = 0;
  const char* engine = "netmcts";
  int simulations = 200;
  double temperature = 0.0;
  const char* model_path = nullptr;
  double dirichlet_alpha = 0.0;
  double dirichlet_epsilon = 0.0;
  int dirichlet_on_first_n_plies = 0;
  PyObject* search_options = nullptr;
  if (!PyArg_ParseTuple(
          args,
          "Lsid|zddiO",
          &seed,
          &engine,
          &simulations,
          &temperature,
          &model_path,
          &dirichlet_alpha,
          &dirichlet_epsilon,
          &dirichlet_on_first_n_plies,
          &search_options)) {
    return nullptr;
  }
  QuoridorSearchOptions parsed_options{};
  if (!parse_quoridor_search_options(search_options, &parsed_options)) return nullptr;
  const bool has_dict = (search_options != nullptr && search_options != Py_None && PyDict_Check(search_options));
  const bool has_alpha = has_dict && (PyDict_GetItemString(search_options, "dirichlet_alpha") != nullptr);
  const bool has_epsilon = has_dict && (PyDict_GetItemString(search_options, "dirichlet_epsilon") != nullptr);
  const bool has_nplies = has_dict && (PyDict_GetItemString(search_options, "dirichlet_on_first_n_plies") != nullptr);
  if (!has_alpha) {
    parsed_options.common.dirichlet_alpha = static_cast<float>(std::max(0.0, dirichlet_alpha));
  }
  if (!has_epsilon) {
    parsed_options.common.dirichlet_epsilon =
        static_cast<float>(std::max(0.0, std::min(1.0, dirichlet_epsilon)));
  }
  if (!has_nplies) {
    parsed_options.common.dirichlet_on_first_n_plies = std::max(0, dirichlet_on_first_n_plies);
  }

  QuoridorState state;
  QuoridorRules rules;
  state.reset_with_seed(static_cast<std::uint64_t>(seed));

  int ply = 0;
  PyObject* samples = PyList_New(0);
  QuoridorFeatureEncoder encoder;
  while (!state.terminal && ply < std::max(1, parsed_options.common.max_episode_plies)) {
    const double runtime_temperature =
        board_ai::search::resolve_linear_temperature(parsed_options.common.temperature_schedule, temperature, ply);
    const bool allow_noise = ply < std::max(0, parsed_options.common.dirichlet_on_first_n_plies);
    const AiDecision d = choose_action_for_state(
        state,
        rules,
        std::string(engine),
        simulations,
        runtime_temperature,
        allow_noise ? parsed_options.common.dirichlet_alpha : 0.0,
        allow_noise ? parsed_options.common.dirichlet_epsilon : 0.0,
        model_path,
        parsed_options);
    if (d.action < 0) {
      if (std::string(engine) == "netmcts") {
        if (PyErr_Occurred()) {
          return nullptr;
        }
        PyErr_SetString(
            PyExc_RuntimeError,
            "netmcts selfplay failed: no valid action produced (model path missing/invalid or evaluator unavailable)");
        return nullptr;
      }
      break;
    }
    const int actor = state.current_player;
    const auto legal = rules.legal_actions(state);
    std::vector<float> features;
    std::vector<float> legal_mask;
    encoder.encode(state, actor, legal, &features, &legal_mask);

    rules.do_action_fast(state, d.action);
    ply += 1;

    PyObject* rec = PyDict_New();
    PyDict_SetItemString(rec, "ply", PyLong_FromLong(ply));
    PyDict_SetItemString(rec, "player", PyLong_FromLong(actor));
    PyDict_SetItemString(rec, "action_id", PyLong_FromLong(d.action));
    PyDict_SetItemString(rec, "state_version", PyLong_FromLong(ply - 1));

    PyObject* feat_list = PyList_New(static_cast<Py_ssize_t>(features.size()));
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(features.size()); ++i) {
      PyList_SET_ITEM(feat_list, i, PyFloat_FromDouble(features[static_cast<size_t>(i)]));
    }
    PyDict_SetItemString(rec, "features", feat_list);
    Py_DECREF(feat_list);

    PyObject* ids = PyList_New(0);
    PyObject* probs = PyList_New(0);
    if (!d.root_actions.empty() && d.root_actions.size() == d.root_visits.size()) {
      const auto p = visit_probs(d.root_actions, d.root_visits);
      for (size_t i = 0; i < d.root_actions.size(); ++i) {
        const ActionId canonical_a = encoder.canonicalize_action(d.root_actions[i], actor);
        PyList_Append(ids, PyLong_FromLong(canonical_a));
        PyList_Append(probs, PyFloat_FromDouble(p[i]));
      }
    } else {
      const ActionId canonical_a = encoder.canonicalize_action(d.action, actor);
      PyList_Append(ids, PyLong_FromLong(canonical_a));
      PyList_Append(probs, PyFloat_FromDouble(1.0));
    }
    PyDict_SetItemString(rec, "policy_action_ids", ids);
    PyDict_SetItemString(rec, "policy_probs", probs);
    Py_DECREF(ids);
    Py_DECREF(probs);
    PyList_Append(samples, rec);
    Py_DECREF(rec);
  }

  PyObject* out = PyDict_New();
  PyDict_SetItemString(out, "samples", samples);
  Py_DECREF(samples);
  PyDict_SetItemString(out, "plies", PyLong_FromLong(ply));
  if (!state.terminal || state.winner < 0) {
    Py_INCREF(Py_None);
    PyDict_SetItemString(out, "winner", Py_None);
  } else {
    PyDict_SetItemString(out, "winner", PyLong_FromLong(state.winner));
  }
  PyDict_SetItemString(out, "shared_victory", Py_False);
  PyObject* scores = PyList_New(2);
  PyList_SET_ITEM(scores, 0, PyLong_FromLong(state.scores[0]));
  PyList_SET_ITEM(scores, 1, PyLong_FromLong(state.scores[1]));
  PyDict_SetItemString(out, "scores", scores);
  Py_DECREF(scores);
  int win_margin_steps = 0;
  if (state.terminal && state.winner >= 0) {
    const int loser = 1 - state.winner;
    win_margin_steps = std::max(0, QuoridorRules::shortest_path_distance(state, loser));
  }
  PyDict_SetItemString(out, "win_margin_steps", PyLong_FromLong(win_margin_steps));
  if (!state.terminal && ply >= std::max(1, parsed_options.common.max_episode_plies)) {
    PyDict_SetItemString(out, "shared_victory", Py_True);
    Py_INCREF(Py_None);
    PyDict_SetItemString(out, "winner", Py_None);
  }
  return out;
}

PyObject* py_run_arena_match_fast(PyObject*, PyObject* args) {
  long long seed = 0;
  const char* p0_engine = "netmcts";
  int p0_simulations = 200;
  double p0_temperature = 0.0;
  const char* p0_model_path = nullptr;
  const char* p1_engine = "netmcts";
  int p1_simulations = 200;
  double p1_temperature = 0.0;
  const char* p1_model_path = nullptr;
  PyObject* p0_search_options = nullptr;
  PyObject* p1_search_options = nullptr;
  if (!PyArg_ParseTuple(
          args,
          "Lsidzsidz|OO",
          &seed,
          &p0_engine,
          &p0_simulations,
          &p0_temperature,
          &p0_model_path,
          &p1_engine,
          &p1_simulations,
          &p1_temperature,
          &p1_model_path,
          &p0_search_options,
          &p1_search_options)) {
    return nullptr;
  }
  QuoridorSearchOptions parsed_p0{};
  QuoridorSearchOptions parsed_p1{};
  if (!parse_quoridor_search_options(p0_search_options, &parsed_p0)) return nullptr;
  if (!parse_quoridor_search_options(p1_search_options, &parsed_p1)) return nullptr;

  QuoridorState state;
  QuoridorRules rules;
  state.reset_with_seed(static_cast<std::uint64_t>(seed));
  int ply = 0;
  const int max_episode_plies = std::max(1, std::min(parsed_p0.common.max_episode_plies, parsed_p1.common.max_episode_plies));
  while (!state.terminal && ply < max_episode_plies) {
    const bool p0_turn = state.current_player == 0;
    const auto& options = p0_turn ? parsed_p0 : parsed_p1;
    const AiDecision d = choose_action_for_state(
        state,
        rules,
        p0_turn ? std::string(p0_engine) : std::string(p1_engine),
        p0_turn ? p0_simulations : p1_simulations,
        board_ai::search::resolve_linear_temperature(
            options.common.temperature_schedule,
            p0_turn ? p0_temperature : p1_temperature,
            ply),
        0.0,
        0.0,
        p0_turn ? p0_model_path : p1_model_path,
        options);
    if (d.action < 0) {
      if ((p0_turn ? std::string(p0_engine) : std::string(p1_engine)) == "netmcts") {
        if (PyErr_Occurred()) {
          return nullptr;
        }
        PyErr_SetString(
            PyExc_RuntimeError,
            "netmcts arena failed: no valid action produced (model path missing/invalid or evaluator unavailable)");
        return nullptr;
      }
      break;
    }
    rules.do_action_fast(state, d.action);
    ply += 1;
  }

  PyObject* out = PyDict_New();
  if (!state.terminal || state.winner < 0) {
    Py_INCREF(Py_None);
    PyDict_SetItemString(out, "winner", Py_None);
  } else {
    PyDict_SetItemString(out, "winner", PyLong_FromLong(state.winner));
  }
  PyDict_SetItemString(out, "shared_victory", (!state.terminal && ply >= max_episode_plies) ? Py_True : Py_False);
  PyObject* scores = PyList_New(2);
  PyList_SET_ITEM(scores, 0, PyLong_FromLong(state.scores[0]));
  PyList_SET_ITEM(scores, 1, PyLong_FromLong(state.scores[1]));
  PyDict_SetItemString(out, "scores", scores);
  Py_DECREF(scores);
  int win_margin_steps = 0;
  if (state.terminal && state.winner >= 0) {
    const int loser = 1 - state.winner;
    win_margin_steps = std::max(0, QuoridorRules::shortest_path_distance(state, loser));
  }
  PyDict_SetItemString(out, "win_margin_steps", PyLong_FromLong(win_margin_steps));
  return out;
}

PyObject* py_onnx_enabled(PyObject*, PyObject*) {
#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

PyObject* py_feature_dim(PyObject*, PyObject*) {
  return PyLong_FromLong(kFeatureDim);
}

PyObject* py_action_space(PyObject*, PyObject*) {
  return PyLong_FromLong(kActionSpace);
}

#include "cpp_quoridor_engine_replay.inc"

PyMethodDef kMethods[] = {
    {"session_new", py_session_new, METH_VARARGS, "Create session"},
    {"session_delete", py_session_delete, METH_VARARGS, "Delete session"},
    {"session_payload", py_session_payload, METH_VARARGS, "Get state payload"},
    {"session_legal_actions", py_session_legal_actions, METH_VARARGS, "Get legal actions"},
    {"session_apply_action", py_session_apply_action, METH_VARARGS, "Apply action"},
    {"session_ai_move", py_session_ai_move, METH_VARARGS, "Apply AI move"},
    {"session_step_back", py_session_step_back, METH_VARARGS, "Step back"},
    {"session_can_step_back", py_session_can_step_back, METH_VARARGS, "Can step back"},
    {"session_seek", py_session_seek, METH_VARARGS, "Seek ply"},
    {"session_replay_events", py_session_replay_events, METH_VARARGS, "Replay events"},
    {"session_frames_range", py_session_frames_range, METH_VARARGS, "Frames range"},
    {"session_frame_at", py_session_frame_at, METH_VARARGS, "Frame at"},
    {"session_rebuild_frames", py_session_rebuild_frames, METH_VARARGS, "Rebuild frames"},
    {"onnx_enabled", py_onnx_enabled, METH_NOARGS, "Return whether ONNX runtime is enabled in this build"},
    {"feature_dim", py_feature_dim, METH_NOARGS, "Return encoded feature dimension"},
    {"action_space", py_action_space, METH_NOARGS, "Return action space size"},
    {"run_selfplay_episode_fast", py_run_selfplay_episode_fast, METH_VARARGS, "Run one full self-play episode in C++"},
    {"run_arena_match_fast", py_run_arena_match_fast, METH_VARARGS, "Run one full arena match in C++"},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModuleDef = {
    PyModuleDef_HEAD_INIT,
    "cpp_quoridor_engine_v1",
    "DinoBoard C++ Quoridor engine",
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit_cpp_quoridor_engine_v1(void) {
  return PyModule_Create(&kModuleDef);
}

