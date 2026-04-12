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

#include "tictactoe_net_adapter.h"
#include "tictactoe_rules.h"
#include "tictactoe_state.h"
#include "infer/onnx_policy_value_evaluator.h"
#include "search/net_mcts.h"
#include "search/root_noise.h"

namespace {

using board_ai::ActionId;
using board_ai::tictactoe::TicTacToeFeatureEncoder;
using board_ai::tictactoe::TicTacToeRules;
using board_ai::tictactoe::TicTacToeStateValueModel;
using board_ai::tictactoe::TicTacToeState;
using board_ai::tictactoe::kBoardSize;
using board_ai::tictactoe::kEmptyCell;
using board_ai::search::select_action_from_visits;

struct EventRec {
  int ply = 0;
  int actor = 0;
  int action_id = 0;
  bool forced = false;
};

struct Session {
  TicTacToeState state;
  TicTacToeRules rules;
  int human_player = 0;
  std::vector<TicTacToeState> timeline_states{};
  std::vector<EventRec> timeline_events{};
  int cursor = 0;

  Session(std::uint64_t seed, int human) : state(), rules(), human_player(human), cursor(0) {
    state.reset_with_seed(seed);
    timeline_states.push_back(state);
  }
};

static std::unordered_map<std::int64_t, std::unique_ptr<Session>> g_sessions;
static std::int64_t g_next_handle = 1;

Session* get_session_from_handle(PyObject* handle_obj) {
  const long long handle = PyLong_AsLongLong(handle_obj);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  auto it = g_sessions.find(static_cast<std::int64_t>(handle));
  if (it == g_sessions.end()) {
    PyErr_SetString(PyExc_KeyError, "session handle not found");
    return nullptr;
  }
  return it->second.get();
}

bool is_corner(ActionId a) {
  return a == 0 || a == 2 || a == 6 || a == 8;
}

void normalize_probs(std::vector<double>& probs) {
  double s = 0.0;
  for (double p : probs) s += p;
  if (s <= 1e-12) {
    const double uni = probs.empty() ? 0.0 : (1.0 / static_cast<double>(probs.size()));
    for (double& p : probs) p = uni;
    return;
  }
  for (double& p : probs) p /= s;
}

ActionId sample_from_probs(const std::vector<ActionId>& actions, std::vector<double> probs, std::mt19937_64& rng) {
  if (actions.empty()) return -1;
  normalize_probs(probs);
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return actions[static_cast<size_t>(dist(rng))];
}

std::vector<double> build_tactical_prior(
    const TicTacToeState& state, const TicTacToeRules& rules, const std::vector<ActionId>& legal, double temperature
) {
  std::vector<double> logits;
  logits.reserve(legal.size());
  const int me = state.current_player;
  const int opp = 1 - me;
  const double t = std::max(1e-3, temperature);
  for (ActionId a : legal) {
    double score = 0.0;
    TicTacToeState tmp = state;
    rules.do_action_fast(tmp, a);
    if (tmp.terminal && tmp.winner == me) {
      score += 4.0;
    }
    if (a == 4) score += 0.7;
    if (is_corner(a)) score += 0.3;

    // Immediate opponent win check (if we don't play this move).
    bool blocks_loss = true;
    for (ActionId alt : legal) {
      if (alt == a) continue;
      TicTacToeState alt_state = state;
      rules.do_action_fast(alt_state, alt);
      if (alt_state.terminal && alt_state.winner == opp) {
        blocks_loss = false;
        break;
      }
    }
    if (blocks_loss) score += 0.2;
    logits.push_back(std::exp(score / t));
  }
  normalize_probs(logits);
  return logits;
}

void mix_dirichlet(std::vector<double>& probs, double alpha, double epsilon, std::mt19937_64& rng) {
  if (probs.empty() || alpha <= 0.0 || epsilon <= 0.0) return;
  std::gamma_distribution<double> gamma(alpha, 1.0);
  std::vector<double> noise(probs.size(), 0.0);
  double sum = 0.0;
  for (double& n : noise) {
    n = gamma(rng);
    sum += n;
  }
  if (sum <= 1e-12) return;
  for (double& n : noise) n /= sum;
  const double eps = std::min(1.0, std::max(0.0, epsilon));
  for (size_t i = 0; i < probs.size(); ++i) {
    probs[i] = (1.0 - eps) * probs[i] + eps * noise[i];
  }
  normalize_probs(probs);
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
  PyDict_SetItemString(common, "round_index", PyLong_FromLong(0));
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

  PyObject* board = PyList_New(kBoardSize);
  for (int i = 0; i < kBoardSize; ++i) {
    const int cell = s.state.board[static_cast<size_t>(i)];
    if (cell == kEmptyCell) {
      PyList_SET_ITEM(board, i, PyUnicode_FromString("."));
    } else if (cell == 0) {
      PyList_SET_ITEM(board, i, PyUnicode_FromString("X"));
    } else {
      PyList_SET_ITEM(board, i, PyUnicode_FromString("O"));
    }
  }
  PyDict_SetItemString(game, "board", board);
  Py_DECREF(board);

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
  return out;
}

struct AiDecision {
  ActionId action = -1;
  bool has_mcts_best_action_value = false;
  double mcts_best_action_value = 0.0;
};

AiDecision choose_action_for_state(
    TicTacToeState& state,
    TicTacToeRules& rules,
    const std::string& engine,
    int simulations,
    int time_budget_ms,
    double temperature,
    double dirichlet_alpha,
    double dirichlet_epsilon,
    int dirichlet_on_first_n_plies,
    const char* model_path
) {
  AiDecision decision;
  (void)time_budget_ms;
  const auto legal = rules.legal_actions(state);
  if (legal.empty()) {
    return decision;
  }
  ActionId block_action = -1;

  // Tactical guardrail: always take immediate win, or block immediate loss.
  for (ActionId a : legal) {
    TicTacToeState tmp = state;
    rules.do_action_fast(tmp, a);
    if (tmp.terminal && tmp.winner == state.current_player) {
      decision.action = a;
      decision.has_mcts_best_action_value = true;
      decision.mcts_best_action_value = 1.0;
      return decision;
    }
  }
  const int opponent = 1 - state.current_player;
  for (ActionId block : legal) {
    TicTacToeState after_block = state;
    rules.do_action_fast(after_block, block);
    const auto opp_legal = rules.legal_actions(after_block);
    bool still_has_win = false;
    for (ActionId opp_move : opp_legal) {
      TicTacToeState opp_tmp = after_block;
      rules.do_action_fast(opp_tmp, opp_move);
      if (opp_tmp.terminal && opp_tmp.winner == opponent) {
        still_has_win = true;
        break;
      }
    }
    if (!still_has_win) {
      block_action = block;
      break;
    }
  }

  if (engine == "heuristic") {
    if (block_action >= 0) {
      decision.action = block_action;
      return decision;
    }
    ActionId best = legal.front();
    int best_score = -100;
    for (ActionId a : legal) {
      TicTacToeState tmp = state;
      rules.do_action_fast(tmp, a);
      int score = 0;
      if (tmp.winner == state.current_player) {
        score = 10;
      } else if (tmp.winner == -1 && tmp.terminal) {
        score = 1;
      }
      if (score > best_score) {
        best_score = score;
        best = a;
      }
    }
    decision.action = best;
    return decision;
  }

  if (engine != "netmcts") {
    return decision;
  }

  if (!(model_path && model_path[0] != '\0')) {
    return decision;
  }

  TicTacToeFeatureEncoder encoder;
  board_ai::infer::OnnxPolicyValueEvaluator evaluator(std::string(model_path), &encoder, {});
  if (evaluator.is_ready()) {
    board_ai::search::NetMctsConfig cfg;
    cfg.simulations = std::max(1, simulations);
    cfg.max_depth = 16;
    cfg.c_puct = 1.4f;
    const auto root_noise = board_ai::search::resolve_root_dirichlet_noise(
        dirichlet_alpha, dirichlet_epsilon, dirichlet_on_first_n_plies, state.move_count);
    cfg.root_dirichlet_alpha = root_noise.alpha;
    cfg.root_dirichlet_epsilon = root_noise.epsilon;
    TicTacToeStateValueModel value_model;
    board_ai::search::NetMcts netmcts(cfg);
    board_ai::search::NetMctsStats stats;
    ActionId a = netmcts.search_root(state, rules, value_model, evaluator, &stats);
    if (a >= 0) {
      const std::uint64_t sample_seed = static_cast<std::uint64_t>(state.rng_salt) ^
                                        (static_cast<std::uint64_t>(state.move_count + 1) << 32U) ^
                                        static_cast<std::uint64_t>(state.current_player + 17);
      decision.action = select_action_from_visits(
          stats.root_actions,
          stats.root_action_visits,
          temperature,
          sample_seed,
          a);
      decision.mcts_best_action_value = std::max(-1.0, std::min(1.0, stats.best_action_value));
      decision.has_mcts_best_action_value = true;
    }
  }
  return decision;
}

PyObject* py_session_new(PyObject*, PyObject* args) {
  long long seed = 0;
  int human_player = 0;
  if (!PyArg_ParseTuple(args, "Li", &seed, &human_player)) {
    return nullptr;
  }
  const std::int64_t handle = g_next_handle++;
  g_sessions.emplace(handle, std::make_unique<Session>(static_cast<std::uint64_t>(seed), human_player));
  return PyLong_FromLongLong(handle);
}

PyObject* py_session_delete(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  if (!PyArg_ParseTuple(args, "O", &h)) {
    return nullptr;
  }
  const long long handle = PyLong_AsLongLong(h);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  g_sessions.erase(static_cast<std::int64_t>(handle));
  Py_RETURN_NONE;
}

PyObject* py_session_payload(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  if (!PyArg_ParseTuple(args, "O", &h)) {
    return nullptr;
  }
  Session* s = get_session_from_handle(h);
  if (!s) return nullptr;
  return build_payload(*s);
}

PyObject* py_session_legal_actions(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  if (!PyArg_ParseTuple(args, "O", &h)) {
    return nullptr;
  }
  Session* s = get_session_from_handle(h);
  if (!s) return nullptr;
  const auto legal = s->rules.legal_actions(s->state);
  PyObject* arr = PyList_New(0);
  for (ActionId a : legal) {
    PyObject* item = PyDict_New();
    PyDict_SetItemString(item, "action_id", PyLong_FromLong(a));
    PyList_Append(arr, item);
    Py_DECREF(item);
  }
  return arr;
}

PyObject* py_session_apply_action(PyObject*, PyObject* args) {
  PyObject* h = nullptr;
  int action_id = 0;
  int forced = 0;
  if (!PyArg_ParseTuple(args, "Oii", &h, &action_id, &forced)) {
    return nullptr;
  }
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
  if (!PyArg_ParseTuple(args, "Osidi|z", &h, &engine, &simulations, &temperature, &time_budget_ms, &model_path)) {
    return nullptr;
  }
  Session* s = get_session_from_handle(h);
  if (!s) return nullptr;
  const AiDecision decision = choose_action_for_state(
      s->state, s->rules, std::string(engine), simulations, time_budget_ms, temperature, 0.0, 0.0, 0, model_path
  );
  ActionId best = decision.action;
  if (best < 0) {
    if (std::string(engine) == "netmcts") {
      PyErr_SetString(
          PyExc_RuntimeError,
          "netmcts requested but no valid action produced (model path missing/invalid or evaluator unavailable)");
      return nullptr;
    }
    Py_RETURN_NONE;
  }
  PyObject* out = apply_action_common(*s, best, false);
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
  if (!PyArg_ParseTuple(
          args,
          "Lsid|zddi",
          &seed,
          &engine,
          &simulations,
          &temperature,
          &model_path,
          &dirichlet_alpha,
          &dirichlet_epsilon,
          &dirichlet_on_first_n_plies
      )) {
    return nullptr;
  }

  TicTacToeState state;
  TicTacToeRules rules;
  state.reset_with_seed(static_cast<std::uint64_t>(seed));
  int cursor = 0;
  PyObject* samples = PyList_New(0);
  while (!state.terminal) {
    const AiDecision decision = choose_action_for_state(
        state,
        rules,
        std::string(engine),
        simulations,
        0,
        temperature,
        dirichlet_alpha,
        dirichlet_epsilon,
        dirichlet_on_first_n_plies,
        model_path
    );
    ActionId action = decision.action;
    if (action < 0) {
      if (std::string(engine) == "netmcts") {
        PyErr_SetString(
            PyExc_RuntimeError,
            "netmcts selfplay failed: no valid action produced (model path missing/invalid or evaluator unavailable)");
        return nullptr;
      }
      break;
    }
    const int actor = state.current_player;
    PyObject* board_before = PyList_New(kBoardSize);
    for (int i = 0; i < kBoardSize; ++i) {
      PyList_SET_ITEM(board_before, i, PyLong_FromLong(static_cast<int>(state.board[static_cast<size_t>(i)])));
    }
    rules.do_action_fast(state, action);
    cursor += 1;
    PyObject* rec = PyDict_New();
    PyDict_SetItemString(rec, "ply", PyLong_FromLong(cursor));
    PyDict_SetItemString(rec, "player", PyLong_FromLong(actor));
    PyDict_SetItemString(rec, "action_id", PyLong_FromLong(action));
    PyDict_SetItemString(rec, "state_version", PyLong_FromLong(cursor - 1));
    PyDict_SetItemString(rec, "board", board_before);
    Py_DECREF(board_before);
    PyList_Append(samples, rec);
    Py_DECREF(rec);
  }
  PyObject* out = PyDict_New();
  PyDict_SetItemString(out, "samples", samples);
  Py_DECREF(samples);
  PyDict_SetItemString(out, "plies", PyLong_FromLong(cursor));
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
  if (!PyArg_ParseTuple(args, "Lsidzsidz", &seed, &p0_engine, &p0_simulations, &p0_temperature, &p0_model_path,
                        &p1_engine, &p1_simulations, &p1_temperature, &p1_model_path)) {
    return nullptr;
  }
  TicTacToeState state;
  TicTacToeRules rules;
  state.reset_with_seed(static_cast<std::uint64_t>(seed));
  while (!state.terminal) {
    const bool p0_turn = state.current_player == 0;
    const std::string engine = p0_turn ? std::string(p0_engine) : std::string(p1_engine);
    const int simulations = p0_turn ? p0_simulations : p1_simulations;
    const double temperature = p0_turn ? p0_temperature : p1_temperature;
    const char* model_path = p0_turn ? p0_model_path : p1_model_path;
    const AiDecision decision = choose_action_for_state(
        state, rules, engine, simulations, 0, temperature, 0.0, 0.0, 0, model_path
    );
    ActionId action = decision.action;
    if (action < 0) {
      if (engine == "netmcts") {
        PyErr_SetString(
            PyExc_RuntimeError,
            "netmcts arena failed: no valid action produced (model path missing/invalid or evaluator unavailable)");
        return nullptr;
      }
      break;
    }
    rules.do_action_fast(state, action);
  }
  PyObject* out = PyDict_New();
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
  return out;
}

#include "cpp_tictactoe_engine_replay.inc"

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
    {"run_selfplay_episode_fast", py_run_selfplay_episode_fast, METH_VARARGS, "Run one full self-play episode in C++"},
    {"run_arena_match_fast", py_run_arena_match_fast, METH_VARARGS, "Run one full arena match in C++"},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModuleDef = {
    PyModuleDef_HEAD_INIT,
    "cpp_tictactoe_engine_v7",
    "DinoBoard C++ TicTacToe engine",
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit_cpp_tictactoe_engine_v7(void) {
  return PyModule_Create(&kModuleDef);
}

