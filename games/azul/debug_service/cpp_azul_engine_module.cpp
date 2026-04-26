#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "azul_net_adapter.h"
#include "azul_rules.h"
#include "azul_state.h"
#include "infer/onnx_policy_value_evaluator.h"
#include "search/net_mcts.h"
#include "search/root_noise.h"
#include "search/search_options_common.h"

namespace {

using board_ai::ActionId;
using board_ai::azul::AzulFeatureEncoder;
using board_ai::azul::AzulRules;
using board_ai::azul::AzulStateValueModel;
using board_ai::azul::AzulState;
using board_ai::azul::kActionSpace;
using board_ai::infer::OnnxPolicyValueEvaluator;
using board_ai::search::NetMcts;
using board_ai::search::NetMctsConfig;
using board_ai::search::NetMctsStats;
using board_ai::search::select_action_from_visits;

using board_ai::azul::kCenterSource;
using board_ai::azul::kColors;
using board_ai::azul::kFactories;
using board_ai::azul::kRows;
using board_ai::azul::kTargetsPerColor;

constexpr std::array<const char*, 5> kColorChars = {"B", "Y", "R", "K", "W"};
constexpr int kFirstPlayerToken = -2;

struct EventRec {
    int ply = 0;
    int actor = 0;
    int action_id = 0;
    bool forced = false;
    int source = 0;
    int color = 0;
    int target_line = 0;
};

struct Session {
    AzulState state;
    AzulRules rules;
    int human_player = 0;
    std::vector<AzulState> timeline_states{};
    std::vector<EventRec> timeline_events{};
    int cursor = 0;

    Session(std::uint64_t seed, int human) : state(), rules(), human_player(human), cursor(0) {
        state.reset_with_seed(seed);
        timeline_states.push_back(state);
    }
};

static std::unordered_map<std::int64_t, std::unique_ptr<Session>> g_sessions;
static std::int64_t g_next_handle = 1;

inline int decode_source(ActionId action) {
    return static_cast<int>(action / (kColors * kTargetsPerColor));
}
inline int decode_color(ActionId action) {
    return static_cast<int>((action / kTargetsPerColor) % kColors);
}
inline int decode_target_line(ActionId action) {
    return static_cast<int>(action % kTargetsPerColor);
}

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

PyObject* build_move_dict(int action_id) {
    const int src = decode_source(action_id);
    const int color = decode_color(action_id);
    const int target = decode_target_line(action_id);

    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    if (src == kCenterSource) {
        PyDict_SetItemString(d, "source", PyUnicode_FromString("center"));
        PyDict_SetItemString(d, "source_idx", PyLong_FromLong(-1));
    } else {
        PyDict_SetItemString(d, "source", PyUnicode_FromString("factory"));
        PyDict_SetItemString(d, "source_idx", PyLong_FromLong(src));
    }
    PyDict_SetItemString(d, "color", PyUnicode_FromString(kColorChars[static_cast<size_t>(color)]));
    PyDict_SetItemString(d, "target_line", PyLong_FromLong(target == 5 ? -1 : target));
    return d;
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
    PyDict_SetItemString(common, "round_index", PyLong_FromLong(s.state.round_index));
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
    PyDict_SetItemString(common, "shared_victory", s.state.shared_victory ? Py_True : Py_False);

    PyObject* factories = PyList_New(kFactories);
    for (int fi = 0; fi < kFactories; ++fi) {
        PyObject* fac = PyList_New(0);
        for (int c = 0; c < kColors; ++c) {
            for (int cnt = 0; cnt < s.state.factories[fi][c]; ++cnt) {
                PyList_Append(fac, PyUnicode_FromString(kColorChars[static_cast<size_t>(c)]));
            }
        }
        PyList_SET_ITEM(factories, fi, fac);
    }
    PyDict_SetItemString(game, "factories", factories);
    Py_DECREF(factories);

    PyObject* center = PyList_New(0);
    for (int c = 0; c < kColors; ++c) {
        for (int cnt = 0; cnt < s.state.center[c]; ++cnt) {
            PyList_Append(center, PyUnicode_FromString(kColorChars[static_cast<size_t>(c)]));
        }
    }
    PyDict_SetItemString(game, "center", center);
    Py_DECREF(center);
    PyDict_SetItemString(
        game, "first_player_token_in_center", s.state.first_player_token_in_center ? Py_True : Py_False
    );
    PyDict_SetItemString(game, "bag_size", PyLong_FromLong(static_cast<long>(s.state.bag.size())));
    PyDict_SetItemString(game, "box_lid_size", PyLong_FromLong(static_cast<long>(s.state.box_lid.size())));

    PyObject* players = PyList_New(2);
    for (int pid = 0; pid < 2; ++pid) {
        const auto& p = s.state.players[pid];
        PyObject* pd = PyDict_New();
        PyObject* pattern_lines = PyList_New(kRows);
        for (int r = 0; r < kRows; ++r) {
            PyObject* line = PyList_New(0);
            if (p.line_color[r] >= 0) {
                for (int k = 0; k < p.line_len[r]; ++k) {
                    PyList_Append(line, PyUnicode_FromString(kColorChars[static_cast<size_t>(p.line_color[r])]));
                }
            }
            PyList_SET_ITEM(pattern_lines, r, line);
        }
        PyDict_SetItemString(pd, "pattern_lines", pattern_lines);
        Py_DECREF(pattern_lines);

        PyObject* wall = PyList_New(kRows);
        for (int r = 0; r < kRows; ++r) {
            PyObject* row = PyList_New(kColors);
            for (int c = 0; c < kColors; ++c) {
                const bool filled = ((p.wall_mask[r] >> c) & 1U) != 0U;
                PyList_SET_ITEM(row, c, filled ? Py_True : Py_False);
                Py_INCREF(filled ? Py_True : Py_False);
            }
            PyList_SET_ITEM(wall, r, row);
        }
        PyDict_SetItemString(pd, "wall", wall);
        Py_DECREF(wall);

        PyObject* floor = PyList_New(0);
        for (int i = 0; i < p.floor_count && i < 7; ++i) {
            const int col = p.floor[i];
            if (col >= 0 && col < kColors) {
                PyList_Append(floor, PyUnicode_FromString(kColorChars[static_cast<size_t>(col)]));
            } else if (col == kFirstPlayerToken) {
                PyList_Append(floor, PyUnicode_FromString("F"));
            }
        }
        PyDict_SetItemString(pd, "floor", floor);
        Py_DECREF(floor);

        PyDict_SetItemString(pd, "score", PyLong_FromLong(p.score));
        PyList_SET_ITEM(players, pid, pd);
    }
    PyDict_SetItemString(game, "players", players);
    Py_DECREF(players);

    PyDict_SetItemString(root, "common", common);
    PyDict_SetItemString(root, "game", game);
    Py_DECREF(common);
    Py_DECREF(game);
    return root;
}

PyObject* build_payload(Session& s) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    PyDict_SetItemString(d, "state_version", PyLong_FromLong(s.cursor));
    PyObject* pub = build_public_state(s);
    if (!pub) {
        Py_DECREF(d);
        return nullptr;
    }
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
    ev.source = decode_source(action_id);
    ev.color = decode_color(action_id);
    ev.target_line = decode_target_line(action_id);
    s.timeline_events.push_back(ev);

    PyObject* out = PyDict_New();
    PyDict_SetItemString(out, "ply", PyLong_FromLong(ev.ply));
    PyDict_SetItemString(out, "actor", PyLong_FromLong(ev.actor));
    PyDict_SetItemString(out, "action_id", PyLong_FromLong(ev.action_id));
    PyDict_SetItemString(out, "forced", ev.forced ? Py_True : Py_False);
    PyObject* move = build_move_dict(ev.action_id);
    PyDict_SetItemString(out, "move", move);
    Py_DECREF(move);
    PyObject* apply_result = PyDict_New();
    PyDict_SetItemString(out, "apply_result", apply_result);
    Py_DECREF(apply_result);
    return out;
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
        PyObject* mv = build_move_dict(a);
        PyDict_SetItemString(item, "move", mv);
        Py_DECREF(mv);
        PyDict_SetItemString(item, "label", PyUnicode_FromFormat("aid=%d", static_cast<int>(a)));
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

PyObject* py_onnx_enabled(PyObject*, PyObject*) {
#if defined(BOARD_AI_WITH_ONNX) && BOARD_AI_WITH_ONNX
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}

#include "cpp_azul_engine_ai.inc"

#include "cpp_azul_engine_replay.inc"

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
    {"run_selfplay_episode_fast", py_run_selfplay_episode_fast, METH_VARARGS, "Run one full self-play episode in C++"},
    {"run_arena_match_fast", py_run_arena_match_fast, METH_VARARGS, "Run one full arena match in C++"},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModuleDef = {
    PyModuleDef_HEAD_INIT,
    "cpp_azul_engine_v7",
    "DinoBoard C++ Azul engine",
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit_cpp_azul_engine_v7(void) {
    return PyModule_Create(&kModuleDef);
}
