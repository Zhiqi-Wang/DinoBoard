PyObject* py_session_step_back(PyObject*, PyObject* args) {
    PyObject* h = nullptr;
    if (!PyArg_ParseTuple(args, "O", &h)) {
        return nullptr;
    }
    Session* s = get_session_from_handle(h);
    if (!s) return nullptr;
    if (s->cursor <= 0) {
        Py_RETURN_FALSE;
    }
    s->cursor -= 1;
    s->state = s->timeline_states[static_cast<size_t>(s->cursor)];
    Py_RETURN_TRUE;
}

PyObject* py_session_can_step_back(PyObject*, PyObject* args) {
    PyObject* h = nullptr;
    if (!PyArg_ParseTuple(args, "O", &h)) {
        return nullptr;
    }
    Session* s = get_session_from_handle(h);
    if (!s) return nullptr;
    if (s->cursor > 0) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyObject* py_session_seek(PyObject*, PyObject* args) {
    PyObject* h = nullptr;
    int ply = 0;
    if (!PyArg_ParseTuple(args, "Oi", &h, &ply)) {
        return nullptr;
    }
    Session* s = get_session_from_handle(h);
    if (!s) return nullptr;
    if (ply < 0 || ply >= static_cast<int>(s->timeline_states.size())) {
        Py_RETURN_FALSE;
    }
    s->cursor = ply;
    s->state = s->timeline_states[static_cast<size_t>(ply)];
    Py_RETURN_TRUE;
}

PyObject* py_session_replay_events(PyObject*, PyObject* args) {
    PyObject* h = nullptr;
    if (!PyArg_ParseTuple(args, "O", &h)) {
        return nullptr;
    }
    Session* s = get_session_from_handle(h);
    if (!s) return nullptr;
    PyObject* arr = PyList_New(0);
    for (int i = 0; i < s->cursor; ++i) {
        const auto& e = s->timeline_events[static_cast<size_t>(i)];
        PyObject* d = PyDict_New();
        PyDict_SetItemString(d, "ply", PyLong_FromLong(e.ply));
        PyDict_SetItemString(d, "actor", PyLong_FromLong(e.actor));
        PyDict_SetItemString(d, "action_id", PyLong_FromLong(e.action_id));
        PyDict_SetItemString(d, "forced", e.forced ? Py_True : Py_False);
        PyObject* mv = build_move_dict(e.action_id);
        PyDict_SetItemString(d, "move", mv);
        Py_DECREF(mv);
        PyList_Append(arr, d);
        Py_DECREF(d);
    }
    return arr;
}

PyObject* build_frame_from_state(const AzulState& st, int ply) {
    PyObject* d = PyDict_New();
    PyDict_SetItemString(d, "ply", PyLong_FromLong(ply));
    PyDict_SetItemString(d, "current_player", PyLong_FromLong(st.current_player));
    PyDict_SetItemString(d, "round_index", PyLong_FromLong(st.round_index));
    PyObject* scores = PyList_New(2);
    PyList_SET_ITEM(scores, 0, PyLong_FromLong(st.scores[0]));
    PyList_SET_ITEM(scores, 1, PyLong_FromLong(st.scores[1]));
    PyDict_SetItemString(d, "scores", scores);
    Py_DECREF(scores);
    PyDict_SetItemString(d, "is_terminal", st.terminal ? Py_True : Py_False);
    if (st.terminal && st.winner >= 0) {
        PyDict_SetItemString(d, "winner", PyLong_FromLong(st.winner));
    } else {
        Py_INCREF(Py_None);
        PyDict_SetItemString(d, "winner", Py_None);
    }
    PyDict_SetItemString(d, "shared_victory", st.shared_victory ? Py_True : Py_False);
    return d;
}

PyObject* py_session_frames_range(PyObject*, PyObject* args) {
    PyObject* h = nullptr;
    int from_ply = 0;
    int to_ply = -1;
    if (!PyArg_ParseTuple(args, "Oii", &h, &from_ply, &to_ply)) {
        return nullptr;
    }
    Session* s = get_session_from_handle(h);
    if (!s) return nullptr;
    int hi = to_ply < 0 ? s->cursor : std::min(to_ply, s->cursor);
    if (from_ply > hi || from_ply < 0) {
        Py_RETURN_NONE;
    }
    PyObject* arr = PyList_New(0);
    for (int ply = from_ply; ply <= hi; ++ply) {
        PyObject* fr = build_frame_from_state(s->timeline_states[static_cast<size_t>(ply)], ply);
        PyList_Append(arr, fr);
        Py_DECREF(fr);
    }
    return arr;
}

PyObject* py_session_frame_at(PyObject*, PyObject* args) {
    PyObject* h = nullptr;
    int ply = 0;
    if (!PyArg_ParseTuple(args, "Oi", &h, &ply)) {
        return nullptr;
    }
    Session* s = get_session_from_handle(h);
    if (!s) return nullptr;
    if (ply < 0 || ply > s->cursor) {
        Py_RETURN_NONE;
    }
    return build_frame_from_state(s->timeline_states[static_cast<size_t>(ply)], ply);
}

PyObject* py_session_rebuild_frames(PyObject*, PyObject* args) {
    PyObject* h = nullptr;
    if (!PyArg_ParseTuple(args, "O", &h)) {
        return nullptr;
    }
    Session* s = get_session_from_handle(h);
    if (!s) return nullptr;
    return PyLong_FromLong(s->cursor + 1);
}
