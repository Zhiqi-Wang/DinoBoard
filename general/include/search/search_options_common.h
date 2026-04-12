#pragma once

#include <Python.h>

#include <algorithm>

#include "search/temperature_schedule.h"

namespace board_ai {
namespace search {

struct CommonSearchOptions {
  int max_search_depth = 10;
  int max_episode_plies = 220;
  float dirichlet_alpha = 0.0f;
  float dirichlet_epsilon = 0.0f;
  int dirichlet_on_first_n_plies = 0;
  TemperatureSchedule temperature_schedule{};
};

inline PyObject* dict_get_with_aliases(
    PyObject* dict_obj,
    const char* key,
    const char* alias1 = nullptr,
    const char* alias2 = nullptr) {
  if (dict_obj == nullptr || !PyDict_Check(dict_obj)) return nullptr;
  if (key != nullptr) {
    if (PyObject* obj = PyDict_GetItemString(dict_obj, key)) return obj;
  }
  if (alias1 != nullptr) {
    if (PyObject* obj = PyDict_GetItemString(dict_obj, alias1)) return obj;
  }
  if (alias2 != nullptr) {
    if (PyObject* obj = PyDict_GetItemString(dict_obj, alias2)) return obj;
  }
  return nullptr;
}

inline bool parse_bool_option(
    PyObject* dict_obj,
    const char* key,
    bool* out_value,
    const char* alias1 = nullptr,
    const char* alias2 = nullptr) {
  if (!out_value) return false;
  PyObject* obj = dict_get_with_aliases(dict_obj, key, alias1, alias2);
  if (!obj) return true;
  const int b = PyObject_IsTrue(obj);
  if (b < 0) return false;
  *out_value = (b == 1);
  return true;
}

inline bool parse_int_option(
    PyObject* dict_obj,
    const char* key,
    int* out_value,
    int min_value,
    const char* alias1 = nullptr,
    const char* alias2 = nullptr) {
  if (!out_value) return false;
  PyObject* obj = dict_get_with_aliases(dict_obj, key, alias1, alias2);
  if (!obj) return true;
  const long v = PyLong_AsLong(obj);
  if (PyErr_Occurred()) return false;
  *out_value = std::max(min_value, static_cast<int>(v));
  return true;
}

inline bool parse_nonnegative_double_option(
    PyObject* dict_obj,
    const char* key,
    double* out_value,
    const char* alias1 = nullptr,
    const char* alias2 = nullptr) {
  if (!out_value) return false;
  PyObject* obj = dict_get_with_aliases(dict_obj, key, alias1, alias2);
  if (!obj) return true;
  const double v = PyFloat_AsDouble(obj);
  if (PyErr_Occurred()) return false;
  *out_value = std::max(0.0, v);
  return true;
}

inline bool parse_probability_option(
    PyObject* dict_obj,
    const char* key,
    double* out_value,
    const char* alias1 = nullptr,
    const char* alias2 = nullptr) {
  if (!out_value) return false;
  PyObject* obj = dict_get_with_aliases(dict_obj, key, alias1, alias2);
  if (!obj) return true;
  const double v = PyFloat_AsDouble(obj);
  if (PyErr_Occurred()) return false;
  *out_value = std::max(0.0, std::min(1.0, v));
  return true;
}

inline bool parse_common_search_options(PyObject* obj, CommonSearchOptions* out) {
  if (!out) return false;
  if (obj == nullptr || obj == Py_None) return true;
  if (!PyDict_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "search_options must be dict when provided");
    return false;
  }

  if (!parse_int_option(obj, "max_search_depth", &out->max_search_depth, 1)) return false;
  if (!parse_int_option(obj, "max_episode_plies", &out->max_episode_plies, 1)) return false;
  {
    double v = static_cast<double>(out->dirichlet_alpha);
    if (!parse_nonnegative_double_option(obj, "dirichlet_alpha", &v)) return false;
    out->dirichlet_alpha = static_cast<float>(v);
  }
  {
    double v = static_cast<double>(out->dirichlet_epsilon);
    if (!parse_probability_option(obj, "dirichlet_epsilon", &v)) return false;
    out->dirichlet_epsilon = static_cast<float>(v);
  }
  if (!parse_int_option(obj, "dirichlet_on_first_n_plies", &out->dirichlet_on_first_n_plies, 0)) return false;

  if (!parse_nonnegative_double_option(obj, "temperature_initial", &out->temperature_schedule.initial)) return false;
  if (dict_get_with_aliases(obj, "temperature_initial")) {
    out->temperature_schedule.has_initial = true;
  }
  if (!parse_nonnegative_double_option(obj, "temperature_final", &out->temperature_schedule.final)) return false;
  if (dict_get_with_aliases(obj, "temperature_final")) {
    out->temperature_schedule.has_final = true;
  }
  if (!parse_int_option(
          obj,
          "temperature_decay_plies",
          &out->temperature_schedule.decay_plies,
          0)) {
    return false;
  }
  out->temperature_schedule.enabled =
      out->temperature_schedule.has_initial ||
      out->temperature_schedule.has_final ||
      out->temperature_schedule.decay_plies > 0;

  return true;
}

}  // namespace search
}  // namespace board_ai
