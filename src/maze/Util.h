/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Copyright (c) 2022 Jakub Rydzewski (jr@fizyka.umk.pl). All rights reserved.

This file is part of maze.

maze is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version.

maze is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.

See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with maze. If not, see <https://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#ifndef __PLUMED_maze_Util_h
#define __PLUMED_maze_Util_h

/**
           __  ___
 |\/|  /\   / |__
 |  | /~~\ /_ |___

  @author Jakub Rydzewski <jr@fizyka.umk.pl>
  @version 2.0
  @file plumed2/src/maze/Util.h
*/

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include "core/ActionPilot.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/PlumedMain.h"

#include "tools/Vector.h"

namespace PLMD {
namespace maze {

template<class T> 
using vector_t = std::vector<T>;
using array_t = Vector;
using value_t = Value;

namespace util {

template <typename T>
static inline bool in_vector(T item, const vector_t<T>& vector) {
  if (std::find(vector.begin(), vector.end(), item) != vector.end()) {
    return true;
  } else {
    return false;
  }
}

template <typename T>
static inline int index_vector(T item, const vector_t<T>& vector) {
  auto it = std::find(vector.begin(), vector.end(), item);
  if (it != vector.end()) {
    int index = it - vector.begin();
    return index;
  } else {
    return -1;
  }
}

template <typename T>
static inline vector_t<T> link_actions(const std::vector<std::string>& action_labels, const ActionSet& action_set) {
  vector_t<T> action_ptrs(action_labels.size(), nullptr);
  for (size_t i = 0; i < action_labels.size(); ++i) {
    action_ptrs[i] = action_set.selectWithLabel<T>(action_labels[i]);
  }
  return action_ptrs;
}

template <typename T>
static inline T link_action(const std::string& action_label, const ActionSet& action_set) {
  T action_ptr = action_set.selectWithLabel<T>(action_label);
  return action_ptr;
}

template <typename T>
static inline value_t* link_value(const std::string& action_label, const ActionSet& action_set) {
  T* value = action_set.selectWithLabel<T*>(action_label);
  return value->copyOutput(value->getLabel());
}

static inline array_t unpack_vector(const vector_t<double>& vector) {
  array_t arr;
  for (size_t i = 0; i < vector.size(); ++i) {
    arr[i] = vector[i];
  }
  return arr;
}

static inline vector_t<double> unpack_array(const array_t& array) {
  vector_t<double> vec;
  for (size_t i = 0; i < 3; ++i) {
    vec[i] = array[i];
  }
  return vec;
}

static inline array_t unpack_arguments(const vector_t<value_t*>& arguments) {
  array_t values;
  for (size_t i = 0; i < arguments.size(); ++i) {
    values[i] = arguments[i]->get();
  }
  return values;
}

// Casting `vector_t` to `array_t` and `vector_t` to `array_t`.
template <class T> 
class vector_cast {};

template <>
class vector_cast<vector_t<double>> {
  const vector_t<double>& v;
 public:
  vector_cast(const vector_t<double>& v): v(v) {}
  operator vector_t<double>() const { return vector_t<double>(v); }
  operator array_t() const { return unpack_vector(v); }
};

template <>
class vector_cast<array_t> {
  const array_t& a;
 public:
  vector_cast(const array_t& a): a(a) {}
  operator array_t() const { return array_t(a); }
  operator vector_t<double>() const { return unpack_array(a); }
};

}  // namespace util

}  // namespace maze
}  // namespace PLMD

#endif  // __PLUMED_maze_Util_h
