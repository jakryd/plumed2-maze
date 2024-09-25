/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
#ifndef __PLUMED_maze_Optimizer_h
#define __PLUMED_maze_Optimizer_h

/**
           __  ___
 |\/|  /\   / |__
 |  | /~~\ /_ |___

  @author Jakub Rydzewski <jr@fizyka.umk.pl>
  @version 2.0
  @file plumed2/src/maze/Optimizer.h
*/

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "colvar/Colvar.h"

#include "tools/NeighborList.h"
#include "tools/OFile.h"
#include "tools/Random.h"
#include "tools/SwitchingFunction.h"
#include "tools/Vector.h"

#include "Util.h"

namespace PLMD {
namespace maze {

/**
\ingroup INHERIT
Abstract class for implementing optimizers. An optimizer must override the
`optimize()` function and return an optimal neighbor accordindly to its 
optimization method. This optimal neighbor can be then used to bias the 
system toward it.
*/

class Optimizer : public colvar::Colvar {
  using distance_t = double (Optimizer::*)(array_t, array_t) const;

 public:
  explicit Optimizer(const ActionOptions& action_options);
  ~Optimizer();

  static void registerKeywords(Keywords& keys);

  void calculate() override;
  void prepare() override;
  void update() override;
  void apply() override {}

  virtual std::pair<double, array_t> optimize() = 0;

  double loss(array_t neighbor) const;
  double dist_neighbor() const;

  // Function interface for `ActionAtomistic::pbcDistance` and `delta` to use
  // either depending on the keyword `NOPBC`.
  double distance_pbc(array_t pk, array_t pl) const;
  double distance_nopbc(array_t pk, array_t pl) const;

  // Uses a pointer to member functions of type `distance_t` to skip checking
  // `NOPBC` every time distance is calculated for an atom pair in the
  // neighbor list. The pointer `distance_ptr` is initialized during keywords
  // processing.
  double distance(size_t ndx_pair) const;
  double distance(size_t ndx_pair, array_t neighbor) const;

  array_t rnd_neighbor();

  array_t neighbor() const { return opt_arg_; }
  size_t stride() const { return opt_stride_; }
  double step() const { return opt_step_; }
  void update_data_order(const vector_t<std::string>& order) { data_order_ = order; }

 private:
  // Ranks of processes.
  size_t mp_rank_;
  // Number of processes.
  size_t mp_stride_;
  // Number of threads.
  size_t mp_n_threads_;

  // Periodic boundary conditions.
  bool pbc_;

  // Neighbor list.
  bool nl_on_;
  // Perform computations on neighbor list in serial.
  bool nl_serial_;
  bool nl_started_;
  bool nl_validated_;
  // Terminate simulation if neighbor list is empty.
  bool nl_committor_;
  size_t nl_stride_;
  double nl_cutoff_;
  std::unique_ptr<NeighborList> nl_list_ptr;

  // A pointer to for member functions computing distances.
  distance_t distance_ptr;

 protected:
  size_t opt_stride_;
  size_t opt_n_iter_;
  double opt_val_;
  double opt_step_;
  bool opt_step_adaptive_;
  bool opt_finished_;
  array_t opt_arg_;

  // Switching function to calculate loss.
  SwitchingFunction switch_func_;

  enum id {kX, kY, kZ, kLoss, kStep, 
      /* number of elements in enum */ n_value = kStep - kX};
  vector_t<value_t*> value_;

  std::map<std::string, vector_t<double>> data_;
  vector_t<std::string> data_order_;

  bool verbose_;
  bool silent_;
  std::string output_fmt_;
  std::string output_filename_;
  std::shared_ptr<OFile> output_file_ptr;

  Random rnd_;
  size_t rnd_seed_;
};

}  // namespace maze
}  // namespace PLMD

#endif  // __PLUMED_maze_Optimizer_h
