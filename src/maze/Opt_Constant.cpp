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

/**
           __  ___
 |\/|  /\   / |__
 |  | /~~\ /_ |___

  @author Jakub Rydzewski <jr@fizyka.umk.pl>
  @version 2.0
  @file plumed2/src/maze/Opt_Constant.cpp
*/

#include "core/ActionRegister.h"

#include "Optimizer.h"
#include "Util.h"

namespace PLMD {
namespace maze {

//+PLUMEDOC MAZE_OPT MAZE_OPT_CONSTANT
/*
Fake optimizer yielding a constant biasing neighbor.

\par Examples
Here, the optimizer is used to find a loss function value between atoms listed
in the `GROUPA` and `GROUPB`, but instead of finding a neighbor correspoding
to the minimal value of the loss function, it returns a constant neighbor
every `STRIDE` and the loss for this neighbor. The loss function is calculated
as the sum over every value corresponding to an atom pair between the defined
groups given by the switching function (`SWITCH`). For more details about
switching functions, see \\ref switchingfunction. The constant neighbor is
passed using the `NEIGHBOR` keyword.

\plumedfile
MAZE_OPT_CONSTANT ...
  LABEL=opt
  GROUPA=2635-2646
  GROUPB=1-2634
  SWITCH={CUSTOM FUNC=exp(-x)/x R_0=0.1}
  STRIDE=1
  NEIGHBOR=1,1,1
...
\endplumedfile
*/
//+ENDPLUMEDOC

class Constant : public Optimizer {
 public:
  explicit Constant(const ActionOptions& action_options);
  ~Constant() {}

  static void registerKeywords(Keywords& keys);

  std::pair<double, array_t> optimize() override;

 private:
  array_t neighbor_;
};

PLUMED_REGISTER_ACTION(Constant, "MAZE_OPT_CONSTANT")

void Constant::registerKeywords(Keywords& keys) {
  Optimizer::registerKeywords(keys);

  keys.remove("N_ITER");
  keys.add("compulsory", "NEIGHBOR", "Constant neighbor of biasing.");
}

Constant::Constant(const ActionOptions& action_options)
  : Action(action_options),
    Optimizer(action_options) {
  log.printf("  Optimizer: Constant\n");

  if (keywords.exists("NEIGHBOR")) {
    vector_t<double> neighbor;
    parseVector("NEIGHBOR", neighbor);
    plumed_massert(neighbor.size() == 3, "Constant neighbor must have three components.");
    neighbor_ = util::unpack_vector(neighbor);
    neighbor_ /= modulo(neighbor_);
    log.printf("  Normalized constant neighbor: (%6.4f, %6.4f, %6.4f)\n", neighbor_[0], neighbor_[1], neighbor_[2]);
  }

  checkRead();
  
  update_data_order({"iter", "loss"});
}

std::pair<double, array_t> Constant::optimize() {
  array_t neighbor = step() * neighbor_;
  double loss_neighbor = loss(neighbor);

  data_["iter"].push_back(0.0);
  data_["loss"].push_back(loss_neighbor);

  return std::make_pair(loss_neighbor, neighbor / modulo(neighbor));
}

}  // namespace maze
}  // namespace PLMD
