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
  @file plumed2/src/maze/Opt_Random.cpp
*/

#include "core/ActionRegister.h"

#include "Optimizer.h"
#include "Util.h"

namespace PLMD {
namespace maze {

//+PLUMEDOC MAZE_OPT MAZE_OPT_RANDOM
/*
Fake optimizer yielding a random biasing direction.

\par Examples
Here, the optimizer is used to find a loss function value between atoms listed
in the `GROUPA` and `GROUPB`, but instead of finding a naighbor correspoding to
the minimal value of the loss function, it returns a random neighbor every
`STRIDE` and the loss for this random configuration. The loss function is
calculated as the sum over every value corresponding to an atom pair between the
defined groups given by the switching function (`SWITCH`). For more details about
switching functions, see \\ref switchingfunction.

\plumedfile
MAZE_OPT_RANDOM ...
  LABEL=opt
  GROUPA=2635-2646
  GROUPB=1-2634
  SWITCH={CUSTOM FUNC=exp(-x)/x R_0=0.1}
  STRIDE=1
...
\endplumedfile
*/
//+ENDPLUMEDOC

class Random : public Optimizer {
 public:
  explicit Random(const ActionOptions& action_options);
  ~Random() {}

  static void registerKeywords(Keywords& keys);

  std::pair<double, array_t> optimize() override;
};

PLUMED_REGISTER_ACTION(Random, "MAZE_OPT_RANDOM")

void Random::registerKeywords(Keywords& keys) {
  Optimizer::registerKeywords(keys);

  keys.remove("N_ITER");
}

Random::Random(const ActionOptions& action_options)
  : Action(action_options),
    Optimizer(action_options) {
  log.printf("  Optimizer: Random\n");
  
  update_data_order({"iter", "loss"});
}

std::pair<double, array_t> Random::optimize() {
  array_t neighbor = rnd_neighbor();
  double loss_neighbor = loss(neighbor);

  data_["iter"].push_back(0.0);
  data_["loss"].push_back(loss_neighbor);

  return std::make_pair(loss_neighbor, neighbor / modulo(neighbor));
}

}  // namespace maze
}  // namespace PLMD
