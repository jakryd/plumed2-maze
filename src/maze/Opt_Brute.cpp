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
  @file plumed2/src/maze/Opt_Brute.cpp
*/

#include "core/ActionRegister.h"

#include "Optimizer.h"
#include "Util.h"

namespace PLMD {
namespace maze {

//+PLUMEDOC MAZE_OPT MAZE_OPT_BRUTE
/*
Optimizer using a brute approach to find an optimal neighbor toward which the
system can be then biased.

The probability of accepting a randomly generated neighbor is 1 when the
neighbor has a lower loss function value and 0 otherwise. The opimization is
performed during the `N_ITER` iterations.

\par Examples
Here, the optimizer is used to find a loss function value between atoms listed
in the `GROUPA` and `GROUPB` corresponding to an optimal neighbor every `STRIDE`.
The loss function is calculated as the sum over every value corresponding to an
atom pair between the defined groups given by the switching function (`SWITCH`).
For more details about switching functions, see \\ref switchingfunction.

\plumedfile
MAZE_OPT_BRUTE ...
  LABEL=opt
  GROUPA=2635-2646
  GROUPB=1-2634
  SWITCH={CUSTOM FUNC=exp(-x)/x R_0=0.1}
  STRIDE=1
  N_ITER=200
...
\endplumedfile

For more information on how the system can be biased toward the optimal neighbor,
plase see \\ref MAZE_BIAS.

*/
//+ENDPLUMEDOC

class Brute : public Optimizer {
 public:
  explicit Brute(const ActionOptions& action_options);
  ~Brute() {}

  static void registerKeywords(Keywords& keys);

  std::pair<double, array_t> optimize() override;
};

PLUMED_REGISTER_ACTION(Brute, "MAZE_OPT_BRUTE")

void Brute::registerKeywords(Keywords& keys) {
  Optimizer::registerKeywords(keys);
}

Brute::Brute(const ActionOptions& action_options)
  : Action(action_options),
    Optimizer(action_options) {
  log.printf("  Optimizer: Brute\n");
  
  update_data_order({"iter", "accept", "loss"});
}

std::pair<double, array_t> Brute::optimize() {
  array_t neighbor = rnd_neighbor();
  double val = loss(neighbor);

  for (size_t i = 0; i < opt_n_iter_; ++i) {
    array_t arg_neighbor = rnd_neighbor();
    double val_neighbor = loss(arg_neighbor);
    bool accept = val_neighbor - val < 0;
    if (accept) {
      neighbor = arg_neighbor;
      val = val_neighbor;
    }

    data_["accept"].push_back(static_cast<double>(accept));
    data_["iter"].push_back(static_cast<double>(i));
    data_["loss"].push_back(val);
  }

  return std::make_pair(val, neighbor / modulo(neighbor));
}

}  // namespace maze
}  // namespace PLMD
