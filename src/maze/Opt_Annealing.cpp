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
  @file plumed2/src/maze/Opt_Annealing.cpp
*/

#include "core/ActionRegister.h"

#include "Optimizer.h"
#include "Util.h"

namespace PLMD {
namespace maze {

//+PLUMEDOC MAZE_OPT MAZE_OPT_ANNEALING
/*
Optimizer using simulated annealing to find an optimal neighbor toward which the
system can be then biased.

The probability of accepting a randomly generated neighbor is the following:
\f[
  p_{kl} = \min(1, \mathrm{e}^{-\beta \Delta L_{kl}}),
\f]
where \f$ \Delta L_{kl}\f$ is the difference of the loss function corresponding
to the \f$k\f$-th and \f$l\f$-th neighbor. The opimization is performed during
the `N_ITER` iterations.

The value of \f$\beta\f$ is increased depending on the update schedule indicated
by the `BETA_SCHEDULE` keyword using the multiplier \f$ \beta_f \f$ (the
`BETA_FACTOR` keyword). It can be the following:
* Geometric (`GEOM`): \f$ \beta \leftarrow \beta\beta_f \f$,
* Exponential (`EXP`): \f$ \beta \leftarrow \beta\beta_f^{i} \f$, where \f$i\f$
  is the iteration,
* Constant (`CONST`): \f$\beta\f$ is constant during the optimization.

\par Examples
Here, the optimizer is used to find a loss function value between atoms listed
in the `GROUPA` and `GROUPB` corresponding to an optimal neighbor every `STRIDE`.
The loss function is calculated as the sum over every value corresponding to an
atom pair between the defined groups given by the switching function (`SWITCH`).
For more details about switching functions, see \\ref switchingfunction.

\plumedfile
MAZE_OPT_ANNEALING ...
  LABEL=opt
  GROUPA=2635-2646
  GROUPB=1-2634
  SWITCH={CUSTOM FUNC=exp(-x)/x R_0=0.1}
  STRIDE=1
  N_ITER=200
  BETA=0.9
  BETA_FACTOR=1.1
  BETA_SCHEDULE=GEOM
...
\endplumedfile

For more information on how the system can be biased toward the optimal neighbor,
plase see \\ref MAZE_BIAS.

*/
//+ENDPLUMEDOC

class Annealing : public Optimizer {
  using beta_schedule_t = void (Annealing::*)(size_t);

 public:
  explicit Annealing(const ActionOptions& action_options);
  ~Annealing() {}

  static void registerKeywords(Keywords& keys);

  std::pair<double, array_t> optimize() override;

  void beta_schedule(size_t iter);
  void beta_schedule_geom(size_t iter);
  void beta_schedule_exp(size_t iter);
  void beta_schedule_const(size_t iter);

 private:
  double beta_;
  double beta_init_;
  double beta_factor_;
  beta_schedule_t beta_schedule_ptr;
};

PLUMED_REGISTER_ACTION(Annealing, "MAZE_OPT_ANNEALING")

void Annealing::registerKeywords(Keywords& keys) {
  Optimizer::registerKeywords(keys);

  keys.add("compulsory", "BETA", "0.9", "Inverse of temperature parameter used in simulated annealing.");
  keys.add("compulsory", "BETA_FACTOR", "1.05", "Multiplier used to increase `BETA` (inverse of temperature "
      "parameter). Decreses the probability of jumping to a neighbor with a lower loss value.");
  keys.add("optional", "BETA_SCHEDULE", "Schedule for modifying `BETA` parameter (using `BETA_FACTOR` and iteration "
      "number). Available schedules are: `GEOM` geometric, `EXP` exponential, and `CONST` in which the parameter is "
      "not modified (equivalent to Metropolis random walk in loss space).");
}

Annealing::Annealing(const ActionOptions& action_options)
  : Action(action_options),
    Optimizer(action_options) {
  log.printf("  Optimizer: Simulated annealing\n");

  if (keywords.exists("BETA")) {
    double beta_init = 0.9;
    parse("BETA", beta_init);
    plumed_massert(beta_init > 0 && beta_init < 1, "`BETA` should be in [0, 1].");
    beta_init_ = beta_init;
    beta_ = beta_init_;
    log.printf("  Beta: %f\n", beta_init_);
  }

  if (keywords.exists("BETA_FACTOR")) {
    double beta_factor = 1.05;
    parse("BETA_FACTOR", beta_factor);
    plumed_massert(beta_factor > 1, "`BETA_FACTOR` should be greater than 1.");
    beta_factor_ = beta_factor;
    log.printf("  Beta factor: %f\n", beta_factor_);
  }

  if (keywords.exists("BETA_SCHEDULE")) {
    std::string schedule = "GEOM";
    parse("BETA_SCHEDULE", schedule);
    log.printf("  Beta update schedule: %s\n", schedule.c_str());
    if (schedule == "GEOM") {
      beta_schedule_ptr = &Annealing::beta_schedule_geom;
    } else if (schedule == "EXP") {
      beta_schedule_ptr = &Annealing::beta_schedule_exp;
    } else if (schedule == "CONST") {
      beta_schedule_ptr = &Annealing::beta_schedule_const;
    } else {
      plumed_merror("`BETA_SCHEDULE` not recognized: " + schedule + ". Available schedules: `GEOM` (default), "
          "`EXP`, and `CONST`.");
    }
  }

  checkRead();

  update_data_order({"iter", "prob", "accept", "beta", "x", "y", "z", "loss"});
}

void Annealing::beta_schedule_const(size_t iter) {
  // `BETA` is not modified.
}

void Annealing::beta_schedule_geom(size_t iter) {
  beta_ *= beta_factor_;
}

void Annealing::beta_schedule_exp(size_t iter) {
  beta_ *= pow(beta_factor_, iter);
}

void Annealing::beta_schedule(size_t iter) {
  (this->*beta_schedule_ptr)(iter);
}

std::pair<double, array_t> Annealing::optimize() {
  beta_ = beta_init_;
  array_t neighbor = rnd_neighbor();
  double val = loss(neighbor);

  for (size_t i = 0; i < opt_n_iter_; ++i) {
    array_t arg_neighbor = rnd_neighbor();
    double val_neighbor = loss(arg_neighbor);
    double prob = std::min(1.0, exp(-beta_ * (val_neighbor - val)));
    bool accept = rnd_.U01() < prob;
    if (accept) {
      neighbor = arg_neighbor;
      val = val_neighbor;
    }

    data_["accept"].push_back(static_cast<double>(accept));
    data_["iter"].push_back(static_cast<double>(i));
    data_["prob"].push_back(prob);
    data_["beta"].push_back(beta_);
    data_["loss"].push_back(val);
    data_["x"].push_back(neighbor[0] / modulo(neighbor));
    data_["y"].push_back(neighbor[1] / modulo(neighbor));
    data_["z"].push_back(neighbor[2] / modulo(neighbor));

    beta_schedule(i);
  }

  return std::make_pair(val, neighbor / modulo(neighbor));
}

}  // namespace maze
}  // namespace PLMD
