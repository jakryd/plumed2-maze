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
  @file plumed2/src/maze/OptimizerBias.cpp
*/

#include <sstream>

#include "core/ActionRegister.h"
#include "core/PlumedMain.h"

#include "bias/Bias.h"

#include "Optimizer.h"

namespace PLMD {
namespace maze {

//+PLUMEDOC MAZE MAZE_OPT_BIAS
/*
Adaptive potential to bias in an optimal biasing direction found by an optimizer
(see \\ref MAZE_OPT).

The adaptive bias requires an optimizer that finds the optimal biasing direction
and a collective variable that is biased toward this biasing direction. The collective
variable must have Cartesian components. As biasing absolute positions of any system
can result in problems (see a detailed explanation in \\ref POSITION) and may require
additional position restraints, it must be a distance with `COMPONENTS`. As such, the
biasing corresponds to a relative position and additional restraints are not required.
See \\ref for an explanation on how to define the `DISTANCE` collective variable
between groups of atoms. The bias height and rate is given by the `HEIGHT` (\f$ h \f$)
and `RATE` (\f$ v \f$) keywords, respectively.

The adaptive bias potential is the following functional form:
\f[
  V({\bf x}_t)=h
      \left(v(t),t - ({\bf x} - {\bf x}^*_{t-\tau}) \cdot
        \frac{{\bf x}^*_t - {\bf x}_t}{\|{\bf x}^*_t-{\bf x}_t\|}
      \right)^2,
\f]
where \f${\bf x}^*_t\f$ is the optimal solution at time \f$t\f$, \f$v\f$ is the
biasing rate, \f$\tau\f$ is the interval at which the loss function is minimized,
and \f$\h\f$ is the bias height.

\par Examples
In this example, the atom groups defined for the distance (`DISTANCE`) and in the
optimizer (`MAZE_OPT_ANNEALING`) are the same, which results in biasing the center
of atoms in `GROUPA` toward the optimal biasing direction found by the optimizer.
However, the distance can be also defined for, e.g., a single atom pair of a
ligand-protein complex.

\plumedfile
CENTER ATOMS=2635-2646 LABEL=group_a
CENTER ATOMS=1-2634 LABEL=group_b

DISTANCE ATOMS=group_a,group_b LABEL=dist COMPONENTS

MAZE_OPT_ANNEALING ...
  LABEL=opt
  GROUPA=2635-2646
  GROUPB=1-2634
  SWITCH={CUSTOM FUNC=exp(-x)/x R_0=0.1}
  STRIDE=1
  N_ITER=200
...

MAZE_OPT_BIAS ...
  LABEL=biaser
  ARG=dist.x,dist.y,dist.z
  OPTIMIZER=opt
  HEIGHT=1
  RATE=0.001
...
\endplumedfile

The biasing rate is kept constant by default which means that, e.g., a ligand is
biased with a constant velocity in the optimal biasing direction. However, the
biasing rate can be also calculated adaptively based on the so-called
acceleration factor:
\f[
  \left< \Delta t \mathrm{e}^{\beta V} \right>,
\f]
where \f$ V \f$ is the value of the adaptive biasing potential and \f$ \Delta t\f$ is
the simulation time step. The calculation of the acceleration factor is disabled
by default and can be switched on using the `CALC_ACC` keyword.

Then, by providing the `RATE_ADAPTIVE` keyword the biasing rate can be adaptively
reduced using the following formula:
\f[

\f]
which is very similar to the frequency adaptive variant of metadynamics
(\\ref FA_). Then the `RATE` keyword defines the initial biasing rate.

The following example illustrates how the adaptive rate can be turned on:

\plumedfile
MAZE_OPT_BIAS ...
  LABEL=biaser
  ARG=dist.x,dist.y,dist.z
  OPTIMIZER=opt
  HEIGHT=1
  CALC_ACC
  TEMP=300
  RATE=0.001
  RATE_ADAPTIVE
  RATE_MIN_ACC=1.5
  RATE_STRIDE=2
...

PRINT ARG=biaser.acc,biaser.rate FILE=colvar.dat FMT=%8.3f
\endplumedfile

Above, the `RATE_MIN_ACC` keyword indicates starting value of the acceleration
factor from which the biasing rate is reduced with a stride of `RATE_STRIDE`. The
resulting output file `colvar.dat` can look like this:

\auxfile{}
#! FIELDS time biaser.acc biaser.rate
 0.000    1.000    0.010
 2.000    1.001    0.010
 4.000    1.002    0.010
 6.000    1.029    0.010
 8.000    1.030    0.010
 10.00    1.031    0.010
 12.00    1.030    0.010
 14.00    1.034    0.010
 16.00    1.045    0.010
 18.00    1.071    0.010
 20.00    1.266    0.010
 22.00    1.496    0.010
 24.00    1.696    0.010
 26.00    1.854    0.009
 28.00    1.967    0.008
\endauxfile

*/
//+ENDPLUMEDOC

class OptimizerBias : public bias::Bias {
 public:
  explicit OptimizerBias(const ActionOptions& action_options);
  ~OptimizerBias() {}

  static void registerKeywords(Keywords& keys);

  void calculate() override;

 private:
  vector_t<value_t*> arguments_;
  array_t arg_;

  Optimizer* optimizer_ptr;

  double height_;
  double time_;
  double bias_;
  double force_;
  double opt_;
  
  // If rate should be decreased when acceleration factor reaches minimal 
  // acceleration factor.
  bool rate_adaptive_;
  double rate_;
  double rate_init_;
  size_t rate_stride_;
  // Minimal acceleration factor value starting from which biasing rate is 
  // decreased.
  double min_acc_;
  double cumdist_;
  
  double kbt_;
  // Calculate hyperdynamics-like acceleration factor.
  bool calc_acceleration_;
  double acceleration_factor_;
  bool calc_work_;
  double work_;

  enum id {kAcc, kWork, kForce, kForce2, kRate, kCumDist, kDelta, 
      /* number of elements in enum */ n_value = kDelta - kAcc};
  vector_t<value_t*> value_;
};

PLUMED_REGISTER_ACTION(OptimizerBias, "MAZE_OPT_BIAS")

void OptimizerBias::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);

  keys.use("ARG");

  keys.add("compulsory", "OPTIMIZER", "Optimization technique that should be selected from {`MAZE_OPT_ANNEALING`, "
      "`MAZE_OPT_CONSTANT`, `MAZE_OPT_RANDOM`, `MAZE_OPT_BRUTE`}. For additional details about defining optimizers, "
      "see \\ref MAZE_OPT.");
  keys.add("compulsory", "HEIGHT", "1.0", "Height of biasing potential.");

  keys.add("compulsory", "RATE", "0.01", "Rate at the collective variable defined by `ARG` should be biased.");
  keys.add("optional", "RATE_MIN_ACC", "Minimal value of the acceleration factor starting from which the biasing rate "
      "is reduced.");
  keys.add("optional", "RATE_STRIDE", "Frequency in steps for updating the biasing rate.");
  keys.addFlag("RATE_ADAPTIVE", false, "Enable for the reduction of the biasing rate based on the calculated "
      "acceleration factor. Requires computing the acceleration factor that can be enabled by keyword `CALC_ACC`)");

  keys.add("optional", "TEMP", "Temperature is needed only if `CALC_ACC` is enabled and PLUMED does not have the "
      "information about \\f$k_BT\\f$.");
  keys.addFlag("CALC_ACC", false, "Enable to compute the acceleration factor similarily like in metadynamics.");
  keys.addFlag("CALC_WORK", false, "Enable to calculate the work performed by biasing `ARG`.");

  componentsAreNotOptional(keys);

  keys.addOutputComponent("force", "default", "Magnitude of force acting on coordinates passed by `ARG`.");
  keys.addOutputComponent("force2", "default", "Instantaneous value of the squared force due to this bias potential.");
  keys.addOutputComponent("delta", "default", "Displacement between coordinates given by `ARG` and a dummy atom moving "
      "with a constant `RATE`.");
  keys.addOutputComponent("cumdist", "default", "Total distance of the projections of displacements on the direction "
      "of biasing.");
  keys.addOutputComponent("acc", "CALC_ACC", "Acceleration factor \\f$\\<exp(\\beta V)\\>\\f$.");
  keys.addOutputComponent("work", "CALC_WORK", "Work in the direction of biasing.");
  keys.addOutputComponent("rate", "RATE_ADAPTIVE", "Biasing rate if the reduction based on the acceleration factor is "
      "enabled.");
}

OptimizerBias::OptimizerBias(const ActionOptions& action_options)
  : PLUMED_BIAS_INIT(action_options),
    time_(0.0),
    bias_(0.0),
    force_(0.0),
    opt_(0.0),
    rate_(0.0), 
    cumdist_(0.0), 
    kbt_(0.0), 
    acceleration_factor_(0.0), 
    work_(0.0),
    value_(static_cast<size_t>(id::n_value), nullptr) {

  log.printf("  maze module for PLUMED; see Refs. ");
  log << plumed.cite("Rydzewski, Valsson, J. Chem. Phys. 150, 22 (2019)");
  log << plumed.cite("Rydzewski, Comput. Phys. Commun. 247 (2020)");
  log.printf("\n");

  addComponent("force");
  componentIsNotPeriodic("force");
  value_[id::kForce] = getPntrToComponent("force");

  addComponent("force2");
  componentIsNotPeriodic("force2");
  value_[id::kForce2] = getPntrToComponent("force2");

  addComponent("delta");
  componentIsNotPeriodic("delta");
  value_[id::kDelta] = getPntrToComponent("delta");

  addComponent("cumdist");
  componentIsNotPeriodic("cumdist");
  value_[id::kCumDist] = getPntrToComponent("cumdist");

  // Check if `ARG` is defined using `DISTANCE` keyword and has Cartesian components.
  std::string available_components = "xyz";
  arguments_ = getArguments();
  plumed_massert(arguments_.size() == available_components.length(), "Wrong number of arguments.");
  size_t i = 0;
  while (available_components.length() != 0) {
    std::string arg_action_name = arguments_[i]->getPntrToAction()->getName();
    if (arg_action_name == "DISTANCE") {
      log.printf("  Argument %s defined by action `DISTANCE` ", arguments_[i]->getName().c_str());
    } else {
      // https://www.plumed.org/doc-v2.8/user-doc/html/_p_o_s_i_t_i_o_n.html
      plumed_merror("Argument not defined by action `DISTANCE`. Coordinates given to `ARG` should be a relative "
          "position to avoid biasing absolute positions. For details about this issue, see warnings for the "
          "`POSITION` collective variable.");
    }
    std::string arg_comp_name = arguments_[i]->getName();
    std::stringstream stream(arg_comp_name);
    vector_t<std::string> stream_split;
    std::string s;
    while (stream.good()) {
      s.clear();
      getline(stream, s, '.');
      stream_split.push_back(s);
    }
    // Find the name of component and check if they are correctly named.
    std::string component_suffix = stream_split.back();
    size_t position = available_components.find(component_suffix);
    if (position != std::string::npos) {
      log.printf("with component: %s\n", component_suffix.c_str());
      available_components.erase(position, component_suffix.length());
    } else {
      // https://www.plumed.org/doc-v2.8/user-doc/html/_d_i_s_t_a_n_c_e.html
      plumed_merror("Component of action `DISTANCE` should be `x`, `y`, or `z`. Ensure that `COMPONENTS` flag is added "
          "to action `DISTANCE` (not `SCALED_COMPONENT`).");
    }
    i++;
  }

  if (keywords.exists("HEIGHT")) {
    double bias_constant = 1;
    parse("HEIGHT", bias_constant);
    plumed_massert(bias_constant > 0, "Bias height must be positive.");
    height_ = bias_constant;
    log.printf("  Bias height: %6.4f\n", height_);
  }

  if (keywords.exists("CALC_ACC")) {
    bool calc_acceleration = false;
    parseFlag("CALC_ACC", calc_acceleration);
    calc_acceleration_ = calc_acceleration;
    log.printf("  Calculate acceleration factor: %s\n", calc_acceleration_ ? "true" : "false");
    if (calc_acceleration_) {
      if (keywords.exists("TEMP")) {
        double temperature = 0;
        parse("TEMP", temperature);
        if (temperature > 0) {
          kbt_ = plumed.getAtoms().getKBoltzmann() * temperature;
          log.printf("  Temperature: %6.4f K read from keyword `TEMP` found in the PLUMED input file\n", temperature);
        } else {
          kbt_ = plumed.getAtoms().getKbT();
          log.printf("  Temperature not needed; kT found in the MD engine\n");
        }
        plumed_massert(kbt_ > 0, "kT is not set; if you are using `plumed driver`, provide `--kt` or `TEMP` keyword in "
            "the PLUMED input file.");
        log.printf("  kT: %6.4f and inverse temperature: %6.4f\n", kbt_, 1.0 / kbt_);
        addComponent("acc");
        componentIsNotPeriodic("acc");
        value_[id::kAcc] = getPntrToComponent("acc");
      }
    }
  }

  if (keywords.exists("RATE")) {
    double bias_rate = 0.01;
    parse("RATE", bias_rate);
    plumed_massert(bias_rate > 0, "Bias rate must be positive.");
    rate_init_ = bias_rate;
    rate_ = bias_rate;
    log.printf("  Bias rate: %6.4f\n", rate_init_);
  }

  if (keywords.exists("RATE_ADAPTIVE")) {
    bool rate_adaptive = false;
    parseFlag("RATE_ADAPTIVE", rate_adaptive);
    rate_adaptive_ = rate_adaptive;
    log.printf("  Adaptive rate: %s\n", rate_adaptive_ ? "true" : "false");
    if (rate_adaptive_) {
      if (calc_acceleration_) {
        if (keywords.exists("RATE_MIN_ACC")) {
          double min_acc = 1.0;
          parse("RATE_MIN_ACC", min_acc);
          plumed_massert(min_acc >= 1.0, "Minimal acceleration must be at least 1.0.");
          min_acc_ = min_acc;
          log.printf("  Minimal acceleration for adaptive rate calculations: %6.4f\n", min_acc_);
        }
        if (keywords.exists("RATE_STRIDE")) {
          size_t rate_stride = 1;
          parse("RATE_STRIDE", rate_stride);
          plumed_massert(rate_stride >= 1, "Rate stride must be positive.");
          rate_stride_ = rate_stride;
          log.printf("  Rate update will be performed every %u steps\n", rate_stride_);
        }
        addComponent("rate");
        componentIsNotPeriodic("rate");
        value_[id::kRate] = getPntrToComponent("rate");
      } else {
        plumed_merror("`CALC_ACC` must be turned on for using `ADAPTIVE_RATE`.");
      }
    }
  }

  if (keywords.exists("CALC_WORK")) {
    bool work = false;
    parseFlag("CALC_WORK", work);
    calc_work_ = work;
    log.printf("  Calculate work: %s\n", calc_work_ ? "true" : "false");
    if (calc_work_) {
      addComponent("work");
      componentIsNotPeriodic("work");
      value_[id::kWork] = getPntrToComponent("work");
    }
  }

  if (keywords.exists("OPTIMIZER")) {
    vector_t<std::string> action_labels;
    parseVector("OPTIMIZER", action_labels);
    plumed_massert(!action_labels.empty(), "Provide the label of `OPTIMIZER` keyword.");
    if (action_labels.size() != 1) {
      log.printf("  Only one optimizer is required; parsing the first provided label.\n");
      log.printf("  Skipping remaining optimizer labels: ");
      for (size_t i = 1; i < action_labels.size(); ++i) {
        log.printf("%s ", action_labels[i].c_str());
      }
      log.printf("\n");
    }
    optimizer_ptr = util::link_action<Optimizer*>(action_labels.front(), plumed.getActionSet());
    if (optimizer_ptr) {
      log.printf("  Optimizer linked: %s\n", action_labels.front().c_str());
    } else {
      plumed_merror("Problem reading the label of `OPTIMIZER` keyword; check if label `" + action_labels.front() + \
          "` exists.");
    }
  }

  checkRead();
}

void OptimizerBias::calculate() {
  size_t opt_stride = optimizer_ptr->stride();
  array_t opt_neighbor = optimizer_ptr->neighbor();
  opt_neighbor /= modulo(opt_neighbor);
  size_t step = getStep();
  double time = getTime();
  size_t stride = getStride();

  array_t arg = util::unpack_arguments(arguments_);

  if (step % opt_stride == 0) {
    arg_ = arg;
    time_ = time;
  }

  double opt_projection = dotProduct(arg - arg_, opt_neighbor);
  double delta = rate_ * (time - time_) - opt_projection;
  double bias = height_ * pow(delta, 2);
  double force = 2.0 * height_ * delta;

  setBias(bias);
  value_[id::kForce]->set(force);
  value_[id::kForce2]->set(pow(force, 2));
  for (size_t i = 0; i < arguments_.size(); ++i) {
    setOutputForce(i, force * opt_neighbor[i]);
  }

  value_[id::kDelta]->set(delta);
  value_[id::kCumDist]->set(fabs(cumdist_));

  if (calc_acceleration_) {
    double acc_mean = 1.0;
    if (step > 0) {
      acceleration_factor_ += static_cast<double>(stride) * exp(bias / kbt_);
      acc_mean = acceleration_factor_ / ((double)step);
    }
    value_[id::kAcc]->set(std::log(acc_mean));

    if (rate_adaptive_) {
      value_[id::kRate]->set(rate_);
      if (step % rate_stride_ == 0) {
        rate_ = rate_init_ * std::min(1.0, min_acc_ / acc_mean);
      }
    }
  }

  if (calc_work_) {
    work_ += delta * force;
    value_[id::kWork]->set(work_);
  }
}

}  // namespace maze
}  // namespace PLMD
