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
  @file plumed2/src/maze/Optimizer.cpp
*/

#include <algorithm>
#include <ctime>
#include <memory>
#include <limits>
#include <tuple>
#include <utility>

#include "tools/Communicator.h"
#include "tools/OpenMP.h"

#include "Optimizer.h"

namespace PLMD {
namespace maze {

void Optimizer::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);

  keys.remove("NUMERICAL_DERIVATIVES");
  keys.remove("RESTART");
  keys.remove("UPDATE_FROM");
  keys.remove("UPDATE_UNTIL");

  keys.addFlag("NLIST", false, "Use a neighbor list to compute the switching function (`SWITCH` keyword) values that "
      "sum to define the loss between atoms in `GROUPA` and `GROUPB`. Using the neighbor list instead of computing "
      "the loss based on all pairs of atoms is more efficient computationally.");
  keys.addFlag("NL_SERIAL", false, "Perform the calculations on the neighbor list in serial (for debugging).");
  keys.addFlag("NL_COMMITTOR", false, "Turn on to stop calculations when there is no atom pairs in the neighbor list.");
  keys.add("optional", "NL_CUTOFF", "Cutoff distance value for pairs of atoms bewteen `GROUPA` and `GROUPB` in the "
      "neighbor list. Atom pairs with distances larger than `NL_CUTOFF` are not stored in the neighbor list.");
  keys.add("optional", "NL_STRIDE", "Frequency of updating the neigbor list in steps. Every `NL_STRIDE` atom pairs "
      "between `GROUPA` and `GROUPB` are recalculated.");

  keys.add("compulsory", "N_ITER", "200", "Number of steps in each optimization run.");
  keys.add("compulsory", "SWITCH", "Switching function describing interactions between atoms in `GROUPA` and `GROUPB` "
      "required by every optimizer. Details of the various switching functions you can use are provided in "
      "\\ref switchingfunction. The loss value of a `GROUPA`--`GROUPB` configuration is calculated as a sum of the "
      "switching function values. This loss value is minimized during the optimization to find an optimal direction "
      "of biasing.");
  keys.add("compulsory", "STRIDE", "5000", "Frequency of running the optimization procedure in steps.");
  keys.add("optional", "STEP", "Step for generating neighbors by the optimization procedure. By default `STEP` is "
      "estimated adaptively as the distance to the nearest neighbor.");

  keys.add("atoms", "GROUPA", "First list of atoms. Warning: the lists are swapped so that the `GROUPA` list has fewer "
      "atoms.");
  keys.add("atoms", "GROUPB", "Second list of atoms. Warning: the lists are swapped so that the `GROUPB` list has more "
      "atoms. `GROUPB` must have indices, in contrast to a standard usage of `GROUPB` in PLUMED, e.g., the `Coordination` "
      "collective variable; see \\ref COORDINATION");

  keys.add("optional", "RANDOM_SEED", "Seed for generating random numbers can be fixed for reproduction purposes. "
      "Affects the optimization procedure.");

  keys.add("optional", "FILE", "Filename for storing additional information about each optimization run (e.g., loss, "
      "acceptance probability).");
  keys.add("optional", "FMT", "Format of data stored in `FILE`.");
  keys.addFlag("SILENT", false, "Do not output additional information about each optimization run in `FILE`.");

  componentsAreNotOptional(keys);

  keys.addOutputComponent("x", "default", "The `x` component of the optimal neighbor.");
  keys.addOutputComponent("y", "default", "The `y` component of the optimal neighbor.");
  keys.addOutputComponent("z", "default", "The `z` component of the optimal neighbor.");
  keys.addOutputComponent("loss", "default", "Optimal loss value calculated based on the switching function.");
  keys.addOutputComponent("step", "default", "Distance to the nearest neighbor used as the optimization step by "
      "default (see `STEP` keyword).");
}

Optimizer::Optimizer(const ActionOptions& action_options)
  : Action(action_options),
    Colvar(action_options),
    mp_rank_(0),
    mp_stride_(0),
    mp_n_threads_(0),
    pbc_(true),
    nl_started_(false),
    nl_validated_(false),
    nl_list_ptr(nullptr),
    distance_ptr(nullptr),
    opt_n_iter_(1),
    opt_val_(0),
    opt_finished_(false),
    value_(static_cast<size_t>(id::n_value), nullptr),
    output_file_ptr(nullptr) {

  log.printf("  maze module for PLUMED; see Refs. ");
  log << plumed.cite("Rydzewski, Valsson, J. Chem. Phys. 150, 22 (2019)");
  log << plumed.cite("Rydzewski, Comput. Phys. Commun. 247 (2020)");
  log.printf("\n");

  addComponent("x");
  componentIsNotPeriodic("x");
  value_[id::kX] = getPntrToComponent("x");

  addComponent("y");
  componentIsNotPeriodic("y");
  value_[id::kY] = getPntrToComponent("y");

  addComponent("z");
  componentIsNotPeriodic("z");
  value_[id::kZ] = getPntrToComponent("z");

  addComponent("loss");
  componentIsNotPeriodic("loss");
  value_[id::kLoss] = getPntrToComponent("loss");

  addComponent("step");
  componentIsNotPeriodic("step");
  value_[id::kStep] = getPntrToComponent("step");

  if (keywords.exists("SILENT")) {
    bool silent = false;
    parseFlag("SILENT", silent);
    verbose_ = !silent;
    log.printf("  Verbose: %s\n", verbose_ ? "true" : "false");
  }

  if (verbose_) {
    if (keywords.exists("FILE")) {
      std::string output_filename = "optimizer.dat";
      parse("FILE", output_filename);
      plumed_massert(!output_filename.empty(), "Cannot read filename from keyword `FILE`.");
      output_filename_ = output_filename;
      log.printf("  Output filename: %s\n", output_filename_.c_str());
    }

    if (keywords.exists("FMT")) {
      std::string output_fmt = "%15.4f";
      parse("FMT", output_fmt);
      plumed_massert(!output_fmt.empty(), "Cannot read format from keyword `FMT`.");
      output_fmt_ = output_fmt;
      log.printf("  Output file format: %s\n", output_fmt_.c_str());
    }

    output_file_ptr.reset(new OFile());
    output_file_ptr->link(*this);
    output_file_ptr->open(output_filename_);
    output_file_ptr->setLinePrefix("");
    output_file_ptr->fmtField(output_fmt_);
    plumed_massert(output_file_ptr, "Cannot create output file.");
  }

  if (keywords.exists("STRIDE")) {
    size_t opt_stride = 5000;
    parse("STRIDE", opt_stride);
    plumed_massert(opt_stride > 0, "`STRIDE` should be positive.");
    opt_stride_ = opt_stride;
    log.printf("  Optimization every %u steps\n", opt_stride_);
  }

  if (keywords.exists("N_ITER")) {
    size_t opt_n_iter = 200;
    parse("N_ITER", opt_n_iter);
    plumed_massert(opt_n_iter > 0, "`N_ITER` should be positive.");
    opt_n_iter_ = opt_n_iter;
    log.printf("  Number of iterations: %u\n", opt_n_iter_);
  }

  if (keywords.exists("STEP")) {
    double step = 0.0;
    parse("STEP", step);
    if (step == 0) {
      opt_step_adaptive_ = true;
      log.printf("  Using adaptive optimizer step\n");
    } else if (step > 0) {
      opt_step_ = step;
      opt_step_adaptive_ = false;
      log.printf("  Using constant optimizer step: %f\n", opt_step_);
    } else {
      plumed_merror("`STEP` should be positive.");
    }
  }

  if (keywords.exists("SWITCH")) {
    std::string label, errors;
    parse("SWITCH", label);
    if (label.length() > 0) {
      switch_func_.set(label, errors);
      if (errors.length() != 0) {
        plumed_merror("Problem reading `SWITCH=" + label + "`. Error: " + errors);
      }
    }
    log.printf("  Switching function defining loss: SWITCH={%s}\n", label.c_str());
  }

  if (keywords.exists("RANDOM_SEED")) {
    size_t rnd_seed = 0;
    parse("RANDOM_SEED", rnd_seed);
    rnd_seed_ = rnd_seed;
    if (rnd_seed_ == 0) {
      rnd_.setSeed(std::time(0));
      log.printf("  Random seed generated with std::time(0)\n");
    } else {
      rnd_.setSeed(rnd_seed_);
      log.printf("  Constant random seed for reproduction purposes: %u\n", rnd_seed_);
    }
  }

  if (keywords.exists("NOPBC")) {
    bool no_pbc = false;
    parseFlag("NOPBC", no_pbc);
    pbc_ = !no_pbc;
    // Select an appropriate distance function depending on PBC.
    if (pbc_) {
      distance_ptr = &Optimizer::distance_pbc;
    } else {
      distance_ptr = &Optimizer::distance_nopbc;
    }
    plumed_massert(distance_ptr, "Pointer to member functions for calculating distances not linked.");
    log.printf("  Account for PBC when calculating distances: %s\n", pbc_ ? "true" : "false");
    log.printf("  Warning: Turning off PBC is rarely needed in this module\n");
  }

  if (keywords.exists("NLIST")) {
    bool nl_on = false;
    parseFlag("NLIST", nl_on);
    nl_on_ = nl_on;
    log.printf("  Neighbor list: %s\n", nl_on_ ? "true" : "false");
  }

  if (nl_on_) {
    if (keywords.exists("NL_STRIDE")) {
      size_t nl_stride = 500;
      parse("NL_STRIDE", nl_stride);
      plumed_massert(nl_stride > 0, "`NL_STRIDE` must be positive.");
      nl_stride_ = nl_stride;
      log.printf("  Neighbor list stride: %u\n", nl_stride_);
    }

    if (keywords.exists("NL_CUTOFF")) {
      double nl_cutoff = 0.7;
      parse("NL_CUTOFF", nl_cutoff);
      plumed_massert(nl_cutoff > 0, "`NL_CUTOFF` must be positive.");
      nl_cutoff_ = nl_cutoff;
      log.printf("  Neighbor list cutoff: %f\n", nl_cutoff_);
    }

    if (keywords.exists("NL_SERIAL")) {
      bool serial = false;
      parseFlag("NL_SERIAL", serial);
      nl_serial_ = serial;
      log.printf("  Serial: %s\n", nl_serial_ ? "true" : "false");
    }

    if (keywords.exists("NL_COMMITTOR")) {
      bool nl_committor = false;
      parseFlag("NL_COMMITTOR", nl_committor);
      nl_committor_ = nl_committor;
      log.printf("  Neighbor list committor: %s\n", nl_committor_ ? "true" : "false");
    }
  } else {
    nl_committor_ = false;
  }

  if (keywords.exists("GROUPA") && keywords.exists("GROUPB")) {
    vector_t<AtomNumber> nl_group_a;
    parseAtomList("GROUPA", nl_group_a);
    vector_t<AtomNumber> nl_group_b;
    parseAtomList("GROUPB", nl_group_b);
    plumed_massert(!nl_group_a.empty() && !nl_group_b.empty(), "At least one atom group is empty. Provide atom "
        "indices for `GROUPA` and `GROUPB`");

    // Avoid checking for duplicates every time distance is calculated for a
    // pair of atoms in the neighbor list.
    std::sort(nl_group_a.begin(), nl_group_a.end());
    std::sort(nl_group_b.begin(), nl_group_b.end());
    vector_t<AtomNumber> nl_duplicates;
    std::set_intersection(nl_group_a.begin(), nl_group_a.end(), nl_group_b.begin(), nl_group_b.end(),
        std::back_inserter(nl_duplicates));
    if (nl_duplicates.size() > 0) {
      log.printf("  Found duplicates in groups `GROUPA` and `GROUPB`: ");
      for (size_t i = 0; i < nl_duplicates.size(); ++i) {
        log.printf("%u ", nl_duplicates[i].serial());
      }
      log.printf("\n");
      plumed_merror("Atom indices in `GROUPA` and `GROUPB` must be mutually exclusive.");
    }

    bool swap = false;
    if (nl_group_a.size() > nl_group_b.size()) {
      std::swap(nl_group_a, nl_group_b);
      swap = true;
    }
    log.printf("  Lists `GROUPA` and `GROUPB` are swapped: %s\n", swap ? "true" : "false");

    auto const printg = [](Log& log, const vector_t<AtomNumber>& group) {
      size_t i = 0;
      while (i < group.size()) {
        size_t j = i + 1;
        while (group[j].serial() - group[j - 1].serial() == 1) j++;
        if (i != j - 1) log.printf("%u-%u ", group[i].serial(), group[j - 1].serial());
        else log.printf("%u ", group[i].serial());
        i = j;
      }
      log.printf("\n");
    };

    log.printf("  Number of atoms in `GROUPA`: %d\n", nl_group_a.size());
    log.printf("  Indices of `GROUPA`: ");
    printg(log, nl_group_a);

    log.printf("  Number of atoms in `GROUPB`: %d\n", nl_group_b.size());
    log.printf("  Indices of `GROUPB`: ");
    printg(log, nl_group_b);

    // Pairing is not used in this module (i.e., pairing considers only 1-1, 2-2, ...)
    if (nl_on_) {
      nl_list_ptr.reset(new NeighborList(nl_group_a, nl_group_b, nl_serial_, /* pairing */ false, pbc_, getPbc(), comm,
          nl_cutoff_, nl_stride_));
    } else {
      nl_list_ptr.reset(new NeighborList(nl_group_a, nl_group_b, nl_serial_, /* pairing */ false, pbc_, getPbc(), comm));
    }
    requestAtoms(nl_list_ptr->getFullAtomList());

    // Estimate whether it is useful to use multiple threads depending on the
    // size of the neighbor list and the number of processes.
    mp_stride_ = comm.Get_size();
    mp_rank_ = comm.Get_rank();
    mp_n_threads_ = OpenMP::getNumThreads();
    if (mp_n_threads_ * mp_stride_ * 10 > nl_list_ptr->size()) {
      mp_n_threads_ = 1;
    }
    log.printf("  Number of processes: %u\n", mp_stride_);
    log.printf("  Number of threads: %u\n", mp_n_threads_);
  }
}

Optimizer::~Optimizer() {
  if (verbose_) {
    output_file_ptr->close();
  }
}

void Optimizer::prepare() {
  if (nl_list_ptr->getStride() > 0) {
    if (nl_started_ || (getStep() % nl_list_ptr->getStride() == 0)) {
      requestAtoms(nl_list_ptr->getFullAtomList());
      nl_validated_ = true;
      nl_started_ = false;
    } else {
      requestAtoms(nl_list_ptr->getReducedAtomList());
      nl_validated_ = false;
      plumed_massert(getExchangeStep() == 0, "Neighbor list should be updated on exchange step. Choose `NL_STRIDE` "
          "that divides the exchange stride.");
    }
    if (getExchangeStep()) {
      nl_started_ = true;
    }
  }
}

void Optimizer::calculate() {
  if (nl_validated_ && nl_list_ptr->getStride() > 0) {
    nl_list_ptr->update(getPositions());
  }
  if (getStep() % opt_stride_ == 0 && !nl_started_) {
    data_.clear();
    opt_finished_ = false;
    if (opt_step_adaptive_) {
      opt_step_ = dist_neighbor();
    }
    std::pair<double, array_t> solution = optimize();
    std::tie(opt_val_, opt_arg_) = solution;
    opt_finished_ = true;
    if (verbose_) {
      for (size_t i = 0; i < opt_n_iter_; ++i) {
        for (size_t j = 0; j < data_.size(); ++j) {
          const std::string& ndx = data_order_[j];
          output_file_ptr->printField(ndx, data_[ndx][i]);
        }
        output_file_ptr->printField();
      }
      output_file_ptr->printf("\n");
    }
  } else {
    nl_started_ = false;
  }
}

void Optimizer::update() {
  if (opt_finished_) {
    value_[id::kX]->set(opt_arg_[0]);
    value_[id::kY]->set(opt_arg_[1]);
    value_[id::kZ]->set(opt_arg_[2]);
    //double val; array_t tmp;
    //std::tie(val, tmp) = optimize();
    array_t tmp;
    value_[id::kLoss]->set(loss(tmp));
    value_[id::kStep]->set(dist_neighbor());
  } else {
    plumed_merror("Result in the `optimize()` function is not set.");
  }
  if (nl_committor_) {
    plumed_massert(nl_list_ptr->size() != 0, "Sending stop signal to MD engine -- neighbor list does not have any "
        "atom pairs, so the system likely dissociated\n");
  }
}

double Optimizer::distance_pbc(array_t pk, array_t pl) const {
  return pbcDistance(pk, pl).modulo();
}

double Optimizer::distance_nopbc(array_t pk, array_t pl) const {
  return delta(pk, pl).modulo();
}

double Optimizer::distance(size_t ndx_pair) const {
  size_t k, l; std::tie(k, l) = nl_list_ptr->getClosePair(ndx_pair);
  return (this->*distance_ptr)(getPosition(k), getPosition(l));
}

double Optimizer::distance(size_t ndx_pair, array_t neighbor) const {
  size_t k, l; std::tie(k, l) = nl_list_ptr->getClosePair(ndx_pair);
  return (this->*distance_ptr)(getPosition(k) + neighbor, getPosition(l));
}

double Optimizer::dist_neighbor() const {
  double min_dist = std::numeric_limits<double>::max();
  #pragma omp parallel num_threads(mp_n_threads_)
  {
    double min_local = min_dist;
    #pragma omp for nowait
    for (size_t i = mp_rank_; i < nl_list_ptr->size(); i += mp_stride_) {
      double dist = distance(i);
      min_local = std::min(min_local, dist);
    }
    #pragma omp critical
    min_dist = std::min(min_dist, min_local);
  }
  if (!nl_serial_) {
    comm.Min(min_dist);
  }
  return min_dist;
}

double Optimizer::loss(array_t neighbor) const {
  double val = 0;
  #pragma omp parallel num_threads(mp_n_threads_)
  {
    #pragma omp for reduction(+:val)
    for (size_t i = mp_rank_; i < nl_list_ptr->size(); i += mp_stride_) {
      double dist = distance(i, neighbor), dfunc;
      val += switch_func_.calculate(dist, /* derivative divided by `dist` */ dfunc);
    }
  }
  if (!nl_serial_) {
    comm.Sum(val);
  }
  if (nl_list_ptr->size() != 0) {
    return val; /// nl_list_ptr->size();
  }
  else {
    return 0.0;
  }
}

array_t Optimizer::rnd_neighbor() {
  double theta = 2 * pi * rnd_.U01();
  double phi = pi * rnd_.U01();
  array_t neighbor = {sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)};
  return opt_step_ * neighbor;
}

}  // namespace maze
}  // namespace PLMD
