// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later
// See LICENSE in the root of this distribution for details.

#pragma once
#include <triqs/stat/log_binning.hpp>
#include "../configuration.hpp"
#include "../results.hpp"
#include "../work_data.hpp"

namespace triqs_ctseg::measures {

  /// Whether a series is the same constant on all ranks, decided from the per-rank data before any reduction: the
  /// reduced variance of a constant is not exactly 0 when the ranks have different sample counts (round-off in the
  /// reweighted mean of log_binning::mpi_all_reduce), so the 0/0 of tau cannot detect it. Collective.
  bool is_constant_series(triqs::stat::log_binning<double> const &acc, mpi::communicator const &c);

  /// Auto-correlation time of one series (in units of samples) and whether it has saturated with the bin size; NaN
  /// and not saturated if no bin level has 64 blocks. Collective.
  std::pair<double, bool> series_tau(triqs::stat::log_binning<double> const &acc, mpi::communicator const &c);

  /// Auto-correlation times of a few series, one sample per MC cycle (tau is in units of MC cycles).
  ///
  /// Series: the sign, the Delta order (if there is a hybridization), the Jperp order (if there is a Jperp
  /// interaction) and the density n_c of each color. The orders track the expansion; the densities also catch slow
  /// modes at fixed order, e.g. the S_z flips of a local moment, which the orders do not see.
  struct auto_corr_time {

    work_data_t const &wdata;
    configuration_t const &config;
    results_t &results;

    // Series names, in the order of log_accs: sign, Delta_order, Jperp_order, density_{block}_{i}
    std::vector<std::string> names;

    // One log-binning accumulator per series (unbounded, block sizes 2^n)
    std::vector<triqs::stat::log_binning<double>> log_accs;

    auto_corr_time(params_t const &params, work_data_t const &wdata, configuration_t const &config,
                   results_t &results);

    void accumulate(double s);
    void collect_results(mpi::communicator const &c);
  };

} // namespace triqs_ctseg::measures
