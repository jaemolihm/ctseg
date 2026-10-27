// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later
// See LICENSE in the root of this distribution for details.

#include "auto_corr_time.hpp"
#include <array>
#include <cmath>
#include <itertools/itertools.hpp>
#include <limits>
#include "../logs.hpp"

namespace triqs_ctseg::measures {

  bool is_constant_series(triqs::stat::log_binning<double> const &acc, mpi::communicator const &c) {
    // A rank without samples (e.g. max_time reached during warmup) holds the zeroed initial sample: ignore it
    bool const empty = acc.count() == 0;
    bool const flat  = mpi::all_reduce(int(empty or acc.var_bins()[0] == 0.0), c, MPI_LAND);
    constexpr double inf = std::numeric_limits<double>::infinity();
    double const m       = acc.mean_bins()[0];
    // {min, -max} of the means over the ranks with samples
    auto const m_ext = mpi::all_reduce(std::array<double, 2>{empty ? inf : m, empty ? inf : -m}, c, MPI_MIN);
    return flat and m_ext[0] == -m_ext[1];
  }

  // -------------------------------------

  std::pair<double, bool> series_tau(triqs::stat::log_binning<double> const &acc, mpi::communicator const &c) {

    // tau saturates at large bin size. We report it at the deepest bin level with >= min_samples blocks and flag it
    // as a lower bound if its finer neighbour (half the bin size) is more than n_sigma errors below it. The error of
    // tau from M blocks is d_tau = (tau + 1/2) * sqrt(2 / (M - 1)) (variance of a variance); the deeper level
    // dominates the uncertainty, so its d_tau is the yardstick. Same rules as cthyb's measure_auto_corr_time.
    constexpr int min_samples = 64;
    constexpr double n_sigma  = 2.0;

    auto d_tau = [](double tau, long M) { return (tau + 0.5) * std::sqrt(2.0 / static_cast<double>(M - 1)); };

    // e.g. the sign of a sign-problem-free run
    bool const flat = is_constant_series(acc, c);

    // All-reduces over the ranks, so every rank obtains the same result
    auto [mean, errs, taus, effs] = acc.mean_errors_and_taus(c, min_samples);

    // No bin level with min_samples blocks: the run is too short to estimate tau
    if (taus.empty()) return {std::numeric_limits<double>::quiet_NaN(), false};

    // Zero variance: tau = 0, saturated
    if (flat) return {0.0, true};

    double const tau = taus.back();
    auto const n     = taus.size();
    // Only level 0 (tau_0 = 0 by definition), not finite, or still rising: lower bound
    bool const converged =
       n >= 2 and std::isfinite(tau) and std::isfinite(taus[n - 2]) and tau - taus[n - 2] <= n_sigma * d_tau(tau, effs[n - 1]);
    return {tau, converged};
  }

  // -------------------------------------

  auto_corr_time::auto_corr_time(params_t const &, work_data_t const &wdata, configuration_t const &config,
                                 results_t &results)
     : wdata{wdata}, config{config}, results{results} {

    names.emplace_back("sign");
    if (wdata.has_Delta) names.emplace_back("Delta_order");
    if (wdata.has_Jperp) names.emplace_back("Jperp_order");
    // Color = block offset + i, as in densities
    for (auto const &[bl_name, bl_size] : wdata.gf_struct)
      for (long i = 0; i < bl_size; ++i) names.emplace_back(fmt::format("density_{}_{}", bl_name, i));

    log_accs.assign(names.size(), triqs::stat::log_binning<double>{0.0, -1});
  }

  // -------------------------------------

  void auto_corr_time::accumulate(double s) {
    // Raw values, not weighted by the sign
    long k = 0;
    log_accs[k++] << s;
    if (wdata.has_Delta) log_accs[k++] << double(config.Delta_order());
    if (wdata.has_Jperp) log_accs[k++] << double(config.Jperp_order());
    for (auto const &seglist : config.seglists) {
      double sum = 0;
      for (auto const &seg : seglist) sum += double(seg.length()); // accounts for cyclicity
      log_accs[k++] << sum / tau_t::beta();
    }
  }

  // -------------------------------------

  void auto_corr_time::collect_results(mpi::communicator const &c) {

    double tau_max     = 0.0;
    bool all_converged = true;
    std::map<std::string, double> taus_out;
    std::vector<bool> converged(names.size());

    for (auto const &[k, name] : itertools::enumerate(names)) {
      auto const [tau, conv] = series_tau(log_accs[k], c);
      if (std::isfinite(tau)) tau_max = std::max(tau_max, tau);
      taus_out[name] = tau;
      converged[k]   = conv;
      all_converged  = all_converged and conv;
    }

    if (c.rank() == 0) {
      SPDLOG_INFO("Auto-correlation times (MC cycles):");
      for (auto const &[k, name] : itertools::enumerate(names)) {
        double tau = taus_out[name];
        if (std::isnan(tau))
          SPDLOG_INFO("  {}: too few cycles to estimate tau (run longer)", name);
        else if (converged[k])
          SPDLOG_INFO("  {}: tau = {:.3g} cycles", name, tau);
        else
          SPDLOG_INFO("  {}: tau >~ {:.3g} cycles (not converged, run longer)", name, tau);
      }
      if (all_converged)
        SPDLOG_INFO("Auto-correlation time: {:.3g} cycles", tau_max);
      else
        SPDLOG_INFO("Auto-correlation time: >~ {:.3g} cycles (not converged, run longer)", tau_max);
    }

    results.auto_corr_time           = tau_max;
    results.auto_corr_time_converged = all_converged;
    results.auto_corr_times          = std::move(taus_out);
  }

} // namespace triqs_ctseg::measures
