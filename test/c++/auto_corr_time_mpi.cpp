// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later
// See LICENSE in the root of this distribution for details.

#include <triqs/test_tools/arrays.hpp>
#include <triqs_ctseg/measures/auto_corr_time.hpp>

#include <array>
#include <cmath>

using triqs::stat::log_binning;
using triqs_ctseg::measures::series_tau;

// Run on 3 ranks. Sample counts 3, 22, 44 per rank: sum_r (n_r / N) * 1.0 != 1.0 in every reduction order.
long n_samples(int rank) { return std::array<long, 3>{3, 22, 44}[rank % 3]; }

// A constant series with unequal sample counts: the reduced variance is round-off, not 0, so tau from the bins is
// finite garbage; series_tau reports tau = 0, saturated. 3 + 22 + 44 = 69 blocks at level 0.
TEST(auto_corr_time, constant_unequal_counts) {
  mpi::communicator c;
  if (c.size() != 3) GTEST_SKIP() << "needs 3 ranks";
  log_binning<double> acc{0.0, -1};
  for (long i = 0; i < n_samples(c.rank()); ++i) acc << 1.0;
  auto [mean, errs, taus, effs] = acc.mean_errors_and_taus(c, 2);
  EXPECT_TRUE(std::isfinite(taus[0])); // not the 0/0 (NaN) of a zero variance
  auto [tau, converged] = series_tau(acc, c);
  EXPECT_EQ(tau, 0.0);
  EXPECT_TRUE(converged);
}

// A rank without samples (max_time reached during warmup) holds the zeroed initial sample; the others hold 1.0
TEST(auto_corr_time, constant_with_empty_rank) {
  mpi::communicator c;
  if (c.size() != 3) GTEST_SKIP() << "needs 3 ranks";
  log_binning<double> acc{0.0, -1};
  long const n = (c.rank() == 0) ? 0 : 22 * c.rank(); // 0, 22, 44
  for (long i = 0; i < n; ++i) acc << 1.0;
  auto [tau, converged] = series_tau(acc, c);
  EXPECT_EQ(tau, 0.0);
  EXPECT_TRUE(converged);
}

// Constant 0 (mean 0): reduces exactly, tau = 0, saturated
TEST(auto_corr_time, constant_zero) {
  mpi::communicator c;
  log_binning<double> acc{0.0, -1};
  for (long i = 0; i < 100; ++i) acc << 0.0;
  auto [tau, converged] = series_tau(acc, c);
  EXPECT_EQ(tau, 0.0);
  EXPECT_TRUE(converged);
}

// +1, -1, +1, ... (mean 0, variance 1) is not constant: blocks of 2 are all 0, so tau_1 = (0 / s_0^2 - 1) / 2 = -0.5,
// the exact value for perfect anti-correlation
TEST(auto_corr_time, alternating_mean_zero) {
  mpi::communicator c;
  log_binning<double> acc{0.0, -1};
  for (long i = 0; i < 200; ++i) acc << ((i % 2 == 0) ? 1.0 : -1.0);
  auto [tau, converged] = series_tau(acc, c);
  EXPECT_EQ(tau, -0.5);
}

// A different constant on each rank (0, 1, 2) is not constant
TEST(auto_corr_time, rank_dependent_constant) {
  mpi::communicator c;
  if (c.size() == 1) GTEST_SKIP() << "needs several ranks";
  log_binning<double> acc{0.0, -1};
  for (long i = 0; i < 100; ++i) acc << double(c.rank());
  auto [tau, converged] = series_tau(acc, c);
  EXPECT_GT(tau, 0.0);
}

MAKE_MAIN;
