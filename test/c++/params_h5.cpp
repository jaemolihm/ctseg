// Copyright (c) 2026--present, The Simons Foundation
// This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later
// See LICENSE in the root of this distribution for details.

#include <triqs/test_tools/arrays.hpp>
#include <triqs_ctseg/params.hpp>

using triqs_ctseg::solve_params_t;

// Solve parameters of an archive written before measure_auto_corr_time existed: the default (true) is kept
TEST(params_h5, missing_measure_auto_corr_time) {
  solve_params_t p_old;
  p_old.measure_auto_corr_time = false;
  {
    h5::file f("params_h5.out.h5", 'w');
    h5_write(f, "solve_params", p_old);
    h5::group(f).open_group("solve_params").unlink("measure_auto_corr_time");
  }
  h5::file f("params_h5.out.h5", 'r');
  solve_params_t p_read;
  h5_read(f, "solve_params", p_read);
  EXPECT_TRUE(p_read.measure_auto_corr_time);
}
