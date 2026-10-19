// Copyright (c) 2022--present, The Simons Foundation
// Copyright (c) 2022--present, Max Planck Institute for Polymer Research, Mainz, Germany
// This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
// SPDX-License-Identifier: GPL-3.0-or-later
// See LICENSE in the root of this distribution for details.

#include "./G_F_tau.hpp"
#include "../logs.hpp"

namespace triqs_ctseg::measures {

  G_F_tau::G_F_tau(params_t const &p, work_data_t const &wdata, configuration_t const &config, results_t &results)
     : wdata{wdata}, config{config}, results{results} {

    beta          = p.beta;
    measure_F_tau = p.measure_F_tau and wdata.rot_inv;
    measure_I_tau = p.measure_I_tau and measure_F_tau;
    gf_struct     = p.gf_struct;

    G_tau     = block_gf<imtime>{triqs::mesh::imtime{beta, Fermion, p.n_tau_G}, p.gf_struct};
    F_tau     = block_gf<imtime>{triqs::mesh::imtime{beta, Fermion, p.n_tau_G}, p.gf_struct};
    F_R_tau   = block_gf<imtime>{triqs::mesh::imtime{beta, Fermion, p.n_tau_G}, p.gf_struct};
    I_tau     = block_gf<imtime>{triqs::mesh::imtime{beta, Fermion, p.n_tau_G}, p.gf_struct};
    G_tau()   = 0;
    F_tau()   = 0;
    F_R_tau() = 0;
    I_tau()   = 0;
    Z         = 0;
  }

  // -------------------------------------

  void G_F_tau::accumulate(double s) {

    LOG("\n =================== MEASURE G(tau) ================ \n");

    Z += s;

    for (auto [bl_idx, det] : itertools::enumerate(wdata.dets)) {
      long N   = det.size();
      auto &g  = G_tau[bl_idx];
      auto &f  = F_tau[bl_idx];
      auto &fr = F_R_tau[bl_idx];
      auto &ii = I_tau[bl_idx];

      // Vertex factors, once per operator
      std::vector<double> Iy, Ix;
      if (measure_F_tau) {
        Iy.resize(N);
        for (long id_y : range(N)) Iy[id_y] = fprefactor(bl_idx, det.get_y(id_y), true);
      }
      if (measure_I_tau) {
        Ix.resize(N);
        for (long id_x : range(N)) Ix[id_x] = fprefactor(bl_idx, det.get_x(id_x), false);
      }

      for (long id_y : range(N)) {
        auto y = det.get_y(id_y);
        for (long id_x : range(N)) {
          auto x    = det.get_x(id_x);
          auto Minv = det.inverse_matrix(id_y, id_x);
          // beta-periodicity is implicit in the argument, just fix the sign properly
          auto val  = (y.first >= x.first ? s : -s) * Minv;
          auto dtau = double(y.first - x.first);
          auto pt   = closest_mesh_pt(dtau);
          g[pt](y.second, x.second) += val;
          if (measure_F_tau) f[pt](y.second, x.second) += val * Iy[id_y];
          if (measure_I_tau) {
            fr[pt](y.second, x.second) += val * Ix[id_x];
            ii[pt](y.second, x.second) += val * Iy[id_y] * Ix[id_x];
          }
        }
      }
    }
  }

  // -------------------------------------

  void G_F_tau::collect_results(mpi::communicator const &c) {

    Z = mpi::all_reduce(Z, c);

    G_tau = mpi::all_reduce(G_tau, c);
    G_tau = G_tau / (-beta * Z * G_tau[0].mesh().delta());

    // Fix the point at zero and beta, for each block
    for (auto &g : G_tau) {
      g[0] *= 2;
      g[g.mesh().size() - 1] *= 2;
    }
    // store the result (not reused later, hence we can move it).
    results.G_tau = std::move(G_tau);

    if (measure_F_tau) {
      F_tau = mpi::all_reduce(F_tau, c);
      F_tau = F_tau / (-beta * Z * F_tau[0].mesh().delta());

      for (auto &f : F_tau) {
        f[0] *= 2;
        f[f.mesh().size() - 1] *= 2;
      }
      results.F_tau = std::move(F_tau);
    }

    if (measure_I_tau) {
      for (auto *gf : {&F_R_tau, &I_tau}) {
        *gf = mpi::all_reduce(*gf, c);
        *gf = *gf / (-beta * Z * (*gf)[0].mesh().delta());
        for (auto &x : *gf) {
          x[0] *= 2;
          x[x.mesh().size() - 1] *= 2;
        }
      }
      results.F_R_tau = std::move(F_R_tau);
      results.I_tau   = std::move(I_tau);
    }
  }

  // -------------------------------------

  double G_F_tau::fprefactor(long const &block, std::pair<tau_t, long> const &op, bool at_annihilator) {
    int color    = wdata.block_to_color(block, op.second);
    double I_tau = 0;
    for (auto const &[c, sl] : itertools::enumerate(config.seglists)) {
      auto ntau = n_tau(op.first, sl); // Density to the right of op.first in sl
      if (c != color) I_tau += wdata.U(c, color) * ntau;
      if (wdata.has_Dt) {
        I_tau -= K_overlap(sl, op.first, false, wdata.Kprime, c, color);
        // K'(0) is evaluated as K'(0^+), which is the right limit only at c^dagger
        if (at_annihilator and c == color) I_tau -= 2 * real(wdata.Kprime(0)(c, c));
      }
      if (wdata.has_Jperp) {
        I_tau -= 4 * real(wdata.Kprime_spin(0)(c, color)) * ntau;
        I_tau -= 2 * K_overlap(sl, op.first, false, wdata.Kprime_spin, c, color);
      }
    }
    return I_tau;
  }

} // namespace triqs_ctseg::measures
