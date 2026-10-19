# Copyright (c) 2026, The Simons Foundation
# This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# Symmetric improved estimator against exact diagonalization.
#
# Two orbitals (one block) with a density-density interaction and a diagonal local
# Hamiltonian, coupled to a two-site bath that makes Delta(iw) off-diagonal. Checks
#  1. the measured F^L, F^R, I against the equation-of-motion values built from the exact
#     Sigma and G: F^L = Sigma G, F^R = G Sigma, I = Sigma - Sigma^HX + Sigma G Sigma,
#  2. F^R = (F^L)^T,
#  3. the symmetric improved estimator (including the static Fock term) against ED.

import numpy as np

import triqs.utility.mpi as mpi
from triqs.gfs import inverse

from triqs_ctseg.solve_generic import solve_generic
from triqs_ctseg.postprocessing import _fourier_wo_moments, density_matrix_from_F

import sie_ed_model as model

n_iw = 100
n_tau = 1001
n_check = 3  # positive Matsubara frequencies compared

ed = model.solve_ed(n_iw)
Delta_iw = ed['Delta_iw']
Sigma_ed = ed['Sigma_iw']['ud'].data[n_iw:n_iw + n_check]

res = solve_generic(
    Delta_iw, [model.h0_mat], model.h_int,
    n_iw=n_iw, n_tau=n_tau,
    n_warmup_cycles=2000, n_cycles=100000, length_cycle=50,
    measure_I_tau=True, random_seed=34788 + 928374 * mpi.rank,
)

r = res.Solver.results
mesh = Delta_iw.mesh
G, FL, FR, I = (_fourier_wo_moments(x, mesh)['ud'].data for x in (r.G_tau, r.F_tau, r.F_R_tau, r.I_tau))
sl = slice(n_iw, n_iw + n_check)

# Equation-of-motion values from the exact Sigma and G
Sigma_ex, G_ex = ed['Sigma_iw']['ud'].data, ed['G_iw']['ud'].data
Sigma_HX = ed['Sigma_H'] + ed['Sigma_X']
FL_ex = Sigma_ex @ G_ex
FR_ex = G_ex @ Sigma_ex
I_ex = Sigma_ex - Sigma_HX + Sigma_ex @ G_ex @ Sigma_ex

dev = {
    'G': np.abs(G[sl] - G_ex[sl]).max(),
    'F^L': np.abs(FL[sl] - FL_ex[sl]).max(),
    'F^R': np.abs(FR[sl] - FR_ex[sl]).max(),
    'I': np.abs(I[sl] - I_ex[sl]).max(),
    'F^R-(F^L)^T': np.abs(FR[sl] - FL[sl].transpose(0, 2, 1)).max(),
    'sIE': np.abs(res.Sigma_iw['ud'].data[sl] - Sigma_ed).max(),
    'Dyson': np.abs(res.Sigma_iw_Dyson['ud'].data[sl] - Sigma_ed).max(),
}
# off-diagonal density from the single bin G(beta^-) against ED
dev['n_01'] = abs(-r.G_tau['ud'].data[-1, 0, 1] - ed['n_mat'][0, 1])
dev['n_aa'] = np.abs(np.array(r.densities['ud']) - ed['densities']).max()
# density matrix from F^L, as used for the Fock term
P = density_matrix_from_F(r.F_tau, ed['G0_iw'], [model.h0_mat])['ud']
dev['P (F^L)'] = np.abs(P - ed['n_mat']).max()

if mpi.is_master_node():
    for k, v in dev.items():
        print(f"max deviation {k:>12s}: {v:.4f}")
    print("exact Sigma^X_01 =", ed['Sigma_X'][0, 1])
    assert abs(ed['Sigma_X'][0, 1]) > 0.1  # the Fock term must matter

    # about 3 times the deviations observed at this number of cycles
    tol = {'G': 0.02, 'F^L': 0.05, 'F^R': 0.05, 'I': 0.1, 'F^R-(F^L)^T': 0.05, 'sIE': 0.1,
           'Dyson': 0.2, 'n_01': 0.05, 'n_aa': 0.01, 'P (F^L)': 0.01}
    for k, t in tol.items():
        assert dev[k] < t, f"{k}: deviation {dev[k]} exceeds {t}"
