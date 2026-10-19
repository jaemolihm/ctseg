# Copyright (c) 2026, The Simons Foundation
# This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# Symmetric improved estimator with a retarded interaction (the model of dynamical_U.py).
# At low Matsubara frequencies, where the Dyson equation is accurate, the sIE and the
# asymmetric improved estimator must agree with it. Dropping the retarded exchange term
# -FT[D(tau) G(tau)] of the sIE must break this agreement.

from types import SimpleNamespace

import numpy as np

import triqs.utility.mpi as mpi
from triqs.gfs import Gf, BlockGf, Block2Gf, MeshImFreq, inverse, iOmega_n
from triqs.gfs.descriptors import Function
from triqs.operators import n

from triqs_ctseg.solve_generic import solve_generic
from triqs_ctseg.postprocessing import compute_sigma_exchange

beta = 10.
U = 4.
mu = U / 2.
eps = 0.2
wp = 1.
g = 1.
n_iw = 200
n_tau = 2001
n_iw_bosonic = 500
n_tau_bosonic = 2001
n_check = 5  # positive Matsubara frequencies compared

gf_struct = [('down', 1), ('up', 1)]
block_names = [bl for bl, _ in gf_struct]
mesh_f = MeshImFreq(beta=beta, S='Fermion', n_iw=n_iw)
Delta_iw = BlockGf(mesh=mesh_f, gf_struct=gf_struct)
Delta_iw << inverse(iOmega_n - eps)

mesh_b = MeshImFreq(beta=beta, S='Boson', n_iw=n_iw_bosonic)
D0_iw = Block2Gf(block_names, block_names,
                 [[Gf(mesh=mesh_b, target_shape=[1, 1]) for _ in block_names] for _ in block_names],
                 make_copies=False)
for n1, n2 in D0_iw.indices:
    D0_iw[n1, n2] << Function(lambda w: g * wp**2 / (w**2 - wp**2))

h_loc0 = -mu * (n('up', 0) + n('down', 0))
h_int = U * n('up', 0) * n('down', 0)

res = solve_generic(
    Delta_iw, h_loc0, h_int, D0_iw=D0_iw,
    n_iw=n_iw, n_tau=n_tau, n_tau_bosonic=n_tau_bosonic,
    length_cycle=50, n_warmup_cycles=2000, n_cycles=100000,
    analytic_hf=True, measure_I_tau=True,
    random_seed=34788 + 928374 * mpi.rank,
)

S = res.Solver
r = S.results
sl = slice(n_iw, n_iw + n_check)

# the model is spin symmetric: average the two blocks
def avg(gf):
    return sum(gf[bl].data[sl, 0, 0] for bl in block_names) / 2

Sigma_sie = avg(res.Sigma_iw)
Sigma_dyson = avg(res.Sigma_iw_Dyson)
Sigma_asym = avg(res.Sigma_iw_aIE)

# sIE without the retarded part of Sigma^X
S_noD = SimpleNamespace(gf_struct=S.gf_struct, h_int=S.h_int, beta=S.beta, n_iw=S.n_iw, D0_tau=None)
# the static Fock term cancels in X_ret - X_stat
P_zero = {bl: np.zeros((g.target_shape[0],) * 2) for bl, g in r.G_tau}
X_ret = compute_sigma_exchange(S, r.G_tau, P_zero)
X_stat = compute_sigma_exchange(S_noD, r.G_tau, P_zero)
Sigma_noX = Sigma_sie - avg(X_ret) + avg(X_stat)

dev = {name: np.abs(s - Sigma_dyson).max()
       for name, s in [('sIE', Sigma_sie), ('asymmetric IE', Sigma_asym), ('sIE w/o retarded X', Sigma_noX)]}
dev_sx = np.abs(avg(X_ret)).max()

if mpi.is_master_node():
    print("max |Sigma_ex - Sigma_Dyson| over low frequencies:")
    for k, v in dev.items():
        print(f"  {k:>20s}: {v:.4f}")
    print(f"max |retarded Sigma^X| = {dev_sx:.4f}")
    assert dev['sIE'] < 0.05, dev
    assert dev["asymmetric IE"] < 0.05, dev
    assert dev['sIE w/o retarded X'] > 3 * dev['sIE'] and dev['sIE w/o retarded X'] > 0.07, dev
