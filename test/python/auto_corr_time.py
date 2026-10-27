# Copyright (c) 2026--present, The Simons Foundation
# This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# Auto-correlation time measure: Anderson model of anderson.py and Jperp model of Jperp.py.
import math
from triqs.gfs import *
import triqs.utility.mpi as mpi
from triqs.gfs.descriptors import Function
from triqs.operators import n
import h5
from triqs_ctseg import Solver

beta = 10
U = 4.0
mu = U / 2
eps = 0.2
n_tau = 2051
n_tau_bosonic = 2001
gf_struct = [('down', 1), ('up', 1)]
constr_params = {"gf_struct": gf_struct, "beta": beta, "n_tau": n_tau, "n_tau_bosonic": n_tau_bosonic}
h_int = U * n("up", 0) * n("down", 0)
h_loc0 = -mu * (n("up", 0) + n("down", 0))


def anderson_solver():
    S = Solver(**constr_params)
    iw_mesh = MeshImFreq(beta, 'Fermion', n_tau // 2)
    delta_iw = GfImFreq(indices=[0], mesh=iw_mesh)
    delta_iw << inverse(iOmega_n - eps)
    S.Delta_tau["up"] << Fourier(delta_iw)
    S.Delta_tau["down"] << Fourier(delta_iw)
    return S


def solve_anderson(**params):
    S = anderson_solver()
    solve_params = {"h_int": h_int, "h_loc0": h_loc0, "length_cycle": 50, "n_warmup_cycles": 1000,
                    "n_cycles": 10000, "random_seed": 34788 + 928374 * mpi.rank}
    solve_params.update(params)
    S.solve(**solve_params)
    return S.results


# Series, finite values, max over series
res50 = solve_anderson(length_cycle=50)
taus50 = res50.auto_corr_times
assert set(taus50.keys()) == {"sign", "Delta_order", "density_down_0", "density_up_0"}, taus50
assert taus50["sign"] == 0.0, taus50
assert all(math.isfinite(t) for t in taus50.values()), taus50
assert res50.auto_corr_time == max(taus50.values())

# Shorter cycles: more correlated samples of the Delta order
res1 = solve_anderson(length_cycle=1)
taus1 = res1.auto_corr_times
assert all(math.isfinite(t) for t in taus1.values()), taus1
assert taus1["Delta_order"] > taus50["Delta_order"], (taus1, taus50)
if mpi.is_master_node():
    print("length_cycle = 1: ", taus1, res1.auto_corr_time_converged)
    print("length_cycle = 50:", taus50, res50.auto_corr_time_converged)

# 100 cycles: no bin level above the finest one has 64 blocks, so saturation cannot be checked.
# Serial only: with several ranks, a rank that is done keeps sampling until all ranks are done.
if mpi.size == 1:
    res_short = solve_anderson(n_cycles=100)
    assert res_short.auto_corr_time_converged is False

# Measure turned off
res_off = solve_anderson(n_cycles=100, measure_auto_corr_time=False)
assert res_off.auto_corr_time is None
assert res_off.auto_corr_time_converged is None
assert res_off.auto_corr_times is None

# h5 round trip
if mpi.is_master_node():
    with h5.HDFArchive("auto_corr_time.out.h5", 'w') as A:
        A['results'] = res50
    with h5.HDFArchive("auto_corr_time.out.h5", 'r') as A:
        res_read = A['results']
    assert res_read.auto_corr_times == taus50
    assert res_read.auto_corr_time == res50.auto_corr_time
    assert res_read.auto_corr_time_converged == res50.auto_corr_time_converged

# Jperp interaction, no hybridization
S = Solver(**constr_params)
wp = 1
g = 4
d0_iw = GfImFreq(indices=[0], beta=beta, statistic="Boson", n_points=n_tau_bosonic // 2)
d0_tau = GfImTime(indices=[0], beta=beta, statistic="Boson", n_points=n_tau_bosonic)
d0_iw << Function(lambda w: g * wp**2 / (w**2 - wp**2))
d0_tau << Fourier(d0_iw)
S.Jperp_tau << d0_tau
S.solve(h_int=h_int, h_loc0=h_loc0, length_cycle=50, n_warmup_cycles=1000, n_cycles=2000)
assert set(S.results.auto_corr_times.keys()) == {"sign", "Jperp_order", "density_down_0", "density_up_0"}, \
    S.results.auto_corr_times
