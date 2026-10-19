# Copyright (c) 2026, The Simons Foundation
# This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# measure_I_tau must be rejected without measure_F_tau and in the presence of Jperp.

from triqs.gfs import GfImFreq, GfImTime, Fourier
from triqs.gfs.descriptors import Function
from triqs.operators import n

from triqs_ctseg import Solver

beta = 10
U = 4.0
mu = U / 2
n_tau = 201
n_tau_bosonic = 201
gf_struct = [('down', 1), ('up', 1)]

solve_params = {
    "h_int": U * n("up", 0) * n("down", 0),
    "h_loc0": -mu * (n("up", 0) + n("down", 0)),
    "length_cycle": 10,
    "n_warmup_cycles": 10,
    "n_cycles": 10,
}


def expect_error(S, message, **params):
    try:
        S.solve(**solve_params, **params)
    except RuntimeError as e:
        assert message in str(e), str(e)
        return
    raise AssertionError(f"solve should have raised ({message})")


# measure_I_tau without measure_F_tau
S = Solver(gf_struct=gf_struct, beta=beta, n_tau=n_tau, n_tau_bosonic=n_tau_bosonic)
expect_error(S, "requires measure_F_tau", measure_I_tau=True)

# measure_I_tau with Jperp
S = Solver(gf_struct=gf_struct, beta=beta, n_tau=n_tau, n_tau_bosonic=n_tau_bosonic)
d0_iw = GfImFreq(indices=[0], beta=beta, statistic="Boson", n_points=n_tau_bosonic // 2)
d0_tau = GfImTime(indices=[0], beta=beta, statistic="Boson", n_points=n_tau_bosonic)
d0_iw << Function(lambda w: 1.0 / (w**2 - 1.0))
d0_tau << Fourier(d0_iw)
S.Jperp_tau << d0_tau
expect_error(S, "not supported with Jperp", measure_F_tau=True, measure_I_tau=True)

print("sie_error_paths: all checks passed")
