# Copyright (c) 2026, The Simons Foundation
# This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

# Symmetric improved estimator without Monte Carlo. Exact G, F^L, F^R, I of the ED model
# are put on the imaginary-time mesh and fed to postprocess_sigma, which must recover the
# exact Sigma (static Hartree and Fock terms included).

from types import SimpleNamespace

import numpy as np

import triqs.utility.mpi as mpi
from triqs.gfs import BlockGf, MeshImTime, make_gf_from_fourier
from triqs_ctseg.postprocessing import postprocess_sigma

import sie_ed_model as model

n_iw = 100
n_iw_big = 3000
n_tau = 10001
beta = model.beta


def to_tau(gf_iw, mesh_tau, first_moment=None):
    """Imaginary-time version of a BlockGf; the 1/(i nu) coefficient is read off the last frequency."""
    out = BlockGf(mesh=mesh_tau, gf_struct=model.gf_struct)
    for bl, g in gf_iw:
        iw_last = 1j * np.pi * (2 * n_iw_big - 1) / beta
        c1 = first_moment if first_moment is not None else iw_last * g.data[-1]
        tail = np.zeros((2,) + g.target_shape, dtype=complex)
        tail[1] = c1
        out[bl] << make_gf_from_fourier(g, mesh_tau, tail)
    return out


def test_postprocess_sigma_sie_recovers_exact_sigma():
    ed = model.solve_ed(n_iw_big)
    G, Sigma = ed['G_iw']['ud'].data, ed['Sigma_iw']['ud'].data
    Sigma_HX = ed['Sigma_H'] + ed['Sigma_X']
    # F^L = Sigma G, F^R = G Sigma, I = Sigma - Sigma^HX + Sigma G Sigma
    def blockgf(data):
        out = ed['G_iw'].copy()
        out['ud'].data[:] = data
        return out

    mesh_tau = MeshImTime(beta, 'Fermion', n_tau)
    G_tau = to_tau(ed['G_iw'], mesh_tau, first_moment=np.eye(2))
    F_tau = to_tau(blockgf(Sigma @ G), mesh_tau)
    F_R_tau = to_tau(blockgf(G @ Sigma), mesh_tau)
    I_tau = to_tau(blockgf(Sigma - Sigma_HX + Sigma @ G @ Sigma), mesh_tau)

    Delta_iw, _ = model.make_delta_and_g0(n_iw_big)
    Delta_tau = to_tau(Delta_iw, mesh_tau)

    solver = SimpleNamespace(
        beta=beta, n_iw=n_iw, gf_struct=model.gf_struct, h_loc0_mat=[model.h0_mat],
        h_int=model.h_int, D0_tau=None, Delta_tau=Delta_tau,
        results=SimpleNamespace(G_tau=G_tau, F_tau=F_tau, F_R_tau=F_R_tau, I_tau=I_tau,
                                densities={'ud': ed['densities']}),
    )
    res = postprocess_sigma(solver, degenerate_blk=None, analytic_hf=True, perform_tail_fit=False)

    sl = slice(n_iw_big, n_iw_big + n_iw)
    err_sie = np.abs(res['Sigma_iw']['ud'].data[n_iw:] - Sigma[sl]).max()
    print(f"max |Sigma_sIE - Sigma_exact| = {err_sie:.2e}")
    assert err_sie < 1e-3, err_sie

    # all available estimators are returned separately; Sigma_iw is the symmetric one
    assert res['Sigma_iw_Dyson'] is not None and res['Sigma_iw_aIE'] is not None
    assert np.abs(res['Sigma_iw_sIE']['ud'].data - res['Sigma_iw']['ud'].data).max() < 1e-12

    # without I and F^R the asymmetric estimator is returned; it misses nothing here either
    solver.results.I_tau = solver.results.F_R_tau = None
    res_asym = postprocess_sigma(solver, degenerate_blk=None, analytic_hf=True, perform_tail_fit=False)
    err_asym = np.abs(res_asym['Sigma_iw']['ud'].data[n_iw:] - Sigma[sl]).max()
    print(f"max |Sigma_asym - Sigma_exact| = {err_asym:.2e}")
    assert err_asym < 5e-3, err_asym
    assert res_asym['Sigma_iw_sIE'] is None
    assert np.abs(res_asym['Sigma_iw_aIE']['ud'].data - res_asym['Sigma_iw']['ud'].data).max() < 1e-12

    # analytic_hf: Sigma_HartreeFock and the tail-fit 0th moment are the exact Sigma^H + Sigma^X,
    # with P from F^L, then from the end points of G(tau)
    for F in (F_tau, None):
        solver.results.F_tau = F
        r = postprocess_sigma(solver, degenerate_blk=None, analytic_hf=True, perform_tail_fit=True,
                              fit_max_moment=3, fit_min_n=None, fit_max_n=None, fit_min_w=None, fit_max_w=None)
        err_hf = np.abs(r['Sigma_HartreeFock'][0] - Sigma_HX).max()
        err_tail = np.abs(r['Sigma_iw']['ud'].fit_hermitian_tail()[0][0] - Sigma_HX).max()
        print(f"P from {'F^L' if F is not None else 'G(tau) end points'}: "
              f"max |Sigma_HF - Sigma^HX| = {err_hf:.2e}, max |Sigma_inf (tail) - Sigma^HX| = {err_tail:.2e}")
        assert err_hf < 1e-6 and err_tail < 1e-6, (err_hf, err_tail)


if mpi.is_master_node():
    test_postprocess_sigma_sie_recovers_exact_sigma()
    print("sie_no_mc: all tests passed")
