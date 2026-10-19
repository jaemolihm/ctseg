# Copyright (c) 2026, The Simons Foundation
# This file is part of TRIQS/ctseg and is licensed under the terms of GPLv3 or later.
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE in the root of this distribution for details.

"""Two-orbital impurity coupled to a two-site bath, solved by exact diagonalization.

The impurity Hamiltonian is diagonal with a density-density interaction, as required by
CTSEG; the off-diagonal elements of Delta(iw) come from the bath. Shared by the tests of
the symmetric improved estimator.
"""

import itertools

import numpy as np
from triqs.atom_diag import AtomDiag, atomic_g_iw, atomic_density_matrix, trace_rho_op
from triqs.gfs import BlockGf, MeshImFreq, inverse
from triqs.operators import c, c_dag, n

beta = 5.0
U = 5.0
n_orb = 2
gf_struct = [('ud', n_orb)]
h0_mat = np.diag([-0.2, -0.3])
V_mat = np.array([[1.0, 0.2], [0.2, 1.0]])
h0_bath_mat = np.array([[-1.0, 0.5], [0.5, -0.3]])
h_int = U * n('ud', 0) * n('ud', 1)


def make_delta_and_g0(n_iw):
    mesh = MeshImFreq(beta=beta, statistic='Fermion', n_iw=n_iw)
    Delta_iw = BlockGf(mesh=mesh, gf_struct=gf_struct)
    G0_iw = Delta_iw.copy()
    for iw in mesh:
        delta = V_mat @ np.linalg.inv(iw.value * np.eye(2) - h0_bath_mat) @ V_mat.T
        Delta_iw['ud'][iw] = delta
        G0_iw['ud'][iw] = np.linalg.inv(iw.value * np.eye(2) - h0_mat - delta)
    return Delta_iw, G0_iw


def solve_ed(n_iw):
    """Exact G_iw, Sigma_iw, Sigma^H + Sigma^X (static), densities and G(beta^-)."""
    Delta_iw, G0_iw = make_delta_and_g0(n_iw)

    fops = [('ud', i) for i in range(n_orb + 2)]
    h_tot = 1.0 * h_int
    for i, j in itertools.product(range(n_orb), repeat=2):
        h_tot += h0_mat[i, j] * c_dag('ud', i) * c('ud', j)
    for i, j in itertools.product(range(2), repeat=2):
        h_tot += h0_bath_mat[i, j] * c_dag('ud', n_orb + i) * c('ud', n_orb + j)
    for i, j in itertools.product(range(n_orb), range(2)):
        h_tot += V_mat[i, j] * (c_dag('ud', i) * c('ud', n_orb + j) + c_dag('ud', n_orb + j) * c('ud', i))
    ad = AtomDiag(h_tot, fops)

    G_tot = atomic_g_iw(ad, beta, [('ud', n_orb + 2)], n_iw)
    G_iw = BlockGf(mesh=G0_iw.mesh, gf_struct=gf_struct)
    G_iw['ud'].data[:] = G_tot['ud'].data[:, :n_orb, :n_orb]
    Sigma_iw = inverse(G0_iw) - inverse(G_iw)

    rho = atomic_density_matrix(ad, beta)
    # n_ab = <c^dag_b c_a> = G_ab(0^-) = -G_ab(beta^-)
    n_mat = np.array([[trace_rho_op(rho, c_dag('ud', b) * c('ud', a), ad) for b in range(n_orb)]
                      for a in range(n_orb)])
    Sigma_H = np.diag([U * n_mat[1, 1].real, U * n_mat[0, 0].real])
    Sigma_X = -U * n_mat * (1 - np.eye(n_orb))
    return dict(G_iw=G_iw, Sigma_iw=Sigma_iw, Sigma_H=Sigma_H, Sigma_X=Sigma_X,
                densities=np.diag(n_mat).real, n_mat=n_mat, Delta_iw=Delta_iw, G0_iw=G0_iw)
