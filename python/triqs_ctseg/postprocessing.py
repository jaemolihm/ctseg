"""
Post-processing routines for the CT-SEG solver.

This module contains self-energy extraction (Dyson or improved estimator),
polarizability and screened interaction calculations, and related helpers.
"""

import triqs.utility.mpi as mpi
import numpy as np
from itertools import product

from triqs.gfs import (
    MeshImFreq, Gf, BlockGf,
    make_gf_dlr, make_gf_imfreq, make_hermitian
)
from triqs.gfs.tools import make_zero_tail
from triqs.operators.util.extractors import extract_U_dict2, dict_to_matrix

from triqs.solver_utils import tail_fit


# =============================================================================
# Helpers: color <-> block index mapping
# =============================================================================

def build_color_tables(gf_struct):
    """Build per-color lookup tables for a gf_struct.

    Returns (block_name, index_in_block, orbital), lists of length n_color.
    The orbital index counts up/down orbitals separately, so that each
    spin sector yields 0..n_orb-1.
    """
    block_name, index_in_block, orbital = [], [], []
    offsets = {"up": 0, "dn": 0}
    for blk_name, blk_dim in gf_struct:
        key = "up" if blk_name[:2] == "up" else "dn"
        for i in range(blk_dim):
            block_name.append(blk_name)
            index_in_block.append(i)
            orbital.append(offsets[key] + i)
        offsets[key] += blk_dim
    return block_name, index_in_block, orbital


# =============================================================================
# Interaction tensor extraction
# =============================================================================

def _dd_to_4idx(direct, exchange):
    """Expand a density-density matrix into a 4-index tensor.

    direct[i, j]   -> entries of type (i, j, i, j)
    exchange[i, j] -> entries of type (i, j, j, i) and (i, i, j, j), for i != j
    """
    n = direct.shape[0]
    out = np.zeros((n, n, n, n), dtype=complex)
    for i, j in product(range(n), repeat=2):
        out[i, j, i, j] = direct[i, j]
        if i != j:
            out[i, j, j, i] = exchange[i, j]
            out[i, i, j, j] = exchange[i, j]
    return out


def _dd_matrix_to_4idx(M):
    """Split a 2n_orb x 2n_orb density-density matrix (up/down layout) into
    direct (opposite-spin) and exchange (direct - same-spin) blocks, then
    expand to a 4-index tensor."""
    n_orb = M.shape[0] // 2
    direct = M[:n_orb, n_orb:2*n_orb]
    exchange = direct - M[:n_orb, :n_orb]
    return _dd_to_4idx(direct, exchange)


def extract_u_tensor_from_h_int(h_int, gf_struct, return_4idx=False):
    """
    Return U tensor in density-density basis from a Coulomb many-body operator h_int.
    If return_4idx is True, construct the full 4-index Uijkl tensor.
    """
    U_dd = dict_to_matrix(extract_U_dict2(h_int), gf_struct=gf_struct)
    return _dd_matrix_to_4idx(U_dd) if return_4idx else U_dd


# =============================================================================
# Spectral truncation
# =============================================================================

def check_spectrum(A, radius=1.0, truncation=False):
    """Check eigenvalues of A and optionally truncate those with |lambda| > radius."""
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    w, V = np.linalg.eig(A)
    keep = np.abs(w) < radius
    w_drop = w[~keep]

    if w_drop.size > 0:
        mpi.report(f"Unqualified eigenvalues (|lambda| > {radius}):")
        for lam in w_drop:
            mpi.report(f"  {lam}")
        mpi.report("")

    if not truncation or w_drop.size == 0:
        return A

    mpi.report(f"Reconstructing matrix with eigenvalues |lambda| <= {radius} only.\n")
    w_keep = w[keep]
    V_keep = V[:, keep]
    if w_keep.size == 0:
        return np.zeros_like(A, dtype=np.result_type(A, 1j))
    return V_keep @ np.diag(w_keep) @ np.linalg.pinv(V_keep)


# =============================================================================
# Screening matrix extraction
# =============================================================================

def _assemble_color_gf(src_block2gf, block_name, index_in_block):
    """Assemble a (n_color, n_color)-target Gf from a Block2Gf by picking
    the (index_in_block[c1], index_in_block[c2]) slot of block
    (block_name[c1], block_name[c2]) for each color pair."""
    n_color = len(block_name)
    mesh = src_block2gf[block_name[0], block_name[0]].mesh
    out = Gf(mesh=mesh, target_shape=(n_color, n_color))
    for c1, c2 in product(range(n_color), repeat=2):
        out.data[:, c1, c2] = (
            src_block2gf[block_name[c1], block_name[c2]].data[:, index_in_block[c1], index_in_block[c2]]
        )
    return out


def extract_screen_matrix_from_D0_tau(blk2_D0_tau, gf_struct, return_4idx=False):
    block_name, index_in_block, _ = build_color_tables(gf_struct)
    D0_tau = _assemble_color_gf(blk2_D0_tau, block_name, index_in_block)

    w0_mesh = MeshImFreq(beta=D0_tau.mesh.beta, statistic="Boson", n_iw=1)
    D0_iw = Gf(mesh=w0_mesh, target_shape=D0_tau.target_shape)
    D0_iw.set_from_fourier(D0_tau, make_zero_tail(D0_iw, n_moments=2))
    Dw0_dd = D0_iw.data[0].real

    return _dd_matrix_to_4idx(Dw0_dd) if return_4idx else Dw0_dd


# =============================================================================
# Static self-energy and the exchange term of the symmetric improved estimator
# =============================================================================

def _fourier_wo_moments(gf_tau, mesh):
    """Fourier transform a BlockGf in imaginary time to the fermionic mesh `mesh`,
    imposing a zero 0th moment (no other tail information is used)."""
    out = BlockGf(mesh=mesh, gf_struct=[(bl, g.target_shape[0]) for bl, g in gf_tau])
    for bl, g in out:
        g.set_from_fourier(gf_tau[bl], make_zero_tail(g, n_moments=1))
    return out


def compute_density_matrix(solver, G0_iw, h_loc0_mat):
    """
    Density matrix P_ab = <c_b^dag c_a> = G_ab(0^-) of each block, made hermitian. If F(tau) is
    measured, from G = G0 + G0 F^L at tau = 0^-,

        P = G0(0^-) - int_0^beta dtau G0(beta - tau) F^L(tau),

    with G0(tau) the Fourier transform of G0_iw with the known moments 1/(i nu) and h_loc0/(i nu)^2;
    this uses the whole F^L(tau) curve. Otherwise from the end bins of G(tau),

        P = [1 + G(0^+) - G(beta^-)] / 2,    since G(0^+) = P - 1 and G(beta^-) = -P.
    """
    F_tau = solver.results.F_tau
    P = {}
    if F_tau is None:
        for bl, g in solver.results.G_tau:
            p = 0.5 * (np.eye(g.target_shape[0]) + g.data[0] - g.data[-1])
            P[bl] = 0.5 * (p + p.conj().T)
        return P
    for i, (bl, F) in enumerate(F_tau):
        dim = F.target_shape[0]
        moments = np.zeros((3, dim, dim), dtype=complex)
        moments[1] = np.eye(dim)
        moments[2] = h_loc0_mat[i]
        G0 = F.copy()
        G0.set_from_fourier(G0_iw[bl], moments)
        taus = np.array([float(t) for t in F.mesh.values()])
        integrand = np.einsum('tac,tcb->tab', G0.data[::-1], F.data)  # G0(beta - tau) F^L(tau)
        p = -G0.data[-1] - np.trapezoid(integrand, taus, axis=0)
        P[bl] = 0.5 * (p + p.conj().T)
    return P


def compute_sigma_static(solver, P):
    """
    Static self-energy Sigma(i nu -> inf) of each block,

        Sigma_ab = delta_ab sum_c [U_ac + D0_ac(i nu = 0)] n_c - (1 - delta_ab) U_ab P_ab,

    with U the instantaneous density-density interaction, D0 = 0 if solver.D0_tau is None, n_c the
    segment densities and P the density matrix of each block (a dict of matrices). The retarded
    exchange term -FT[D_ab(tau) G_ab(tau)] decays as 1/(i nu) and is not part of it.
    """
    U = extract_u_tensor_from_h_int(h_int=solver.h_int, gf_struct=solver.gf_struct).real
    V = U.copy()
    if solver.D0_tau is not None:
        V += extract_screen_matrix_from_D0_tau(blk2_D0_tau=solver.D0_tau, gf_struct=solver.gf_struct)
    densities = np.concatenate([np.asarray(solver.results.densities[bl], dtype=float) for bl, _ in solver.gf_struct])
    hartree = V @ densities

    Sigma_static = {}
    c0 = 0
    for bl, dim in solver.gf_struct:
        sigma = -U[c0:c0 + dim, c0:c0 + dim] * P[bl]
        np.fill_diagonal(sigma, hartree[c0:c0 + dim])
        Sigma_static[bl] = sigma
        c0 += dim
    return Sigma_static


def compute_sigma_retarded_exchange(solver, G_tau):
    """
    Retarded exchange self-energy of the symmetric improved estimator,

        Sigma^X_ret,ab(i nu) = -FT[D_ab(tau) G_ab(tau)]    (no sum over a, b),

    zero if solver.D0_tau is None. Returns a BlockGf on the fermionic mesh with n_iw = solver.n_iw.
    """
    from scipy.interpolate import CubicSpline

    mesh = MeshImFreq(beta=solver.beta, statistic="Fermion", n_iw=solver.n_iw)
    Sigma_X = BlockGf(mesh=mesh, gf_struct=solver.gf_struct)
    Sigma_X.zero()

    # D0 interpolated onto the G(tau) mesh
    if solver.D0_tau is not None:
        for bl, g_tau in G_tau:
            D0 = solver.D0_tau[bl, bl]
            n_tau_D = len(D0.mesh)
            n_tau_G = len(g_tau.mesh)
            tau_D = np.linspace(0, solver.beta, n_tau_D)
            tau_G = np.linspace(0, solver.beta, n_tau_G)
            # D_ab(tau) -> (D_ab(tau) + D_ba(tau)) / 2: the action depends only on this part, and the
            # solver symmetrizes D0 in the same way (work_data.cpp)
            D_data = 0.5 * (D0.data + D0.data.transpose(0, 2, 1))
            D_on_G = CubicSpline(tau_D, D_data, axis=0)(tau_G)
            if np.abs(D_on_G).max() == 0:
                continue
            DG = g_tau.copy()
            DG.data[:] = D_on_G * g_tau.data
            DG_iw = Sigma_X[bl].copy()
            DG_iw.set_from_fourier(DG, make_zero_tail(DG_iw, n_moments=1))
            Sigma_X[bl] << Sigma_X[bl] - DG_iw

    return Sigma_X


# =============================================================================
# Self-energy post-processing
# =============================================================================

def postprocess_sigma(
    solver,
    symmetrize_func=None,
    **post_proc_params
):
    """
    Post-process the self-energy from the CT-SEG solver.

    Steps:
    1. Fourier transform G(tau) to G(iw)
    2. Compute fermionic Weiss field g(iw) from Delta(tau)
    3. Static self-energy Sigma(i nu -> inf) (compute_sigma_static), if analytic_hf or for the
       symmetric improved estimator
    4. Self-energy estimators: Dyson (always), the asymmetric improved estimator F G^{-1} (if F(tau)
       is measured) and the symmetric improved estimator Sigma^static + Sigma^X_ret + I - F^L G^{-1} F^R
       (if F_R(tau) and I(tau) are measured), returned as Sigma_iw_Dyson, Sigma_iw_aIE and
       Sigma_iw_sIE (None if not available). Sigma_iw is the last available one.
    5. Tail fit of Sigma(iw), with the static self-energy as known 0th moment if analytic_hf, and
       G(iw) from the Dyson equation
    6. Static part Sigma_HartreeFock (step 3 if analytic_hf, else the tail of Sigma(iw)) and the
       dynamic part Sigma - Sigma_HartreeFock
    """
    from triqs.gfs import iOmega_n
    from triqs.gfs.tools import inverse

    degenerate_blk = post_proc_params['degenerate_blk']
    analytic_hf = post_proc_params['analytic_hf']
    res = solver.results

    def symmetrize(gf):
        if symmetrize_func is not None and degenerate_blk:
            gf << symmetrize_func(gf, degenerate_blk)

    mesh = MeshImFreq(beta=solver.beta, statistic="Fermion", n_iw=solver.n_iw)
    G_iw = BlockGf(mesh=mesh, gf_struct=solver.gf_struct)
    G0_iw = G_iw.copy()

    # 1. Fourier transform G(tau) to G(iw)
    Gf_known_moments = make_zero_tail(G_iw, n_moments=2)
    for i, bl in enumerate(G_iw.indices):
        Gf_known_moments[i][1] = np.eye(G_iw[bl].target_shape[0])
        G_iw[bl].set_from_fourier(res.G_tau[bl], Gf_known_moments[i])
    G_iw << make_hermitian(G_iw)
    symmetrize(G_iw)

    # 2. Compute fermionic Weiss field g(iw)
    # The solver uses only the real diagonal of h_loc0 (the segment picture requires a diagonal
    # local Hamiltonian), so G0 is built from the same part.
    # TODO: reject an h_loc0 with off-diagonal elements at the input side (solve / work_data.cpp)
    # instead of dropping them silently; then the diagonal projection here is no longer needed.
    h_loc0_diag = [np.diag(np.diag(np.asarray(m)).real) for m in solver.h_loc0_mat]
    max_offdiag = max(np.abs(np.asarray(m) - np.diag(np.diag(np.asarray(m)))).max() for m in solver.h_loc0_mat)
    if max_offdiag > 1e-10:
        mpi.report(f"WARNING: h_loc0 has off-diagonal elements (max {max_offdiag:.2e}); the solver ignores them "
                   "and so does the post-processing.\n")
    Delta_iw = BlockGf(mesh=mesh, gf_struct=solver.gf_struct)
    Delta_known_moments = make_zero_tail(Delta_iw, n_moments=1)
    for i, bl in enumerate(solver.Delta_tau.indices):
        Delta_iw[bl].set_from_fourier(solver.Delta_tau[bl], Delta_known_moments[i])
        G0_iw[bl] << inverse(iOmega_n - h_loc0_diag[i] - Delta_iw[bl])
    G0_iw << make_hermitian(G0_iw)
    symmetrize(G0_iw)

    # If F_tau, F_R_tau and I_tau are measured, use the symmetric improved estimator (sIE)
    F_R_tau = getattr(res, 'F_R_tau', None)
    I_tau = getattr(res, 'I_tau', None)
    use_sIE = res.F_tau is not None and F_R_tau is not None and I_tau is not None

    # 3. Static self-energy
    if analytic_hf or use_sIE:
        P = compute_density_matrix(solver, G0_iw, h_loc0_diag)
        Sigma_static = compute_sigma_static(solver, P)
        if symmetrize_func is not None and degenerate_blk:
            Sigma_static = dict(zip(Sigma_static.keys(), symmetrize_func(list(Sigma_static.values()), degenerate_blk)))

    # 4. Self-energy estimators
    Sigma_iw_Dyson = inverse(G0_iw) - inverse(G_iw)
    Sigma_iw_aIE = None
    Sigma_iw_sIE = None
    if res.F_tau is not None:
        # Asymmetric improved estimator (aIE)
        F_iw = _fourier_wo_moments(res.F_tau, mesh)
        symmetrize(F_iw)
        # Sigma = F G^{-1} as a matrix product within each block. F is not hermitian
        # (F(-iw)^dagger = G Sigma, not Sigma G), so only Sigma is made hermitian.
        Sigma_iw_aIE = F_iw * inverse(G_iw)
    if use_sIE:
        # Symmetric improved estimator (sIE)
        # The symmetric estimator uses G, F^L, F^R and I Fourier transformed with the same
        # (zero) known 0th moment, not the tail-corrected G_iw used for Dyson and F G^{-1}.
        G_iw_wo, F_R_iw, I_iw = (_fourier_wo_moments(x, mesh) for x in (res.G_tau, F_R_tau, I_tau))
        for g in (G_iw_wo, F_R_iw, I_iw):
            symmetrize(g)
        Sigma_iw_sIE = compute_sigma_retarded_exchange(solver, res.G_tau) + I_iw - F_iw * inverse(G_iw_wo) * F_R_iw
        for bl, g in Sigma_iw_sIE:
            g.data[:] += Sigma_static[bl]
    for Sigma_est in (Sigma_iw_Dyson, Sigma_iw_aIE, Sigma_iw_sIE):
        if Sigma_est is not None:
            Sigma_est << make_hermitian(Sigma_est)
            symmetrize(Sigma_est)

    if use_sIE:
        mpi.report("F(tau), F_R(tau) and I(tau) are measured -> Compute the self-energy via the "
                   "symmetric improved estimator.\n")
        Sigma_iw = Sigma_iw_sIE.copy()
    elif res.F_tau is not None:
        mpi.report("F(tau) is measured -> Compute the self-energy via the improved estimator.\n")
        Sigma_iw = Sigma_iw_aIE.copy()
    else:
        mpi.report("F(tau) is not measured -> Compute the self-energy via the Dyson equation.\n")
        Sigma_iw = Sigma_iw_Dyson.copy()

    # 5. Tail fit, and G(iw) from the Dyson equation
    solver.Sigma_moments = (
        {bl: np.array([s], dtype=complex) for bl, s in Sigma_static.items()} if analytic_hf else None
    )
    if post_proc_params['perform_tail_fit']:
        Sigma_iw = tail_fit(
            Sigma_iw,
            fit_min_n=post_proc_params['fit_min_n'],
            fit_max_n=post_proc_params['fit_max_n'],
            fit_min_w=post_proc_params['fit_min_w'],
            fit_max_w=post_proc_params['fit_max_w'],
            fit_max_moment=post_proc_params['fit_max_moment'],
            fit_known_moments=solver.Sigma_moments,
        )
        Sigma_iw << make_hermitian(Sigma_iw)
        symmetrize(Sigma_iw)

    G_iw << inverse(inverse(G0_iw) - Sigma_iw)
    G_iw << make_hermitian(G_iw)
    symmetrize(G_iw)

    # 6. Static and dynamic parts
    if analytic_hf:
        mpi.report("Static self-energy, analytic:")
        Sigma_HartreeFock = Sigma_static
    else:
        mpi.report("Static self-energy, from the tail of Sigma(iw):")
        Sigma_HartreeFock = {bl: g.fit_hermitian_tail()[0][0] for bl, g in Sigma_iw}
    for bl, hf_val in Sigma_HartreeFock.items():
        mpi.report(f"Sigma_HF {bl}:\n    {hf_val}")
    mpi.report("")

    Sigma_dynamic = Sigma_iw.copy()
    for bl, g in Sigma_dynamic:
        g << g - Sigma_HartreeFock[bl]

    return {
        'G_iw': G_iw,
        'G_tau': res.G_tau,
        'Sigma_iw': Sigma_iw,
        'Sigma_iw_Dyson': Sigma_iw_Dyson,
        'Sigma_iw_aIE': Sigma_iw_aIE,
        'Sigma_iw_sIE': Sigma_iw_sIE,
        'Sigma_dynamic': Sigma_dynamic,
        'Sigma_HartreeFock': list(Sigma_HartreeFock.values()),
    }


# =============================================================================
# Polarizability and screened interaction
# =============================================================================

def postprocess_pi(
    solver,
    degenerate_blk=None,
    symmetrize_func=None,
    output_in_4idx=False,
    truncate_uchi=False,
):
    """Post-process the charge susceptibility to obtain the impurity polarizability and screened interaction."""
    mpi.report("Charge susceptibility is measured for a impurity with dynamic interactions")
    mpi.report('--> Post-processing the density-density susceptibility to obtain the impurity polarizability.\n')

    def symmetrize(gf):
        if symmetrize_func is not None and degenerate_blk is not None:
            gf << symmetrize_func(gf, degenerate_blk)

    block_name, index_in_block, color_to_orbital = build_color_tables(solver.gf_struct)
    n_color = len(block_name)
    assert n_color % 2 == 0, "n_color is an odd number."
    n_orb = n_color // 2

    D0_tau = _assemble_color_gf(solver.D0_tau, block_name, index_in_block)
    nn_iw_dlr = _assemble_color_gf(solver.results.nn_nu_dlr, block_name, index_in_block)
    iw_mesh_dlr = nn_iw_dlr.mesh

    # 1. Subtract the constant part of the charge susceptibility
    densities = np.array([
        solver.results.densities[block_name[c]][index_in_block[c]]
        for c in range(n_color)
    ], dtype=float)
    mpi.report(f"Average of time-dependent occupations: {densities}")

    mpi.report("Subtracting the constant component, and then symmetrizing the density-density susceptibility: \n"
               "  1. nn(t).imag = 0.0\n"
               "  2. nn(i, j) = nn(j, i)\n")

    w0_idx = next(k for k, iw in enumerate(iw_mesh_dlr) if iw.index == 0)
    beta = iw_mesh_dlr.beta
    for c1 in range(n_color):
        for c2 in range(c1 + 1):
            dd = beta * densities[c1] * densities[c2]
            nn_iw_dlr[c1, c2].data[w0_idx] -= dd
            nn_iw_dlr[c1, c2].data.imag = 0.0
            if c1 != c2:
                nn_iw_dlr[c2, c1].data[w0_idx] -= dd
                nn_iw_dlr[c1, c2].data[:] += nn_iw_dlr[c2, c1].data[:].real
                nn_iw_dlr[c1, c2].data[:] /= 2.0
                nn_iw_dlr[c2, c1] << nn_iw_dlr[c1, c2]

    nn_iw = make_gf_imfreq(make_gf_dlr(nn_iw_dlr), n_iw=solver.n_iw)

    # Convert to density-density basis
    nn_iw_dd = Gf(mesh=nn_iw.mesh, target_shape=[n_orb, n_orb])
    for c1, c2 in product(range(n_color), repeat=2):
        nn_iw_dd[color_to_orbital[c1], color_to_orbital[c2]].data[:] += nn_iw[c1, c2].data[:]

    symmetrize(nn_iw_dd)

    # Convert to product basis
    nn_iw_pb = Gf(mesh=nn_iw.mesh, target_shape=[n_orb*n_orb, n_orb*n_orb])
    for i, j in product(range(n_orb), repeat=2):
        if i < j:
            continue
        ii, jj = i*n_orb+i, j*n_orb+j
        nn_iw_pb[ii, jj] << nn_iw_dd[i, j].real
        if i != j:
            nn_iw_pb[jj, ii] << nn_iw_pb[ii, jj]

    # 2. Construct bosonic Weiss field in product basis
    D0_iw = nn_iw.copy()
    D0_iw << 0.0
    u_known_moments = make_zero_tail(D0_iw, n_moments=2)
    D0_iw.set_from_fourier(D0_tau, u_known_moments)

    D0_iijj = D0_iw[:n_orb, n_orb:2*n_orb]
    D0_ijij = D0_iijj - D0_iw[:n_orb, 0:n_orb]

    Vijkl = extract_u_tensor_from_h_int(h_int=solver.h_int, gf_struct=solver.gf_struct, return_4idx=True)

    U_iw_pb = Gf(mesh=nn_iw_pb.mesh, target_shape=nn_iw_pb.target_shape)
    for i, j in product(range(n_orb), repeat=2):
        if i < j:
            continue
        ii, jj = i*n_orb+i, j*n_orb+j
        U_iw_pb[ii, jj] << D0_iijj[i, j].real
        U_iw_pb[ii, jj].data[:] += Vijkl[i, j, i, j]
        if i == j:
            continue
        ij, ji = i*n_orb+j, j*n_orb+i
        U_iw_pb[jj, ii] << U_iw_pb[ii, jj]
        U_iw_pb[ij, ij] << D0_ijij[i, j].real
        U_iw_pb[ij, ij].data[:] += Vijkl[i, j, j, i]
        U_iw_pb[ji, ji] << U_iw_pb[ij, ij]
        U_iw_pb[ij, ji] << D0_ijij[i, j].real
        U_iw_pb[ij, ji].data[:] += Vijkl[i, i, j, j]
        U_iw_pb[ji, ij] << U_iw_pb[ij, ji]

    # 3. Dyson equation: Pi(w) = Chi(w) * [U(w)*Chi(w) - I]^-1
    Pi_iw_pb = Gf(mesh=nn_iw_pb.mesh, target_shape=nn_iw_pb.target_shape)
    ones = np.eye(n_orb*n_orb, dtype=complex)
    for iwn in nn_iw_pb.mesh:
        UX = U_iw_pb[iwn] @ nn_iw_pb[iwn]
        UX = check_spectrum(UX, truncation=truncate_uchi)
        denom = UX - ones
        cond = np.linalg.cond(denom)
        if cond > 20:
            mpi.report(f"WARNING: Large condition number for [U(w) * Chi(w) - I] = {cond} at n = {iwn.index}.")
        Pi_iw_pb[iwn] = nn_iw_pb[iwn] @ np.linalg.pinv(denom)
        Pi_iw_pb[iwn].imag = 0.0

    # 4. Screened interaction W(w) = U(w) - U(w) * Chi(w) * U(w)
    W_iw_pb = Gf(mesh=nn_iw_pb.mesh, target_shape=nn_iw_pb.target_shape)
    for iwn in nn_iw_pb.mesh:
        W_iw_pb[iwn] = U_iw_pb[iwn] - U_iw_pb[iwn] @ nn_iw_pb[iwn] @ U_iw_pb[iwn]

    # 5. Remove the static part
    for i, j in product(range(n_orb), repeat=2):
        if i < j:
            continue
        ii, jj = i*n_orb+i, j*n_orb+j
        W_iw_pb[ii, jj].data[:] -= Vijkl[i, j, i, j]
        if i == j:
            continue
        ij, ji = i*n_orb+j, j*n_orb+i
        W_iw_pb[jj, ii] << W_iw_pb[ii, jj]
        W_iw_pb[ji, ji].data[:] -= Vijkl[i, j, j, i]
        W_iw_pb[ij, ij] << W_iw_pb[ji, ji]
        W_iw_pb[ji, ij].data[:] -= Vijkl[i, i, j, j]
        W_iw_pb[ij, ji] << W_iw_pb[ji, ij]

    # Transform back
    if not output_in_4idx:
        Chi_iw = nn_iw_dd
        Pi_iw = Gf(mesh=nn_iw_pb.mesh, target_shape=(n_orb, n_orb))
        W_iw = Pi_iw.copy()
        for i, j in product(range(n_orb), repeat=2):
            Pi_iw[i, j] << Pi_iw_pb[i*n_orb+i, j*n_orb+j]
            W_iw[i, j] << W_iw_pb[i*n_orb+i, j*n_orb+j]
        symmetrize(Pi_iw)
        symmetrize(W_iw)
    else:
        Chi_iw = nn_iw_pb
        Pi_iw = Gf(mesh=nn_iw_pb.mesh, target_shape=(n_orb, n_orb, n_orb, n_orb))
        W_iw = Pi_iw.copy()
        for i, j, k, l in product(range(n_orb), repeat=4):
            Pi_iw[i, j, k, l] << Pi_iw_pb[i*n_orb+j, k*n_orb+l]
            W_iw[i, j, k, l] << W_iw_pb[i*n_orb+j, k*n_orb+l]

    return {
        'Pi_iw': Pi_iw,
        'W_iw': W_iw,
        'Chi_iw': Chi_iw,
    }


# =============================================================================
# Top-level post-processing dispatcher
# =============================================================================

def postprocess(
    solver,
    symmetrize_func=None,
    **post_proc_params,
):
    pp_results = postprocess_sigma(solver, symmetrize_func, **post_proc_params)

    has_chi = solver.D0_tau is not None and getattr(solver.results, 'nn_nu_dlr', None) is not None
    if has_chi:
        deg_blk = post_proc_params['degenerate_blk']
        deg_blk_2e = (
            [np.array(blks[:len(blks) // 2]) for blks in deg_blk]
            if deg_blk else None
        )
        pp_results.update(
            postprocess_pi(
                solver, deg_blk_2e, symmetrize_func,
                truncate_uchi=post_proc_params['truncate_uchi'],
            )
        )

    wmax = post_proc_params.get('dlr_w_max')
    eps = post_proc_params.get('dlr_eps')
    if wmax is not None and eps is not None:
        from triqs.solver_utils import make_gf_dlr_imfreq
        keys = ['Sigma_iw', 'Sigma_iw_Dyson', 'Sigma_iw_aIE', 'Sigma_iw_sIE', 'G_iw'] \
            + (['Pi_iw', 'W_iw', 'Chi_iw'] if has_chi else [])
        for k in keys:
            if pp_results[k] is not None:
                pp_results[k] = make_gf_dlr_imfreq(pp_results[k], w_max=wmax, eps=eps)

    return pp_results
