"""Unknown Multi-node Interventional CRL (UMNI-CRL)

Setting: X = G.Z, i.e. linear transform. 
unknown multi-node interventions.

Paper: Linear Causal Representation Learning from Unknown Multi-node Interventions
(https://arxiv.org/abs/2406.05937) at NeurIPS 2024

TODO: Add some file-wide comments.
"""

__all__ = ["umni_crl"]

import sys
import os
# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from itertools import product
import logging
logger = logging.getLogger(__name__)

import numpy as np
import numpy.typing as npt

from causaldag import partial_correlation_suffstat, partial_correlation_test

import utils


def setminus(lst1: list, lst2: list) -> list:
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3


def umni_crl(
    x_samples: npt.NDArray[np.float_],
    dsx_samples: npt.NDArray[np.float_],
    hard_intervention: bool,
    hard_graph_postprocess: bool,
    atol_eigv: float,
    atol_ci_test: float,
    kappa : int,
) -> tuple[
    npt.NDArray[np.int_],
    npt.NDArray[np.float_],
    npt.NDArray[np.int_],
    npt.NDArray[np.float_],
    npt.NDArray[np.bool_],
    None | tuple[
        npt.NDArray[np.float_],
        npt.NDArray[np.bool_],
    ],
]:
    # TODO: Choosing n from M input envs (obs always is included)
    assert dsx_samples.ndim == 4 and dsx_samples.shape[3] == 1
    n, nsamples, d, _ = dsx_samples.shape
    assert x_samples.shape == (n + 1, nsamples, d, 1)

    # Preprocessing:
    # x and relevant parts of dsx lives in `n` dimensional column space of the decoder.
    # Find this subspace and write x and dsx samples in this basis.
    # Note that `n` random samples almost surely suffice for this task. I use `n + d`
    # samples just in case.
    x_cov = utils.cov(x_samples[:, : n + d, :, 0])
    _, dec_svec = np.linalg.eigh(np.sum(x_cov, 0))
    dec_colbt = dec_svec[:, -n:].T
    x_n_samples = dec_colbt @ x_samples
    dsx_n_samples = dec_colbt @ dsx_samples

    # We can express the algorithm entirely in terms of covariance & correlation matrices
    rx_ij = np.stack([
        np.stack([
            utils.cov(dsx_n_samples[i, :, :, 0], dsx_n_samples[j, :, :, 0], center_data=False)
            for i in range(n)
        ])
        for j in range(n)
    ])

    # Run algorithm steps
    hat_enc_n_c, w_mat_c = _causal_order(rx_ij, atol_eigv,kappa=kappa)

    # Transform the encoder estimates back up to `d` dimensions
    hat_enc_c = hat_enc_n_c @ dec_colbt

    hat_g_s, hat_enc_n_s, w_mat_s = _ancestors(rx_ij, hat_enc_n_c, w_mat_c, atol_eigv)

    # Transform the encoder estimates back up to `d` dimensions
    hat_enc_s = hat_enc_n_s @ dec_colbt

    # Optional hard intervention routine
    hard_results = None
    if hard_intervention:
        hat_g_h, hat_enc_n_h, suffstat = _unmixing_cov(
            x_n_samples,
            hat_enc_n_s,
            hat_g_s,
            w_mat_c, # note the c, not s.
            atol_ci_test
        )

        # Transform the encoder estimates back up to `d` dimensions
        hat_enc_h = hat_enc_n_h @ dec_colbt
        hard_results = (hat_enc_h, hat_g_h, suffstat)

        if hard_graph_postprocess:
            hat_g_h *= hat_g_s

    return w_mat_c, hat_enc_c, w_mat_s, hat_enc_s, hat_g_s, hard_results


def _causal_order(
    rx_ij: npt.NDArray[np.float_],
    atol_eigv: float,
    kappa: int
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.int_]]:
    n, _, d, _ = rx_ij.shape
    assert n == d
    logger.debug(f"Starting `_causal_order`.")

    #KAPPA_ARR = [-1, 1, 1, 1, 2, 3, 5, 9, 32]
    #assert(n < len(KAPPA_ARR))
    #kappa = KAPPA_ARR[n]

    h_mat = np.zeros((n, d))
    w_mat = np.zeros((n, n), dtype=np.int64)

    for t in range(n):
        for w in utils.umn.weight_set_generator(kappa, n):
            r_w = sum(w[i] * w[j] * rx_ij[i, j, :, :] for i in range(n) for j in range(n))

            cm_eig = np.linalg.eigh(r_w)
            cm_eigval: npt.NDArray[np.float_] = cm_eig.eigenvalues
            cm_eigvec: npt.NDArray[np.float_] = cm_eig.eigenvectors
            cm_rank = np.sum(cm_eigval > atol_eigv, dtype=np.int_)
            logger.debug(f"Rank of vt column space: {cm_rank}")
            rw_colb = cm_eigvec[:, -cm_rank:]

            # if cm_rank > t + K: yada yada

            ht_b = np.linalg.qr(h_mat[:t, :].T, "complete").Q
            ht_nullb = ht_b[:, t:]
            ht_nullp = ht_nullb @ ht_nullb.T
            proj_h_null_dsx_w_colb = ht_nullp @ rw_colb

            # TODO: Think on this
            cm_svs = np.linalg.svd(proj_h_null_dsx_w_colb, compute_uv=False)
            cm_rank = np.sum(cm_svs > atol_eigv, dtype=np.int_)
            logger.debug(f"Rank of proj null etc etc: {cm_rank}")

            if cm_rank == 1:
                u, _, _ = np.linalg.svd(rw_colb @ rw_colb.T @ ht_nullb)
                h_mat[t, :] = u[:, 0]
                w_mat[:, t] = np.array(w)
                break

        assert np.any(w_mat[:,t] != 0)

    w_mat = utils.umn.divide_by_gcd_matrix(w_mat)
    
    return h_mat, w_mat


def _ancestors(
    rx_ij: npt.NDArray[np.float_],
    h_mat_c: npt.NDArray[np.float_],
    w_mat_c: npt.NDArray[np.int_],
    atol_eigv: float,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float_], npt.NDArray[np.int_]]:
    n, _, d, _ = rx_ij.shape
    assert (n == d)
    logger.debug(f"Starting `_ancestors`.")

    hat_g_s = np.zeros((n, n), dtype=np.bool_)
    h_mat_s = h_mat_c.copy()
    w_mat_s = w_mat_c.copy()

    for t in reversed(range(n)):
        for j in range(t + 1, n):
            if hat_g_s[t, j]: continue
            if_parent = True
            mtj = [i for i in range(j) if i != t and not hat_g_s[t, i]]

            max_entry_w_t = int(np.sum(np.abs(w_mat_s[:,j])))
            max_entry_w_j = int(np.sum(np.abs(w_mat_s[:,t])))

            #for b, a in product(range(1, n * kappa + 1),utils.umn.generate_alternating_range(new_kappa,start_with_zero=True)):
            #for a, b in product(range(-new_kappa, new_kappa + 1), range(1, n * kappa + 1)):
            for (a, b) in product(range(-max_entry_w_t, max_entry_w_t + 1), range(1, max_entry_w_j + 1)):
                w_new = a * w_mat_s[:, t] + b * w_mat_s[:, j]
                r_w = sum(w_new[i] * w_new[j] * rx_ij[i, j, :, :] for i in range(n) for j in range(n))

                cm_eig = np.linalg.eigh(r_w)
                cm_eigval: npt.NDArray[np.float_] = cm_eig.eigenvalues
                cm_eigvec: npt.NDArray[np.float_] = cm_eig.eigenvectors
                cm_rank = np.sum(cm_eigval > atol_eigv, dtype=np.int_)
                # cm_rank = np.sum(cm_eigval > np.sum(w_new) * atol_eigv, dtype=np.int_)
                rw_colb = cm_eigvec[:, -cm_rank:]

                # if cm_rank > t + K: yada yada

                h_mtj_b = np.linalg.qr(h_mat_s[mtj, :].T, "complete").Q
                h_mtj_nullb = h_mtj_b[:, len(mtj):]
                h_mtj_nullp = h_mtj_nullb @ h_mtj_nullb.T
                proj_h_null_dsx_w_colb = h_mtj_nullp @ rw_colb

                # TODO: Think on this
                cm_svs = np.linalg.svd(proj_h_null_dsx_w_colb, compute_uv=False)
                cm_rank = np.sum(cm_svs > atol_eigv, dtype=np.int_)
                # cm_rank = np.sum(cm_svs > np.sum(w_new) * atol_eigv, dtype=np.int_)

                if cm_rank == 1:
                    u, _, _ = np.linalg.svd(rw_colb @ rw_colb.T @ h_mtj_nullb)

                    h_mat_s[j, :] = u[:, 0]
                    w_mat_s[:, j] = np.array(w_new)

                    if_parent = False
                    break

            if if_parent:
                hat_g_s[t, j] = True
                hat_g_s[t, :] |= hat_g_s[j, :]

    return hat_g_s, h_mat_s, w_mat_s


def _unmixing_cov(
    x_samples: npt.NDArray[np.float_],
    hat_enc_s: npt.NDArray[np.float_],
    hat_g_s: npt.NDArray[np.bool_],
    w_mat_s: npt.NDArray[np.int_],
    atol_ci_test: float,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float_]]:
    logger.debug(f"Starting `_unmixing`.")

    n1, _, d, _ = x_samples.shape
    n = n1 - 1
    hat_enc_h = hat_enc_s.copy()
    hat_g_h = hat_g_s.copy()
    x_cov = utils.cov(x_samples[..., 0])

    ### ENCODER UPDATE
    for t in range(1,n):
        an_t = np.where(hat_g_s[:,t])[0]
        if len(an_t) == 0:
            # t has no ancestors, already identified
            continue
        else:       
            hat_z_cov_all = hat_enc_h @ x_cov @ hat_enc_h.T
            u_obs = np.linalg.solve(hat_z_cov_all[0][an_t][:,an_t], -hat_z_cov_all[0][an_t][:,t])                
            # candidate environments
            m_list = np.where(w_mat_s[:,t])[0]
            ### TODO: Pick the m with largest difference -- gonna be easier I think
            for m in m_list:
                u_m = np.linalg.solve(hat_z_cov_all[m+1][an_t][:,an_t], -hat_z_cov_all[m+1][an_t][:,t])         
                if np.linalg.norm(u_m - u_obs) > 1e-1:
                    # ok we found an environment in which node pi_t is intervened
                    hat_enc_h[t,:] += u_m @ hat_enc_h[an_t][:]
                    break

    ### GRAPH UPDATE
    hat_z_obs_samples = np.squeeze(hat_enc_h @ x_samples[0])
    suffstat = partial_correlation_suffstat(hat_z_obs_samples)
    hat_g_h = np.empty((n, n), dtype=bool)        
    # start by leveraging the soft graph
    # transitive reduction is already done
    hat_g_tr = utils.dag.transitive_reduction(hat_g_s)
    #hat_g_h[np.where(hat_g_tr)] = True

    for t in range(n):
        for j in range(n):
            # if hat_g_tr[t,j] == True:
            #     hat_g_h[t,j] = True
            # elif hat_g_s[t,j] == False:
            #     hat_g_h[t,j] = False
            if hat_g_s[t,j] == False:
                hat_g_h[t,j] = False
            else:
                hat_g_h[t,j] = True
                # check the remaining edges
                hat_pa_j = list(np.where(hat_g_s[:,j])[0])
                hat_pa_j_minus_t = setminus(hat_pa_j,[t])

                if partial_correlation_test(suffstat,t,j,hat_pa_j_minus_t)['p_value'] > atol_ci_test:
                    hat_g_h[t, j] = False

    return hat_g_h, hat_enc_h, suffstat
