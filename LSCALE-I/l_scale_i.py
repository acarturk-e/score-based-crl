"""Linear Score-based Causal Latent Estimation via Interventions (LSCALE-I)

Setting: X = G.Z, i.e. linear transform. 
single-node interventions.

Paper: Score-based Causal Representation Learning: Linear and General Transformations
(https://arxiv.org/abs/2402.00849), published at JMLR 05/2025.
"""

__all__ = ["l_scale_i"]


import sys
import os
# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

import logging

import numpy as np
import numpy.typing as npt

import utils

def l_scale_i(
    x_samples: npt.NDArray[np.floating],
    dsx_samples: npt.NDArray[np.floating],
    hard_intervention: bool,
    hard_graph_postprocess: bool,
    soft_graph_postprocess: bool,
    full_rank_scores: bool,
    ATOL_EIGV: float,
    ATOL_EDGE: float,
) -> tuple[
    npt.NDArray[np.int_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    tuple[
        npt.NDArray[np.int_],
        npt.NDArray[np.bool_],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]
    | None,
]:
    '''
    LSCALE-I (Algorithm 1) in the paper.

    Inputs:
    - x_samples: (n + 1, nsamples, d, 1) array of samples. First dimension is observational, the rest are n single-node interventions.
    - dsx_samples: (n, nsamples, d, 1) array of samples. Score differences for n single-node interventions w.r.t. observational environment. Samples from the obs. environment are used.
    - hard_intervention: True if the interventions are hard, False if they are soft.
    - hard_graph_postprocess: True if the hard int. graph estimate should be postprocessed with the soft int. graph estimate.
    - soft_graph_postprocess: True if the full-rank soft int. graph estimate should be postprocessed with the vanilla soft int. graph estimate.
    - full_rank_scores: True if the score differences are known to be full-rank, False otherwise. (see Assumption 2 of the paper)
    - ATOL_EIGV: (only for full-rank scores case) Tolerance for eigenvalue thresholding.
    - ATOL_EDGE: Tolerance for edge thresholding in graph estimate.
    '''

    assert dsx_samples.ndim == 4 and dsx_samples.shape[3] == 1
    n, nsamples, d, _ = dsx_samples.shape
    assert x_samples.shape == (n + 1, nsamples, d, 1)

    # Preprocessing:
    # x and relevant parts of dsx lives in `n` dimensional column space of the decoder.
    # Find this subspace and write x and dsx samples in this basis.
    # Note that `n` random samples almost surely suffice for this task. We use `n + d` samples just in case.
    x_cov = utils.cov(x_samples[:, : n + d, :, 0])
    _, dec_svec = np.linalg.eigh(np.sum(x_cov, 0))
    dec_colbt = dec_svec[:, -n:].T
    x_n_samples = dec_colbt @ x_samples
    dsx_n_samples = dec_colbt @ dsx_samples

    # We can express the algorithm entirely in terms of covariance & correlation matrices
    x_n_cov = utils.cov(x_n_samples[..., 0])
    dsx_n_cor = utils.cov(dsx_n_samples[..., 0], center_data=False)

    # Run algorithm steps
    hat_enc_n_s = _estimate_encoder(dsx_n_cor)
    hat_dec_n_s = np.linalg.pinv(hat_enc_n_s)
    # optional: normalize the decoder
    #hat_dec_n_s /= np.max(np.abs(hat_dec_n_s), 0)
    #hat_enc_n_s = np.linalg.pinv(hat_dec_n_s)

    A = dec_colbt.T @ hat_dec_n_s
    hat_dec_n_s = dec_colbt @ (A / np.max(np.abs(A), 0))
    hat_enc_n_s = np.linalg.pinv(hat_dec_n_s)

    # Compute the "estimated latent" score diffs
    dsz_cor_s = hat_dec_n_s.T @ dsx_n_cor @ hat_dec_n_s
    # Estimate the graph
    hat_g_s, top_order_s = _estimate_graph(dsz_cor_s, ATOL_EDGE)
    # Transform the encoder estimates back up to `d` dimensions
    hat_enc_s = hat_enc_n_s @ dec_colbt
    soft_results = (hat_g_s, hat_enc_s, top_order_s, dsz_cor_s)
    # placeholders
    hard_results = None
    soft_full_rank_results = None

    # See Assumption 2 of the paper: if the score differences are known to be full-rank, then we can use a slightly different algorithm to get better estimates.
    if full_rank_scores == True:
        hat_g_f, hat_enc_n_f = l_scale_i_full_rank_scores(dsx_n_cor, hat_g_s, top_order_s, hat_enc_n_s, ATOL_EIGV)
        if soft_graph_postprocess == True:
            hat_g_f *= hat_g_s

            
        hat_dec_n_f = np.linalg.pinv(hat_enc_n_f)
        # WLOG, normalize the decoder
        hat_dec_n_f /= np.max(np.abs(hat_dec_n_f), 0)
        # Transform the encoder estimates back up to `d` dimensions
        hat_enc_f = hat_enc_n_f @ dec_colbt
        # Compute the "estimated latent" score diffs
        dsz_cor_f = hat_dec_n_f.T @ dsx_n_cor @ hat_dec_n_f
        soft_full_rank_results = (hat_g_f, hat_enc_f, top_order_s, dsz_cor_f)
    
    # Optional: hard intervention routine. If the interventions are hard, then we can refine the estimates further.
    if hard_intervention:
        hat_enc_n_h, hat_g_h, top_order_h, dsz_cor_h = _unmixing_cov(x_n_cov, dsx_n_cor, top_order_s, hat_enc_n_s , hat_g_s, ATOL_EDGE)
        # Transform the encoder estimates back up to `d` dimensions
        hat_enc_h = hat_enc_n_h @ dec_colbt
        
        if hard_graph_postprocess:
            hat_g_h *= hat_g_s

        hard_results = (hat_g_h, hat_enc_h, top_order_h, dsz_cor_h)

    return soft_results, hard_results, soft_full_rank_results


def _estimate_encoder(
    dsx_cor: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    n, d, _ = dsx_cor.shape
    logging.info(f"Starting `_estimate_encoder`.")
    """Estimate the encoder.

    Stage L1 of Algorithm 1 in the paper.

    Inputs:
    - dsx_cor: (n, d, d) array of score differences covariance matrices for n single-node interventions w.r.t. observational environment. See R_X matrices in the paper: dsx_cor[i] = R_X^i.

    The original algorithm randomly picks a vector from column space of `rxs[m]` and assigns it to the m-th row of the encoder.

    Here, we use the eigenvector corresponding to the largest eigenvalue to make the algorithm deterministic."""

    # We disregard any possible "trivial" null space components:
    # If the original decoder has a null space, _all_ dsx_cor null spaces include it.
    assert n == d

    encoder_hat = np.zeros((n, d))
    for m in range(n):
        # Estimate the encoder row by row from score differences
        _, eigvectors = np.linalg.eigh(dsx_cor[m])
        encoder_hat[m] = eigvectors[:, -1]

    return encoder_hat

def _estimate_graph(dsz_cor: npt.NDArray[np.floating],ATOL_EDGE:float
                    ) -> npt.NDArray[np.bool_]:
    logging.info(f"Starting `_estimate_graph`.")
    """Estimate the graph.

    Stage L2 of Algorithm 1 in the paper.

    Given that the encoder is estimated (at worst) up to mixing with parents, the corresponding score differences are non-zero only at locations corresponding to the intervened node's ancestors."""
    # Root-mean-square of the estimated latent score diffs
    # can be computed from correlation matrices as follows
    n = dsz_cor.shape[0]

    hat_g = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        hat_g[:,i] = np.diagonal(dsz_cor[i]) >= ATOL_EDGE

    hat_g &= ~np.eye(n, dtype=np.bool_)
    # Make the estimated graph acyclic
    hat_g = utils.dag.closest_dag(hat_g)
    # Find and save a topological order
    top_order = utils.dag.topological_order(hat_g)

    assert top_order is not None

    return hat_g, top_order



def _unmixing_cov(
    x_cov: npt.NDArray[np.floating],
    dsx_cor: npt.NDArray[np.floating],
    top_order: npt.NDArray[np.int_],
    hat_enc_s: npt.NDArray[np.floating],
    hat_g_s: npt.NDArray[np.bool_],
    ATOL_EDGE: float,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.floating]]:
    logging.info(f"Starting `_unmixing_procedure`.")
    logging.debug(f"{ATOL_EDGE = }")

    '''
    Unmixing procedure for hard interventions.
    Stage L3 of Algorithm 1 in the paper.
    
    Refines the encoder and graph estimates.
    '''

    hat_enc_h = hat_enc_s.copy()
    #hat_g_h = hat_g_s.copy()
    n = hat_g_s.shape[0]

    ## ENCODER UPDATE
    for t in range(1,n):
        m = top_order[t]
        an_m = np.where(hat_g_s[:,m])[0]
        if len(an_m) == 0:
            # t has no ancestors, already identified
            continue
        else:       
            # get the covariance of Z for m-th int. environment
            hat_z_cov = hat_enc_s @ x_cov[m+1] @ hat_enc_s.T
            # solve for the unmixing vector
            u = np.linalg.solve(hat_z_cov[an_m][:,an_m], hat_z_cov[an_m][:,m])
            hat_enc_h[m,:] -= u @ hat_enc_s[an_m][:]

    ## DECODER UPDATE
    hat_dec_h = np.linalg.pinv(hat_enc_h)
    dsz_cor = hat_dec_h.T @ dsx_cor @ hat_dec_h

    ## GRAPH UPDATE
    hat_g_h, top_order_h = _estimate_graph(dsz_cor, ATOL_EDGE)

    return hat_enc_h, hat_g_h, top_order_h, dsz_cor

def l_scale_i_full_rank_scores(
    dsx_cor: npt.NDArray[np.floating],
    hat_g_s: npt.NDArray[np.bool_],
    top_order: npt.NDArray[np.int_],
    hat_enc_n_s: npt.NDArray[np.floating],
    ATOL_EIGV: float,
) -> tuple[
    npt.NDArray[np.int_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    tuple[
        npt.NDArray[np.int_],
        npt.NDArray[np.bool_],
        npt.NDArray[np.floating],
        npt.NDArray[np.floating],
    ]
    | None,
]:
    '''
    LSCALE-I for sufficiently nonlinear latent causal models.
    Algorithm 2 in the paper.
    '''

    assert dsx_cor.ndim == 3
    _, d, d = dsx_cor.shape
    n = len(top_order)

    hat_g_s_tr = utils.dag.transitive_reduction(hat_g_s)
    hat_g_s_tc = utils.dag.transitive_closure(hat_g_s)

    ### LATENT GRAPH ESTIMATION
    hat_g_f = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        k = top_order[i]
        for j in range(i):
            t = top_order[j]
            # OPTIONAL: leverage known causal relationships from previous steps
            if hat_g_s_tr[t,k] == True:
                hat_g_f[t,k] = 1
                continue
            elif hat_g_s_tc[t,k] == False:
                hat_g_f[t,k] = 0
                continue


            subspace_tk = utils.linalg.subspace_intersection_from_cor([dsx_cor[t],dsx_cor[k]],tol=ATOL_EIGV)
            rank_tk_int = subspace_tk.shape[1]

            hat_pa_tk_int = np.where(hat_g_f[:,t]*hat_g_f[:,k])[0]
            if rank_tk_int > len(hat_pa_tk_int):
                hat_g_f[t,k] = True 

    # ENCODER ESTIMATION
    hat_enc_f = np.zeros((n, d))
    R_list = [dsx_cor[i] for i in range(n)]
    for m in range(n):
        # Estimate the encoder row by row from score differences
        ch_m = np.where(hat_g_f[m,:])[0]
        chplus_m = np.append(ch_m, m)
        Q_int = utils.linalg.subspace_intersection_from_cor([R_list[i] for i in chplus_m], tol=1e-6)
        if Q_int.shape[1] == 0:
            # this should not happen, but if it happens, we just keep the old vector from the previous step
            hat_enc_f[m] = hat_enc_n_s[m]
        else:
            # note that np.linalg.eigh returns eigenvectors in ascending order, so last basis is the strongest one.
            hat_enc_f[m] = Q_int[:,-1]


    return hat_g_f, hat_enc_f
