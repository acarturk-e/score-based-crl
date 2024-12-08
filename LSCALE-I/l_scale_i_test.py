"""Linear Score-based Causal Latent Estimation via Interventions (LSCALE-I)

Setting: X = G.Z, i.e. linear transform. 
single-node interventions.

Paper: Score-based Causal Representation Learning: Linear and General Transformations
(https://arxiv.org/abs/2402.00849)

Runs the algorithm (l_scale_i.py) on chosen settings.
"""


import sys
import os
# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

#import datetime
import logging
import pickle
from typing import Callable
import numpy as np
import numpy.typing as npt
#import torch
#import scipy.stats
import time

import utils
from l_scale_i import l_scale_i

scm_type = "linear" # "linear" or "quadratic"

# single-node or multi-node
run_dir = os.path.join("results","SN")
if not os.path.exists(run_dir):
    os.makedirs(run_dir)


DECODER_MIN_COND_NUM = 1e-1
ATOL_EIGV = 5e-2

ATOL_ORTH = 1e-1 
# e.g. for single-node interventions: linear SCM: n=5, use 1e-1 | n=8, use 2e-1
# quadratic SCM: n=5, use 5e-1
ATOL_EDGE = 0.1 # can be updated during analysis
ATOL_EFF_NZ = 1e-1 # to threshold the effective mixing matrix

if scm_type == "linear":
    from scm.linear import LinearSCM as SCM
elif scm_type == "quadratic":
    from scm.quadratic import QuadraticSCM as SCM
else:
    raise ValueError(f"{scm_type=} is not recognized!")


if __name__ == "__main__":    
    t0 = time.time()
    nd_list = [(5,11)]
    results_all = {(n,d) : [] for (n,d) in nd_list}

    fill_rate = 0.5  # graph density
    nsamples = 10_000
    nruns = 2
    np_rng = np.random.default_rng()
    hard_intervention = True

    # Score computation/estimation settings
    estimate_score_fns = True
    if estimate_score_fns == True:
        nsamples_for_se = nsamples
    else:
        nsamples_for_se = 0

    # SCM settings
    if scm_type == "linear":
        full_rank_scores = False
        enable_gaussian_score_est = True
        n_score_epochs = 0
    elif scm_type == "quadratic":
        full_rank_scores = True
        enable_gaussian_score_est = False
        # for estimating scores (via SSM: sliced score matching)
        n_score_epochs = 20

    if hard_intervention == True:
        type_int = "hard int"
        hard_graph_postprocess = True
    else:
        type_int = "soft int"
        hard_graph_postprocess = False

    # so this is for intervention mechanism, for both of multi-node and single-node
    var_change_mech = "scale"
    var_change = 0.25

    randomize_top_order = True
    randomize_intervention_order = True

    # Logger setup
    log_file = os.path.join(run_dir, "out.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    log_formatter = logging.Formatter(
        "%(asctime)s %(process)d %(levelname)s %(message)s"
    )
    log_file_h = logging.FileHandler(log_file)
    log_file_h.setFormatter(log_formatter)
    log_file_h.setLevel(logging.DEBUG)
    log_console_h = logging.StreamHandler()
    log_console_h.setFormatter(log_formatter)
    log_console_h.setLevel(logging.WARNING)
    log_root_l = logging.getLogger()
    log_root_l.setLevel(logging.INFO)
    log_root_l.addHandler(log_file_h)
    log_root_l.addHandler(log_console_h)

    logging.info(f"Logging to {log_file}")

    # shared config for all runs
    config_top = {
        "int_size": "SN",
        "ATOL_EIGV": ATOL_EIGV,
        "ATOL_ORTH": ATOL_ORTH,
        "ATOL_EDGE": ATOL_EDGE,
        "ATOL_EFF_NZ": ATOL_EFF_NZ,
        "DECODER_MIN_COND_NUM": DECODER_MIN_COND_NUM,
        "scm_type": scm_type,
        "intervention_type": type_int,
        "var_change_mech": var_change_mech,
        "var_change": var_change,
        "nruns" : nruns,
        "nsamples": nsamples,
        "estimate_score_fns": estimate_score_fns,
        "n_score_epochs": n_score_epochs,
        "nsamples_for_se": nsamples_for_se,
    }

    for nd_idx, (n, d) in enumerate(nd_list):
        print(f"Starting {(n, d) = }")

        # sample run_name: linear_soft_n5_d20_ns10k_nr100_gt
        run_name = (
            scm_type + "_"
            + ("hard" if hard_intervention else "soft") + "_"
            + f"n{n}" + "_"
            + f"d{d}" + "_"
            + f"ns{nsamples/1000:g}k" + "_"
            + f"nr{nruns}" + "_"
            + (
                "gt"
                if not estimate_score_fns
                else (
                    "ssm"
                    if not enable_gaussian_score_est or scm_type != "linear"
                    else "gaus"
                )
            )
        )

        # TO SAVE THE RESULTS
        results_nd = {}
        results_nd["config"] = config_top
        results_nd["config"]["zx_dim"] = (n,d)
        # and create saving results for nruns
        results_nd["results"] = [
            { 
                ### Common Inputs
                "decoder": np.empty((d, n)),
                "encoder": np.empty((n, d)),
                "dag_gt_s": np.empty((n,n)),
                "dag_gt_h": np.empty((n,n)),
                ### Outputs
                "is_run_ok": False,
                # `_obtain_top_order`
                "top_order": np.empty((n,), dtype=np.int_), 
                # `_minimize_score_variations`
                "hat_enc_s": np.empty((n, d)),
                "hat_enc_h": np.empty((n, n)),
                "dshatz_cor": np.empty((n, n, n)),
                "mcc_s": 0.0,
                "mcc_h": 0.0,
                "eff_transform_s": np.empty((n, n)),
                "eff_transform_h": np.empty((n, n)),
                "hat_g_s": np.empty((n, n), dtype=bool),
                "hat_g_h": np.empty((n, n), dtype=bool),
                "shd_s": 0.0,
                "shd_h": 0.0,
                "edge_precision_s": 0.0,
                "edge_precision_h": 0.0,
                "edge_recall_s": 0.0,
                "edge_recall_h": 0.0,
                "norm_z_err_s": 0.0,
                "norm_z_err_h": 0.0,
                "extra_nz_in_eff_s": 0,
                "extra_nz_in_eff_h": 0,
            } for _ in range(nruns)
        ]

        for run_idx in range(nruns):
                
            if run_idx % 10 == 10 - 1:
                print(f"{(n, d) = }, {run_idx = }")

            # generate a decoder (mixing matrix = linear transformation)
            decoder, encoder = utils.generate_mixing(n=n,d=d,np_rng=np_rng,DECODER_MIN_COND_NUM=DECODER_MIN_COND_NUM)

            # generate an SCM
            scm = SCM(
                n,
                fill_rate,
                randomize_top_order=randomize_top_order,
                np_rng=np_rng,
            )

            intervention_order = np_rng.permutation(n) if randomize_intervention_order else np.arange(n)
            envs = [list[int]()] + [[i] for i in intervention_order]

            # generate latent (z) and observed (x) samples for each environment
            z_samples = np.stack(
                [
                    scm.sample(
                        (nsamples,),
                        nodes_int=env,
                        type_int=type_int,
                        var_change_mech=var_change_mech,
                        var_change=var_change,
                    )
                    for env in envs
                ]
            )
            z_samples_norm = (z_samples.__pow__(2).sum() ** (0.5))
            x_samples = decoder @ z_samples

            # needed for dim_reduction of x (from d to n)
            x_samples_cov = utils.cov(x_samples[0, : n + d, :, 0])
            xsc_eigval, xsc_eigvec = np.linalg.eigh(x_samples_cov)
            basis_of_x_supp = xsc_eigvec[:, -n:]

            if estimate_score_fns == True:
                hat_sx_fns = utils.estimate_score_fns_from_data(scm=scm,scm_type=scm_type,envs=envs,decoder=decoder,basis_of_x_supp=basis_of_x_supp,nsamples_for_se=nsamples_for_se, n_score_epochs=n_score_epochs,type_int=type_int,var_change_mech=var_change_mech,var_change=var_change)

                sx_samples = np.stack(
                    [
                        hat_sx_fns[env_idx](x_samples[0, ...])
                        for env_idx in range(len(envs))
                    ]
                )
                sz_samples = decoder.T @ sx_samples

            else:
                # Use ground truth score functions
                sz_samples = np.stack(
                    [
                        scm.score_fn(
                            z_samples[0, ...],
                            nodes_int=env,
                            type_int=type_int,
                            var_change_mech=var_change_mech,
                            var_change=var_change,
                        )
                        for env in envs
                    ]
                )
                sx_samples = encoder.T @ sz_samples
   
            # ok, here are dsz and dsx
            dsz_samples = sz_samples[0, ...] - sz_samples[1:, ...]
            dsz_cor = (
                np.swapaxes(dsz_samples[..., 0], -1, -2) @ dsz_samples[..., 0]
            ) / nsamples

            dsx_samples = sx_samples[0, ...] - sx_samples[1:, ...]
            dsx_cor = (
                np.swapaxes(dsx_samples[..., 0], -1, -2) @ dsx_samples[..., 0]
            ) / nsamples

            # Record input state (except data samples)
            results_run = results_nd["results"][run_idx]
            results_run["scm"] = scm
            #results_run["dsz_cor"] = dsz_cor
            #results_run["dsx_cor"] = dsx_cor
            results_run["decoder"] = decoder
            results_run["encoder"] = encoder


            ##########  RUN LSCALE-I  ##########
            results_run["intervention_order"] = intervention_order

            # We aim to recover the latents permuted with intervention order
            goal_dag = scm.adj_mat[intervention_order, :][:, intervention_order]
            goal_z_samples = z_samples[..., intervention_order, :]
            goal_decoder = decoder[:, intervention_order]
            goal_encoder = encoder[intervention_order, :]

            try:
                top_order, hat_g_s, hat_enc_s, hard_ests = l_scale_i(
                    x_samples,
                    dsx_samples,
                    hard_intervention=hard_intervention,
                    full_rank_scores=full_rank_scores,
                    hard_graph_postprocess=hard_graph_postprocess,
                    atol_eigv=ATOL_EIGV,
                    atol_orth=ATOL_ORTH,
                    atol_edge=ATOL_EDGE
                )
            except Exception as err:
                logging.error(f"Unexpected {err=}, masking entry out")
                results_run["is_run_ok"] = False
                continue

            # This run succeeded
            results_run["is_run_ok"] = True
            results_run["top_order"] = top_order
            results_run["hat_enc_s"] = hat_enc_s
            results_run["hat_g_s"] = hat_g_s                

            ### ANALYSIS

            # Graph accuracy metrics: SHD, precision, recall
            # We compare the graph itself if full rank OR hard intervention
            # Otherwise, we compare transitive closures    
            if full_rank_scores:
                dag_gt_s = goal_dag
            else:
                dag_gt_s = utils.dag.transitive_closure(goal_dag)
                assert dag_gt_s is not None

            results_run["dag_gt_s"] = dag_gt_s

            tp_s, fp_s, fn_s, tn_s = utils.cm_graph_entries(dag_gt_s,hat_g_s)
            eff_transform_s = hat_enc_s @ goal_decoder
            eff_transform_s *= (np.sign(np.diagonal(eff_transform_s)) / np.linalg.norm(eff_transform_s, ord=2, axis=1))[:, None]

            hat_z_samples_s = eff_transform_s @ z_samples

            z_samples_obs = np.squeeze(z_samples[0])
            hat_z_samples_obs_s = np.squeeze(hat_z_samples_s[0])
            # check MCC for observational data
            mcc_s = utils.mcc(z_samples_obs,hat_z_samples_obs_s)
            shd_s =  utils.structural_hamming_distance(dag_gt_s,hat_g_s)

            results_run["mcc_s"] = mcc_s
            results_run["shd_s"] = shd_s
            results_run["edge_precision_s"] = tp_s / (tp_s + fp_s)
            results_run["edge_recall_s"] = tp_s / (tp_s + fn_s)
            results_run["eff_transform_s"] = eff_transform_s
            results_run["norm_z_err_s"] = ((hat_z_samples_s - z_samples).__pow__(2).sum() ** (0.5)) / z_samples_norm

            # Construct maximal theoretically allowed mixing pattern: either up to ancestors or up to surrounding. Note that sur(g_anc) = g_anc already. So we can apply surrounding effect on dag_gt_s directly.
            dag_gt_s_chp = [np.nonzero((dag_gt_s + np.eye(n, dtype=bool))[i, :])[0] for i in range(n)]  
            dag_gt_s_surp = [
                [
                    j for j in range(n)
                    if all([k in dag_gt_s_chp[j] for k in dag_gt_s_chp[i]])
                ]
                for i in range(n)
            ]
            soft_mixing_mat = np.array([[j in dag_gt_s_surp[i] for j in range(n)] for i in range(n)])
            # for the given threshold ATOL_EFF_NZ, check the ratio & number of incorrect nonzero entries in effective mixing
            results_run["extra_nz_in_eff_s"] = np.sum(
                (np.abs(eff_transform_s) >= ATOL_EFF_NZ) & ~soft_mixing_mat, dtype=int
            )
            results_run["extra_nz_ratio_in_eff_s"] = results_run["extra_nz_in_eff_s"]/(n**2 - np.sum(soft_mixing_mat))


            if hard_intervention:
                assert hard_ests is not None
                hat_g_h, hat_enc_h, dshatz_cor = hard_ests
                results_run["hat_enc_h"] = hat_enc_h
                results_run["hat_g_h"] = hat_g_h
                results_run["dshatz_cor"] = dshatz_cor
                # DAG related
                dag_gt_h = goal_dag
                results_run["dag_gt_h"] = dag_gt_h
                tp_h, fp_h, fn_h, tn_h = utils.cm_graph_entries(dag_gt_h,hat_g_h)
                results_run["shd_h"] = utils.structural_hamming_distance(dag_gt_h,hat_g_h)
                results_run["edge_precision_h"] = tp_h / (tp_h + fp_h)
                results_run["edge_recall_h"] = tp_h / (tp_h + fn_h)
                # Z related
                eff_transform_h = hat_enc_h @ goal_decoder
                eff_transform_h *= (np.sign(np.diagonal(eff_transform_h)) / np.linalg.norm(eff_transform_h, ord=2, axis=1))[:, None]
                hat_z_samples_h = eff_transform_h @ z_samples
                hat_z_samples_obs_h = np.squeeze(hat_z_samples_h[0])
                # check MCC for observational data
                results_run["mcc_h"] = utils.mcc(z_samples_obs,hat_z_samples_obs_h)
                results_run["eff_transform_h"] = eff_transform_h
                results_run["norm_z_err_h"] = ((hat_z_samples_h - z_samples).__pow__(2).sum() ** (0.5)) / z_samples_norm
                results_run["extra_nz_in_eff_h"] = np.sum((np.abs(eff_transform_h) >= ATOL_EFF_NZ) & ~np.eye(n, dtype=bool), dtype=int)
                results_run["extra_nz_ratio_in_eff_h"] = results_run["extra_nz_in_eff_h"]/(n**2 - n)

            results_nd["results"][run_idx] = results_run

        results_all[(n,d)] = results_nd
        # save after nruns for (n,d) is done
        with open(os.path.join(run_dir,run_name+".pkl"), "wb") as f:
            pickle.dump(results_nd, f)

    t1 = time.time() - t0

    # Transpose the results dict to make it more functional
    results_all = {
        (n,d): {
            k: [results_run[k] for results_run in results_all[(n,d)]["results"]]
            for k in results_all[(n,d)]["results"][0].keys()
        }
        for (n,d) in nd_list
    }

    print("")
    print("")
    print(f"LSCALE-I Results ({nruns=}, {nsamples=}, {nsamples_for_se=}), hard_int={hard_intervention}, noisy_scores={estimate_score_fns}, scm = {scm_type}")
    print(f"  (n, d) pairs = {nd_list}")
    print(f"Algo finished in {t1} sec")


    is_run_ok = np.array([results_all[n, d]["is_run_ok"] for (n, d) in nd_list])
    n_ok_runs = is_run_ok.sum(-1)

    mcc_s = np.array([results_all[n, d]["mcc_s"] for (n, d) in nd_list])
    shd_s = np.array([results_all[n, d]["shd_s"] for (n, d) in nd_list])
    edge_precision_s = np.array([results_all[n, d]["edge_precision_s"] for (n, d) in nd_list])
    edge_recall_s = np.array([results_all[n, d]["edge_recall_s"] for (n, d) in nd_list])
    norm_z_err_s = np.array([results_all[n, d]["norm_z_err_s"] for (n, d) in nd_list])
    extra_nz_in_eff_s = np.array([results_all[n, d]["extra_nz_in_eff_s"] for (n, d) in nd_list])
    extra_nz_ratio_in_eff_s = np.array([results_all[n, d]["extra_nz_ratio_in_eff_s"] for (n, d) in nd_list])

    ## DISPLAY RESULTS
    # report means and standard errors
    print(f"    Ratio of failed runs = {1.0 - n_ok_runs / nruns}")
    print(f"== Score minimization == ")
    print(f" = [Means], ([standard errors]) = ")
    print(f"    Structural Hamming dist = {np.around(shd_s.sum(-1) / n_ok_runs, 3)}, ({np.around(shd_s.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    Edge precision = {np.around(edge_precision_s.sum(-1) / n_ok_runs, 3)}, ({np.around(edge_precision_s.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    Edge recall = {np.around(edge_recall_s.sum(-1) / n_ok_runs, 3)}, ({np.around(edge_recall_s.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    Normalized Z error = {np.around(norm_z_err_s.sum(-1) / n_ok_runs, 3)}, ({np.around(norm_z_err_s.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    # of incorrect mixing = {np.around(extra_nz_in_eff_s.sum(-1) / n_ok_runs, 3)}, ({np.around(extra_nz_in_eff_s.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    ratio of incorrect mixing = {np.around(extra_nz_ratio_in_eff_s.sum(-1) / n_ok_runs, 3)}, ({np.around(extra_nz_ratio_in_eff_s.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    MCC Soft = {np.around(mcc_s.sum(-1) / n_ok_runs, 3)}, ({np.around(mcc_s.std(-1) / np.sqrt(n_ok_runs), 3)})")

    if hard_intervention:
        mcc_h = np.array([results_all[n, d]["mcc_h"] for (n, d) in nd_list])
        shd_h = np.array([results_all[n, d]["shd_h"] for (n, d) in nd_list])
        edge_precision_h = np.array([results_all[n, d]["edge_precision_h"] for (n, d) in nd_list])
        edge_recall_h = np.array([results_all[n, d]["edge_recall_h"] for (n, d) in nd_list])
        norm_z_err_h = np.array([results_all[n, d]["norm_z_err_h"] for (n, d) in nd_list])
        extra_nz_in_eff_h = np.array([results_all[n, d]["extra_nz_in_eff_h"] for (n, d) in nd_list])
        extra_nz_ratio_in_eff_h = np.array([results_all[n, d]["extra_nz_ratio_in_eff_h"] for (n, d) in nd_list])

        print(f"== Unmixing (for hard int) == ")
        print(f" = [Means], ([standard errors]) = ")
        print(f"    Structural Hamming dist = {np.around(shd_h.sum(-1) / n_ok_runs, 3)}, ({np.around(shd_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    Edge precision = {np.around(edge_precision_h.sum(-1) / n_ok_runs, 3)}, ({np.around(edge_precision_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    Edge recall = {np.around(edge_recall_h.sum(-1) / n_ok_runs, 3)}, ({np.around(edge_recall_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    Normalized Z error = {np.around(norm_z_err_h.sum(-1) / n_ok_runs, 3)}, ({np.around(norm_z_err_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    # of incorrect mixing = {np.around(extra_nz_in_eff_h.sum(-1) / n_ok_runs, 3)}, ({np.around(extra_nz_in_eff_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    ratio of incorrect mixing = {np.around(extra_nz_ratio_in_eff_h.sum(-1) / n_ok_runs, 3)}, ({np.around(extra_nz_ratio_in_eff_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    MCC Hard = {np.around(mcc_h.sum(-1) / n_ok_runs, 3)}, ({np.around(mcc_h.std(-1) / np.sqrt(n_ok_runs), 3)})")







