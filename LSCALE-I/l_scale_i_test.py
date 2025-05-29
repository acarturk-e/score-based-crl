"""Linear Score-based Causal Latent Estimation via Interventions (LSCALE-I)

Setting: X = G.Z, i.e. linear transform. 
single-node interventions.

Paper: Score-based Causal Representation Learning: Linear and General Transformations
(https://arxiv.org/abs/2402.00849)

Runs the algorithm (l_scale_i.py) on chosen settings.

To replicate the results in the paper (Tables 3, 4, 5, 6, 7, 8, 11), adjust the parameters (e.g., `n`, `d`, `n_samples`, etc.) in this script accordingly.
"""

import sys
import os
from l_scale_i_analyze import read_and_display_results, read_results, display_results, load_pickle

# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

#import datetime
import logging
import pickle
#from typing import Callable
import numpy as np
import numpy.typing as npt
#import torch
import time
import utils

from l_scale_i import l_scale_i
#from l_scale_i import _estimate_graph
#from l_scale_i import _estimate_encoder

scm_type = "quadratic" # "linear" or "quadratic" or "mlp"

# single-node interventions for LSCALE-I
run_dir = os.path.join("results","SN")
if not os.path.exists(run_dir):
    os.makedirs(run_dir)


DECODER_MIN_COND_NUM = 1e-1 # for stable decoder generation

# For main algo, the only threshold parameter we need is ATOL_EDGE for graph recovery, which also can be updated during analysis if need be. As expected, slightly adjusting ATOL_EDGE when using ground truth or estimated scores may be needed. See Table 17 in the paper for the parameter values used in the experiments.
ATOL_EDGE = 1e-3

# For full_rank scores algorithm only, we also need ATOL_EIGV for numerical rank estimation
ATOL_EIGV = 1e-2

ATOL_EFF_NZ = 5e-2 # to threshold the effective mixing matrix

if scm_type == "linear":
    from scm.linear import LinearSCM as SCM
elif scm_type == "quadratic":
    from scm.quadratic import QuadraticSCM as SCM
elif scm_type == "mlp":
    from scm.mlp import MlpSCM as SCM
else:
    raise ValueError(f"{scm_type=} is not recognized!")


if __name__ == "__main__":    
    ### EXPERIMENTATION SETUP
    t0 = time.time()
    # ENTER WHICH (n, d) settings to run
    nd_list = [(5,100)]
    results_all = {(n,d) : [] for (n,d) in nd_list}

    fill_rate = 0.5  # graph density for G(nnodes,density) model
    nsamples = 50_000
    nruns = 1
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
        soft_graph_postprocess = None
        enable_gaussian_score_est = True
        n_score_epochs = 0
    elif scm_type == "quadratic":
        # in this case, we can use the full rank scores (Assumption 2 in the paper)
        full_rank_scores = True
        soft_graph_postprocess = False
        enable_gaussian_score_est = False
        # for estimating scores (via SSM: sliced score matching)
        n_score_epochs = 20
    elif scm_type == "mlp":
        full_rank_scores = False
        soft_graph_postprocess = None
        enable_gaussian_score_est = False
        # for estimating scores (via SSM: sliced score matching)
        n_score_epochs = 10

    if hard_intervention == True:
        type_int = "hard int"
        hard_graph_postprocess = True
    else:
        type_int = "soft int"
        hard_graph_postprocess = False

    # for intervention mechanism (for exogenous variables)
    var_change_mech = "scale"
    #var_change = 0.25
    var_change = 5.0

    # set False for easier debugging and interpretation
    randomize_top_order = False
    randomize_intervention_order = False

    ### Logger setup
    log_file = os.path.join(run_dir, "out.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    log_formatter = logging.Formatter(
        "%(asctime)s %(process)d %(levelname)s %(message)s")
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

    #### START THE EXPERIMENTS
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
        results_nd["results"] = utils.generate_results_dict(nruns=nruns,n=n,d=d)

        # to debug the estimation of dsz_cor_diag
        err_est_dsz_cor_diag = np.zeros(nruns)

        for run_idx in range(nruns):
                
            if run_idx % 5 == 0:
                print(f"{(n, d) = }, {run_idx = }")

            # intervention should be fairly impactful, so that score of the parent coordinate changes. Otherwise, it is not a good test for graph estimation. If you do not care about this, set flag_valid_model = True
            flag_valid_model = False
            while flag_valid_model == False:
                print(f"generating run_idx = {run_idx}")
                # generate decoder = mixing func. = linear transform
                decoder, encoder = utils.generate_mixing(n=n,d=d,np_rng=np_rng,DECODER_MIN_COND_NUM=DECODER_MIN_COND_NUM,normalize_decoder=True,normalize_encoder=False)

                # generate a latent SCM
                scm = SCM(
                    n,
                    fill_rate,
                    randomize_top_order=randomize_top_order,
                    np_rng=np_rng,
                )

                intervention_order = np_rng.permutation(n) if randomize_intervention_order else np.arange(n)
                envs = [list[int]()] + [[i] for i in intervention_order]

                goal_dag = scm.adj_mat[intervention_order, :][:, intervention_order]

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
                x_samples = decoder @ z_samples

                # needed for dim_reduction of x (from d to n)
                x_samples_cov = utils.cov(x_samples[0, : n + d, :, 0])
                xsc_eigval, xsc_eigvec = np.linalg.eigh(x_samples_cov)
                basis_of_x_supp = xsc_eigvec[:, -n:]

                # ground truth scores: compute for debug purposes, even if we will perform score estimation from data
                true_sz_samples = np.stack(
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
                true_sx_samples = encoder.T @ true_sz_samples
    
                # ok, here are dsz and dsx
                true_dsz_samples = true_sz_samples[0, ...] - true_sz_samples[1:, ...]
                true_dsz_cor = (
                    np.swapaxes(true_dsz_samples[..., 0], -1, -2) @ true_dsz_samples[..., 0]
                ) / nsamples

                true_dsz_cor_diag = np.zeros((n,n))
                for idx_env in range(n):
                    true_dsz_cor_diag[:,idx_env] = np.diag(true_dsz_cor[idx_env])

                # if edges are not strong in score function differences, resample the model. If the edges are good (at least above a small threshold, say 0.02), then we are good to continue with this model.
                if np.sum(goal_dag) == 0:
                    flag_valid_model = True

                elif np.min(true_dsz_cor_diag[np.where(goal_dag)]) >= 0.02:
                    flag_valid_model = True

            # noisy scores
            hat_sx_fns = utils.estimate_score_fns_from_data(scm=scm,scm_type=scm_type,envs=envs,decoder=decoder,basis_of_x_supp=basis_of_x_supp,nsamples_for_se=nsamples_for_se, n_score_epochs=n_score_epochs,type_int=type_int,var_change_mech=var_change_mech,var_change=var_change)

            est_sx_samples = np.stack(
                    [
                        hat_sx_fns[env_idx](x_samples[0, ...])
                        for env_idx in range(len(envs))
                    ]
                )
            est_sz_samples = decoder.T @ est_sx_samples


            est_dsz_samples = est_sz_samples[0, ...] - est_sz_samples[1:, ...]
            est_dsz_cor = (
                np.swapaxes(est_dsz_samples[..., 0], -1, -2) @ est_dsz_samples[..., 0]
            ) / nsamples

            est_dsz_cor_diag = np.zeros((n,n))
            for idx_env in range(n):
                est_dsz_cor_diag[:,idx_env] = np.diag(est_dsz_cor[idx_env])

            err_est_dsz_cor_diag[run_idx] = np.linalg.norm(true_dsz_cor_diag-est_dsz_cor_diag,ord='fro')

            # choose to use either "ground truth" or "estimated" scores. 
            if estimate_score_fns == True:
                sx_samples = est_sx_samples
                sz_samples = est_sz_samples
                dsz_cor = est_dsz_cor
                dsz_cor_diag = est_dsz_cor_diag
            else:
                sx_samples = true_sx_samples
                sz_samples = true_sz_samples
                dsz_cor = true_dsz_cor
                dsz_cor_diag = true_dsz_cor_diag        


            dsx_samples = sx_samples[0, ...] - sx_samples[1:, ...]
            dsx_cor = (
                np.swapaxes(dsx_samples[..., 0], -1, -2) @ dsx_samples[..., 0]
            ) / nsamples

            # Record input state (except data samples)
            results_run = results_nd["results"][run_idx]
            results_run["scm"] = scm
            results_run["dsz_cor"] = dsz_cor
            results_run["decoder"] = decoder
            results_run["encoder"] = encoder
            results_run["intervention_order"] = intervention_order


            ##########  RUN LSCALE-I  ##########

            # We aim to recover the latents permuted with intervention order
            goal_dag = scm.adj_mat[intervention_order, :][:, intervention_order]
            goal_z_samples = z_samples[..., intervention_order, :]
            goal_decoder = decoder[:, intervention_order]
            goal_encoder = encoder[intervention_order, :]

            try:
                soft_results, hard_results, soft_full_rank_results = l_scale_i(
                        x_samples,
                        dsx_samples,
                        hard_intervention=hard_intervention,
                        hard_graph_postprocess=hard_graph_postprocess,
                        soft_graph_postprocess=soft_graph_postprocess,
                        full_rank_scores=full_rank_scores,
                        ATOL_EIGV=ATOL_EIGV,
                        ATOL_EDGE=ATOL_EDGE
                    )

                # unpack results
                hat_g_s, hat_enc_s, top_order_s, dshatz_cor_s = soft_results

                dshatz_cor_diag_s = np.zeros((n,n))
                for idx_env in range(n):
                    dshatz_cor_diag_s[:,idx_env] = np.diag(dshatz_cor_s[idx_env])

                # just get transitive closure of hat_g_s
                hat_g_s_tc = utils.dag.transitive_closure(hat_g_s)
                hat_g_s_tr = utils.dag.transitive_reduction(hat_g_s)

                if hard_intervention == True:
                    assert hard_results is not None
                    hat_g_h, hat_enc_h, top_order_h, dshatz_cor_h = hard_results

                    dshatz_cor_diag_h = np.zeros((n,n))
                    for idx_env in range(n):
                        dshatz_cor_diag_h[:,idx_env] = np.diag(dshatz_cor_h[idx_env])

                if full_rank_scores == True:
                    hat_g_f, hat_enc_f, top_order_f, dshatz_cor_f = soft_full_rank_results

                    dshatz_cor_diag_f = np.zeros((n,n))
                    for idx_env in range(n):
                        dshatz_cor_diag_f[:,idx_env] = np.diag(dshatz_cor_f[idx_env])

            except Exception as err:
                logging.error(f"Unexpected {err=}, masking entry out")
                results_run["is_run_ok"] = False
                continue

            # This run succeeded
            results_run["is_run_ok"] = True
            results_run["hat_g_s"] = hat_g_s                
            results_run["hat_g_s_tc"] = hat_g_s_tc                
            results_run["hat_g_s_tr"] = hat_g_s_tr                
            results_run["hat_enc_s"] = hat_enc_s
            results_run["top_order_s"] = top_order_s
            results_run["dshatz_cor_s"] = dshatz_cor_s
            if hard_intervention == True:
                results_run["hat_g_h"] = hat_g_h
                results_run["hat_enc_h"] = hat_enc_h
                results_run["top_order_h"] = top_order_h
                results_run["dshatz_cor_h"] = dshatz_cor_h
            if full_rank_scores == True:
                results_run["hat_g_f"] = hat_g_f
                results_run["hat_enc_f"] = hat_enc_f
                results_run["dshatz_cor_f"] = dshatz_cor_f

            ### ANALYSIS

            # Graph accuracy metrics: SHD, precision, recall
            # We compare the graph itself if full rank OR hard intervention. Otherwise, we compare transitive closures    
            dag_gt_tc = utils.dag.transitive_closure(goal_dag)
            dag_gt_tr = utils.dag.transitive_reduction(goal_dag)
            dag_gt = goal_dag
            results_run["dag_gt_tc"] = dag_gt_tc
            results_run["dag_gt_tr"] = dag_gt_tr
            results_run["dag_gt"] = dag_gt


            ### LSCALE-I results: for soft interventions (general)
            # check graph recovery: for soft, use transitive reduction
            results_run["shd_s_tr"] = utils.dag.structural_hamming_distance(dag_gt_tr,hat_g_s_tr)
            results_run["shd_s_tc"] = utils.dag.structural_hamming_distance(dag_gt_tc,hat_g_s_tc)
            precision_s, recall_s, f1_s = utils.dag.precision_recall_f1_graph(dag_gt_tr,hat_g_s_tr)
            results_run["edge_precision_s"] = precision_s
            results_run["edge_recall_s"] = recall_s

            # check latent variable recovery
            eff_transform_s = hat_enc_s @ goal_decoder
            eff_transform_s *= (np.sign(np.diagonal(eff_transform_s)) / np.linalg.norm(eff_transform_s, ord=2, axis=1))[:, None]
            results_run["eff_transform_s"] = eff_transform_s

            hat_z_samples_s = eff_transform_s @ z_samples
            z_samples_obs = np.squeeze(z_samples[0])
            hat_z_samples_obs_s = np.squeeze(hat_z_samples_s[0])
            # check MCC for observational data
            results_run["mcc_s"] = utils.mcc(z_samples_obs,hat_z_samples_obs_s)

            # Theoretically allowed mixing pattern: up to parents (for general soft)
            mixing_parent_mat = goal_dag.T | np.eye(n, dtype=bool)
            # for the given threshold ATOL_EFF_NZ, check the ratio & number of incorrect nonzero entries in effective mixing
            results_run["extra_nz_in_eff_s"] = np.sum(
                (np.abs(eff_transform_s) >= ATOL_EFF_NZ) & ~mixing_parent_mat, dtype=int
            )
            results_run["extra_nz_ratio_in_eff_s"] = results_run["extra_nz_in_eff_s"]/(n**2 - np.sum(mixing_parent_mat))
            # also record the norm of the error in effective mixing
            results_run["eff_transform_err_norm_s"] = np.linalg.norm(eff_transform_s * ~mixing_parent_mat, ord=2)


            if hard_intervention:
                # check graph recovery
                results_run["shd_h"] = utils.dag.structural_hamming_distance(dag_gt,hat_g_h)
                precision_h, recall_h, f1_h = utils.dag.precision_recall_f1_graph(dag_gt,hat_g_h)
                results_run["edge_precision_h"] = precision_h
                results_run["edge_recall_h"] = recall_h
                # check variable recovery
                eff_transform_h = hat_enc_h @ goal_decoder
                eff_transform_h *= (np.sign(np.diagonal(eff_transform_h)) / np.linalg.norm(eff_transform_h, ord=2, axis=1))[:, None]
                results_run["eff_transform_h"] = eff_transform_h
                hat_z_samples_h = eff_transform_h @ z_samples
                hat_z_samples_obs_h = np.squeeze(hat_z_samples_h[0])
                # check MCC for observational data
                results_run["mcc_h"] = utils.mcc(z_samples_obs,hat_z_samples_obs_h)

                results_run["extra_nz_in_eff_h"] = np.sum((np.abs(eff_transform_h) >= ATOL_EFF_NZ) & ~np.eye(n, dtype=bool), dtype=int)
                results_run["extra_nz_ratio_in_eff_h"] = results_run["extra_nz_in_eff_h"]/(n**2 - n)
                results_run["eff_transform_err_norm_h"] = np.linalg.norm(eff_transform_h * ~np.eye(n,dtype=bool), ord=2)

            if full_rank_scores:
                # check graph recovery
                results_run["shd_f"] = utils.dag.structural_hamming_distance(dag_gt,hat_g_f)
                precision_f, recall_f, f1_f = utils.dag.precision_recall_f1_graph(dag_gt,hat_g_f)
                results_run["edge_precision_f"] = precision_f
                results_run["edge_recall_f"] = recall_f
                # check variable recovery
                eff_transform_f = hat_enc_f @ goal_decoder
                eff_transform_f *= (np.sign(np.diagonal(eff_transform_f)) / np.linalg.norm(eff_transform_f, ord=2, axis=1))[:, None]
                results_run["eff_transform_f"] = eff_transform_f
                hat_z_samples_f = eff_transform_f @ z_samples
                hat_z_samples_obs_f = np.squeeze(hat_z_samples_f[0])
                # check MCC for observational data
                results_run["mcc_f"] = utils.mcc(z_samples_obs,hat_z_samples_obs_f)

                # Theoretically allowed mixing pattern: up to surrounding parents
                mixing_surrounding_mat = utils.dag.surrounding_mat(goal_dag).T | np.eye(n, dtype=bool)

                results_run["extra_nz_in_eff_f"] = np.sum(
                    (np.abs(eff_transform_f) >= ATOL_EFF_NZ) & ~mixing_surrounding_mat, dtype=int)
                results_run["extra_nz_ratio_in_eff_f"] = results_run["extra_nz_in_eff_f"]/(n**2 - np.sum(mixing_surrounding_mat))
                results_run["eff_transform_err_norm_f"] = np.linalg.norm(eff_transform_s * ~mixing_surrounding_mat, ord=2)


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
    print(f"LSCALE-I Results ({nruns=}, {nsamples=}, {nsamples_for_se=}), hard_int={hard_intervention}, noisy_scores={estimate_score_fns}, scm= {scm_type}, atol_edge = {ATOL_EDGE}",)
    if full_rank_scores == True:
        print(f"atol_eigv = {ATOL_EIGV}")
        
    print(f"  (n, d) pairs = {nd_list}")
    print(f"Algo finished in {t1} sec")

    ## read the saved results in a clean way.

    read_soft, read_hard, read_full_rank = read_results(results_all, nd_list, hard_intervention=hard_intervention, full_rank_scores=full_rank_scores)

    mcc_s, shd_s_tr, shd_s_tc, edge_precision_s, edge_recall_s, extra_nz_in_eff_s, extra_nz_ratio_in_eff_s, eff_transform_err_norm_s = read_soft
    # None if hard_intervention=False
    mcc_h, shd_h, edge_precision_h, edge_recall_h, extra_nz_in_eff_h, extra_nz_ratio_in_eff_h, eff_transform_err_norm_h = read_hard
    # None if full_rank_scores=False
    mcc_f, shd_f, edge_precision_f, edge_recall_f, extra_nz_in_eff_f, extra_nz_ratio_in_eff_f, eff_transform_err_norm_f = read_full_rank

    ### DISPLAY RESULTS 
    display_results(results_all, nd_list, nruns, hard_intervention=hard_intervention, full_rank_scores=full_rank_scores)


    #print(np.mean(err_est_dsz_cor_diag))
    #print(np.median(err_est_dsz_cor_diag))



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Table 11 in the paper: comparison with Squires et al. (2023) results on linear SCMs and linear transform.
# uncomment below to load the results 

# results_dir = os.getcwd() + "/results/SN/"

# res_n5_ns5k = load_pickle(results_dir + 'linear_hard_n5_d100_ns5k_nr100_gaus.pkl')["results"]
# n5_ns5k_err_norm = np.asarray([res_n5_ns5k[i]['eff_transform_err_norm_h'] for i in range(100)])
# res_n5_ns10k = load_pickle(results_dir + 'linear_hard_n5_d100_ns10k_nr100_gaus.pkl')["results"]
# n5_ns10k_err_norm = np.asarray([res_n5_ns10k[i]['eff_transform_err_norm_h'] for i in range(100)])
# res_n5_ns50k = load_pickle(results_dir + 'linear_hard_n5_d100_ns50k_nr100_gaus.pkl')["results"]
# n5_ns50k_err_norm = np.asarray([res_n5_ns50k[i]['eff_transform_err_norm_h'] for i in range(100)])
# res_n8_ns5k = load_pickle(results_dir + 'linear_hard_n8_d100_ns5k_nr100_gaus.pkl')["results"]
# n8_ns5k_err_norm = np.asarray([res_n8_ns5k[i]['eff_transform_err_norm_h'] for i in range(100)])
# res_n8_ns10k = load_pickle(results_dir + 'linear_hard_n8_d100_ns10k_nr100_gaus.pkl')["results"]
# n8_ns10k_err_norm = np.asarray([res_n8_ns10k[i]['eff_transform_err_norm_h'] for i in range(100)])
# res_n8_ns50k = load_pickle(results_dir + 'linear_hard_n8_d100_ns50k_nr100_gaus.pkl')["results"]
# n8_ns50k_err_norm = np.asarray([res_n8_ns50k[i]['eff_transform_err_norm_h'] for i in range(100)])
# squires_n5_err_norm = load_pickle(results_dir + 'squires_err_n5.pkl')
# squires_n5_ns5k_err_norm = squires_n5_err_norm[:,0]
# squires_n5_ns10k_err_norm = squires_n5_err_norm[:,1]
# squires_n5_ns50k_err_norm = squires_n5_err_norm[:,2]
# squires_n8_err_norm = load_pickle(results_dir + 'squires_err_n8.pkl')
# squires_n8_ns5k_err_norm = squires_n8_err_norm[:,0]
# squires_n8_ns10k_err_norm = squires_n8_err_norm[:,1]
# squires_n8_ns50k_err_norm = squires_n8_err_norm[:,2]



# %%
