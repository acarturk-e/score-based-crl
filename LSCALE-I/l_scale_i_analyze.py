#from typing import Any
import numpy as np
import pickle
import os
import sys
'''
use read_and_display_results(file_path) to read and display results from a pickle file.

example usage: file_path = 'codebase/LSCALE-I/results/SN/quadratic_hard_n5_d100_ns50k_nr50_ssm.pkl'

read_and_display_results(file_path)

'''
#data = util_analysis.load_pickle('/Users/burak/Desktop/all-CRL-code/JMLR-final-codebase/LSCALE-I/results/SN/quadratic_hard_n5_d100_ns50k_nr50_ssm.pkl')

# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def read_and_display_results(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)    
    
    results = data["results"]
    config = data["config"]
    nruns = config["nruns"]
    (n,d) = config["zx_dim"]
    if config["intervention_type"] == "hard int":
        hard_intervention = True
    else:
        hard_intervention = False

    if config["scm_type"] == "full rank":
        full_rank_scores = True
    else:
        full_rank_scores = False

    # Transpose the results dict to make it more functional
    results = {
        (n,d): {
            k: [results_run[k] for results_run in results]
            for k in results[0].keys()
        }
    }
    #read_soft_results, read_hard_results, read_full_rank_results = read_results(results, [(n,d)], hard_intervention, full_rank_scores)
    print(f"CONFIG: {config} \n")
    print("RESULTS: \n")

    display_results(results, [(n,d)], nruns, hard_intervention=hard_intervention, full_rank_scores=full_rank_scores)



def read_results(results_all, nd_list, hard_intervention=False, full_rank_scores=False):
    mcc_s = np.array([results_all[n, d]["mcc_s"] for (n, d) in nd_list])
    shd_s_tr = np.array([results_all[n, d]["shd_s_tr"] for (n, d) in nd_list])
    shd_s_tc = np.array([results_all[n, d]["shd_s_tc"] for (n, d) in nd_list])
    edge_precision_s = np.array([results_all[n, d]["edge_precision_s"] for (n, d) in nd_list])
    edge_recall_s = np.array([results_all[n, d]["edge_recall_s"] for (n, d) in nd_list])
    extra_nz_in_eff_s = np.array([results_all[n, d]["extra_nz_in_eff_s"] for (n, d) in nd_list])
    extra_nz_ratio_in_eff_s = np.array([results_all[n, d]["extra_nz_ratio_in_eff_s"] for (n, d) in nd_list])
    eff_transform_err_norm_s = np.array([results_all[n, d]["eff_transform_err_norm_s"] for (n, d) in nd_list])


    read_soft_results = [mcc_s, shd_s_tr, shd_s_tc, edge_precision_s, edge_recall_s, extra_nz_in_eff_s, extra_nz_ratio_in_eff_s, eff_transform_err_norm_s]

    if hard_intervention == True:
        mcc_h = np.array([results_all[n, d]["mcc_h"] for (n, d) in nd_list])
        shd_h = np.array([results_all[n, d]["shd_h"] for (n, d) in nd_list])
        edge_precision_h = np.array([results_all[n, d]["edge_precision_h"] for (n, d) in nd_list])
        edge_recall_h = np.array([results_all[n, d]["edge_recall_h"] for (n, d) in nd_list])
        extra_nz_in_eff_h = np.array([results_all[n, d]["extra_nz_in_eff_h"] for (n, d) in nd_list])
        extra_nz_ratio_in_eff_h = np.array([results_all[n, d]["extra_nz_ratio_in_eff_h"] for (n, d) in nd_list])
        eff_transform_err_norm_h = np.array([results_all[n, d]["eff_transform_err_norm_h"] for (n, d) in nd_list])

        read_hard_results = [mcc_h, shd_h, edge_precision_h, edge_recall_h, extra_nz_in_eff_h, extra_nz_ratio_in_eff_h, eff_transform_err_norm_h]
    else:
        read_hard_results = [None for _ in range(len(read_soft_results)-1)]

    if full_rank_scores == True:
        mcc_f = np.array([results_all[n, d]["mcc_f"] for (n, d) in nd_list])
        shd_f = np.array([results_all[n, d]["shd_f"] for (n, d) in nd_list])
        edge_precision_f = np.array([results_all[n, d]["edge_precision_f"] for (n, d) in nd_list])
        edge_recall_f = np.array([results_all[n, d]["edge_recall_f"] for (n, d) in nd_list])
        extra_nz_in_eff_f = np.array([results_all[n, d]["extra_nz_in_eff_f"] for (n, d) in nd_list])
        extra_nz_ratio_in_eff_f = np.array([results_all[n, d]["extra_nz_ratio_in_eff_f"] for (n, d) in nd_list])
        eff_transform_err_norm_f = np.array([results_all[n, d]["eff_transform_err_norm_f"] for (n, d) in nd_list])

        read_full_rank_results = [mcc_f, shd_f, edge_precision_f, edge_recall_f, extra_nz_in_eff_f, extra_nz_ratio_in_eff_f, eff_transform_err_norm_f]
    else:
        read_full_rank_results = [None for _ in range(len(read_soft_results)-1)]

    return read_soft_results, read_hard_results, read_full_rank_results


def display_results(results_all, nd_list, nruns, hard_intervention=False, full_rank_scores=False):
    ## DISPLAY RESULTS
    # report means and standard errors
    is_run_ok = np.array([results_all[n, d]["is_run_ok"] for (n, d) in nd_list])
    n_ok_runs = is_run_ok.sum(-1)

    read_soft, read_hard, read_full_rank = read_results(results_all, nd_list, hard_intervention, full_rank_scores)

    mcc_s, shd_s_tr, shd_s_tc, edge_precision_s, edge_recall_s, extra_nz_in_eff_s, extra_nz_ratio_in_eff_s, eff_transform_err_norm_s = read_soft
    mcc_h, shd_h, edge_precision_h, edge_recall_h, extra_nz_in_eff_h, extra_nz_ratio_in_eff_h, eff_transform_err_norm_h = read_hard
    mcc_f, shd_f, edge_precision_f, edge_recall_f, extra_nz_in_eff_f, extra_nz_ratio_in_eff_f, eff_transform_err_norm_f = read_full_rank


    print(f"    Ratio of failed runs = {1.0 - n_ok_runs / nruns}")
    print(f"== Main stage (for general soft int) == ")
    print(f" = [Means], ([standard errors]) = ")
    print(f"    SHD : tr.reduction = {np.around(shd_s_tr.sum(-1) / n_ok_runs, 3)}, ({np.around(shd_s_tr.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    SHD : tr.closure = {np.around(shd_s_tc.sum(-1) / n_ok_runs, 3)}, ({np.around(shd_s_tc.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    Edge precision = {np.around(edge_precision_s.sum(-1) / n_ok_runs, 3)}, ({np.around(edge_precision_s.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    Edge recall = {np.around(edge_recall_s.sum(-1) / n_ok_runs, 3)}, ({np.around(edge_recall_s.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    # of incorrect mixing = {np.around(extra_nz_in_eff_s.sum(-1) / n_ok_runs, 3)}, ({np.around(extra_nz_in_eff_s.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    ratio of incorrect mixing = {np.around(extra_nz_ratio_in_eff_s.sum(-1) / n_ok_runs, 3)}, ({np.around(extra_nz_ratio_in_eff_s.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    norm of incorrect mixing = {np.around(eff_transform_err_norm_s.sum(-1) / n_ok_runs, 3)}, ({np.around(eff_transform_err_norm_s.std(-1) / np.sqrt(n_ok_runs), 3)})")

    print(f"    MCC Soft = {np.around(mcc_s.sum(-1) / n_ok_runs, 3)}, ({np.around(mcc_s.std(-1) / np.sqrt(n_ok_runs), 3)})")

    if hard_intervention:
        print(f"== Unmixing (for hard int) == ")
        print(f" = [Means], ([standard errors]) = ")
        print(f"    Structural Hamming dist = {np.around(shd_h.sum(-1) / n_ok_runs, 3)}, ({np.around(shd_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    Edge precision = {np.around(edge_precision_h.sum(-1) / n_ok_runs, 3)}, ({np.around(edge_precision_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    Edge recall = {np.around(edge_recall_h.sum(-1) / n_ok_runs, 3)}, ({np.around(edge_recall_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    # of incorrect mixing = {np.around(extra_nz_in_eff_h.sum(-1) / n_ok_runs, 3)}, ({np.around(extra_nz_in_eff_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    ratio of incorrect mixing = {np.around(extra_nz_ratio_in_eff_h.sum(-1) / n_ok_runs, 3)}, ({np.around(extra_nz_ratio_in_eff_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    norm of incorrect mixing = {np.around(eff_transform_err_norm_h.sum(-1) / n_ok_runs, 3)}, ({np.around(eff_transform_err_norm_h.std(-1) / np.sqrt(n_ok_runs), 3)})")  
        print(f"    MCC Hard = {np.around(mcc_h.sum(-1) / n_ok_runs, 3)}, ({np.around(mcc_h.std(-1) / np.sqrt(n_ok_runs), 3)})")

    if full_rank_scores:
        print(f"== Full rank scores (for full rank scores) == ")
        print(f" = [Means], ([standard errors]) = ")
        print(f"    Structural Hamming dist = {np.around(shd_f.sum(-1) / n_ok_runs, 3)}, ({np.around(shd_f.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    Edge precision = {np.around(edge_precision_f.sum(-1) / n_ok_runs, 3)}, ({np.around(edge_precision_f.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    Edge recall = {np.around(edge_recall_f.sum(-1) / n_ok_runs, 3)}, ({np.around(edge_recall_f.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    # of incorrect mixing = {np.around(extra_nz_in_eff_f.sum(-1) / n_ok_runs, 3)}, ({np.around(extra_nz_in_eff_f.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    ratio of incorrect mixing = {np.around(extra_nz_ratio_in_eff_f.sum(-1) / n_ok_runs, 3)}, ({np.around(extra_nz_ratio_in_eff_f.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    norm of incorrect mixing = {np.around(eff_transform_err_norm_f.sum(-1) / n_ok_runs, 3)}, ({np.around(eff_transform_err_norm_f.std(-1) / np.sqrt(n_ok_runs), 3)})")
        print(f"    MCC Full rank = {np.around(mcc_f.sum(-1) / n_ok_runs, 3)}, ({np.around(mcc_f.std(-1) / np.sqrt(n_ok_runs), 3)})")
