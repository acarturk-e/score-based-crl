
import sys
import os
# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)


import pickle
import numpy as np
from utils import *

#th = 0.1

# TODO: adopt Emre's earlier analysis code for l_scale_i for applying varying ATOL_EDGE thresholds for graph recovery in the end.

run_dir = "results/MN"

nd_list = [(4,10), (5,10), (6,10), (7,10), (8,10)]
nruns = 100
intervention_type = "hard"
scm_type = "linear"
nsamples = "100k"
score_estimation = "gaus"

#%%

# read all files

res_all = {}

for (n, d) in nd_list:
    print(f"(n, d) = {(n, d)}")

    with open(f"{run_dir}/{scm_type}_{intervention_type}_n{n}_d{d}_ns{nsamples}_nr{nruns}_{score_estimation}.pkl", "rb") as f:
        pkl_dict = pickle.load(f)
        #config = pkl_dict["config"]
        res_all[(n,d)] = pkl_dict["results"]


# transpose the results file
key_list = list(res_all[(n,d)][0].keys())
key_list.remove('suff_stat')
key_list.remove('suffstat')

res_all = {
    (n,d): {
        k: [results_run[k] for results_run in res_all[(n,d)]]
        for k in key_list
    }
    for (n,d) in nd_list
}


print("")
print("")
print(f"Results ({nruns=}, {nsamples=}, int. type = {intervention_type}, score = {score_estimation}, scm = {scm_type}")
print(f"  (n, d) pairs = {nd_list}")

is_run_ok = np.array([res_all[n, d]["is_run_ok"] for (n, d) in nd_list])
n_ok_runs = is_run_ok.sum(-1)

mcc_s = np.array([res_all[n, d]["mcc_s"] for (n, d) in nd_list])
shd_s = np.array([res_all[n, d]["shd_s"] for (n, d) in nd_list])
edge_precision_s = np.array([res_all[n, d]["edge_precision_s"] for (n, d) in nd_list])
edge_recall_s = np.array([res_all[n, d]["edge_recall_s"] for (n, d) in nd_list])
norm_z_err_s = np.array([res_all[n, d]["norm_z_err_s"] for (n, d) in nd_list])
extra_nz_in_eff_s = np.array([res_all[n, d]["extra_nz_in_eff_s"] for (n, d) in nd_list])
extra_nz_ratio_in_eff_s = np.array([res_all[n, d]["extra_nz_ratio_in_eff_s"] for (n, d) in nd_list])

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


if intervention_type == "hard":
    mcc_h = np.array([res_all[n, d]["mcc_h"] for (n, d) in nd_list])
    shd_h = np.array([res_all[n, d]["shd_h"] for (n, d) in nd_list])
    edge_precision_h = np.array([res_all[n, d]["edge_precision_h"] for (n, d) in nd_list])
    edge_recall_h = np.array([res_all[n, d]["edge_recall_h"] for (n, d) in nd_list])
    norm_z_err_h = np.array([res_all[n, d]["norm_z_err_h"] for (n, d) in nd_list])
    extra_nz_in_eff_h = np.array([res_all[n, d]["extra_nz_in_eff_h"] for (n, d) in nd_list])
    extra_nz_ratio_in_eff_h = np.array([res_all[n, d]["extra_nz_ratio_in_eff_h"] for (n, d) in nd_list])

    print(f"== Unmixing (for hard int) == ")
    print(f" = [Means], ([standard errors]) = ")
    print(f"    Structural Hamming dist = {np.around(shd_h.sum(-1) / n_ok_runs, 3)}, ({np.around(shd_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    Edge precision = {np.around(edge_precision_h.sum(-1) / n_ok_runs, 3)}, ({np.around(edge_precision_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    Edge recall = {np.around(edge_recall_h.sum(-1) / n_ok_runs, 3)}, ({np.around(edge_recall_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    Normalized Z error = {np.around(norm_z_err_h.sum(-1) / n_ok_runs, 3)}, ({np.around(norm_z_err_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    # of incorrect mixing = {np.around(extra_nz_in_eff_h.sum(-1) / n_ok_runs, 3)}, ({np.around(extra_nz_in_eff_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    ratio of incorrect mixing = {np.around(extra_nz_ratio_in_eff_h.sum(-1) / n_ok_runs, 3)}, ({np.around(extra_nz_ratio_in_eff_h.std(-1) / np.sqrt(n_ok_runs), 3)})")
    print(f"    MCC Hard = {np.around(mcc_h.sum(-1) / n_ok_runs, 3)}, ({np.around(mcc_h.std(-1) / np.sqrt(n_ok_runs), 3)})")

