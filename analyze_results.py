import argparse
import os
import pickle

import numpy as np

# Result dir setup: Take it from command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("run_dir", help="Path to the specific run directory")
parser.add_argument("--nruns", type=int, help="Override the `nruns` saved in config file")
parser.add_argument("--atol-edge-sweep", action="store_true", help="Try different `atol_edge` values for `hat_g_h` estimation")
args = parser.parse_args()
run_dir = args.run_dir
assert os.path.exists(run_dir)

# Load config
with open(os.path.join(run_dir, "config.pkl"), "rb") as f:
    config = pickle.load(f)

# Extract config parameters
ATOL_EIGV = config["ATOL_EIGV"]
ATOL_ORTH = config["ATOL_ORTH"]
ATOL_EDGE = config["ATOL_EDGE"]
SCM = config["scm_type"]
full_rank_scores = config["full_rank_scores"]
hard_intervention = config["hard_intervention"]
hard_graph_postprocess = config["hard_graph_postprocess"]
var_change_mech = config["var_change_mech"]
var_change = config["var_change"]
nd_list = config["nd_list"]
nruns = config["nruns"]
nsamples = config["nsamples"]
estimate_score_fns = config["estimate_score_fns"]
nsamples_for_se = config["nsamples_for_se"]

# If nruns override was provided, apply it
if args.nruns is not None:
    nruns = args.nruns

# Load results
results = {nd: [{} for _ in range(nruns)] for nd in nd_list}
for (n, d) in nd_list:
    for run_idx in range(nruns):
        with open(os.path.join(run_dir, f"{n}_{d}_{run_idx}.pkl"), "rb") as f:
            results[n, d][run_idx] = pickle.load(f)

# Transpose the results dict to make it more functional
results = {
    nd: {
        k: [results_run[k] for results_run in results_run_list]
        for k in results_run_list[0].keys()
    }
    for (nd, results_run_list) in results.items()
}

# Extract results
scm = {nd: results_nd["scm"] for (nd, results_nd) in results.items()}
intervention_order = {nd: results_nd["intervention_order"] for (nd, results_nd) in results.items()}
dsz_cor = {nd: results_nd["dsz_cor"] for (nd, results_nd) in results.items()}
decoder = {nd: results_nd["decoder"] for (nd, results_nd) in results.items()}
encoder = {nd: results_nd["encoder"] for (nd, results_nd) in results.items()}
dsx_cor = {nd: results_nd["dsx_cor"] for (nd, results_nd) in results.items()}
is_run_ok = {nd: results_nd["is_run_ok"] for (nd, results_nd) in results.items()}
top_order = {nd: results_nd["top_order"] for (nd, results_nd) in results.items()}
hat_g_s = {nd: results_nd["hat_g_s"] for (nd, results_nd) in results.items()}
hat_enc_s = {nd: results_nd["hat_enc_s"] for (nd, results_nd) in results.items()}
shd_s = {nd: results_nd["shd_s"] for (nd, results_nd) in results.items()}
edge_precision_s = {nd: results_nd["edge_precision_s"] for (nd, results_nd) in results.items()}
edge_recall_s = {nd: results_nd["edge_recall_s"] for (nd, results_nd) in results.items()}
norm_z_err_s = {nd: results_nd["norm_z_err_s"] for (nd, results_nd) in results.items()}
extra_nz_in_eff_s = {nd: results_nd["extra_nz_in_eff_s"] for (nd, results_nd) in results.items()}
hat_g_h = {nd: results_nd["hat_g_h"] for (nd, results_nd) in results.items()}
hat_enc_h = {nd: results_nd["hat_enc_h"] for (nd, results_nd) in results.items()}
shd_h = {nd: results_nd["shd_h"] for (nd, results_nd) in results.items()}
edge_precision_h = {nd: results_nd["edge_precision_h"] for (nd, results_nd) in results.items()}
edge_recall_h = {nd: results_nd["edge_recall_h"] for (nd, results_nd) in results.items()}
norm_z_err_h = {nd: results_nd["norm_z_err_h"] for (nd, results_nd) in results.items()}
extra_nz_in_eff_h = {nd: results_nd["extra_nz_in_eff_h"] for (nd, results_nd) in results.items()}

# Display results
print(f"Results ({nruns=}, {nsamples_for_se=})")
print(f"    (n, d) pairs = {nd_list}")

is_run_ok = np.array([results[n, d]["is_run_ok"] for (n, d) in nd_list])
n_ok_runs = is_run_ok.sum(-1)

shd_s = np.array([results[n, d]["shd_s"] for (n, d) in nd_list])
edge_precision_s = np.array([results[n, d]["edge_precision_s"] for (n, d) in nd_list])
edge_recall_s = np.array([results[n, d]["edge_recall_s"] for (n, d) in nd_list])
norm_z_err_s = np.array([results[n, d]["norm_z_err_s"] for (n, d) in nd_list])
extra_nz_in_eff_s = np.array([results[n, d]["extra_nz_in_eff_s"] for (n, d) in nd_list])

print(f"    Ratio of failed runs = {1.0 - n_ok_runs / nruns}")
print(f"Score minimization")
print(f"    Structural Hamming dist = {np.around(shd_s.sum(-1) / n_ok_runs, 3)}")
print(f"    Edge precision          = {np.around(edge_precision_s.sum(-1) / n_ok_runs, 3)}")
print(f"    Edge recall             = {np.around(edge_recall_s.sum(-1) / n_ok_runs, 3)}")
print(f"    Normalized Z error      = {np.around(norm_z_err_s.sum(-1) / n_ok_runs, 3)}")
print(f"    # of pa minus sur mixed = {np.around(extra_nz_in_eff_s.sum(-1) / n_ok_runs, 3)}")

if hard_intervention:
    shd_h = np.array([results[n, d]["shd_h"] for (n, d) in nd_list])
    edge_precision_h = np.array([results[n, d]["edge_precision_h"] for (n, d) in nd_list])
    edge_recall_h = np.array([results[n, d]["edge_recall_h"] for (n, d) in nd_list])
    norm_z_err_h = np.array([results[n, d]["norm_z_err_h"] for (n, d) in nd_list])
    extra_nz_in_eff_h = np.array([results[n, d]["extra_nz_in_eff_h"] for (n, d) in nd_list])

    print(f"Unmixing")
    print(f"    Structural Hamming dist = {np.around(shd_h.sum(-1) / n_ok_runs, 3)}")
    print(f"    Edge precision          = {np.around(edge_precision_h.sum(-1) / n_ok_runs, 3)}")
    print(f"    Edge recall             = {np.around(edge_recall_h.sum(-1) / n_ok_runs, 3)}")
    print(f"    Normalized Z error      = {np.around(norm_z_err_h.sum(-1) / n_ok_runs, 3)}")
    print(f"    # of pa minus sur mixed = {np.around(extra_nz_in_eff_h.sum(-1) / n_ok_runs, 3)}")

# Re-compute hard SHD using a different threshold
if args.atol_edge_sweep:
    for (n, d) in nd_list:
        results[n, d]["new_shd_h"] = [0 for _ in range(nruns)]
        results[n, d]["new_edge_precision_h"] = [0.0 for _ in range(nruns)]
        results[n, d]["new_edge_recall_h"] = [0.0 for _ in range(nruns)]

    for new_atol_edge in [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7]:
        for (n, d) in nd_list:
            for run_idx in range(nruns):
                dag_gt_h = scm[n, d][run_idx].adj_mat[intervention_order[n, d][run_idx], :][:, intervention_order[n, d][run_idx]]

                new_hat_g_h = np.zeros((n, n), dtype=np.bool_)
                hat_enc_h_pinvt = np.linalg.pinv(hat_enc_h[n, d][run_idx]).T
                dshatz_cor = hat_enc_h_pinvt @ dsx_cor[n, d][run_idx] @ hat_enc_h_pinvt.T
                for i in range(n):
                    new_hat_g_h[:, i] = np.diagonal(dshatz_cor[i]) >= new_atol_edge
                # Remove the i == j case -- these are not actual edges
                new_hat_g_h &= ~np.eye(n, dtype=np.bool_)

                if hard_graph_postprocess:
                    new_hat_g_h *= hat_g_s[n, d][run_idx]

                edge_cm_h = [
                    [
                        ( dag_gt_h &  new_hat_g_h).sum(dtype=int),
                        (~dag_gt_h &  new_hat_g_h).sum(dtype=int),
                    ], [
                        ( dag_gt_h & ~new_hat_g_h).sum(dtype=int),
                        (~dag_gt_h & ~new_hat_g_h).sum(dtype=int),
                    ]
                ]
                results[n, d]["new_shd_h"]           [run_idx] = edge_cm_h[0][1] + edge_cm_h[1][0]
                results[n, d]["new_edge_precision_h"][run_idx] = (edge_cm_h[0][0] / (edge_cm_h[0][0] + edge_cm_h[0][1])) if (edge_cm_h[0][0] + edge_cm_h[0][1]) != 0 else 1.0
                results[n, d]["new_edge_recall_h"]   [run_idx] = (edge_cm_h[0][0] / (edge_cm_h[0][0] + edge_cm_h[1][0])) if (edge_cm_h[0][0] + edge_cm_h[1][0]) != 0 else 1.0

            new_shd_h = np.array([results[n, d]["new_shd_h"] for (n, d) in nd_list])
            new_edge_precision_h = np.array([results[n, d]["new_edge_precision_h"] for (n, d) in nd_list])
            new_edge_recall_h = np.array([results[n, d]["new_edge_recall_h"] for (n, d) in nd_list])
            print(f"With {new_atol_edge = }")
            print(f"    Structural Hamming dist = {np.around(new_shd_h.sum(-1) / n_ok_runs, 3)}")
            print(f"    Edge precision          = {np.around(new_edge_precision_h.sum(-1) / n_ok_runs, 3)}")
            print(f"    Edge recall             = {np.around(new_edge_recall_h.sum(-1) / n_ok_runs, 3)}")
