#%%
import sys
import os
import pickle
import numpy as np

try:
    current_file = __file__
except NameError:
    # __file__ is not defined, e.g., interactive mode
    current_file = os.path.abspath("")

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(current_file)))
sys.path.append(repo_root)

import utils

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data



run_dir = "results/gscale_camera_abs"
scm_type, n, d, nsamples, nruns, score_est = "quadratic", 5, 100, 200, 20, "ssm"


# threshold for obtaining DAG
ATOL_EDGE = 0.5

filepath = f'{run_dir}/{scm_type}_n{n}_d{d}_ns{nsamples/1000}k_nr{nruns}_{score_est}.pkl'

res_and_config = load_pickle(filepath)
res = res_and_config['results']
config = res_and_config['config']

mcc_list = np.zeros(nruns)
shd_list = np.zeros((nruns,2))
err_fro_list = np.zeros(nruns)

tp_list = np.zeros((nruns,2), dtype=int)
fp_list = np.zeros((nruns,2), dtype=int)
fn_list = np.zeros((nruns,2), dtype=int)


for idx_run in range(nruns):
    mcc_list[idx_run] = res[idx_run]['MCC']
    err_fro_list[idx_run] = res[idx_run]['err_fro'][0]
    dag_gt_run = res[idx_run]['dag_gt']
    dhat_obs_hard_1 = res[idx_run]['dhat_obs_hard_1'][0]
    dhat_obs_hard_2 = res[idx_run]['dhat_obs_hard_2'][0]

    dag_hat_th_1 = dhat_obs_hard_1 >= ATOL_EDGE
    np.fill_diagonal(dag_hat_th_1,0)
    dag_hat_th_1 = utils.dag.closest_dag(dag_hat_th_1)

    dag_hat_th_2 = dhat_obs_hard_2 >= ATOL_EDGE
    np.fill_diagonal(dag_hat_th_2,0)
    dag_hat_th_2 = utils.dag.closest_dag(dag_hat_th_2)

    shd_list[idx_run,0] = utils.dag.structural_hamming_distance(dag_gt_run,dag_hat_th_1)
    shd_list[idx_run,1] = utils.dag.structural_hamming_distance(dag_gt_run,dag_hat_th_2)

    tp_run1, fp_run1, fn_run1, tn_run1 = utils.dag.cm_graph_entries(dag_gt_run, dag_hat_th_1)
    tp_run2, fp_run2, fn_run2, tn_run2 = utils.dag.cm_graph_entries(dag_gt_run, dag_hat_th_2)

    tp_list[idx_run,0] = tp_run1
    tp_list[idx_run,1] = tp_run2
    fp_list[idx_run,0] = fp_run1
    fp_list[idx_run,1] = fp_run2
    fn_list[idx_run,0] = fn_run1
    fn_list[idx_run,1] = fn_run2


# print(np.mean(mcc_list), np.std(mcc_list) / np.sqrt(nruns))
# print(np.mean(shd_list,0), np.std(shd_list,0) / np.sqrt(nruns))

mcc_mean = np.mean(mcc_list)
mcc_stderr = np.std(mcc_list) / np.sqrt(nruns)
shd_mean = np.mean(shd_list, axis=0)
shd_stderr = np.std(shd_list, axis=0) / np.sqrt(nruns)
edge_precision = np.sum(tp_list, axis=0) / (np.sum(tp_list, axis=0) + np.sum(fp_list, axis=0))
edge_recall = np.sum(tp_list, axis=0) / (np.sum(tp_list, axis=0) + np.sum(fn_list, axis=0))

print(f"n = {n}, d = {d}, nruns = {nruns}")
print(f"nsamples = {config['nsamples']}")
print(f"nsteps_max for learning: {config['nsteps_max']}")
print(f"l0_threshold for early stop: {config['l0_threshold']}")
print(f"MCC mean: {mcc_mean}, S.E : {mcc_stderr} ")
print(f"Edge Threshold: {ATOL_EDGE}")
print(f"Average SHD: {shd_mean}, se: {shd_stderr}")
print(f"DAG recovery ratio: {(np.array(shd_list) == 0).mean()}")
#print(f"Average precision: {edge_precision}",f"recall: {edge_recall}\n\n")
