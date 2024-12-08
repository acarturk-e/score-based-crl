import sys
import os
# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

#%%
import pickle
import numpy as np
import utils
import torch

# TODO: this code is written for an earlier version of the experiments (the ones reported in the paper)
# need to revise it for the released code.

run_dir = "results/glm_quad_ssm"
nd_list = [(5,5)]
#nd_list = [(8,8),(8,25),(8,40)]
nruns = 1
# e.g., lambdaG=0.1 for n=5 and 0.2 for n=8
lambdaG = 0.1


#%%
for (n, d) in nd_list:
    print(f"(n, d) = {(n, d)}")
    err_fro = [0.0 for _ in range(nruns)]
    A_mat = [np.empty((n, n), dtype=np.bool_) for _ in range(nruns)]
    dag_hat_th = [np.empty((n, n), dtype=np.bool_) for _ in range(nruns)]
    tp = [0 for _ in range(nruns)]
    fp = [0 for _ in range(nruns)]
    fn = [0 for _ in range(nruns)]
    edge_precisions = [0.0 for _ in range(nruns)]
    edge_recalls = [0.0 for _ in range(nruns)]
    shd_th = [0 for _ in range(nruns)]
    mse = [0 for _ in range(nruns)]
    corr_coefs = [0 for _ in range(nruns)]
    mcc_list = [0 for _ in range(nruns)]

    for run_idx in range(nruns):
        with open(f"{run_dir}/{n}_{d}_{run_idx}.pkl", "rb") as f:
            pkl_dict = pickle.load(f)
            l0_threshold = pkl_dict["l0_threshold"]
            nsteps_max = pkl_dict["nsteps_max"]
            nsamples = pkl_dict["nsamples"]
            A_mat[run_idx] = pkl_dict["scm"].adj_mat
            t = pkl_dict["t"]
            u = pkl_dict["u"]

            t_u_cosines_diag = pkl_dict["t_u_cosines_diag"]
            u = u * np.sign(t_u_cosines_diag)

            t_u_fro = np.linalg.norm(t - u, "fro")
            t_fro = np.linalg.norm(t, "fro")
            err_fro[run_idx] = t_u_fro / t_fro

            dhat = pkl_dict["dhat"]
            dhat_obs_hard_1 = pkl_dict["dhat_obs_hard_1"]
            dhat_obs_hard_2 = pkl_dict["dhat_obs_hard_2"]

            dag_hat_th[run_idx] = dhat_obs_hard_1 > lambdaG
            np.fill_diagonal(dag_hat_th[run_idx],0)

            (tp[run_idx],fp[run_idx],fn[run_idx]) = utils.graph_diff2(dag_hat_th[run_idx], A_mat[run_idx])

            shd_th[run_idx] = utils.structural_hamming_distance(dag_hat_th[run_idx], A_mat[run_idx])

            zs_obs = pkl_dict["zs_obs"]
            zhats_obs = np.linalg.pinv(u) @ t @ zs_obs
            corr_coefs[run_idx] = np.mean([np.corrcoef(zs_obs[i].T,zhats_obs[i].T)[0,1] for i in range(len(zs_obs))])
            mse[run_idx] = np.linalg.norm(np.squeeze(zhats_obs-zs_obs),2) / np.linalg.norm(np.squeeze(zs_obs),2)

            mcc_list[run_idx] = utils.mcc(np.squeeze(zs_obs),np.squeeze(zhats_obs))



    avg_precision = sum(tp) / (sum(tp) + sum(fp))
    avg_recall = sum(tp) / (sum(tp) + sum(fn))
    avg_shd = np.mean(shd_th)
    se_shd = np.std(shd_th) / np.sqrt(nruns)
    avg_err_fro = np.mean(err_fro)
    avg_mse = np.mean(mse)
    se_mse = np.std(mse) / np.sqrt(nruns)
    avg_mcc = np.mean(mcc_list)
    se_mcc = np.std(mcc_list) / np.sqrt(nruns)


    print(f"n = {n}, d = {d}, nruns = {nruns}")
    print(f"nsamples = {nsamples}")
    print(f"nsteps_max for learning: {nsteps_max}")
    print(f"l0_threshold for early stop: {l0_threshold}")
    print(f"Threshold for obtaining DAG: {lambdaG}")
    print(f"Error Frob. (T-U)_F/T_F: {avg_err_fro}")
    print(f"Mean-square error. (Z- hat Z)_2/Z_2: {avg_mse},  s.e: {se_mse}")
    print(f"MCC mean: {avg_mcc}, S.E : {se_mcc} ")
    print(f"DAG accuracy:{len([i for i in range(nruns) if shd_th[i] == 0]) / nruns}")
    print(f"Average SHD: {avg_shd}, se: {se_shd}")
    print(f"DAG recovery ratio: {(np.array(shd_th) == 0).mean()}")
    print(f"Average precision: {avg_precision}",f"recall: {avg_recall}\n\n")
