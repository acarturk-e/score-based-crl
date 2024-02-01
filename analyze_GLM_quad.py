import pickle
import numpy as np
import torch
import utils

th = 0.2

run_dir = "results/glm_quad_gt"
nd_list = [
    (5, 5),
]
nruns = 10

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

    for run_idx in range(nruns):
        with open(f"{run_dir}/{n}_{d}_{run_idx}.pkl", "rb") as f:
            pkl_dict = pickle.load(f)
            A_mat[run_idx] = pkl_dict["scm"].adj_mat
            t = pkl_dict["t"]
            u = pkl_dict["u"]
            #err_fro[run_idx] = pkl_dict["err_fro"]
            t = t * np.sign(t[0])
            u = u * np.sign(t[0])
            t_u_fro = np.linalg.norm(t - u, "fro")
            t_fro = np.linalg.norm(t, "fro")
            err_fro[run_idx] = t_u_fro / t_fro

            t_u_cosines_diag = pkl_dict["t_u_cosines_diag"]
            dhat = pkl_dict["dhat"]
            dhat_obs_hard_1 = pkl_dict["dhat_obs_hard_1"]
            dhat_obs_hard_2 = pkl_dict["dhat_obs_hard_2"]

            # just look at upper-triangular part
            dag_hat_th[run_idx] = np.triu(dhat_obs_hard_1,k=1) > th 

            (tp[run_idx],fp[run_idx],fn[run_idx]) = utils.graph_diff2(dag_hat_th[run_idx], A_mat[run_idx])

            shd_th[run_idx] = utils.structural_hamming_distance(dag_hat_th[run_idx], A_mat[run_idx])


            # dag_hat_th[run_idx] = dhat_obs_hard_1 > th 

            # (tp[run_idx],fp[run_idx],fn[run_idx]) = graph_diff2(dag_hat_th[run_idx], A_mat[run_idx] | np.eye(n, dtype=np.bool_))
            
            # shd_th[run_idx] = structural_hamming_distance(
            #     dag_hat_th[run_idx], A_mat[run_idx] | np.eye(n, dtype=np.bool_))


    avg_precision = sum(tp) / (sum(tp) + sum(fp))
    avg_recall = sum(tp) / (sum(tp) + sum(fn))
    avg_shd = np.mean(shd_th)
    avg_err_fro = np.mean(err_fro)

    print(f"n = {n}, d = {d}, nruns = {nruns}")
    print(f"Threshold value: {th}")
    print(f"Error Frob. (T-U)_F/T_F: {avg_err_fro}")
    print(f"DAG accuracy:{len([i for i in range(nruns) if shd_th[i] == 0]) / nruns}")
    print(f"Average SHD of T: {avg_shd}")
    print(f"DAG recovery ratio: {(np.array(shd_th) == 0).mean()}")
    print(f"Average precision: {avg_precision}",f"recall: {avg_recall}\n\n")
