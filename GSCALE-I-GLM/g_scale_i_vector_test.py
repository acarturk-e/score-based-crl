#%%
r'''
GSCALE-I algorithm

- Two hard intervention per node
- Generalized Linear transformation (with tanh link function): X = tanh(G.Z)
- Quadratic (or linear) latent models
- Use score oracle or estimated scores (via SSM)
'''

import sys
import os
# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

import logging
import pickle
import time

import numpy as np
import torch
from torch import Tensor
from torch.func import jacrev, vmap  # type: ignore
from score_estimators.ssm import score_fn_from_data


import utils
from g_scale_i_algos import g_scale_i_glm_tanh


# Speeds things up by a lot compared to working in CPU
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# computes pseudo-inverse
def pinv(x: Tensor) -> Tensor:
    return torch.linalg.pinv(x)

scm_type = "quadratic" # "linear" or "quadratic"

if scm_type == "linear":
    from scm.linear import LinearSCM as SCM
elif scm_type == "quadratic":
    from scm.quadratic import QuadraticSCM as SCM
else:
    raise ValueError(f"{scm_type=} is not recognized!")

run_dir = os.path.join("results","gscalei")

# Result dir setup
run_name = "gscale_camera_abs"
run_dir = os.path.join("results", run_name)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

if __name__ == "__main__":    
    t0 = time.time()
    nd_list = [(8,100)]
    results_all = {(n,d) : [] for (n,d) in nd_list}

    fill_rate = 0.5  # graph density
    nsamples = 500
    nruns = 20
    #np_rng = np.random.default_rng()
    hard_intervention = True
    assert hard_intervention == True
    type_int = "hard int"
    use_exact_loss = True

    # Score computation/estimation settings
    estimate_score_fns = True
    if estimate_score_fns == True:
        nsamples_for_se = 10000
        n_score_epochs = 10
    else:
        nsamples_for_se = 0
        n_score_epochs = 0


    DECODER_MIN_COND_NUM = 0.1
    lambda_recon = 1.0
    nsteps_max = 20_000
    learning_rate = 1e-3
    
    float_precision = 1e-6
    l0_threshold = 0.02
    lambdaG = 1.0 # change for noisy vs. true scores

    # since we consider a non-linear transform
    enable_gaussian_score_est = False 

    # this is for intervention mechanism
    var_change_mech = "scale"
    var_change_1 = 4.0
    var_change_2 = 0.25

    randomize_top_order = False

    # Logger setup
    log_file = os.path.join(run_dir, "out.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    log_formatter = logging.Formatter("%(asctime)s %(module)s %(levelname)s %(message)s")
    log_file_h = logging.FileHandler(log_file)
    log_file_h.setFormatter(log_formatter)
    log_file_h.setLevel(logging.DEBUG)
    log_console_h = logging.StreamHandler()
    log_console_h.setFormatter(log_formatter)
    log_console_h.setLevel(logging.INFO)
    log_root_l = logging.getLogger()
    log_root_l.setLevel(logging.DEBUG)
    log_root_l.addHandler(log_file_h)
    log_root_l.addHandler(log_console_h)

    logging.info(f"Logging to {log_file}")
    logging.info("Starting")

    # shared config for all runs
    config_top = {
        "int_size": "SN",
        "DECODER_MIN_COND_NUM": DECODER_MIN_COND_NUM,
        "scm_type": scm_type,
        "intervention_type": type_int,
        "var_change_mech": var_change_mech,
        "var_change_1": var_change_1,
        "var_change_2": var_change_2,
        "nruns" : nruns,
        "nsamples": nsamples,
        "estimate_score_fns": estimate_score_fns,
        "n_score_epochs": n_score_epochs,
        "nsamples_for_se": nsamples_for_se,
        "nsteps_max": nsteps_max,
        "learning_rate": learning_rate,
        "l0_threshold": l0_threshold,
        "lambdaG": lambdaG # or ATOL_EDGE
    }

    for nd_idx, (n, d) in enumerate(nd_list):
        print(f"Starting {(n, d) = }")

        # sample run_name: linear_n5_d20_ns10k_nr100_gt
        run_name = (
            scm_type + "_"
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
                "t": np.empty((d, n)),
                "dag_gt": np.empty((n,n)),
                "intervention_order" : np.empty((n,), dtype=np.int_), 
                ### Outputs
                "is_run_ok": False,
                "u": np.empty((d, n)),
                "dag_hat_th": np.empty((n,n)),
                "edge_precision": 0.0,
                "edge_recall" : 0.0, 
                "dhat": np.empty((n, n)),
                "dhat_obs_hard_1": np.empty((n, n)),
                "dhat_obs_hard_2": np.empty((n, n)),
                "err_fro" : 100,
                "t_u_cosines":  np.empty((n, n)),
                "t_u_cosines_diag":  np.empty((n,)),
                "MSE": 0.0,
                "MCC": 0.0, 
            } for _ in range(nruns)
        ]

        for run_idx in range(nruns):
            seed = int(time.time())
            torch.manual_seed(seed)
            np_rng = np.random.default_rng(seed=seed)


            logging.info(f"n={n}, d={d}, run_idx={run_idx}")
            logging.info(f"seed={seed}")
            logging.info(f"stop learning when l0 norm is minimized under threshold {l0_threshold}")
            logging.info(f"Setup:{scm_type} latent model, Z sampling, computing X, {estimate_score_fns = }")

            results_run = results_nd["results"][run_idx]

            # Intervention order, set to 1..n now
            intervention_order = torch.arange(n)

            # generate a decoder
            t, t_pinv = utils.generate_mixing(n=n,d=d,np_rng=np_rng,DECODER_MIN_COND_NUM=DECODER_MIN_COND_NUM)

            t = torch.tensor(t)
            t = t.float()
            t = t.to(device=dev)
            t_pinv = pinv(t)

            # let's define encoder-decoder functions
            def decoder(z: Tensor, u: Tensor) -> Tensor:
                return torch.clamp((u @ z).tanh(),min=-1+float_precision, max=1-float_precision)

            def encoder(x: Tensor, u_pinv: Tensor) -> Tensor:
                return u_pinv @ x.arctanh()

            # to take autograd, little trick
            def true_decoder(z: Tensor) -> Tensor:
                    return decoder(z, t)

            def true_encoder(x: Tensor) -> Tensor:
                return encoder(x, t_pinv)
            
            # SET THE LATENT CAUSAL MODEL
            scm = SCM(n, fill_rate, randomize_top_order=randomize_top_order, np_rng=np_rng)

            # environments - intervention targets
            env_obs = []
            envs_hard_1 = [[rho_i] for rho_i in intervention_order]
            envs_hard_2 = [[rho_i] for rho_i in intervention_order]

            # create z samples
            zs_obs    =              torch.Tensor(scm.sample((nsamples,), nodes_int=[])).to(device=dev)
            zs_hard_1 = torch.stack([torch.Tensor(scm.sample((nsamples,), nodes_int=env, var_change_mech=var_change_mech, var_change=var_change_1)).to(device=dev) for env in envs_hard_1])
            zs_hard_2 = torch.stack([torch.Tensor(scm.sample((nsamples,), nodes_int=env, var_change_mech=var_change_mech, var_change=var_change_2)).to(device=dev) for env in envs_hard_2])

            # create x samples from z using t
            xs_obs = decoder(zs_obs, t)
            #xs_hard_1 = decoder(zs_hard_1, t)
            #xs_hard_2 = decoder(zs_hard_2, t)

            if estimate_score_fns == True:
                # note that the algorithm requires not too many samples for learning the decoder (e.g. 300-500 in our experiments). However, score estimation requires far too many samples, so we create samples only for score estimation part here.

                # create z samples for score estimation 
                zs_obs_se    =              torch.Tensor(scm.sample((nsamples_for_se,), nodes_int=[])).to(device=dev)
                zs_hard_1_se = torch.stack([torch.Tensor(scm.sample((nsamples_for_se,), nodes_int=env, var_change_mech=var_change_mech, var_change=var_change_1)).to(device=dev) for env in envs_hard_1])
                zs_hard_2_se = torch.stack([torch.Tensor(scm.sample((nsamples_for_se,), nodes_int=env, var_change_mech=var_change_mech, var_change=var_change_2)).to(device=dev) for env in envs_hard_2])
                # create x samples from z using t for score estimation
                xs_obs_se = decoder(zs_obs_se, t)
                xs_hard_1_se = decoder(zs_hard_1_se, t)
                xs_hard_2_se = decoder(zs_hard_2_se, t)

                # estimated score functions of x
                sx_fn_obs = score_fn_from_data(xs_obs_se, epochs=n_score_epochs).to(device=dev)
                sx_fns_hard_1 = [score_fn_from_data(xs_hard_1_env, epochs=n_score_epochs).to(device=dev) for xs_hard_1_env in xs_hard_1_se]
                sx_fns_hard_2 = [score_fn_from_data(xs_hard_2_env, epochs=n_score_epochs).to(device=dev) for xs_hard_2_env in xs_hard_2_se]

                # score estimates of x evaluated at observational data
                sxs_obs = sx_fn_obs(xs_obs[:, :, 0]).detach().unsqueeze(-1)
                sxs_hard_1 = torch.stack([sx_fn(xs_obs[:, :, 0]).detach() for sx_fn in sx_fns_hard_1]).unsqueeze(-1)
                sxs_hard_2 = torch.stack([sx_fn(xs_obs[:, :, 0]).detach() for sx_fn in sx_fns_hard_2]).unsqueeze(-1)

                # score differences between hard_1 and hard_2 environments at latent points X.
                dsxs_bw_hards = sxs_hard_1 - sxs_hard_2
                # score differences between hard_1 and observational environments at data points X.
                dsxs_obs_hard_1 = sxs_hard_1 - sxs_obs
                # score differences between hard_2 and observational environments at data points X.
                dsxs_obs_hard_2 = sxs_hard_2 - sxs_obs

            else:
                # compute ground truth score differences of X.

                # score functions of z
                sz_fn_obs     =  lambda z:      (torch.from_numpy(scm.score_fn((z).detach().cpu().numpy(), nodes_int=[])).to(device=z.device, dtype=z.dtype))
                sz_fns_hard_1 = [lambda z, i=i: (torch.from_numpy(scm.score_fn((z).detach().cpu().numpy(), nodes_int=envs_hard_1[i], var_change_mech="increase", var_change=1.0)).to(device=z.device, dtype=z.dtype)) for i in range(n)]
                sz_fns_hard_2 = [lambda z, i=i: (torch.from_numpy(scm.score_fn((z).detach().cpu().numpy(), nodes_int=envs_hard_2[i], var_change_mech="increase", var_change=2.0)).to(device=z.device, dtype=z.dtype)) for i in range(n)]

                # scores of z evaluated at observational data
                szs_obs = sz_fn_obs(zs_obs).detach()
                szs_hard_1 = torch.stack([sz_fn(zs_obs).detach() for sz_fn in sz_fns_hard_1])
                szs_hard_2 = torch.stack([sz_fn(zs_obs).detach() for sz_fn in sz_fns_hard_2])
                # score differences between hard_1 and hard_2 environments at latent z (for obs. env. data)
                dszs_bw_hards = szs_hard_1 - szs_hard_2
                # score differences between hard_1 and observational environments at latent z (for obs. env. data)
                dszs_obs_hard_1 = szs_hard_1 - szs_obs
                # score differences between hard_2 and observational environments at latent z (for obs. env. data)
                dszs_obs_hard_2 = szs_hard_2 - szs_obs

                # and reduced to matrix form for future usage
                dsz_bw_hards = (torch.linalg.vector_norm(dszs_bw_hards, ord=1, dim=1) / nsamples).mT
                dsz_obs_hard_1 = (torch.linalg.vector_norm(dszs_obs_hard_1, ord=1, dim=1) / nsamples).mT
                dsz_obs_hard_2 = (torch.linalg.vector_norm(dszs_obs_hard_2, ord=1, dim=1) / nsamples).mT

                # evaluate jacobians of true decoder at obs data in order to compute true score differences for X
                # jacobian of g:R^n \to R^d function at point z \in R^n is d by n.
                true_jacobians_at_samples = vmap(jacrev(true_decoder))(zs_obs).squeeze(-1,-3)

                J = pinv(true_jacobians_at_samples).mT # has shape (nsamples,d,n)

                # score differences between hard_1 and hard_2 environments at x (for obs. env. data)
                dsxs_bw_hards = J @ dszs_bw_hards
                # score differences between hard_1 and observational environments at x (for obs. env. data)
                dsxs_obs_hard_1 = J @ dszs_obs_hard_1
                # score differences between hard_2 and observational environments at x (for obs. env. data)
                dsxs_obs_hard_2 = J @ dszs_obs_hard_2

            # Record input state (except data samples)
            results_run = results_nd["results"][run_idx]
            results_run["scm"] = scm
            results_run["t"] = t

            # NOW READY TO START ALGORITHM
            logging.info("Starting algo")


            try:
                u = g_scale_i_glm_tanh(d=d,n=n,nsamples=nsamples,dsxs_bw_hards=dsxs_bw_hards,xs_obs=xs_obs,learning_rate=learning_rate,nsteps_max=nsteps_max,lambda_recon=lambda_recon,l0_threshold=l0_threshold,zs_obs=zs_obs)
            except Exception as err:
                logging.error(f"Unexpected {err=}, masking entry out")
                results_run["is_run_ok"] = False
                continue

            # this run succeeded
            results_run["is_run_ok"] = True

            # now, infer DAG using obs vs. hard_1 OR obs vs. hard_2 score difference matrix
            def candidate_decoder(zhat: Tensor) -> Tensor:
                return decoder(zhat, u)

            u_pinv = torch.linalg.pinv(u)
            zhats_obs = encoder(xs_obs, u_pinv)
            #zhats_hard_1 = encoder(xs_hard_1, u_pinv)
            #zhats_hard_2 = encoder(xs_hard_2, u_pinv)

            jacobians_at_samples = vmap(jacrev(candidate_decoder))(zhats_obs).squeeze(-1,-3)

            dszhats = jacobians_at_samples.mT @ dsxs_bw_hards
            dszhats_obs_hard_1 = jacobians_at_samples.mT @ dsxs_obs_hard_1
            dszhats_obs_hard_2 = jacobians_at_samples.mT @ dsxs_obs_hard_2

            dhat = (torch.linalg.vector_norm(dszhats, ord=1, dim=1)[:, :, 0] / nsamples).mT
            dhat_obs_hard_1 = (torch.linalg.vector_norm(dszhats_obs_hard_1, ord=1, dim=1)[:, :, 0] / nsamples).mT
            dhat_obs_hard_2 = (torch.linalg.vector_norm(dszhats_obs_hard_2, ord=1, dim=1)[:, :, 0] / nsamples).mT

            # permute the results, use dhat itself to do so.
            perm_to_order = torch.arange(n)
            tmp = dhat.abs()
            for _ in range(n):
                values, indices = tmp.max(dim=0)
                j = values.argmax()
                i = indices[j]
                perm_to_order[j] = i
                tmp[i, :] = 0.0
                tmp[:, j] = 0.0

            # u is estimate of t
            u = u[:, perm_to_order]
            u_pinv = pinv(u)

            dhat = dhat[perm_to_order, :]
            dhat_obs_hard_1 = dhat_obs_hard_1[perm_to_order, :]
            dhat_obs_hard_2 = dhat_obs_hard_2[perm_to_order, :]

            dag_hat_th = dhat_obs_hard_1 > lambdaG

            dag_hat_th = dag_hat_th.numpy()
            np.fill_diagonal(dag_hat_th,0)
            dag_hat_th = utils.dag.closest_dag(dag_hat_th)

            dag_gt = scm.adj_mat[intervention_order, :][:, intervention_order]
            shd =  utils.dag.structural_hamming_distance(dag_gt,dag_hat_th)
            tp, fp, fn, tn = utils.dag.cm_graph_entries(dag_gt,dag_hat_th)

            t_u_cosines = t.T @ u
            t_u_cosines_diag = t_u_cosines.diag()
            t_u_fro = torch.linalg.matrix_norm(t - u, "fro")
            t_fro = torch.linalg.matrix_norm(t, "fro")
            err_fro = t_u_fro / t_fro

            # let's get back to np arrays
            t = t.detach().cpu().numpy()
            u = u.detach().cpu().numpy()
            u_pinv = np.linalg.pinv(u)

            # more importantly, save MCC
            zs_obs = zs_obs.detach().cpu().numpy()
            zhats_obs = u_pinv @ t @ zs_obs

            mse = np.linalg.norm(np.squeeze(zhats_obs-zs_obs),2) / np.linalg.norm(np.squeeze(zs_obs),2)
            mcc = utils.mcc(np.squeeze(zs_obs),np.squeeze(zhats_obs))

            logging.info(f"pa_plus=\n{scm.adj_mat | np.eye(n, dtype=bool)}")
            logging.info(f"dhat=\n{dhat}")
            logging.info(f"t_u_cosines_diag=\n{t_u_cosines_diag}")
            logging.info(f"error fro={err_fro}")
            logging.info(f"MSE={mse}")
            logging.info(f"MCC={mcc}")
            logging.info(f"SHD={shd}")


            results_run["scm"] = scm,
            results_run["intervention_order"] = intervention_order.detach().cpu().numpy(),
            results_run["t"] =  t,
            results_run["u"] =  u,
            results_run["dhat"] = dhat.detach().cpu().numpy(),
            results_run["dhat_obs_hard_1"] = dhat_obs_hard_1.detach().cpu().numpy(),
            results_run["dhat_obs_hard_2"] = dhat_obs_hard_2.detach().cpu().numpy(),
            results_run["err_fro"] = err_fro.detach().cpu().numpy(),
            results_run["t_u_cosines"] = t_u_cosines.detach().cpu().numpy(),
            results_run["t_u_cosines_diag"] = t_u_cosines_diag.detach().cpu().numpy(),
            results_run["MSE"] = mse 
            results_run["MCC"] = mcc
            results_run["dag_gt"] = dag_gt
            results_run["dag_hat_th"] = dag_hat_th
            results_run["edge_precision"] = tp / (tp + fp + 1e-6)
            results_run["edge_recall"] = tp / (tp + fn + 1e-6)
            results_run["shd"] = shd

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

logging.info("ALL DONE")
