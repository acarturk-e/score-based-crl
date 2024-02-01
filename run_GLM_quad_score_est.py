r'''
Two hard intervention per node
Generalized Linear transformation (with tanh link function)
Quadratic latent models
Use score estimator (SSM) from Neurips SCALE-I
'''

import logging
import os
import pickle

import numpy as np
import torch
from torch import Tensor
from torch.func import jacrev, vmap  # type: ignore

from scm.quadratic import QuadraticSCM
from score_estimators.ssm import score_fn_from_data

nruns = 10
nd_list = [
    (5, 5),
]
nsamples = 300
nsamples_for_score_est = 30_000
fill_rate = 0.5
lambda1 = 1
lambda2 = 1.0e-4
nsteps_max = 20_000
print_status_step = 500
learning_rate = 1e-3
lr_update_step = 1000
lr_update_gamma = 0.95
float_precision = 1e-6
l0_threshold = 0.02
score_est_steps = 3

# computes pseudo-inverse
def pinv(x: Tensor) -> Tensor:
    return torch.linalg.pinv(x)

# Speeds things up by a lot compared to working in CPU
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
np_rng = np.random.default_rng()

# Result dir setup
run_name = "glm_quad_w_se"
run_dir = os.path.join("results", run_name)
if not os.path.exists(run_dir):
    os.makedirs(run_dir)

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


logging.info("Starting")

for nd_idx, (n, d) in enumerate(nd_list):
    for run_idx in range(nruns):
        logging.info(f"n={n}, d={d}, run_idx={run_idx}")
        logging.info(f"stop learning when l0 norm is minimized under threshold {l0_threshold}")
        logging.info("Setup:Quadratic latent model generation, Z sampling, computing X, computing ground truth score differences")

        # Intervention order, set to 1..n now
        p_i = torch.arange(n)

        # ground truth decoder : Generalized linear transform
        # ground truth decoder parameters. make all columns have norm=1, and first entry be non-zero
        t = torch.rand((d, n)) - 0.5
        t = t / t.norm(dim=0, keepdim=True)
        t = t * torch.sign(t[0])
        # The transform must be well-conditioned wrt matrix L2 norm for low reconstruction loss
        tsvs = torch.linalg.svdvals(t)
        while tsvs[-1] / tsvs[0] < 1e-1:
            t = torch.rand((d, n)) - 0.5
            t = t / t.norm(dim=0, keepdim=True)
            t = t * torch.sign(t[0])
            tsvs = torch.linalg.svdvals(t)
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
        scm = QuadraticSCM(n, fill_rate, True, np_rng=np_rng)

        # environments - intervention targets
        env_obs = []
        envs_hard_1 = [[rho_i] for rho_i in p_i]
        envs_hard_2 = [[rho_i] for rho_i in p_i]

        # create z samples
        zs_obs    =              torch.Tensor(scm.sample((nsamples,), nodes_int=[])).to(device=dev)
        zs_hard_1 = torch.stack([torch.Tensor(scm.sample((nsamples,), nodes_int=env, var_change_mech="increase", var_change=1.0)).to(device=dev) for env in envs_hard_1])
        zs_hard_2 = torch.stack([torch.Tensor(scm.sample((nsamples,), nodes_int=env, var_change_mech="increase", var_change=2.0)).to(device=dev) for env in envs_hard_2])

        # create x samples from z using t
        xs_obs = decoder(zs_obs, t)
        xs_hard_1 = decoder(zs_hard_1, t)
        xs_hard_2 = decoder(zs_hard_2, t)

        ### Compute GT score functions of x

        # GT score functions of z
        gt_sz_fn_obs     =  lambda z:      (torch.from_numpy(scm.score_fn((z).detach().cpu().numpy(), nodes_int=[])).to(device=z.device, dtype=z.dtype))
        gt_sz_fns_hard_1 = [lambda z, i=i: (torch.from_numpy(scm.score_fn((z).detach().cpu().numpy(), nodes_int=envs_hard_1[i], var_change_mech="increase", var_change=1.0)).to(device=z.device, dtype=z.dtype)) for i in range(n)]
        gt_sz_fns_hard_2 = [lambda z, i=i: (torch.from_numpy(scm.score_fn((z).detach().cpu().numpy(), nodes_int=envs_hard_2[i], var_change_mech="increase", var_change=2.0)).to(device=z.device, dtype=z.dtype)) for i in range(n)]

        # scores of z evaluated at observational data
        gt_szs_obs = gt_sz_fn_obs(zs_obs).detach()
        gt_szs_hard_1 = torch.stack([sz_fn(zs_obs).detach() for sz_fn in gt_sz_fns_hard_1])
        gt_szs_hard_2 = torch.stack([sz_fn(zs_obs).detach() for sz_fn in gt_sz_fns_hard_2])
        # score differences between hard_1 and hard_2 environments at latent points X.
        gt_dszs_bw_hards = gt_szs_hard_1 - gt_szs_hard_2
        # score differences between hard_1 and observational environments at data points X.
        gt_dszs_obs_hard_1 = gt_szs_hard_1 - gt_szs_obs
        # score differences between hard_2 and observational environments at data points X.
        gt_dszs_obs_hard_2 = gt_szs_hard_2 - gt_szs_obs

        # and reduced to matrix form for future usage
        gt_dsz_bw_hards = (torch.linalg.vector_norm(gt_dszs_bw_hards, ord=1, dim=1) / nsamples).T
        gt_dsz_obs_hard_1 = (torch.linalg.vector_norm(gt_dszs_obs_hard_1, ord=1, dim=1) / nsamples).T
        gt_dsz_obs_hard_2 = (torch.linalg.vector_norm(gt_dszs_obs_hard_2, ord=1, dim=1) / nsamples).T

        # evaluate jacobians of true decoder at obs data in order to compute true score differences for X
        # jacobian of g:R^n \to R^d function at point z \in R^n is d by n.
        true_jacobians_at_samples = vmap(jacrev(true_decoder))(zs_obs).squeeze(-1,-3)

        J = pinv(true_jacobians_at_samples).mT # has shape nsamples * d * n

        # score differences between hard_1 and hard_2 environments at data points X.
        gt_dsxs_bw_hards = J @ gt_dszs_bw_hards
        # score differences between hard_1 and observational environments at data points X.
        gt_dsxs_obs_hard_1 = J @ gt_dszs_obs_hard_1
        # score differences between hard_2 and observational environments at data points X.
        gt_dsxs_obs_hard_2 = J @ gt_dszs_obs_hard_2

        ### Compute score estimators of X using more data

        # create z samples for score estimation
        zs_obs_se    =              torch.Tensor(scm.sample((nsamples_for_score_est,), nodes_int=[])).to(device=dev)
        zs_hard_1_se = torch.stack([torch.Tensor(scm.sample((nsamples_for_score_est,), nodes_int=env, var_change_mech="increase", var_change=1.0)).to(device=dev) for env in envs_hard_1])
        zs_hard_2_se = torch.stack([torch.Tensor(scm.sample((nsamples_for_score_est,), nodes_int=env, var_change_mech="increase", var_change=2.0)).to(device=dev) for env in envs_hard_2])

        # create x samples from z using t for score estimation
        xs_obs_se = decoder(zs_obs_se, t)
        xs_hard_1_se = decoder(zs_hard_1_se, t)
        xs_hard_2_se = decoder(zs_hard_2_se, t)

        ### Compute GT score functions of x

        # estimated score functions of x
        sx_fn_obs = score_fn_from_data(xs_obs_se, epochs=score_est_steps).to(device=dev)
        sx_fns_hard_1 = [score_fn_from_data(xs_hard_1_env, epochs=score_est_steps).to(device=dev) for xs_hard_1_env in xs_hard_1_se]
        sx_fns_hard_2 = [score_fn_from_data(xs_hard_2_env, epochs=score_est_steps).to(device=dev) for xs_hard_2_env in xs_hard_2_se]

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

        # NOW READY TO START ALGORITHM
        logging.info("Starting algo")

        # Randomly start from a U
        u = torch.rand((d, n), device=xs_obs.device) - 0.5
        u.detach_()
        u.requires_grad_(True)

        dhat = torch.empty((n, n))
        loss = torch.empty((1,))
        loss_x_reconstruction = torch.empty((1,))
        loss_l1 = torch.empty((1,))
        loss_l2_zhat = torch.empty((1,))

        optimizer = torch.optim.RMSprop([u], lr=learning_rate)

        for step in range(nsteps_max):
            u_pinv = pinv(u)

            zhats_obs = encoder(xs_obs, u_pinv)
            zhats_hard_1 = encoder(xs_hard_1, u_pinv)
            zhats_hard_2 = encoder(xs_hard_2, u_pinv)
            xhats_hard_1 = decoder(zhats_hard_1, u)
            xhats_hard_2 = decoder(zhats_hard_2, u)

            def candidate_decoder(zhat: Tensor) -> Tensor:
                return decoder(zhat, u)

            # evaluate at obs data
            jacobians_at_samples = vmap(jacrev(candidate_decoder))(zhats_obs).squeeze(-1,-3)

            # main loss, use l1 instead of l0
            dszhats = jacobians_at_samples.mT @ dsxs_bw_hards
            dhat = (torch.linalg.vector_norm(dszhats, ord=1, dim=1)[:, :, 0] / nsamples).T

            loss_l1 = lambda1 * dhat.mean()

            # ensure reconstruction of x (for invertible transform condition)
            loss_x_reconstruction = (torch.norm(torch.flatten(xhats_hard_1 - xs_hard_1)) ** 2 +
                                    torch.norm(torch.flatten(xhats_hard_2 - xs_hard_2)) ** 2) / nsamples

            # to restrain the magnitude
            loss_l2_zhat = (lambda2 * (torch.norm(torch.flatten(zhats_hard_1)) ** 2 +
                                    torch.norm(torch.flatten(zhats_hard_2)) ** 2) / nsamples)

            # final loss
            loss = loss_x_reconstruction + loss_l1 + loss_l2_zhat
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % print_status_step == 0:
                thresholded_l0_norm = torch.sum(dhat > l0_threshold)
                logging.info(f"step #{step}, loss={loss.item()}, th-l0={thresholded_l0_norm}")
                if thresholded_l0_norm <= n:
                    logging.info(f"thresholded l0 norm is minimized, learning is stopped at #{step} steps")
                    break

            if step % lr_update_step == 0:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*lr_update_gamma


        u = u / u.norm(dim=0, keepdim=True)
        u = u * torch.sign(u[0])
        logging.info(
            f"loss={loss.item()}, "
            + f"loss_x_reconstruction={loss_x_reconstruction.item()}, "
            + f"loss_l1={loss_l1.item()}, "
            + f"loss_l2_zhat={loss_l2_zhat.item()}")


        # infer DAG using obs vs. hard_1 OR obs vs. hard_2 score difference matrix
        u_pinv = torch.linalg.pinv(u)
        zhats_obs = encoder(xs_obs, u_pinv)
        zhats_hard_1 = encoder(xs_hard_1, u_pinv)
        zhats_hard_2 = encoder(xs_hard_2, u_pinv)

        jacobians_at_samples = vmap(jacrev(candidate_decoder))(zhats_obs).squeeze(-1,-3)

        dszhats = jacobians_at_samples.mT @ dsxs_bw_hards
        dszhats_obs_hard_1 = jacobians_at_samples.mT @ dsxs_obs_hard_1
        dszhats_obs_hard_2 = jacobians_at_samples.mT @ dsxs_obs_hard_2

        dhat = (torch.linalg.vector_norm(dszhats, ord=1, dim=1)[:, :, 0] / nsamples).T
        dhat_obs_hard_1 = (torch.linalg.vector_norm(dszhats_obs_hard_1, ord=1, dim=1)[:, :, 0] / nsamples).T
        dhat_obs_hard_2 = (torch.linalg.vector_norm(dszhats_obs_hard_2, ord=1, dim=1)[:, :, 0] / nsamples).T

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

        u = u[:, perm_to_order]
        dhat = dhat[perm_to_order, :]
        dhat_obs_hard_1 = dhat_obs_hard_1[perm_to_order, :]
        dhat_obs_hard_2 = dhat_obs_hard_2[perm_to_order, :]

        t_u_cosines = t.T @ u
        t_u_cosines_diag = t_u_cosines.diag()
        t_u_fro = torch.linalg.matrix_norm(t - u, "fro")
        t_fro = torch.linalg.matrix_norm(t, "fro")
        err_fro = t_u_fro / t_fro

        logging.info(f"pa_plus=\n{scm.adj_mat | np.eye(n, dtype=bool)}")
        logging.info(f"dhat=\n{dhat}")
        logging.info(f"t_u_cosines_diag=\n{t_u_cosines_diag}")
        logging.info(f"error fro={err_fro}")

        with open(
            os.path.join(run_dir, f"{n}_{d}_{run_idx}.pkl"), "wb"
        ) as f:
            pickle.dump(
                {
                    "scm": scm,
                    "p_i": p_i.detach().cpu().numpy(),
                    "t": t.detach().cpu().numpy(),
                    "u": u.detach().cpu().numpy(),
                    "err_fro": err_fro.detach().cpu().numpy(),
                    "t_u_cosines": t_u_cosines.detach().cpu().numpy(),
                    "t_u_cosines_diag": t_u_cosines_diag.detach().cpu().numpy(),
                    "dhat": dhat.detach().cpu().numpy(),
                    "dhat_obs_hard_1": dhat_obs_hard_1.detach().cpu().numpy(),
                    "dhat_obs_hard_2": dhat_obs_hard_2.detach().cpu().numpy(),
                    "nsamples": nsamples,
                    "nsteps_max": nsteps_max,
                    "l0_threshold": l0_threshold,
                },
                f,
            )

        logging.info("Algo finished")

logging.info("ALL DONE")
