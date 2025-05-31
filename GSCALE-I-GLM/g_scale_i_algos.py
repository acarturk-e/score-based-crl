"""General Score-based Causal Latent Estimation via Interventions (GSCALE-I)

Setting: X = g(Z), i.e. general transform. 
single-node interventions.

Paper: Score-based Causal Representation Learning: Linear and General Transformations
(https://arxiv.org/abs/2402.00849), published at JMLR 05/2025.

In Section 7.2.1: we consider single-layer MLPs with tanh activation function, that is
    X = tanh(G.Z), where G is a d x n matrix, Z is a n-dimensional latent variable.

"""


import sys
import os
# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)


import numpy as np
import torch
from torch import Tensor
from torch.func import jacrev, vmap  # type: ignore
import logging
import utils 

# Speeds things up by a lot compared to working in CPU
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

relu = torch.nn.ReLU()

# computes pseudo-inverse
def pinv(x: Tensor) -> Tensor:
    return torch.linalg.pinv(x)


def g_scale_i_glm_tanh(d,n,nsamples,dsxs_bw_hards,xs_obs,learning_rate=0.001,nsteps_max=20000,lambda_recon=1.0,l0_threshold=0.02,zs_obs=None):
    '''
    this implementation is specialized to X = tanh(G.Z) case. Modify the following definitions accordingly for more general cases.

    Inputs:
    - d: int, dimension of observed variable X
    - n: int, dimension of latent variable Z
    - nsamples: int, number of samples
    - dsxs_bw_hards = score difference (of observed X) between hard interventions
    - xs_obs: observable data, observational environment.
    - learning_rate: float, learning rate for the optimizer
    - nsteps_max: int, maximum number of optimization steps
    - lambda_recon: float, coefficient for the reconstruction loss
    - l0_threshold: float, threshold for early stopping based on the l0 norm of the learned matrix, just to gauge the training progress
    '''
    
    print_status_step = 1000
    lr_update_step = 1000
    lr_update_gamma = 0.95
    float_precision = 1e-6
    
    def decoder(z: Tensor, u: Tensor) -> Tensor:
        return torch.clamp((u @ z).tanh(),min=-1+float_precision, max=1-float_precision)

    def encoder(x: Tensor, u_pinv: Tensor) -> Tensor:
        return u_pinv @ x.arctanh()

    # Randomly start from a U
    u = torch.rand((d, n), device=dev) - 0.5
    u.detach_()
    u.requires_grad_(True)
    optimizer = torch.optim.RMSprop([u], lr=learning_rate)

    # DEBUG
    zs_obs = zs_obs.detach().cpu().numpy()

    dhat = torch.empty((n, n))
    loss = torch.empty((1,))
    loss_x_reconstruction = torch.empty((1,))
    loss_main = torch.empty((1,))
    
    diag_mask = torch.eye(n)
    off_diag_mask = 1 - torch.eye(n)
    
    for step in range(nsteps_max):
        u_pinv = pinv(u)
        # we only need observational data now (and the score functions of interventional env., not the interv. data itself)
        zhats_obs = encoder(xs_obs, u_pinv)
        xhats_obs = decoder(zhats_obs, u)

        def candidate_decoder(zhat: Tensor) -> Tensor:
            return decoder(zhat, u)

        # evaluate at obs data
        jacobians_at_samples = vmap(jacrev(candidate_decoder))(zhats_obs).squeeze(-1,-3)

        dszhats = jacobians_at_samples.mT @ dsxs_bw_hards
        dhat = (torch.linalg.vector_norm(dszhats, ord=1, dim=1)[:, :, 0] / nsamples).T

        # ensure reconstruction of x (for invertible transform condition)
        loss_x_reconstruction = (torch.norm(torch.flatten(xhats_obs - xs_obs)) ** 2 ) / nsamples

        #loss_main = (dhat - 0.5*torch.eye(n)).abs().sum()

        loss_diag = relu((0.5*torch.ones(n) - dhat.diagonal())).mean()
        loss_off_diag = ((dhat * off_diag_mask)).sum() / (n * (n - 1))
        
        loss_main = 1.0 * loss_diag + 1.0 * loss_off_diag

        #loss_main = dhat.fill_diagonal_(0.0).abs().sum()
        #loss_main = (dhat * off_diag_mask).abs().sum()
        #loss_main = torch.linalg.norm(dhat * off_diag_mask,ord='fro')
        
        loss = loss_main + lambda_recon*loss_x_reconstruction

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


        if step % print_status_step == 0:
            thresholded_l0_norm = torch.sum(dhat > l0_threshold)
            #logging.info(f"step #{step}, loss={loss.item()}, loss_main={loss_main.item()}, loss_x_recon={loss_x_reconstruction.item()},  th-l0={thresholded_l0_norm}")
            logging.info(f"step #{step}, loss={loss.item()}, loss_main={loss_main.item()}, loss_diag={loss_diag.item()}, loss_off_diag={loss_off_diag.item()}, loss_x_recon={loss_x_reconstruction.item()},  th-l0={thresholded_l0_norm}")

            # DEBUG
            mcc = utils.mcc(np.squeeze(zs_obs),np.squeeze(zhats_obs.detach().cpu().numpy()))
            logging.info(f"step #{step}, mcc={mcc}")
            logging.info(f"dhat={dhat.detach().cpu().numpy()}")

            if loss_main < 1e-2:
                logging.info(f"learning is stopped at #{step} steps")
                break

        if step % lr_update_step == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*lr_update_gamma

    u = u / u.norm(dim=0, keepdim=True)
    u = u * torch.sign(u[0])
    logging.info(
        f"loss={loss.item()}, "
        + f"loss_x_recon={loss_x_reconstruction.item()}, "
        + f"loss_main={loss_main.item()}")
        
    return u


