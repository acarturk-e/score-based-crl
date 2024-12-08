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

# Speeds things up by a lot compared to working in CPU
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# computes pseudo-inverse
def pinv(x: Tensor) -> Tensor:
    return torch.linalg.pinv(x)


def g_scale_i_glm_tanh(d,n,nsamples,dsxs_bw_hards,xs_obs,xs_hard_1,xs_hard_2,learning_rate=0.001,nsteps_max=20000,lambda_recon=1.0,l0_threshold=0.02):
    '''
    this implementation is specialized to X = tanh(G.Z) case. Modify the following definitions accordingly for more general cases.
    '''
    print_status_step = 500
    lr_update_step = 1000
    lr_update_gamma = 1.0

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

    dhat = torch.empty((n, n))
    loss = torch.empty((1,))
    loss_x_reconstruction = torch.empty((1,))
    loss_main = torch.empty((1,))
    loss_l1 = torch.empty((1,))
    loss_l2_zhat = torch.empty((1,))


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

        dszhats = jacobians_at_samples.mT @ dsxs_bw_hards
        dhat = (torch.linalg.vector_norm(dszhats, ord=1, dim=1)[:, :, 0] / nsamples).T
        # main loss: make dhat equal to Id_n
        loss_main = (dhat - torch.eye(n)).abs().sum()

        # ensure reconstruction of x (for invertible transform condition)
        loss_x_reconstruction = (torch.norm(torch.flatten(xhats_hard_1 - xs_hard_1)) ** 2 +
                                torch.norm(torch.flatten(xhats_hard_2 - xs_hard_2)) ** 2) / nsamples
        
        # # to restrain the magnitude (optional regularizer)
        # loss_l2_zhat = (lambda2 * (torch.norm(torch.flatten(zhats_hard_1)) ** 2 +
        #                         torch.norm(torch.flatten(zhats_hard_2)) ** 2) / nsamples)

        # loss_l1 = lambda1 * dhat.mean()  # alternative formulation, l1 relaxation

        # total loss: exact version
        loss = loss_main + lambda_recon*loss_x_reconstruction
        
        # loss = loss_x_reconstruction + loss_l1 + loss_l2_zhat # alternative versionn with l1 relaxation.

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % print_status_step == 0:
            thresholded_l0_norm = torch.sum(dhat > l0_threshold)
            logging.info(f"step #{step}, loss={loss.item()}, loss_main={loss_main.item()}, loss_x_recon={loss_x_reconstruction.item()},  th-l0={thresholded_l0_norm}")
            if thresholded_l0_norm <= n:
                logging.info(f"thresholded l0 norm is minimized, learning is stopped at #{step} steps")
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