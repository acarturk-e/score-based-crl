import argparse
import logging
from typing import Callable

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class SlicedScoreMatching(torch.nn.Sequential):
    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        super().__init__(
            torch.nn.Linear(args.n, args.p),
            torch.nn.Tanh(),
            torch.nn.Linear(args.p, args.p),
            torch.nn.Tanh(),
            torch.nn.Linear(args.p, args.n),
        )
        self.args = args
        if args.optimizer == "SGD":
            self.opt = torch.optim.SGD(
                self.parameters(), lr=args.lr, momentum=args.momentum
            )
        elif self.args.optimizer == "RMSprop":
            self.opt = torch.optim.RMSprop(self.parameters(), lr=args.lr)
        elif self.args.optimizer == "Adam":
            self.opt = torch.optim.Adam(self.parameters(), lr=args.lr, betas=args.betas)
        else:
            raise ValueError(f"Unrecognized optimizer: {self.args.optimizer}")

    def loss_func(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x.requires_grad_(True)
        s: torch.Tensor = self.forward(x)
        loss = 0.5 * s.pow(2).sum(dim=-1).mean()
        if self.args.compute_hessian_trace_sliced:
            new_loss = torch.empty(self.args.m).to(device=self.args.device)
            for mi in range(self.args.m):
                v = torch.randn_like(x)
                v_grad_s_v = (
                    v * torch.autograd.grad((s * v).sum(), x, create_graph=True)[0]
                ).sum(dim=-1)
                new_loss[mi] = v_grad_s_v.mean()
            loss += new_loss.mean()
        else:
            s_jac_diag: torch.Tensor = torch.autograd.functional.jacobian(
                lambda x: self.forward(x).sum(dim=0),
                x,
                create_graph=True,
                vectorize=True,
            ).diagonal(  # type: ignore
                offset=0, dim1=-2, dim2=-1
            )
            loss += s_jac_diag.sum(dim=-1).mean()
            if self.args.regularize_hessian_diag:
                loss += self.args.lambda2 * s_jac_diag.pow(2).sum(dim=-1).mean()
        return loss

    def loss_batch(
        self,
        xb: torch.Tensor,
        optimize: bool = True,
    ) -> tuple[float, int]:
        loss = self.loss_func(xb)
        if self.args.regularize_model_parameters:
            loss += self.args.lambda1 * sum(
                map(lambda param: param.pow(2).sum(), self.parameters())
            )
        if optimize:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return loss.item(), len(xb)

    def fit(
        self,
        epochs: int,
        train_dl: DataLoader,
        valid_dl: DataLoader,
    ) -> None:
        eval_step = epochs // 100 if epochs >= 100 else 1
        for epoch in range(epochs):
            for xb in train_dl:
                xb = xb[0].to(
                    device=self.args.device, non_blocking=self.args.dataloader_multith
                )
                self.train()
                self.loss_batch(xb)
            if (epoch + 1) % eval_step == 0 or (epoch + 1) == epochs:
                self.eval()
                val_losses, val_nums = zip(
                    *[
                        self.loss_batch(
                            xb[0].to(
                                device=self.args.device,
                                non_blocking=self.args.dataloader_multith,
                            ),
                            optimize=False,
                        )
                        for xb in valid_dl
                    ]
                )
                val_loss = np.sum(np.multiply(val_losses, val_nums)) / np.sum(val_nums)
                if self.args.print_train_losses:
                    train_losses, train_nums = zip(
                        *[
                            self.loss_batch(
                                xb[0].to(
                                    device=self.args.device,
                                    non_blocking=self.args.dataloader_multith,
                                ),
                                optimize=False,
                            )
                            for xb in valid_dl
                        ]
                    )
                    train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(
                        train_nums
                    )
                logging.info(
                    f"Epoch:{epoch + 1:6}/{epochs:6}, val_loss={val_loss:+.10f}"
                    + (
                        f", train_loss={train_loss:+.10f}"  # type: ignore
                        if self.args.print_train_losses
                        else ""
                    )
                )


def score_fn_from_data(
    xi: torch.Tensor,
    epochs: int = 15,
    add_noise: bool = False,
    noise_var: float = 0.25,
) -> Callable[[torch.Tensor], torch.Tensor]:
    args = argparse.Namespace()

    args.disable_cuda = False
    args.cuda_enabled = not args.disable_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda_enabled else "cpu")

    args.dataloader_multith = False
    args.dataloader_worker_cnt = 8 if args.dataloader_multith else 0

    args.print_train_losses = False

    args.regularize_model_parameters = True
    args.lambda1 = 1e-5

    args.compute_hessian_trace_sliced = True
    args.m = 2

    args.regularize_hessian_diag = True  # lambda2, from Kingma and LeCun, 2010
    args.lambda2 = 1e-3

    args.n = xi.shape[1]
    args.p = 64

    n_valid = xi.shape[0] // 10

    args.optimizer = "Adam"
    args.lr = 1e-3
    args.betas = (0.5, 0.999)
    args.number_of_epochs = epochs
    args.batch_size = 512

    scoreModel = SlicedScoreMatching(args).to(device=args.device)

    xi_train = xi[n_valid:, :, :].detach().clone()
    xi_valid = xi[:n_valid, :, :].detach().clone()

    if add_noise:
        xi_train += torch.randn_like(xi_train) * noise_var

    train_dl = DataLoader(
        TensorDataset(xi_train.squeeze(dim=-1)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_dl = DataLoader(
        TensorDataset(xi_valid.squeeze(dim=-1)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    scoreModel.fit(args.number_of_epochs, train_dl, valid_dl)
    return scoreModel.cpu()
