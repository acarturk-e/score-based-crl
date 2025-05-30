from argparse import Namespace
from copy import deepcopy
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Setting these flags True makes A100 training a lot faster:
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from model_score_diff import DenseLdr, ScoreDiffCls, GaussianLdr


def train_score_diff(
    dataloader: DataLoader,
    opts: Namespace,
    savedir: str,
    runname: str,
) -> ScoreDiffCls:
    """Train the score difference model.

    Args:
        dataloader: DataLoader for the data.
        opts: Namespace containing the hyperparameters.
        savedir: Directory to save the model.
        runname: Name of the run (use to differentiate different pairs).
        log: Whether to log the training to wandb.

    Returns:
        The trained score difference model.

    Notes on data:
        The dataloader should return a tuple of (x, y) where x is the input data and y is the target labels. Both x and y should be torch.Tensor with floating point data types.

    Required fields for opts:
        load: Whether to load a pre-trained model.
        load_path: Path to the pre-trained model (conditional).
        nu: Ratio of class priors ($Pr(0) / Pr(1)$)
        epochs: Number of epochs to train.
        lr: Learning rate.
        disable_cuda: Whether to disable CUDA.
        use_gaussian_ldr: Whether to use parametric Gaussian LDR model. If False, uses generic, nonparametric DenseLdr.
        dense_ldr_width: Width of the dense LDR model (conditional).
        dense_ldr_depth: Depth of the dense LDR model (conditional).
    """
    on_cuda = torch.cuda.is_available() and not opts.disable_cuda
    device = torch.device("cuda" if on_cuda else "cpu")

    # Data dimension
    d = dataloader.dataset[0][0].shape[-1]
    num_epochs: int = opts.epochs

    # Model
    ldr_model: nn.Module
    if opts.use_gaussian_ldr:
        ldr_model = GaussianLdr(d)
    else:
        ldr_model = DenseLdr(d, opts.dense_ldr_width, opts.dense_ldr_depth)
    model = ScoreDiffCls(ldr_model, opts.nu)
    model.to(device)

    if opts.load:
        model.load_state_dict(torch.load(opts.load_path))

    criterion = nn.BCELoss()
    opt = optim.Adam(params=model.parameters(), lr=opts.lr)  # type: ignore
    lrsch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, (num_epochs // 5)+1, 4)

    model.train()
    print("Training for {} epochs...".format(str(num_epochs)))

    min_train_loss = np.inf
    best_model = deepcopy(model)
    for n in range(0, num_epochs):
        ct = 0
        lossAv = 0.0
        skipped = False
        for i, batch in enumerate(dataloader):
            x = batch[0].to(device)
            y = batch[1].to(device)

            opt.zero_grad()
            yhat = model(x)
            if not math.isfinite(yhat.sum()):
                skipped = True
                break

            loss = criterion(yhat, y)
            loss.backward()
            opt.step()

            ct += 1
            lossAv += loss.detach().cpu().item()

        if skipped:
            print(f"Finiteness issue in epoch {n}. Exiting training.")
            break

        print("Epoch " + str(n) + ": Loss=" + str(lossAv / ct))

        if (lossAv) / ct < min_train_loss:
            min_train_loss = (lossAv) / ct
            best_model = deepcopy(model)
            torch.save(best_model, os.path.join(savedir, runname + "best_model.pt"))

        lrsch.step()

    last_model = deepcopy(model)
    torch.save(last_model, os.path.join(savedir, runname + "last_model.pt"))

    return model


def score_diff_gaussian(
    dataloader: DataLoader,
    opts: Namespace,
    savedir: str,
    runname: str,
) -> ScoreDiffCls:
    """Learn Gaussian parameters for the score difference model.

    Args:
        dataloader: DataLoader for the data.
        opts: Namespace containing the hyperparameters.
        savedir: Directory to save the model.
        runname: Name of the run (use to differentiate different pairs).
        log: Whether to log the training to wandb.

    Returns:
        The trained score difference model with a Gaussian LDR.

    Notes on data:
        The dataloader should return a tuple of (x, y) where x is the input data and y is the target labels. Both x and y should be torch.Tensor with floating point data types.

    Required fields for opts:
        disable_cuda: Whether to disable CUDA."""
    on_cuda = torch.cuda.is_available() and not opts.disable_cuda
    device = torch.device("cuda" if on_cuda else "cpu")

    # Data dimension
    d = dataloader.dataset[0][0].shape[-1]

    # Initialize accumulators
    sum_x0 = torch.zeros(d, device=device)
    sum_x1 = torch.zeros(d, device=device)
    sum_xx0 = torch.zeros(d, d, device=device)
    sum_xx1 = torch.zeros(d, d, device=device)
    count0 = 0
    count1 = 0

    # Accumulate sums
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        mask0 = (y == 0).float().unsqueeze(-1)
        mask1 = (y == 1).float().unsqueeze(-1)
        x0 = x * mask0
        x1 = x * mask1
        sum_x0 += x0.sum(dim=0)
        sum_x1 += x1.sum(dim=0)
        sum_xx0 += torch.einsum("ki,kj->ij", x0, x0)
        sum_xx1 += torch.einsum("ki,kj->ij", x1, x1)
        count0 += mask0.sum().item()
        count1 += mask1.sum().item()

    # Compute means
    mean0 = sum_x0 / count0
    mean1 = sum_x1 / count1

    # Compute covariances
    cov0 = sum_xx0 / count0 - mean0.unsqueeze(-1) * mean0.unsqueeze(-2)
    cov1 = sum_xx1 / count1 - mean1.unsqueeze(-1) * mean1.unsqueeze(-2)

    prec0 = torch.inverse(cov0)
    prec1 = torch.inverse(cov1)

    ldr_model = GaussianLdr.from_gaussian_params(mean0, mean1, prec0, prec1)
    model = ScoreDiffCls(ldr_model, count0 / count1)
    model.to(device)

    # Save the model
    torch.save(model, os.path.join(savedir, runname + "model.pt"))

    return model
