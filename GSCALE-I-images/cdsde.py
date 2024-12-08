from collections import OrderedDict
from typing import Callable, Self

import torch
import torch.nn as nn
from torch import Tensor


def class_density_from_ldr(ldrs: Tensor) -> Tensor:
    r"""Class density computation from log density ratio

    Computes $Pr(C = 1 | X) \in [0, 1]$ based on the given log
    density ratio $\log (Pr(X | C = 1) / Pr(X | C = 0))$ samples.

    ## Usage:
    - Construct a log density ratio function.
    - Minimize binary cross entropy between output of `cd` and class labels.
    - `grad_log_dr` computes the gradient of the estimated log density
        ratio, which corresponds to the (Stein) score difference.

    See:
    Gutmann and Hyvärinen,
    Noise-Contrastive Estimation of Unnormalized Statistical Models with
    Applications to Natural Image Statistics,
    JMLR, 2012.
    """
    return (-ldrs).sigmoid()


def score_diff_from_ldr(log_dr: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Stein score difference, i.e., gradient of log density ratio"""
    return torch.autograd.functional.jacobian(  # type: ignore
        func=lambda y: log_dr(y).sum(),
        inputs=x,
        create_graph=False,
        strict=False,
        vectorize=True)


class LdrCnn(nn.Sequential):
    """Models the log density ratio b/w 64x64 RGB image datasets
    using a CNN with ReLU activation.

    Input: B x 3 x 64 x 64
    Output: B"""
    def __init__(self):
        super(LdrCnn, self).__init__(
            # 3 x 64^2
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(False),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2, 2),
            # 32 x 32^2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(False),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2, 2),
            # 64 x 16^2
            nn.Flatten(1, -1),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(False),
            # 128
            nn.Linear(128, 1),
            nn.Flatten(0, -1))


class LdrNn(nn.Sequential):
    r"""ReLU network for modelling log density ratio of
    `n` dimensional vector inputs.
    
    NOTE: This module auto-flattens the input, i.e.,
    only the first axis is treated as a batch dim."""

    def __init__(self, n: int, n_layers: int = 3, width: int = 512):
        assert n >= 1 and n_layers >= 2
        self.n = n
        self.n_layers = n_layers
        self.width = width
        od = OrderedDict[str, nn.Module]()
        od["flatten_in"] = nn.Flatten()
        od["linear1"] = nn.Linear(n, width)
        od["sigmoid1"] = nn.Sigmoid()
        od["batchnorm1"] = nn.BatchNorm1d(width)
        for i in range(2, n_layers):
            od[f"linear{i}"] = nn.Linear(width, width)
            od[f"sigmoid{i}"] = nn.Sigmoid()
            od[f"batchnorm{i}"] = nn.BatchNorm1d(width)
        od[f"linear{n_layers}"] = nn.Linear(width, 1)
        od[f"squeeze_out"] = nn.Flatten(-2, -1)
        super().__init__(od)


class LdrGaussian(nn.Module):
    r"""Parametric log density ratio model for Gaussians

    This class provides boilerplate code for constructing a `autograd` and
    batch processing amenable object that represents a 2nd degree polynomial.

    Log density ratio between two `n` dimensional multivariate Gaussian
    distributions is a 2nd degree polynomial in `n` variates.
    See `from_gauss` factory method for direct coefficient computation
    from known distribution parameters."""

    def __init__(self, n: int):
        super().__init__()
        assert n >= 1
        self.n = n
        self.c1 = nn.Linear(n, 1, bias=True)
        self.c2 = nn.Linear(n, n, bias=False)

    @classmethod
    def from_gaussian_params(cls, m0: Tensor, m1: Tensor, q0: Tensor, q1: Tensor) -> Self:
        r"""Constructor for log density ratio between two multivariate
        Gaussians with means `m0` and `m1` and precision matrices `q0` and `q1`."""
        assert (
            m0.ndim == 1
            and q0.ndim == 2
            and m0.shape == m1.shape
            and q0.shape == q1.shape
        )
        n = m0.shape[0]
        assert n >= 1 and q0.shape == (n, n)
        res = cls(n)
        res.c1.bias = nn.Parameter(
            0.5 * (q0.logdet() - q1.logdet() - m0 @ q0 @ m0 + m1 @ q1 @ m1))
        res.c1.weight = nn.Parameter((m0 @ q0 - m1 @ q1).unsqueeze(0))
        res.c2.weight = nn.Parameter(0.5 * (-q0 + q1))
        return res

    def forward(self, x: Tensor) -> Tensor:
        return self.c1(x)[..., 0] + (x * self.c2(x)).sum(-1)
