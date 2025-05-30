"""Classification based score difference estimation models"""
from typing import Self

import torch
import torch.nn as nn
from torch import Tensor


class ScoreDiffCls(nn.Module):
    r"""Classification based score difference model

    This class estimates the score difference using a binary classifier.

    ### How to:

    Forward call returns class probabilities for the given samples. The output should be used to train a binary classifier via **binary cross entropy** minimization.

    Internally, the class computes class probabilities from a provided log density ratio model, which can be used to compute score differences.

    Approach:
    1. Construct a log density ratio model (should be a `torch.nn.Module`).
    2. Initialize an object of this class using the log density ratio model.
    3. Minimize binary cross entropy between class probability and class labels.
    4. Gradient of log density ratio of the trained model corresponds to the (Stein) score differences (access it via `score_diff` method).

    Forward call returns class probability based on a given log density ratio model.

    The score difference is computed using the provided log density ratio model and the gradient of the log density ratio corresponds to the score difference, which is used to estimate the classifier.

    For more details, see the paper by Gutmann and Hyvärinen, 2012."""

    def __init__(self, ldr_model: nn.Module, nu: float = 1.0):
        r"""Constructor for score difference model

        Args:
            ldr_model: Log density ratio model, i.e., $\log (Pr(X | C = 1) / Pr(X | C = 0))$
            nu: Ratio of class priors ($Pr(0) / Pr(1)$) (default: 1.0)"""
        super().__init__()
        self.ldr_model = ldr_model
        self.nu = nu

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward call to compute class probability for class 1.

        Computes $Pr(C = 1 | X) \in [0, 1]$ based on the model's log density ratio $\log (Pr(X | C = 1) / Pr(X | C = 0))$.
        """
        ans = 1.0 / (1.0 + self.nu * (-self.ldr_model(x)).exp())
        return ans

    def score_diff(self, x: Tensor) -> Tensor:
        """Stein score difference, i.e., gradient of log density ratio"""
        return torch.autograd.functional.jacobian(  # type: ignore
            func=lambda y: self.ldr_model(y).sum(),
            inputs=x,
            create_graph=False,
            strict=False,
            vectorize=True,
        )


class GaussianLdr(nn.Module):
    r"""Parametric log density ratio model for Gaussians

    This class models a 2nd degree polynomial. Log density ratio between two `d` dimensional multivariate Gaussian distributions is a 2nd degree polynomial in `d` variates. See `from_gauss` factory method for direct coefficient computation from known distribution parameters.
    """

    def __init__(self, d: int):
        super().__init__()
        assert d >= 1
        self.d = d
        self.c1 = nn.Linear(d, 1, bias=True)
        self.c2 = nn.Linear(d, d, bias=False)
        # 0 initialization
        self.c1.bias.data.fill_(0.0)
        self.c1.weight.data.fill_(0.0)
        self.c2.weight.data.fill_(0.0)

    @classmethod
    def from_gaussian_params(
        cls, m0: Tensor, m1: Tensor, q0: Tensor, q1: Tensor
    ) -> Self:
        r"""Constructor for log density ratio log p1/p0 between two multivariate Gaussians with means `m0` and `m1` and precision matrices `q0` and `q1`."""
        assert (
            m0.ndim == 1
            and q0.ndim == 2
            and m0.shape == m1.shape
            and q0.shape == q1.shape
        )
        d = m0.shape[0]
        assert d >= 1 and q0.shape == (d, d)
        res = cls(d)
        res.c1.bias = nn.Parameter(
            0.5 * (q1.logdet() - q0.logdet() - m1 @ q1 @ m1 + m0 @ q0 @ m0)
        )
        res.c1.weight = nn.Parameter((m1 @ q1 - m0 @ q0).unsqueeze(0))
        res.c2.weight = nn.Parameter(0.5 * (-q1 + q0))
        return res

    def forward(self, x: Tensor) -> Tensor:
        return self.c1(x)[..., 0] + (x * self.c2(x)).sum(-1)


class DenseLdr(nn.Sequential):
    """Log density ratio modelled by a shallow, dense neural network"""

    def __init__(
        self,
        d: int,
        width: int = 32,
        depth: int = 2,
    ):
        """Constructor for dense log density ratio model

        Args:
            d: Dimension of the input
            width: Number of hidden units (default: 32)
            depth: Number of layers (default: 2)

        The model is a dense neural network with `depth` layers and `width` hidden units in each layer.

        Last layer is a linear layer without activation with scalar output."""

        self.d = d
        self.width = width
        self.depth = depth
        layers = [nn.Linear(d, width), nn.ReLU()]
        for _ in range(1, depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 1))
        layers.append(nn.Flatten(-2, -1))
        super().__init__(*layers)
