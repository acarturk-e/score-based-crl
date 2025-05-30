"""Utility functions for GSCALE-I"""
import numpy as np
import numpy.typing as npt
import scipy
from scipy.optimize import linear_sum_assignment  # type: ignore
import torch
import torch.nn as nn
from torch import Tensor


def mcc(x_est: npt.NDArray[np.floating], x_gt: npt.NDArray[np.floating]) -> float:
    """Computes mean correlation coefficient between `x_est` and `x_gt`

    Data dimension: `(n_samples, n)`. Computes the correlation coefficients
    between entries of `x_est` and `x_gt`, and solves the maximum linear sum
    assignment problem."""
    assert x_est.ndim == 2 and x_est.shape == x_gt.shape
    x_est -= x_est.mean(axis=0, keepdims=True)
    x_gt -= x_gt.mean(axis=0, keepdims=True)
    xy_abs_corrs = np.abs(
        (x_est.T @ x_gt)
        / (((x_est.T @ x_est).diagonal()[:, None] * (x_gt.T @ x_gt).diagonal()) ** 0.5)
    )
    row_ind, col_ind = linear_sum_assignment(-xy_abs_corrs)
    return xy_abs_corrs[row_ind, col_ind].mean()


def generate_mixing(n, d, np_rng=np.random.default_rng(), DECODER_MIN_COND_NUM=0.1):
    # Build the decoder in two steps:
    # 1: Uniformly random selection of column subspace
    decoder_q: npt.NDArray[np.floating] = scipy.stats.ortho_group(d, np_rng).rvs()[:, :n]  # type: ignore

    # 2: Random mixing within the subspace
    decoder_r = np_rng.random((n, n)) - 0.5
    decoder_r_svs = np.linalg.svd(decoder_r, compute_uv=False)
    while decoder_r_svs[-1] / decoder_r_svs[0] < DECODER_MIN_COND_NUM:
        decoder_r = np_rng.random((n, n)) - 0.5
        decoder_r_svs = np.linalg.svd(decoder_r, compute_uv=False)

    # Then, the full decoder is the composition of these transforms
    decoder = decoder_q @ decoder_r
    encoder = np.linalg.pinv(decoder)
    return decoder, encoder


class InvertibleMlp(nn.Module):
    """Invertible MLP with tanh activation"""
    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 2):
        assert num_layers >= 2, "At least 2 layers are required for less code clutter :)"
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = min(in_dim, out_dim)
        self.num_layers = num_layers

        encoders = []
        decoders = []
        enc, dec = generate_mixing(self.in_dim, self.hidden_dim)
        encoders.append(nn.Parameter(torch.from_numpy(enc).float()))
        decoders.append(nn.Parameter(torch.from_numpy(dec).float()))
        for _ in range(1, num_layers - 1):
            enc, dec = generate_mixing(self.hidden_dim, self.hidden_dim)
            encoders.append(nn.Parameter(torch.from_numpy(enc).float()))
            decoders.append(nn.Parameter(torch.from_numpy(dec).float()))
        enc, dec = generate_mixing(self.hidden_dim, self.out_dim)
        encoders.append(nn.Parameter(torch.from_numpy(enc).float()))
        decoders.append(nn.Parameter(torch.from_numpy(dec).float()))

        self.encoders = nn.ParameterList(encoders)
        self.decoders = nn.ParameterList(decoders)

    def encode(self, x: Tensor) -> Tensor:
        for enc in self.encoders:
            x = torch.tanh(x @ enc.T)
        return x

    def decode(self, x: Tensor) -> Tensor:
        for dec_i in self.decoders[::-1]:
            x = torch.arctanh(x) @ dec_i.T
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))
