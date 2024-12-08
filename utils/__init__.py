__all__ = [
    # Submodules
    "dag",
    "linalg",
    "umn",
    # Functions
    "mcc",
    "cov",
    "gaussian_score_est",
    "structural_hamming_distance",
    "cm_graph_entries",
    "generate_mixing",
    "estimate_score_fns_from_data"
]

from scipy.optimize import linear_sum_assignment  # type: ignore
from score_estimators.ssm import score_fn_from_data


from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
import scipy

from . import dag
from . import linalg
from . import umn


def cov(
    x_samples: npt.NDArray[np.float_],
    y_samples: npt.NDArray[np.float_] | None = None,
    center_data: bool = True,
) -> npt.NDArray[np.float_]:
    """Computes batch covariance.

    - Input shapes: `(..., nsamples, n)` and `(..., nsamples, m)`
    - Output shape: `(..., n, m)`

    Second argument is optional; if not provided, computes `x_samples` vs `x_samples`.

    Third argument optionally disables subtracting product of means."""
    if y_samples is None:
        y_samples = x_samples

    assert (
        x_samples.ndim >= 2
        and x_samples.ndim == y_samples.ndim
        and x_samples.shape[:-1] == y_samples.shape[:-1]
    )

    if center_data:
        x_samples -= np.mean(x_samples, axis=-2, keepdims=True)
        y_samples -= np.mean(y_samples, axis=-2, keepdims=True)

    assert y_samples is not None  # pylance...
    return np.mean(x_samples[..., :, None] * y_samples[..., None, :], axis=-3)


def gaussian_score_est(
    x_samples: npt.NDArray[np.float_],
) -> Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]]:
    """Estimates score function of multivariate Gaussian R.V.s using inverse covariance.

    - Input shape: `(nsamples, n, 1)`
    - Output: Function with input & output shapes `(..., n, 1)`."""
    assert x_samples.ndim == 3 and x_samples.shape[-1] == 1

    neg_x_precision_mat = -np.linalg.inv(cov(x_samples[:, :, 0]))

    def score_est(
        x_samples: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        return neg_x_precision_mat @ x_samples

    return score_est


def graph_diff(
    g1: npt.NDArray[np.bool_],
    g2: npt.NDArray[np.bool_],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    edges1 = set(zip(np.where(g1)[0], np.where(g1)[1]))
    edges2 = set(zip(np.where(g2)[0], np.where(g2)[1]))

    g1_reversed = {(j, i) for (i, j) in edges1 if i != j}
    g2_reversed = {(j, i) for (i, j) in edges2 if i != j}

    additions = edges2 - edges1 - g1_reversed
    deletions = edges1 - edges2 - g2_reversed
    reversals = edges1 & g2_reversed

    return additions, deletions, reversals


def cm_graph_entries(
    g1: npt.NDArray[np.bool_],
    g2: npt.NDArray[np.bool_],
) -> tuple[int, int, int]:
    """Computes TP, FP, FN, TN for the edges of
    g1: true graph
    g2: estimated graph
    """
    # g1: true graph
    # g2: estimated graph
    tp =  g1 &  g2
    fp =  g1 & ~g2
    fn = ~g1 &  g2
    tn = ~g1 & ~g2
    return tp.sum(dtype=int), fp.sum(dtype=int), fn.sum(dtype=int), tn.sum(dtype=int)


def structural_hamming_distance(g1: np.ndarray, g2: np.ndarray) -> int:
    additions, deletions, reversals = graph_diff(g1, g2)
    return len(additions) + len(deletions) + len(reversals)

def mcc(x_est: npt.NDArray[np.floating], x_gt: npt.NDArray[np.floating]) -> float:
    """Computes mean correlation coefficient between `x_est` and `x_gt`

    Data dimension: `(n_samples, n)`. Computes the correlation coefficients
    between entries of `x_est` and `x_gt`, and solves the maximum linear sum
    assignment problem."""
    assert x_est.ndim == 2 and x_est.shape == x_gt.shape
    x_est -= x_est.mean(axis=0, keepdims=True)
    x_gt -= x_gt.mean(axis=0, keepdims=True)
    xy_abs_corrs = np.abs((x_est.T @ x_gt) / ((
            (x_est.T @ x_est).diagonal()[:, None] *
            (x_gt.T @ x_gt).diagonal()
        ) ** 0.5))
    row_ind, col_ind = linear_sum_assignment(-xy_abs_corrs)
    return xy_abs_corrs[row_ind, col_ind].mean()


def generate_mixing(n,d,np_rng,DECODER_MIN_COND_NUM=0.1):
    # Build the decoder in two steps:
    # 1: Uniformly random selection of column subspace
    decoder_q: npt.NDArray[np.float_] = scipy.stats.ortho_group(d, np_rng).rvs()[:, :n]  # type: ignore

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

### TO-DO: add SSM to the function below.
def estimate_score_fns_from_data(scm,scm_type,envs,decoder,basis_of_x_supp,nsamples_for_se,n_score_epochs,type_int,var_change_mech,var_change):
    z_samples_for_se = np.stack(
        [
            scm.sample(
                (nsamples_for_se,),
                nodes_int=env,
                type_int=type_int,
                var_change_mech=var_change_mech,
                var_change=var_change,
            )
            for env in envs
        ]
    )       
    x_samples_for_se = decoder @ z_samples_for_se
    x_samples_for_se_on_x_supp = basis_of_x_supp.T @ x_samples_for_se

    hat_sx_fns = list[Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]]]()

    for i in range(len(envs)):
        # If we know the latent model is Linear Gaussian, score estimation
        # is essentially just precision matrix --- a parameter --- estimation

        if scm_type == "linear":
            hat_sx_fn_i_on_x_supp = gaussian_score_est(x_samples_for_se_on_x_supp[i])
            def hat_sx_fn_i(
                x_in: npt.NDArray[np.float_],
                # python sucks... capture value with this since loops are NOT scopes
                hat_sx_fn_i_on_x_supp: Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]] = hat_sx_fn_i_on_x_supp,
            ) -> npt.NDArray[np.float_]:
                """Reduce input down to support of x, compute estimate, transform the result back up."""
                return basis_of_x_supp @ hat_sx_fn_i_on_x_supp(basis_of_x_supp.T @ x_in)
        else:
                # Since parametric approach doesn't work, use SSM.
                hat_sx_fn_i_on_x_supp = score_fn_from_data(
                    torch.from_numpy(x_samples_for_se_on_x_supp[i]).to(torch.float32),
                    epochs=n_score_epochs,
                    add_noise=False,
                )
                def hat_sx_fn_i(
                    x_in: npt.NDArray[np.float_],
                    hat_sx_fn_i_on_x_supp: Callable[[torch.Tensor], torch.Tensor] = hat_sx_fn_i_on_x_supp,
                ) -> npt.NDArray[np.float_]:
                    return basis_of_x_supp @ (
                        hat_sx_fn_i_on_x_supp(
                            torch.from_numpy(basis_of_x_supp.T @ x_in).to(torch.float32)[..., 0]
                        ).detach().numpy()[..., None]
                    )

        hat_sx_fns.append(hat_sx_fn_i)

    return hat_sx_fns