__all__ = [
    # Submodules
    "dag",
    "linalg",
    "umn",
    "util_analysis",
    # Functions
    "mcc",
    "cov",
    "gaussian_score_est",
    "generate_mixing",
    "estimate_score_fns_from_data"
]

from scipy.optimize import linear_sum_assignment  # type: ignore
from score_estimators.ssm import score_fn_from_data
import dcor

from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
import scipy

from . import dag
from . import linalg
from . import umn
from . import util_analysis


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


def mdcc(x_est: npt.NDArray[np.floating], x_gt: npt.NDArray[np.floating]) -> float:
    """Computes mean distance correlation between `x_est` and `x_gt`

    Data dimension: `(n_samples, n)`. Computes the correlation coefficients
    between entries of `x_est` and `x_gt`, and solves the maximum linear sum
    assignment problem."""
    assert x_est.ndim == 2 and x_est.shape == x_gt.shape
    x_est -= x_est.mean(axis=0, keepdims=True)
    x_gt -= x_gt.mean(axis=0, keepdims=True)

    n = x_est.shape[-1]
    dcor_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            dcor_mat[i,j] = dcor.distance_correlation(x_est[:,i],x_gt[:,j])
            dcor_mat[j,i] = dcor_mat[i,j]

    row_ind, col_ind = linear_sum_assignment(-dcor_mat)

    return dcor_mat[row_ind, col_ind].mean()


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


def generate_mixing(n,d,np_rng,DECODER_MIN_COND_NUM=0.1,normalize_decoder=True,normalize_encoder=False):

    # can only normalize of them    
    assert (normalize_decoder & normalize_encoder) == False
 
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

    if normalize_decoder == True:
        decoder = decoder / np.max(np.abs(decoder),0)
        encoder = np.linalg.pinv(decoder)
    elif normalize_encoder == True:
        encoder = np.linalg.pinv(decoder)
        encoder = (encoder.T / np.max(np.abs(encoder),1)).T
        decoder = np.linalg.pinv(encoder)

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


def generate_results_dict(nruns:int, n:int, d:int):
    res_dict  = [{ 
                ### Common Inputs
                "decoder": np.empty((d, n)),
                "encoder": np.empty((n, d)),
                "dag_gt_tc": np.empty((n,n)),
                "dag_gt_tr": np.empty((n,n)),
                "dag_gt": np.empty((n,n)),
                ### Outputs
                "is_run_ok": False,
                # Soft interventions
                "hat_g_s": np.empty((n, n), dtype=bool),
                "hat_g_s_tc": np.empty((n, n), dtype=bool),
                "hat_g_s_tr": np.empty((n, n), dtype=bool),
                "hat_enc_s": np.empty((n, d)),
                "top_order_s": np.empty((n,), dtype=np.int_), 
                "dshatz_cor_s": np.empty((n, n, n)),
                # Hard interventions
                "hat_g_h": np.empty((n, n), dtype=bool),
                "hat_enc_h": np.empty((n, d)),
                "top_order_h": np.empty((n,), dtype=np.int_), 
                "dshatz_cor_h": np.empty((n, n, n)),
                # for special case: full rank scores
                "hat_g_f": np.empty((n, n), dtype=bool),
                "hat_enc_f": np.empty((n, d)),
                "dshatz_cor_f": np.empty((n, n, n)),   
                # analysis
                "shd_s_tr": 0.0,
                "shd_s_tc": 0.0,
                "shd_h": 0.0,
                "shd_f": 0.0,
                "edge_precision_s": 0.0,
                "edge_precision_h": 0.0,
                "edge_precision_f": 0.0,
                "edge_recall_s": 0.0,
                "edge_recall_h": 0.0,
                "edge_recall_f": 0.0,
                "mcc_s": 0.0,
                "mcc_h": 0.0,
                "mcc_f": 0.0,
                "eff_transform_s": np.empty((n, n)),
                "eff_transform_h": np.empty((n, n)),
                "eff_transform_f": np.empty((n, n)),
                "extra_nz_in_eff_s": 0,
                "extra_nz_in_eff_h": 0,
                "extra_nz_in_eff_f": 0,
                "extra_nz_ratio_in_eff_s": 0,
                "extra_nz_ratio_in_eff_h": 0,
                "extra_nz_ratio_in_eff_f": 0,
                "eff_transform_err_norm_s": 0.0,
                "eff_transform_err_norm_h": 0.0,
                "eff_transform_err_norm_f": 0.0,
            } for _ in range(nruns)]
    return res_dict
