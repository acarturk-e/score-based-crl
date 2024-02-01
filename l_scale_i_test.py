import datetime
import os
import logging
import pickle
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch

from l_scale_i import l_scale_i
from score_estimators.ssm import score_fn_from_data
import utils

scm_type = "quadratic"

if scm_type == "linear":
    from scm.linear import LinearSCM as SCM
elif scm_type == "quadratic":
    from scm.quadratic import QuadraticSCM as SCM
else:
    raise ValueError(f"{scm_type=} is not recognized!")

ATOL_EIGV = 1e-2
ATOL_ORTH = 1e-1
ATOL_EDGE = 5

ATOL_EFF_NZ = 1e-1
DECODER_MIN_COND_NUM = 1e-1


if __name__ == "__main__":
    nd_list = [
        (5, 5),
        (5, 25),
        (5, 50),
        (5, 100),
        (8, 8),
        (8, 25),
        (8, 50),
        (8, 100),
    ]

    fill_rate = 0.5
    nsamples = 10_000
    nruns = 100
    np_rng = np.random.default_rng()

    # Score computation/estimation settings
    estimate_score_fns = True
    nsamples_for_se = 25_000
    enable_gaussian_score_est = False
    n_score_epochs = 20
    add_noise_to_ssm = False

    # SCM settings
    full_rank_scores = True
    hard_intervention = True
    hard_graph_postprocess = True
    type_int = "hard int"
    var_change_mech = "scale"
    var_change = 0.25

    # DEBUG:
    randomize_top_order = True
    randomize_intervention_order = True

    # Result dir setup
    run_name = (
        scm_type
        + "_"
        + ("hard" if hard_intervention else "soft")
        + "_"
        + f"ns{nsamples_for_se/1000:g}k"
        + "_"
        + f"nr{nruns}"
        + "_"
        + (
            "gt"
            if not estimate_score_fns
            else (
                "ss"
                if not enable_gaussian_score_est or scm_type != "linear"
                else "gaus"
            )
        )
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    run_dir = os.path.join("results", run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Logger setup
    log_file = os.path.join(run_dir, "out.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    log_formatter = logging.Formatter(
        "%(asctime)s %(process)d %(levelname)s %(message)s"
    )
    log_file_h = logging.FileHandler(log_file)
    log_file_h.setFormatter(log_formatter)
    log_file_h.setLevel(logging.DEBUG)
    log_console_h = logging.StreamHandler()
    log_console_h.setFormatter(log_formatter)
    log_console_h.setLevel(logging.WARNING)
    log_root_l = logging.getLogger()
    log_root_l.setLevel(logging.INFO)
    log_root_l.addHandler(log_file_h)
    log_root_l.addHandler(log_console_h)

    logging.info(f"Logging to {log_file}")

    config = {
        "ATOL_EIGV": ATOL_EIGV,
        "ATOL_ORTH": ATOL_ORTH,
        "ATOL_EDGE": ATOL_EDGE,
        "scm_type": scm_type,
        "full_rank_scores": full_rank_scores,
        "hard_intervention": hard_intervention,
        "hard_graph_postprocess": hard_graph_postprocess,
        "var_change_mech": var_change_mech,
        "var_change": var_change,
        "nd_list": nd_list,
        "nruns" : nruns,
        "nsamples": nsamples,
        "estimate_score_fns": estimate_score_fns,
        "n_score_epochs": n_score_epochs,
        "nsamples_for_se": nsamples_for_se,
    }
    with open(os.path.join(run_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    results = {
        (n, d): [
            {
                # Inputs
                "scm": SCM.__new__(SCM),
                "intervention_order": np.empty(n, dtype=np.int_),
                "dsz_cor": np.empty((n, n)),
                "decoder": np.empty((d, n)),
                "encoder": np.empty((n, d)),
                "dsx_cor": np.empty((n, n)),
                # Outputs
                "is_run_ok": False,
                # `_obtain_top_order`
                "top_order": np.empty((n,), dtype=np.int_),
                # `_minimize_score_variations`
                "hat_g_s": np.empty((n, n), dtype=bool),
                "hat_enc_s": np.empty((n, d)),
                # Analysis
                "shd_s": 0.0,
                "edge_precision_s": 0.0,
                "edge_recall_s": 0.0,
                "norm_z_err_s": 0.0,
                "extra_nz_in_eff_s": 0,
                # `_unmixing_procedure`
                "hat_g_h": np.empty((n, n), dtype=bool),
                "hat_enc_h": np.empty((n, d)),
                # Analysis
                "shd_h": 0.0,
                "edge_precision_h": 0.0,
                "edge_recall_h": 0.0,
                "norm_z_err_h": 0.0,
                "extra_nz_in_eff_h": 0,
            }
            for _ in range(nruns)
        ]
        for (n, d) in nd_list
    }

    for nd_idx, (n, d) in enumerate(nd_list):
        print(f"Starting {(n, d) = }")

        for run_idx in range(nruns):
            if run_idx % 10 == 10 - 1:
                print(f"{(n, d) = }, {run_idx = }")

            results_run = results[n, d][run_idx]

            # Build the decoder in two steps:
            # 1: Uniformly random selection of column subspace
            # TODO: Theoretically ensure this is indeed uniform
            import scipy.stats  # type: ignore
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

            scm = SCM(
                n,
                fill_rate,
                randomize_top_order=randomize_top_order,
                np_rng=np_rng,
            )

            intervention_order = np_rng.permutation(n) if randomize_intervention_order else np.arange(n)
            envs = [list[int]()] + [[i] for i in intervention_order]

            z_samples = np.stack(
                [
                    scm.sample(
                        (nsamples,),
                        nodes_int=env,
                        type_int=type_int,
                        var_change_mech=var_change_mech,
                        var_change=var_change,
                    )
                    for env in envs
                ]
            )
            z_samples_norm = (z_samples.__pow__(2).sum() ** (0.5))

            x_samples = decoder @ z_samples

            # Evaluate score functions on the same data points
            if estimate_score_fns:
                # Use estimated (noisy) score functions

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

                x_samples_cov = utils.cov(x_samples[0, : n + d, :, 0])
                xsc_eigval, xsc_eigvec = np.linalg.eigh(x_samples_cov)
                basis_of_x_supp = xsc_eigvec[:, -n:]
                x_samples_for_se_on_x_supp = basis_of_x_supp.T @ x_samples_for_se

                hat_sx_fns = list[Callable[[npt.NDArray[np.float_]], npt.NDArray[np.float_]]]()
                for i in range(len(envs)):
                    # If we know the latent model is Linear Gaussian, score estimation
                    # is essentially just precision matrix --- a parameter --- estimation
                    if enable_gaussian_score_est and scm_type == "linear":
                        hat_sx_fn_i_on_x_supp = utils.gaussian_score_est(x_samples_for_se_on_x_supp[i])
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
                            add_noise=add_noise_to_ssm,
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

                sx_samples = np.stack(
                    [
                        hat_sx_fns[env_idx](x_samples[0, ...])
                        for env_idx in range(len(envs))
                    ]
                )
                sz_samples = decoder.T @ sx_samples

            else:
                # Use ground truth score functions
                sz_samples = np.stack(
                    [
                        scm.score_fn(
                            z_samples[0, ...],
                            nodes_int=env,
                            type_int=type_int,
                            var_change_mech=var_change_mech,
                            var_change=var_change,
                        )
                        for env in envs
                    ]
                )
                sx_samples = encoder.T @ sz_samples

            dsz_samples = sz_samples[0, ...] - sz_samples[1:, ...]
            dsz_cor = (
                np.swapaxes(dsz_samples[..., 0], -1, -2) @ dsz_samples[..., 0]
            ) / nsamples

            dsx_samples = sx_samples[0, ...] - sx_samples[1:, ...]
            dsx_cor = (
                np.swapaxes(dsx_samples[..., 0], -1, -2) @ dsx_samples[..., 0]
            ) / nsamples

            # We aim to recover the latents permuted with intervention order
            goal_dag = scm.adj_mat[intervention_order, :][:, intervention_order]
            goal_z_samples = z_samples[..., intervention_order, :]
            goal_decoder = decoder[:, intervention_order]
            goal_encoder = encoder[intervention_order, :]

            # Record input state (except data samples)
            results_run["scm"] = scm
            results_run["intervention_order"] = intervention_order
            results_run["dsz_cor"] = dsz_cor
            results_run["decoder"] = decoder
            results_run["encoder"] = encoder
            results_run["dsx_cor"] = dsx_cor

            try:
                top_order, hat_g_s, hat_enc_s, hard_ests = l_scale_i(
                    x_samples,
                    dsx_samples,
                    hard_intervention=hard_intervention,
                    full_rank_scores=full_rank_scores,
                    hard_graph_postprocess=hard_graph_postprocess,
                    atol_eigv=ATOL_EIGV,
                    atol_orth=ATOL_ORTH,
                    atol_edge=ATOL_EDGE
                )
            except Exception as err:
                logging.error(f"Unexpected {err=}, masking entry out")
                results_run["is_run_ok"] = False
                continue

            # This run succeeded
            results_run["is_run_ok"] = True
            results_run["top_order"] = top_order
            results_run["hat_g_s"] = hat_g_s
            results_run["hat_enc_s"] = hat_enc_s

            if hard_intervention:
                assert hard_ests is not None
                hat_g_h, hat_enc_h = hard_ests

                results_run["hat_g_h"] = hat_g_h
                results_run["hat_enc_h"] = hat_enc_h

            ### ANALYSIS

            # Graph accuracy metrics: SHD, precision, recall
            # We compare the graph itself if full rank OR hard intervention
            # Otherwise, we compare transitive closures
            if full_rank_scores:
                dag_gt_s = goal_dag
            else:
                dag_gt_s = utils.dag.transitive_closure(goal_dag)
                assert dag_gt_s is not None
            edge_cm_s = [
                [
                    ( dag_gt_s &  hat_g_s).sum(dtype=np.int_),
                    (~dag_gt_s &  hat_g_s).sum(dtype=np.int_),
                ], [
                    ( dag_gt_s & ~hat_g_s).sum(dtype=np.int_),
                    (~dag_gt_s & ~hat_g_s).sum(dtype=np.int_),
                ]
            ]
            results[n, d][run_idx]["shd_s"]            = edge_cm_s[0][1] + edge_cm_s[1][0]
            results[n, d][run_idx]["edge_precision_s"] = (edge_cm_s[0][0] / (edge_cm_s[0][0] + edge_cm_s[0][1])) if (edge_cm_s[0][0] + edge_cm_s[0][1]) != 0 else 1.0
            results[n, d][run_idx]["edge_recall_s"]    = (edge_cm_s[0][0] / (edge_cm_s[0][0] + edge_cm_s[1][0])) if (edge_cm_s[0][0] + edge_cm_s[1][0]) != 0 else 1.0

            eff_transform_s = hat_enc_s @ goal_decoder
            eff_transform_s *= (np.sign(np.diagonal(eff_transform_s)) / np.linalg.norm(eff_transform_s, ord=2, axis=1))[:, None]
            hat_z_samples_s = eff_transform_s @ z_samples
            results_run["norm_z_err_s"] = ((hat_z_samples_s - z_samples).__pow__(2).sum() ** (0.5)) / z_samples_norm

            # Construct maximal theoretically allowed mixing pattern:
            # Surrounded nodes can be surrounded by their surrounding nodes.
            dag_gt_s_chp = [np.nonzero((dag_gt_s + np.eye(n, dtype=bool))[i, :])[0] for i in range(n)]
            dag_gt_s_surp = [
                [
                    j for j in range(n)
                    if all([k in dag_gt_s_chp[j] for k in dag_gt_s_chp[i]])
                ]
                for i in range(n)
            ]
            max_mixing_mat = np.array([[j in dag_gt_s_surp[i] for j in range(n)] for i in range(n)])
            results_run["extra_nz_in_eff_s"] = np.sum(
                (np.abs(eff_transform_s) >= ATOL_EFF_NZ) & ~max_mixing_mat, dtype=int
            )

            if hard_intervention:
                assert hard_ests is not None
                hat_g_h, hat_enc_h = hard_ests

                dag_gt_h = goal_dag
                edge_cm_h = [
                    [
                        ( dag_gt_h &  hat_g_h).sum(dtype=int),
                        (~dag_gt_h &  hat_g_h).sum(dtype=int),
                    ], [
                        ( dag_gt_h & ~hat_g_h).sum(dtype=int),
                        (~dag_gt_h & ~hat_g_h).sum(dtype=int),
                    ]
                ]
                results[n, d][run_idx]["shd_h"]            = edge_cm_h[0][1] + edge_cm_h[1][0]
                results[n, d][run_idx]["edge_precision_h"] = (edge_cm_h[0][0] / (edge_cm_h[0][0] + edge_cm_h[0][1])) if (edge_cm_h[0][0] + edge_cm_h[0][1]) != 0 else 1.0
                results[n, d][run_idx]["edge_recall_h"]    = (edge_cm_h[0][0] / (edge_cm_h[0][0] + edge_cm_h[1][0])) if (edge_cm_h[0][0] + edge_cm_h[1][0]) != 0 else 1.0

                # The maximal theoretically allowed mixing pattern is the identity matrix.
                eff_transform_h = hat_enc_h @ goal_decoder
                eff_transform_h *= (np.sign(np.diagonal(eff_transform_h)) / np.linalg.norm(eff_transform_h, ord=2, axis=1))[:, None]
                hat_z_samples_h = eff_transform_h @ z_samples
                results_run["norm_z_err_h"] = ((hat_z_samples_h - z_samples).__pow__(2).sum() ** (0.5)) / z_samples_norm
                results_run["extra_nz_in_eff_h"] = np.sum((np.abs(eff_transform_h) >= ATOL_EFF_NZ) & ~np.eye(n, dtype=bool), dtype=int)

            # Save run
            results[n, d][run_idx] = results_run
            with open(os.path.join(run_dir, f"{n}_{d}_{run_idx}.pkl"), "wb") as f:
                pickle.dump(results_run, f)

    # Transpose the results dict to make it more functional
    results = {
        nd: {
            k: [results_run[k] for results_run in results_run_list]
            for k in results_run_list[0].keys()
        }
        for (nd, results_run_list) in results.items()
    }

    print("")
    print(f"Results ({nruns=}, {nsamples_for_se=})")
    print(f"    (n, d) pairs = {nd_list}")

    is_run_ok = np.array([results[n, d]["is_run_ok"] for (n, d) in nd_list])
    n_ok_runs = is_run_ok.sum(-1)

    shd_s = np.array([results[n, d]["shd_s"] for (n, d) in nd_list])
    edge_precision_s = np.array([results[n, d]["edge_precision_s"] for (n, d) in nd_list])
    edge_recall_s = np.array([results[n, d]["edge_recall_s"] for (n, d) in nd_list])
    norm_z_err_s = np.array([results[n, d]["norm_z_err_s"] for (n, d) in nd_list])
    extra_nz_in_eff_s = np.array([results[n, d]["extra_nz_in_eff_s"] for (n, d) in nd_list])

    print(f"    Ratio of failed runs = {1.0 - n_ok_runs / nruns}")
    print(f"Score minimization")
    print(f"    Structural Hamming dist = {np.around(shd_s.sum(-1) / n_ok_runs, 3)}")
    print(f"    Edge precision          = {np.around(edge_precision_s.sum(-1) / n_ok_runs, 3)}")
    print(f"    Edge recall             = {np.around(edge_recall_s.sum(-1) / n_ok_runs, 3)}")
    print(f"    Normalized Z error      = {np.around(norm_z_err_s.sum(-1) / n_ok_runs, 3)}")
    print(f"    # of pa minus sur mixed = {np.around(extra_nz_in_eff_s.sum(-1) / n_ok_runs, 3)}")

    if hard_intervention:
        shd_h = np.array([results[n, d]["shd_h"] for (n, d) in nd_list])
        edge_precision_h = np.array([results[n, d]["edge_precision_h"] for (n, d) in nd_list])
        edge_recall_h = np.array([results[n, d]["edge_recall_h"] for (n, d) in nd_list])
        norm_z_err_h = np.array([results[n, d]["norm_z_err_h"] for (n, d) in nd_list])
        extra_nz_in_eff_h = np.array([results[n, d]["extra_nz_in_eff_h"] for (n, d) in nd_list])

        print(f"Unmixing")
        print(f"    Structural Hamming dist = {np.around(shd_h.sum(-1) / n_ok_runs, 3)}")
        print(f"    Edge precision          = {np.around(edge_precision_h.sum(-1) / n_ok_runs, 3)}")
        print(f"    Edge recall             = {np.around(edge_recall_h.sum(-1) / n_ok_runs, 3)}")
        print(f"    Normalized Z error      = {np.around(norm_z_err_h.sum(-1) / n_ok_runs, 3)}")
        print(f"    # of pa minus sur mixed = {np.around(extra_nz_in_eff_h.sum(-1) / n_ok_runs, 3)}")
