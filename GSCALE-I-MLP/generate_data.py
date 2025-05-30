"""
General Score-based Causal Latent Estimation via Interventions (GSCALE-I)

Setting: X = g(Z), i.e. general transform. 
single-node hard interventions.

Paper: Score-based Causal Representation Learning: Linear and General Transformations
(https://arxiv.org/abs/2402.00849), published at JMLR 05/2025.

In Section 7.2: we consider transformation functions g that are 1 or 3-layer MLPs with tanh activation function. For instance, for 1-layer, we have X = tanh(G.Z).

This scripts generates data for the desired settings.

Note that for 1-layer case, we can more directly estimate the parameter matrix G (refer to `g_scale_i_glm_tanh` function).
"""

import argparse
import os
import pickle
import sys

import gzip
import numpy as np
import torch

# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

# Import the desired SCM class: LinearSCM, QuadraticSCM, or MlpSCM
from scm.mlp import MlpSCM as SCM
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data-dir", type=str, default="data/", metavar="DIR", help="Directory to store data and logs.")
    args = parser.parse_args()

    ### NOTE HARD-CODED!
    torch.set_num_threads(32)

    data_dir: str = args.data_dir

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    scm_type = "mlp"
    n = 5
    d = 10
    num_nonlinearities = 3  # number of layers in MLP-based transformation g
    fill_rate = 0.5  # graph density for G(nnodes,density) model
    nsamples = 10_000
    var_change_mech = "scale"
    randomize_top_order = False
    randomize_intervention_order = False
    rng = np.random.default_rng()

    # generate a nonlinear mixing model
    inv_autoenc = utils.InvertibleMlp(n, d, num_layers=num_nonlinearities)

    # generate a latent SCM
    scm = SCM(
        n,
        fill_rate,
        randomize_top_order=randomize_top_order,
        np_rng=rng,
    )

    intervention_order = rng.permutation(n) if randomize_intervention_order else np.arange(n)
    envs = [list[int]()] + [[i] for i in intervention_order] + [[i] for i in intervention_order]
    var_changes = [1.0] + [0.25 for _ in intervention_order] + [4.0 for _ in intervention_order]

    # Save the configuration
    with open(
        os.path.join(data_dir, "generate_data_cfg.pkl"),
        "wb", pickle.HIGHEST_PROTOCOL
    ) as f:
        pickle.dump({
            "data_dir": data_dir,
            "scm_type": scm_type,
            "n": n,
            "d": d,
            "num_nonlinearities": num_nonlinearities,
            "fill_rate": fill_rate,
            "nsamples": nsamples,
            "var_change_mech": var_change_mech,
            "randomize_top_order": randomize_top_order,
            "randomize_intervention_order": randomize_intervention_order,
            "intervention_order": intervention_order,
            "envs": envs,
            "var_changes": var_changes,
            "scm": scm,
            "inv_autoenc": inv_autoenc,
        }, f)

    # generate latent (z) and observed (x) samples for each environment
    # NOTE: IMPORTANT: we "encode" the z samples to get the x samples
    # in contrast to the usual autoencoder language
    z_samples = np.stack([
        scm.sample(
            (nsamples,),
            nodes_int=env,
            type_int="hard int",
            var_change_mech=var_change_mech,
            var_change=var_change)
        for env, var_change in zip(envs, var_changes)])
    z_samples = z_samples.squeeze(-1)
    x_samples = inv_autoenc.encode(torch.from_numpy(z_samples).float()).detach().double().numpy()

    # save the data in compressed format
    with gzip.open(os.path.join(data_dir, "data.pkl.gz"), "wb") as f:
        pickle.dump((z_samples, x_samples), f)
