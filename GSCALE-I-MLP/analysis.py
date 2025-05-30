import argparse
import os
import pickle
import sys
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.func import jacfwd  # type: ignore

# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

import utils
from autoencoders import DenseAutoencoder as Autoencoder

def main(data_dir: str):
    # Load data generation config
    with open(
        os.path.join(data_dir, "generate_data_cfg.pkl"),
        "rb", pickle.HIGHEST_PROTOCOL
    ) as f:
        data_config = pickle.load(f)

    n: int = data_config["n"]
    d : int = data_config["d"]
    intervention_order: npt.NDArray[np.int_] = data_config["intervention_order"]
    n_samples: int = data_config["nsamples"]
    scm = data_config["scm"]

    # Ground truth graph
    # Note the permutation by the intervention order
    dag_gt = scm.adj_mat
    dag_gt = dag_gt[intervention_order, :][:, intervention_order]

    # Load data samples
    with open(
        os.path.join(data_dir, "data.pkl"),
        "rb", pickle.HIGHEST_PROTOCOL
    ) as f:
        data = pickle.load(f)

    zs_obs = data[0][0] # observational env. latent variables
    xs_obs = data[1][0] # observational env. observable variables
    zs_obs = torch.from_numpy(zs_obs).float()
    xs_obs = torch.from_numpy(xs_obs).float()
    print(f"Loaded z and x data.")
    print(f"{zs_obs.shape = }, {xs_obs.shape = }")

    # Load dsx data
    with open(os.path.join(data_dir, "dsxs_bw_hards.pth"), "rb") as f:
        dsxs_bw_hards = torch.load(f, weights_only=True)
    with open(os.path.join(data_dir, "dsxs_hard_obs.pth"), "rb") as f:
        dsxs_hard_obs = torch.load(f, weights_only=True)
    assert isinstance(dsxs_bw_hards, Tensor)
    assert isinstance(dsxs_hard_obs, Tensor)
    print(f"Loaded dsx data.")
    print(f"{dsxs_bw_hards.shape = }, {dsxs_hard_obs.shape = }")

    # Load the trained autoencoder
    autoenc = Autoencoder(n,d)
    autoenc.load_state_dict(torch.load(os.path.join(data_dir, "gscalei.pth"), weights_only=True))
    autoenc.requires_grad_(False)

    encoder = autoenc.get_submodule("encoder").requires_grad_(False)
    decoder = autoenc.get_submodule("decoder").requires_grad_(False)

    ### Part 2: Analysis

    ## Latent variables recovery
    # Mean correlation coefficient
    zhats_obs = encoder(xs_obs)
    assert isinstance(zhats_obs, Tensor)
    z_mcc = utils.mcc(zhats_obs.detach().cpu().numpy(), zs_obs.detach().cpu().numpy())
    print(f"{z_mcc = }")

    ## Latent graph recovery

    # We first need to estimate the latent graph. We do so using the
    # "recovered latent" score difference between the hard int and obs envs:

    # Jac evaluated at obs data. Shape: batch size x input dim x output dim (n)
    js_obs = torch.zeros(xs_obs.shape + (n,))
    for (_idx, zhat) in enumerate(zhats_obs):
        jac = jacfwd(decoder.forward)(zhat.unsqueeze(0))
        assert isinstance(jac, Tensor)
        js_obs[_idx] = jac.squeeze(0, -2)

    # dszhat = jac of decoder * dsx
    dszhats_hard_obs = (js_obs.unsqueeze(1) * dsxs_hard_obs.movedim(0, 1).unsqueeze(-1)).sum(tuple(range(2, xs_obs.ndim + 1)))

    # E| | of this "latent" score difference should be nonzero for only edges in the graph.
    dt1 = dszhats_hard_obs.abs().mean(0)
    print(f"{dag_gt = }")
    print(f"{dt1 = }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", type=str, metavar="DIR", help="Directory to load data and logs from.")
    args = parser.parse_args()
    data_dir: str = args.data_dir
    main(data_dir)
