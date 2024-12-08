"""Transform score diff in `x` (64x64x3) to `y` (of size `latent_dim`)"""
import argparse
import os

import numpy as np

import torch
from torch import Tensor
from torch.func import jacfwd  # type: ignore

from autoencoders import DenseAutoencoder as Autoencoder


def get_data(data_dir: str) -> tuple[Tensor, Tensor, Tensor]:
    """Returns tuple `(xs_obs, dsxs_bw_hards, dsxs_hard_obs)`

    Shape of tensors: `(n, nsamples, 3, w, h)`"""
    data = np.load(os.path.join(data_dir, "z_and_x.npz"))
    xs_obs = data["xs_obs"]
    xs_obs = torch.from_numpy(xs_obs).float().moveaxis(-1, -3) / 255.0

    dsxs_bw_hards = torch.load(os.path.join(data_dir, "dsxs_bw_hards.pth"), weights_only=True)
    dsxs_hard_obs = torch.load(os.path.join(data_dir, "dsxs_hard_obs.pth"), weights_only=True)

    return xs_obs, dsxs_bw_hards, dsxs_hard_obs


def main(args: argparse.Namespace):
    data_dir: str = args.data_dir
    latent_dim: int = args.latent_dim

    # Load data (i) observation x, and (ii) score difference on x domain
    xs_obs, dsxs_bw_hards, dsxs_hard_obs = get_data(data_dir)
    print(f"{dsxs_bw_hards.shape = }")

    # Load the trained dimensionality reduction autoencoder
    autoenc = Autoencoder(latent_dim)
    autoenc.load_state_dict(torch.load(os.path.join(data_dir, f"autoenc_reconstruct_{latent_dim}.pth"), weights_only=True))
    autoenc.requires_grad_(False)

    encoder = autoenc.get_submodule("encoder").requires_grad_(False)
    decoder = autoenc.get_submodule("decoder").requires_grad_(False)

    ys_obs = encoder(xs_obs)

    dsys_shape = dsxs_bw_hards.shape[:2] + (latent_dim,)
    dsys_bw_hards = torch.zeros(dsys_shape)
    dsys_hard_obs = torch.zeros(dsys_shape)
    for (_idx, yi) in enumerate(ys_obs):
        ji = jacfwd(decoder.forward)(yi.unsqueeze(0))
        assert isinstance(ji, Tensor)
        ji = ji.squeeze(0, -2)
        dsxi_bw_hards = dsxs_bw_hards[:, _idx]
        dsxi_hard_obs = dsxs_hard_obs[:, _idx]
        dsys_bw_hards[:, _idx] = (ji * dsxi_bw_hards.unsqueeze(-1)).sum((1, 2, 3))
        dsys_hard_obs[:, _idx] = (ji * dsxi_hard_obs.unsqueeze(-1)).sum((1, 2, 3))

    print(f"{dsys_bw_hards.shape = }")

    torch.save(dsys_bw_hards, os.path.join(args.data_dir, f"dsys_bw_hards_{latent_dim}.pth"))
    torch.save(dsys_hard_obs, os.path.join(args.data_dir, f"dsys_hard_obs_{latent_dim}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", type=str, metavar="DIR", help="Directory to store data and logs.")
    parser.add_argument("latent_dim", type=int, metavar="DIM", help="Latent dimension of dim-reduction step.")
    args = parser.parse_args()

    main(args)
