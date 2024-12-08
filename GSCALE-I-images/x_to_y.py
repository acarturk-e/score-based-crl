"""Transform `x` (64x64x3) data to `y` (of size `latent_dim`)"""
import argparse
import os

import numpy as np

import torch
from torch import Tensor

from autoencoders import DenseAutoencoder as Autoencoder


def get_x(data_dir: str) -> tuple[Tensor, Tensor, Tensor]:
    """Returns tuple `(xs_obs, xs_hard_1, xs_hard_2)`

    NOTE: `x` data is as images i.e. uint8. I map them to [0, 1] float32."""
    data = np.load(os.path.join(data_dir, "z_and_x.npz"))
    xs_obs = data["xs_obs"]
    xs_hard_1 = data["xs_hard_1"]
    xs_hard_2 = data["xs_hard_2"]

    xs_obs = torch.from_numpy(xs_obs).float().moveaxis(-1, -3) / 255.0
    xs_hard_1 = torch.from_numpy(xs_hard_1).float().moveaxis(-1, -3) / 255.0
    xs_hard_2 = torch.from_numpy(xs_hard_2).float().moveaxis(-1, -3) / 255.0

    return xs_obs, xs_hard_1, xs_hard_2


def get_dsx(data_dir: str) -> tuple[Tensor, Tensor]:
    """Returns tuple `(dsxs_bw_hards, dsxs_hard_obs)`

    Shape of tensors: `(n, nsamples, 3, w, h)`"""
    dsxs_bw_hards = torch.load(os.path.join(data_dir, "dsxs_bw_hards.pth"), weights_only=True)
    dsxs_hard_obs = torch.load(os.path.join(data_dir, "dsxs_hard_obs.pth"), weights_only=True)

    return dsxs_bw_hards, dsxs_hard_obs


def main(args: argparse.Namespace):
    data_dir: str = args.data_dir
    latent_dim: int = args.latent_dim

    # Load data (i) observation x, and (ii) score difference on x domain
    xs_obs, xs_hard_1, xs_hard_2 = get_x(data_dir)
    print(f"{xs_obs.shape = }")

    n_samples = xs_obs.shape[0]
    n = xs_hard_1.shape[0]

    # Load the trained dimensionality reduction autoencoder
    autoenc = Autoencoder(latent_dim)
    autoenc.load_state_dict(torch.load(os.path.join(data_dir, f"autoenc_reconstruct_{latent_dim}.pth"), weights_only=True))
    autoenc.requires_grad_(False)

    encoder = autoenc.get_submodule("encoder").requires_grad_(False)
    decoder = autoenc.get_submodule("decoder").requires_grad_(False)

    ys_obs = encoder(xs_obs)
    ys_hard_1 = encoder(xs_hard_1.flatten(0, 1)).unflatten(0, (n, n_samples))
    ys_hard_2 = encoder(xs_hard_2.flatten(0, 1)).unflatten(0, (n, n_samples))

    torch.save(ys_obs, os.path.join(args.data_dir, f"ys_obs_{latent_dim}.pth"))
    torch.save(ys_hard_1, os.path.join(args.data_dir, f"ys_hard_1_{latent_dim}.pth"))
    torch.save(ys_hard_2, os.path.join(args.data_dir, f"ys_hard_2_{latent_dim}.pth"))
    print(f"{ys_obs.shape = }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", type=str, metavar="DIR", help="Directory to store data and logs.")
    parser.add_argument("latent_dim", type=int, metavar="DIM", help="Latent dimension of dim-reduction step.")
    args = parser.parse_args()

    main(args)
