"""Transform `x` (64x64x3) data to `y` (of size `latent_dim`)"""
import argparse
import os
from tqdm import tqdm

import numpy as np
import torch
from torch.func import jacfwd, vmap  # type: ignore
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from autoencoders import CnnAutoencoder, DenseAutoencoder


def get_data(data_dir: str) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return all data: z, x, and both dsxs"""
    data = np.load(os.path.join(data_dir, "z_and_x.npz"))
    n = data["zs_obs"].shape[1]
    xs_obs    = torch.from_numpy(data["xs_obs"   ]).float() / 255.0
    xs_hard_1 = torch.from_numpy(data["xs_hard_1"]).float() / 255.0
    xs_hard_2 = torch.from_numpy(data["xs_hard_2"]).float() / 255.0
    print(f"Loaded x data with shapes {xs_obs.shape = }, {xs_hard_1.shape = } (for int)")
    dsxs_bw_hards = torch.load(os.path.join(data_dir, f"dsxs_bw_hards.pth"), weights_only=True)
    dsxs_hard_obs = torch.load(os.path.join(data_dir, f"dsxs_hard_obs.pth"), weights_only=True)
    print(f"Loaded dsx data with shapes {dsxs_bw_hards.shape = }")
    return xs_obs, xs_hard_1, xs_hard_2, dsxs_bw_hards, dsxs_hard_obs


def main(args: argparse.Namespace):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_dir: str = args.data_dir
    latent_dim: int = args.latent_dim

    xs_obs, xs_hard_1, xs_hard_2, dsxs_bw_hards, dsxs_hard_obs = get_data(data_dir)
    n, nsamples = xs_hard_1.shape[:2]

    # Load the trained dimensionality reduction autoencoder
    autoenc = CnnAutoencoder(latent_dim)
    autoenc.load_state_dict(torch.load(os.path.join(data_dir, f"autoenc_reconstruct_{latent_dim}.pth"), weights_only=True))
    autoenc.requires_grad_(False).to(device)

    encoder = autoenc.get_submodule("encoder")
    decoder = autoenc.get_submodule("decoder")

    # Transform x to y
    print("Transforming xs_obs")
    ys_obs = torch.zeros((nsamples, latent_dim))
    for i, (xb,) in tqdm(enumerate(DataLoader(TensorDataset(xs_obs), batch_size=args.batch_size, shuffle=False))):
        ys_obs[i*args.batch_size: i*args.batch_size+len(xb)] = encoder(xb.to(device)).cpu()

    torch.save(ys_obs, os.path.join(data_dir, f"ys_obs_{latent_dim}.pth"))

    print("Transforming xs_hard_1")
    ys_hard_1 = torch.zeros((n * nsamples, latent_dim))
    for i, (xb,) in tqdm(enumerate(DataLoader(TensorDataset(xs_hard_1.flatten(0, 1)), batch_size=args.batch_size, shuffle=False))):
        ys_hard_1[i*args.batch_size: i*args.batch_size+len(xb)] = encoder(xb.to(device)).cpu()
    torch.save(ys_hard_1, os.path.join(data_dir, f"ys_hard_1_{latent_dim}.pth"))

    print("Transforming xs_hard_2")
    ys_hard_2 = torch.zeros((n * nsamples, latent_dim))
    for i, (xb,) in tqdm(enumerate(DataLoader(TensorDataset(xs_hard_2.flatten(0, 1)), batch_size=args.batch_size, shuffle=False))):
        ys_hard_2[i*args.batch_size: i*args.batch_size+len(xb)] = encoder(xb.to(device)).cpu()
    torch.save(ys_hard_2, os.path.join(data_dir, f"ys_hard_2_{latent_dim}.pth"))

    print(f"Reduced data to y of shape {ys_obs.shape = }")

    # Transform dsxs to dsys
    # js_obs = vmap(jacfwd(decoder))(ys_obs)
    js_obs = torch.stack([torch.zeros_like(xs_obs).flatten(1, -1)] * latent_dim, -1)
    print(js_obs.shape)
    autoenc.to(device)
    for i, yi in tqdm(enumerate(ys_obs)):
        # js_obs[i] = jacfwd(decoder)(yi).squeeze(0, 2)
        js_obs[i] = jacfwd(decoder)(yi.unsqueeze(0).to(device)).cpu().squeeze(0, -2).flatten(0, -2)
    dsys_bw_hards = (dsxs_bw_hards.flatten(2, -1).moveaxis(0, 1) @ js_obs).moveaxis(0, 1)
    dsys_hard_obs = (dsxs_hard_obs.flatten(2, -1).moveaxis(0, 1) @ js_obs).moveaxis(0, 1)

    torch.save(dsys_bw_hards, os.path.join(data_dir, f"dsys_bw_hards_{latent_dim}.pth"))
    torch.save(dsys_hard_obs, os.path.join(data_dir, f"dsys_hard_obs_{latent_dim}.pth"))
    print(f"Reduced dsxs to dsys of shapes {dsys_bw_hards.shape = }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", type=str, metavar="DIR", help="Directory to store data and logs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the training on.")
    parser.add_argument("--latent-dim", type=int, metavar="DIM", default=64, help="Latent dimension of dim-reduction step.")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    ### NOTE HARD-CODED!
    torch.set_num_threads(32)

    main(args)
