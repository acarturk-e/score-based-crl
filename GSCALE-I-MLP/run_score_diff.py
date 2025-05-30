"""Do score difference estimation on the GSCALE-I data."""
import argparse
import copy
import json
import os
import pickle
import sys

import gzip
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Setting these flags True makes A100 training a lot faster:
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from model_score_diff import ScoreDiffCls
from train_score_diff import train_score_diff


class CombinedDataset4X(Dataset):
    """Combine two datasets into one dataset with corresponding label.
    
    Returns the 2nd item of the dataset and a label indicating which dataset it came from."""
    def __init__(self, dataset0: TensorDataset, dataset1: TensorDataset):
        self.dataset0 = dataset0
        self.dataset1 = dataset1

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        if idx < len(self.dataset0):
            return self.dataset0[idx][1], torch.tensor(0.0)
        else:
            return self.dataset1[idx - len(self.dataset0)][1], torch.tensor(1.0)

    def __len__(self):
        return len(self.dataset0) + len(self.dataset1)


def main(args: argparse.Namespace):

    runname = "run_score_diff"

    opts = argparse.Namespace(
        use_gaussian_ldr=False,
        batch_size=256,
        lr=1e-2,
        epochs=50,
        seed=12,
        m=4,
        disable_cuda=False,
        use_gaussian=False,
        dense_ldr_width=64,
        dense_ldr_depth=3,
    )

    # Load the data generation configuration
    with open(
        os.path.join(args.data_dir, "generate_data_cfg.pkl"),
        "rb", pickle.HIGHEST_PROTOCOL
    ) as f:
        _ = pickle.load(f)
        data_dir = _["data_dir"]
        scm_type = _["scm_type"]
        n = _["n"]
        d = _["d"]
        num_nonlinearities = _["num_nonlinearities"]
        fill_rate = _["fill_rate"]
        nsamples = _["nsamples"]
        var_change_mech = _["var_change_mech"]
        randomize_top_order = _["randomize_top_order"]
        randomize_intervention_order = _["randomize_intervention_order"]
        intervention_order = _["intervention_order"]
        envs = _["envs"]
        var_changes = _["var_changes"]
        scm = _["scm"]
        inv_autoenc = _["inv_autoenc"]

    # Load the data
    with gzip.open(os.path.join(data_dir, "data.pkl.gz"), "rb") as f:
        z_samples, x_samples = pickle.load(f)

    z_samples = torch.from_numpy(z_samples).float()
    x_samples = torch.from_numpy(x_samples).float()

    # Create the datasets
    dataset_obs = TensorDataset(z_samples[0], x_samples[0])
    datasets_int_0 = [TensorDataset(z_samples[1+i], x_samples[1+i]) for i in range(n)]
    datasets_int_1 = [TensorDataset(z_samples[1+n+i], x_samples[1+n+i]) for i in range(n)]

    with open(f"{args.data_dir}/run_score_diff_args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    with open(f"{args.data_dir}/run_score_diff_opts.json", "w") as f:
        json.dump(opts.__dict__, f, indent=4)

    with gzip.open(f"{args.data_dir}/run_score_diff_datasets.pkl.gz", "wb") as f:
        pickle.dump((dataset_obs, datasets_int_0, datasets_int_1), f, protocol=pickle.HIGHEST_PROTOCOL)

    ## Compute log density ratio between
    # 1. Hard intervention pairs
    ldr_bw_hards = list[ScoreDiffCls]()
    for env_idx in range(n):
        print(f"Training score diff b/w int and obs, #{env_idx}")
        dataset_int = datasets_int_0[env_idx]
        combined_dataset = CombinedDataset4X(dataset_obs, dataset_int)
        dataloader = DataLoader(
            combined_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
        )

        opts_m = copy.deepcopy(opts)
        opts_m.load = False
        # set class priors
        opts_m.nu = len(dataset_obs) / len(dataset_int)

        # Train score difference function between the two datasets
        ldr_bw_hards.append(train_score_diff(dataloader, opts_m, data_dir, runname + f"_bw_hards_{env_idx}").cpu().eval().requires_grad_(False))

    # 2. Hard interventions (one set suffices) and observational domain
    ldr_hard_obs = list[ScoreDiffCls]()
    for env_idx in range(n):
        print(f"Training score diff b/w ints, #{env_idx}")
        dataset_int0 = datasets_int_0[env_idx]
        dataset_int1 = datasets_int_1[env_idx]
        combined_dataset = CombinedDataset4X(dataset_int0, dataset_int1)
        dataloader = DataLoader(
            combined_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
        )

        opts_m = copy.deepcopy(opts)
        opts_m.load = False
        # set class priors
        opts_m.nu = len(dataset_int0) / len(dataset_int1)

        # Train score difference function between the two datasets
        ldr_hard_obs.append(train_score_diff(dataloader, opts_m, data_dir, runname + f"_hard_obs_{env_idx}").cpu().eval().requires_grad_(False))

    ## Use the LDR models to estimate the score difference function
    ## on observational data points.
    # NOTE: Score difference has the same shape as the original data.
    # Since we compute one score difference per environment, the overall
    # shape of `dsxs` is `(n, n_samples) + image_shape`, where `n_samples` is
    # the number of samples in `xs_obs`, the points we evaluate the models on.

    # 1. Hard intervention pairs
    dsxs_bw_hards = torch.zeros((n,) + x_samples[0].shape)
    for env_idx in range(n):
        ldr_model = ldr_bw_hards[env_idx]
        with torch.no_grad():
            dsxs_bw_hards[env_idx] = ldr_model.score_diff(x_samples[0])
    torch.save(dsxs_bw_hards, os.path.join(args.data_dir, "dsxs_bw_hards.pth"))

    # 2. Hard interventions (one set suffices) and observational domain
    dsxs_hard_obs = torch.zeros((n,) + x_samples[0].shape)
    for env_idx in range(n):
        ldr_model = ldr_bw_hards[env_idx]
        with torch.no_grad():
            dsxs_hard_obs[env_idx] = ldr_model.score_diff(x_samples[0])
    torch.save(dsxs_hard_obs, os.path.join(args.data_dir, "dsxs_hard_obs.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("-d", "--data-dir", type=str, default="data/", help="directory to load data and save the results")
    parser.add_argument("--device", type=str, default=None, help="device to run the training")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    main(args)
