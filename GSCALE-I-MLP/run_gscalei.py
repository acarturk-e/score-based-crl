"""Run GSCALE-I algorithm on pre-computed score differences."""
import argparse
import json
import os
import pickle
import sys

import gzip
import torch
from torch.func import vmap, jacfwd  # type: ignore
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# Setting these flags True makes A100 training a lot faster:
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# Add the repo root to sys.path so files in parent directory are accessible
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from autoencoders import DenseAutoencoderTanh
import utils


def main(args: argparse.Namespace):

    runname = "run_gscalei"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    lambda1: int = args.lambda1
    num_epochs: int = args.max_epochs

    # Load the data generation configuration
    with open(
        os.path.join(args.data_dir, "generate_data_cfg.pkl"),
        "rb", pickle.HIGHEST_PROTOCOL
    ) as f:
        _ = pickle.load(f)
        data_dir = _["data_dir"]
        n = _["n"]
        d = _["d"]

    # Load the data
    with gzip.open(os.path.join(data_dir, "data.pkl.gz"), "rb") as f:
        z_samples, x_samples = pickle.load(f)

    z_samples = torch.from_numpy(z_samples).float()
    x_samples = torch.from_numpy(x_samples).float()

    # Load the score differences
    # The axis swap moves the sample dimension to the first axis.
    # This is despite the fact that data from different "environments" are not related.
    dsxs_bw_hards = torch.load(os.path.join(args.data_dir, "dsxs_bw_hards.pth"), weights_only=True).swapaxes(0, 1)

    # Create the dataset
    dataset = TensorDataset(x_samples[0], dsxs_bw_hards)

    with open(f"{args.data_dir}/run_gscalei_args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    with gzip.open(f"{args.data_dir}/run_gscalei_datasets.pkl.gz", "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Create model: Generic MLP based autoencoder with softmax activation
    autoenc = DenseAutoencoderTanh(d, n).to(device)

    if args.load_checkpoint:
        autoenc.load_state_dict(torch.load(os.path.join(args.data_dir, f"gscalei.pth"), weights_only=True))

    encoder = autoenc.get_submodule("encoder")
    decoder = autoenc.get_submodule("decoder")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size if args.batch_size is not None else len(dataset),
        shuffle=True,
    )

    opt = torch.optim.Adam(
        autoenc.parameters(),
        lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100])


    def loss_fn(xb: Tensor, dsxb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """The loss function for our autoencoder. Note that the `dsxb`
        is the score difference between hard intervention environment
        pairs for this batch."""
        assert xb.ndim == 2 and dsxb.ndim == 3
        zhatb = encoder(xb)
        xhatb = decoder(zhatb)
        assert isinstance(zhatb, Tensor)
        assert isinstance(xhatb, Tensor)

        # # Jac evaluated at obs data. Shape: batch size x input dim (n) x output dim (d)
        # jb = torch.zeros((len(xb), d, n), device=device)
        # for (i, zhati) in enumerate(zhatb):
        #     ji = jacfwd(decoder.forward)(zhati.unsqueeze(0))
        #     assert isinstance(ji, Tensor)
        #     jb[i] = ji.squeeze(0, -2)
        jb = vmap(jacfwd(decoder.forward))(zhatb)

        dszhatb = dsxb @ jb
        dt = dszhatb.abs().mean(0)
        #loss_main = dt.fill_diagonal_(0.0).sum()
        loss_main = (dt - torch.eye(n, device=device)).abs().sum()

        loss_reconstr = lambda1 * (xhatb - xb).pow(2).mean(0).sum()
        loss = loss_main + loss_reconstr
        return loss, loss_reconstr.detach(), loss_main.detach()


    for epoch in range(num_epochs):

        # Training
        running_loss = running_loss_reconstr = running_loss_main = log_steps = 0
        for (xb, dsxb) in dataloader:
            xb, dsxb = xb.to(device), dsxb.to(device)
            opt.zero_grad()
            assert isinstance(xb, Tensor)
            assert isinstance(dsxb, Tensor)
            loss, loss_reconstr, loss_main = loss_fn(xb, dsxb)

            loss.backward()
            opt.step()

            log_steps += 1
            running_loss += loss.item()
            running_loss_reconstr += loss_reconstr.item()
            running_loss_main += loss_main.item()

        avg_loss = running_loss / log_steps
        avg_loss_reconstr = running_loss_reconstr / log_steps
        avg_loss_main = running_loss_main / log_steps
        # End of training

        print(f"Epoch:{epoch + 1:3}/{num_epochs:3}, train Loss={avg_loss:+.6f} (reconstr={avg_loss_reconstr:+.6f}, main={avg_loss_main:+.6f})")
        lr_sched.step()

        if args.checkpoint_epochs != -1 and epoch % args.checkpoint_epochs == 0:
            torch.save(autoenc.state_dict(), os.path.join(args.data_dir, f"gscalei.pth"))

            # also add a MCC check for every checkpoint.
            zhat_obs = encoder(x_samples[0].to(device))
            assert isinstance(zhat_obs, Tensor)
            z_mcc = utils.mcc(zhat_obs.detach().cpu().numpy(), z_samples[0].detach().cpu().numpy())
            print(f"\tEpoch:{epoch + 1:3}/{num_epochs:3}, MCC={z_mcc:.4f}")

    autoenc.eval()
    autoenc.requires_grad_(False)
    torch.save(autoenc.state_dict(), os.path.join(args.data_dir, f"gscalei.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data-dir", type=str, default="data/", metavar="DIR", help="Directory to store data and logs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the training on.")
    parser.add_argument("--lambda1", type=float, default=100.0, help="Scale for reconstruction loss.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate of optimizer.")
    parser.add_argument("--weight-decay", type=float, default=0.0001, metavar="LAMBDA", help="Weight decay.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--max-epochs", type=int, default=200, metavar="EPOCHS", help="Number of epochs to run for each LDR model.")
    parser.add_argument("--checkpoint-epochs", type=int, default=10, metavar="EPOCHS", help="Epoch period of checkpoint saves. Set to -1 to not save checkpoints.")
    parser.add_argument("--load-checkpoint", action="store_true", help="Loads all model parameters from checkpoints.")
    parser.add_argument("--global-seed", type=int, default=9724)
    args = parser.parse_args()

    ### NOTE HARD-CODED!
    torch.set_num_threads(32)

    main(args)
