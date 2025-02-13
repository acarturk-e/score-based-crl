import argparse
import logging
import os

import numpy as np
import torch
from torch import Tensor
from torch.func import jacfwd, vmap  # type: ignore

# Setting these flags True makes A100 training a lot faster:
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# For data
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Local imports
from autoencoders import DenseAutoencoder2
import utils


def get_data(data_dir: str, latent_dim: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return all data: z, y, and dsys_bw_hards"""
    data = np.load(os.path.join(data_dir, "z_and_x.npz"))
    n = data["zs_obs"].shape[1]
    zs_obs = data["zs_obs"]
    zs_obs = torch.from_numpy(zs_obs).float()
    ys_obs = torch.load(os.path.join(data_dir, f"ys_obs_{latent_dim}.pth"), weights_only=True)
    logging.info(f"Loaded z and y data with shapes {zs_obs.shape = }, {ys_obs.shape = }")

    dsys_bw_hards = torch.load(os.path.join(data_dir, f"dsys_bw_hards_{latent_dim}.pth"), weights_only=True)
    logging.info(f"Loaded dsys data with shape {dsys_bw_hards.shape = }")
    return zs_obs, ys_obs, dsys_bw_hards


def create_logger(log_dir: str) -> logging.Logger:
    """Create a logger that writes to a log file (DEBUG level) and stdout (INFO level)"""
    os.makedirs(log_dir, exist_ok=True)
    h_out = logging.StreamHandler()
    h_out.setLevel(logging.INFO)
    h_file = logging.FileHandler(os.path.join(log_dir, "train_autoenc_disentangle_cpu.log"), mode="w")
    h_file.setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(module)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[h_out, h_file])
    logger = logging.getLogger(__name__)
    return logger


def main(args: argparse.Namespace):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_dir: str = args.data_dir
    latent_dim: int = args.latent_dim

    lambda1: float = args.lambda1

    # Set your random seed for experiment reproducibility.
    if args.seed is not None: torch.manual_seed(args.seed)

    logger = create_logger(data_dir)
    logger.info(f"Experiment directory created at {data_dir}")

    # Get data
    zs_obs, ys_obs, dsys_bw_hards = get_data(data_dir, latent_dim)

    # One intervention pair per latent dimension
    n, n_samples = dsys_bw_hards.shape[:2]

    [train_dataset, valid_dataset] = random_split(TensorDataset(ys_obs, dsys_bw_hards.moveaxis(0, 1)), [0.9, 0.1])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True) # Set "True" to prevent uneven splits
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True) # Set "True" to prevent uneven splits


    # Create model
    autoenc = DenseAutoencoder2(n, latent_dim, [256]).to(device)

    if args.load_checkpoint:
        autoenc.load_state_dict(torch.load(os.path.join(args.data_dir, f"autoenc_disentangle_{latent_dim}.pth"), weights_only=True))

    encoder = autoenc.get_submodule("encoder")
    decoder = autoenc.get_submodule("decoder")

    opt = torch.optim.Adam(
        autoenc.parameters(),
        lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=(args.max_epochs // 15) + 1, T_mult=2)


    def loss_fn(yb: Tensor, dsyb: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """The loss function for our autoencoder. Note that the `dsyb`
        is the score difference between hard intervention environment
        pairs for this batch."""
        assert yb.ndim == 2 and dsyb.ndim == 3
        zhatb = encoder(yb)
        yhatb = decoder(zhatb)
        assert isinstance(zhatb, Tensor)
        assert isinstance(yhatb, Tensor)

        # Jac evaluated at obs data. Shape: batch size x input dim x output dim (n)
        jb = vmap(jacfwd(decoder))(zhatb)
        dszhatb = dsyb @ jb
        dt = dszhatb.abs().mean(0)
        # loss_main = dt.abs().sum()
        # loss_main = dt.abs().sum() + zhatb.pow(2).mean(0).mean()
        loss_main = (dt - torch.eye(n, device=yb.device)).abs().sum()

        loss_reconstr = lambda1 * (yhatb - yb).pow(2).mean(0).sum()
        loss = loss_main + loss_reconstr
        return loss, loss_reconstr.detach(), loss_main.detach()


    for epoch in range(args.max_epochs):

        # Training
        autoenc.train()
        running_loss = running_loss_reconstr = running_loss_main = log_steps = 0
        for (yb, dsyb) in train_loader:
            yb, dsyb = yb.to(device), dsyb.to(device)
            opt.zero_grad()
            assert isinstance(yb, Tensor)
            assert isinstance(dsyb, Tensor)
            loss, loss_reconstr, loss_main = loss_fn(yb, dsyb)

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
        logger.info(f"({epoch=}), Train Loss: {avg_loss:.5f} (reconstr={avg_loss_reconstr:.5f}, main={avg_loss_main:.5f})")

        # Validation
        autoenc.eval()
        running_loss = running_loss_reconstr = running_loss_main = log_steps = 0
        for (yb, dsyb) in valid_loader:
            yb, dsyb = yb.to(device), dsyb.to(device)
            assert isinstance(yb, Tensor)
            assert isinstance(dsyb, Tensor)

            with torch.no_grad():
                loss, loss_reconstr, loss_main = loss_fn(yb, dsyb)

            log_steps += 1
            running_loss += loss.item()
            running_loss_reconstr += loss_reconstr.item()
            running_loss_main += loss_main.item()

        avg_loss = running_loss / log_steps
        avg_loss_reconstr = running_loss_reconstr / log_steps
        avg_loss_main = running_loss_main / log_steps

        # End of validation
        lr_sched.step()

        logger.info(f"({epoch=}), Validation Loss: {avg_loss:.5f} (reconstr={avg_loss_reconstr:.5f}, main={avg_loss_main:.5f})")
        if args.checkpoint_epochs != -1 and epoch % args.checkpoint_epochs == 0:
            torch.save(autoenc.state_dict(), os.path.join(args.data_dir, f"autoenc_disentangle_{latent_dim}.pth"))

            # also add a MCC check for every checkpoint.
            zhats_obs = autoenc.encoder(ys_obs.to(device))
            assert isinstance(zhats_obs, Tensor)
            z_mcc = utils.mcc(zhats_obs.detach().cpu().numpy(), zs_obs.detach().cpu().numpy())
            logger.info(f"(MCC={z_mcc:.4f})")

    autoenc.eval()
    autoenc.requires_grad_(False)
    torch.save(autoenc.state_dict(), os.path.join(args.data_dir, f"autoenc_disentangle_{latent_dim}.pth"))
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", type=str, metavar="DIR", help="Directory to store data and logs.")
    parser.add_argument("--latent-dim", type=int, metavar="DIM", default=64, help="Dimension of the dim-reduced representations.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the training on.")
    parser.add_argument("--lambda1", type=float, default=1, help="Scale for reconstruction loss.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate of optimizer.")
    parser.add_argument("--weight-decay", type=float, default=0.01, metavar="LAMBDA", help="Weight decay.")
    parser.add_argument("--max-epochs", type=int, default=150, metavar="EPOCHS", help="Number of epochs to run for each LDR model.")
    parser.add_argument("--checkpoint-epochs", type=int, default=5, metavar="EPOCHS", help="Epoch period of checkpoint saves. Set to -1 to not save checkpoints.")
    parser.add_argument("--load-checkpoint", action="store_true", help="Loads all model parameters from checkpoints.")
    parser.add_argument("--batch-size", type=int, default=256, metavar="SIZE", help="Global, i.e., across all processes, batch size")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    ### NOTE HARD-CODED!
    torch.set_num_threads(32)

    main(args)
