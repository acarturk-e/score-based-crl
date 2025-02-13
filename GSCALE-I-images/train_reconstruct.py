import argparse
import logging
import os

import numpy as np

import torch
from torch import Tensor

# Setting these flags True makes A100 training a lot faster:
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# For data
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Local imports
from autoencoders import CnnAutoencoder, DenseAutoencoder


def get_x(data_dir: str) -> tuple[int, Tensor]:
    """Returns tuple `(zs_obs, xs_obs)`

    Shapes: `(nsamples, n)`, `(nsamples, 3, w, h)`,
    where `w` and `h` are image width and height.
    
    NOTE: `x` data is as images i.e. uint8. I map them to [0, 1] float32."""
    data = np.load(os.path.join(data_dir, "z_and_x.npz"))
    n = data["zs_obs"].shape[1]
    xs_obs = data["xs_obs"]
    xs_obs = torch.from_numpy(xs_obs).float() / 255.0
    logging.debug(f"{xs_obs.shape = }")

    return n, xs_obs


def create_logger(log_dir: str, latent_dim: int) -> logging.Logger:
    """Create a logger that writes to a log file (DEBUG level) and stdout (INFO level)"""
    os.makedirs(log_dir, exist_ok=True)
    h_out = logging.StreamHandler()
    h_out.setLevel(logging.INFO)
    h_file = logging.FileHandler(os.path.join(log_dir, f"train_autoenc_reconstruct_{latent_dim}_cpu.log"), mode="w")
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

    latent_dim: int = args.latent_dim

    # Set your random seed for experiment reproducibility.
    if args.seed is not None: torch.manual_seed(args.seed)

    logger = create_logger(args.data_dir, latent_dim)
    logger.info(f"Experiment directory created at {args.data_dir}")

    # Read data and create distributive data loader
    n, xs_obs = get_x(args.data_dir)
    n_samples = xs_obs.shape[0]
    image_shape = xs_obs.shape[1:]
    [train_dataset, valid_dataset] = random_split(TensorDataset(xs_obs), [0.9, 0.1])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False)


    # Create model
    autoenc = CnnAutoencoder(latent_dim).to(device)

    if args.load_checkpoint:
        autoenc.load_state_dict(torch.load(os.path.join(args.data_dir, f"autoenc_reconstruct_{latent_dim}.pth"), weights_only=True))

    encoder = autoenc.get_submodule("encoder")
    decoder = autoenc.get_submodule("decoder")

    opt = torch.optim.Adam(
        autoenc.parameters(),
        lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=(args.max_epochs // 5) + 1, T_mult=4)


    for epoch in range(args.max_epochs):

        # Training
        autoenc.train()
        running_loss = log_steps = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            assert isinstance(xb, Tensor)

            zhatb = encoder(xb)
            xhatb = decoder(zhatb)
            assert isinstance(xhatb, Tensor)

            loss = (xb - xhatb).pow(2).mean(0).sum()
            loss.backward()
            opt.step()

            log_steps += 1
            running_loss += loss.item()

        avg_loss = running_loss / log_steps

        # End of training
        logger.info(f"(step={epoch}), Train Loss: {avg_loss:.5f}")

        # Validation
        autoenc.eval()
        running_loss = log_steps = 0
        for (xb,) in valid_loader:
            xb = xb.to(device)
            assert isinstance(xb, Tensor)

            zhatb = encoder(xb)
            xhatb = decoder(zhatb)
            assert isinstance(xhatb, Tensor)

            with torch.no_grad():
                loss = (xb - xhatb).pow(2).mean(0).sum()

            log_steps += 1
            running_loss += loss.item()

        avg_loss = running_loss / log_steps

        # End of validation
        lr_sched.step()

        logger.info(f"(step={epoch}), Validation Loss: {avg_loss:.5f}")
        if args.checkpoint_epochs != -1 and epoch % args.checkpoint_epochs == 0:
            torch.save(autoenc.state_dict(), os.path.join(args.data_dir, f"autoenc_reconstruct_{latent_dim}.pth"))

    autoenc.eval()
    autoenc.requires_grad_(False)
    torch.save(autoenc.state_dict(), os.path.join(args.data_dir, f"autoenc_reconstruct_{latent_dim}.pth"))
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", type=str, metavar="DIR", help="Directory to store data and logs.")
    parser.add_argument("--latent-dim", type=int, metavar="DIM", default=64, help="Dimension of the autoencoder latent space.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, metavar="LAMBDA", help="Weight decay.")
    parser.add_argument("--max-epochs", type=int, default=100, metavar="EPOCHS", help="Number of epochs to run for 1st step.")
    parser.add_argument("--checkpoint-epochs", type=int, default=5, metavar="EPOCHS", help="Epoch period of checkpoint saves. Set to -1 to not save checkpoints.")
    parser.add_argument("--load-checkpoint", action="store_true", help="Loads all model parameters from checkpoints.")
    parser.add_argument("--batch-size", type=int, default=256, metavar="SIZE", help="Global, i.e., across all processes, batch size")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    ### NOTE HARD-CODED!
    torch.set_num_threads(32)

    main(args)
