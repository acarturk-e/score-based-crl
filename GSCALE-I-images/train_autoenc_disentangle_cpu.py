import argparse
import logging
import os

# For DDP utils
from socket import gethostname

import numpy as np

import torch
from torch import Tensor
from torch.func import jacfwd  # type: ignore

# Setting these flags True makes A100 training a lot faster:
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# For ddp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# For data
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Local imports
from autoencoders import DenseAutoencoder2 as Autoencoder
import utils


def get_data(data_dir: str, latent_dim: int) -> tuple[Tensor, Tensor]:
    """Returns tuple `(ys_obs, dsys_bw_hards)`"""
    ys_obs = torch.load(os.path.join(data_dir, f"ys_obs_{latent_dim}.pth"), weights_only=True)
    logging.info(f"Loaded y data with shape {ys_obs.shape = }")
    dsys_bw_hards = torch.load(os.path.join(data_dir, f"dsys_bw_hards_{latent_dim}.pth"), weights_only=True)
    logging.info(f"Loaded dsy data with shape {dsys_bw_hards.shape = }")
    return ys_obs, dsys_bw_hards


def get_z_data(data_dir: str) -> Tensor:
    """Returns zs_obs"""
    data = np.load(os.path.join(data_dir, "z_and_x.npz"))
    zs_obs = data["zs_obs"]
    zs_obs = torch.from_numpy(zs_obs).float()
    return zs_obs


def create_logger(log_dir: str) -> logging.Logger:
    """Create a logger that writes to a log file (DEBUG level) and stdout (INFO level)"""
    os.makedirs(log_dir, exist_ok=True)
    h_out = logging.StreamHandler()
    h_out.setLevel(logging.INFO)
    h_file = logging.FileHandler(os.path.join(log_dir, "train_autoenc_disentangle2_cpu.log"), mode="w")
    h_file.setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(module)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[h_out, h_file])
    logger = logging.getLogger(__name__)
    return logger


def main(
    rank: int, # Your node rank
    args: argparse.Namespace
):
    data_dir: str = args.data_dir
    latent_dim: int = args.latent_dim

    assert args.global_batch_size % dist.get_world_size() == 0, "Global batch size must split evenly among ranks."
    lambda1: float = args.lambda1

    # Set your random seed for experiment reproducibility.
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    logger = None
    if rank == 0:
        logger = create_logger(data_dir)
        logger.info(f"Experiment directory created at {data_dir}")
        logger.info(f"Batch size per rank: {args.global_batch_size // dist.get_world_size()}")

    # Read data and create distributive data loader
    ys_obs, dsys_bw_hards = get_data(data_dir, latent_dim)

    # Get z data for validation MCC
    zs_obs = get_z_data(data_dir)

    # One intervention pair per latent dimension
    n = dsys_bw_hards.shape[0]

    # Note the `movedim`: TensorDataset, understandably, requires the sample dimension to come first
    [train_dataset, valid_dataset] = random_split(TensorDataset(ys_obs, dsys_bw_hards.moveaxis(0, 1)), [0.9, 0.1])

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed)
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False, # IMPORTANT set this to "False" since sampler's shuffle is True
        sampler=train_sampler,
        num_workers=args.num_workers, # This should be equal to the number of CPUs set per task
        pin_memory=True,
        drop_last=True) # Set "True" to prevent uneven splits
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=valid_sampler,
        num_workers=args.num_workers, # This should be equal to the number of CPUs set per task
        pin_memory=True,
        drop_last=True) # Set "True" to prevent uneven splits


    # Create model
    autoenc = Autoencoder(n, latent_dim)

    if args.load_checkpoint:
        autoenc.load_state_dict(torch.load(os.path.join(args.data_dir, f"autoenc_disentangle2_{latent_dim}.pth"), weights_only=True))

    encoder = DDP(autoenc.get_submodule("encoder"))
    decoder = DDP(autoenc.get_submodule("decoder"))

    opt = torch.optim.Adam(
        autoenc.parameters(),
        lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100])


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
        jb = torch.zeros(yb.shape + (n,))
        for (i, zhati) in enumerate(zhatb):
            ji = jacfwd(decoder.forward)(zhati.unsqueeze(0))
            assert isinstance(ji, Tensor)
            jb[i] = ji.squeeze(0, -2)

        dszhatb = dsyb @ jb
        dt = dszhatb.abs().mean(0)
        loss_main = (dt - torch.eye(n)).abs().sum()

        loss_reconstr = lambda1 * (yhatb - yb).pow(2).mean(0).sum()
        loss = loss_main + loss_reconstr
        return loss, loss_reconstr.detach(), loss_main.detach()


    for epoch in range(args.max_epochs):

        # Training
        autoenc.train()
        running_loss = running_loss_reconstr = running_loss_main = log_steps = 0
        train_sampler.set_epoch(epoch)
        for (yb, dsyb) in train_loader:
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

        avg_loss = torch.tensor(running_loss / log_steps)
        avg_loss_reconstr = torch.tensor(running_loss_reconstr / log_steps)
        avg_loss_main = torch.tensor(running_loss_main / log_steps)

        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_loss_reconstr, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_loss_main, op=dist.ReduceOp.SUM)

        avg_loss = avg_loss.item() / dist.get_world_size()
        avg_loss_reconstr = avg_loss_reconstr.item() / dist.get_world_size()
        avg_loss_main = avg_loss_main.item() / dist.get_world_size()

        # End of training
        dist.barrier()

        if rank == 0:
            assert logger is not None
            logger.info(f"({epoch=}), Train Loss: {avg_loss:.5f} (reconstr={avg_loss_reconstr:.5f}, main={avg_loss_main:.5f})")

        # Validation
        autoenc.eval()
        running_loss = log_steps = 0
        valid_sampler.set_epoch(epoch)
        for (yb, dsyb) in valid_loader:
            assert isinstance(yb, Tensor)
            assert isinstance(dsyb, Tensor)

            with torch.no_grad():
                loss, loss_reconstr, loss_main = loss_fn(yb, dsyb)

            log_steps += 1
            running_loss += loss.item()
            running_loss_reconstr += loss_reconstr.item()
            running_loss_main += loss_main.item()

        avg_loss = torch.tensor(running_loss / log_steps)
        avg_loss_reconstr = torch.tensor(running_loss_reconstr / log_steps)
        avg_loss_main = torch.tensor(running_loss_main / log_steps)

        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_loss_reconstr, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_loss_main, op=dist.ReduceOp.SUM)

        avg_loss = avg_loss.item() / dist.get_world_size()
        avg_loss_reconstr = avg_loss_reconstr.item() / dist.get_world_size()
        avg_loss_main = avg_loss_main.item() / dist.get_world_size()

        # End of validation
        dist.barrier()
        lr_sched.step()

        if rank == 0:
            assert logger is not None
            logger.info(f"({epoch=}), Validation Loss: {avg_loss:.5f} (reconstr={avg_loss_reconstr:.5f}, main={avg_loss_main:.5f})")
            if args.checkpoint_epochs != -1 and epoch % args.checkpoint_epochs == 0:
                torch.save(autoenc.state_dict(), os.path.join(args.data_dir, f"autoenc_disentangle2_{latent_dim}.pth"))

                # also add a MCC check for every checkpoint.
                zhats_obs = autoenc.encoder(ys_obs)
                assert isinstance(zhats_obs, Tensor)
                z_mcc = utils.mcc(zhats_obs.detach().cpu().numpy(), zs_obs.detach().cpu().numpy())
                logger.info(f"(MCC={z_mcc:.4f})")

    dist.barrier()
    autoenc.eval()
    autoenc.requires_grad_(False)
    if rank == 0:
        assert logger is not None
        torch.save(autoenc.state_dict(), os.path.join(args.data_dir, f"autoenc_disentangle2_{latent_dim}.pth"))
        logger.info("Done!")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", type=str, metavar="DIR", help="Directory to store data and logs.")
    parser.add_argument("latent_dim", type=int, metavar="DIM", help="Dimension of the dim-reduced representations.")
    parser.add_argument("--lambda1", type=float, default=1.0, help="Scale for reconstruction loss.")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate of optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, metavar="LAMBDA", help="Weight decay.")
    parser.add_argument("--max-epochs", type=int, default=250, metavar="EPOCHS", help="Number of epochs to run for each LDR model.")
    parser.add_argument("--checkpoint-epochs", type=int, default=10, metavar="EPOCHS", help="Epoch period of checkpoint saves. Set to -1 to not save checkpoints.")
    parser.add_argument("--load-checkpoint", action="store_true", help="Loads all model parameters from checkpoints.")
    parser.add_argument("--num-workers", type=int, default=8, metavar="N", help="Number of CPUs per process")
    parser.add_argument("--global-batch-size", type=int, default=128, metavar="SIZE", help="Global, i.e., across all processes, batch size")
    parser.add_argument("--global-seed", type=int, default=9724)
    args = parser.parse_args()

    rank          = int(os.environ["SLURM_PROCID"])
    world_size    = int(os.environ["WORLD_SIZE"])
    file_store    = dist.FileStore(os.path.join(args.data_dir, "_train_autoenc_disentangle2_file_store"), 1)  # type: ignore

    print(f"Hello from rank {rank} of {world_size} on {gethostname()}", flush=True)

    dist.init_process_group("gloo", store=file_store, rank=rank, world_size=world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    main(rank, args)
