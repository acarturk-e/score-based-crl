import argparse
import os
import pickle

import numpy as np

from scm.box import BoxSCM
from utils import Renderer


def main(args: argparse.Namespace):
    rng = np.random.default_rng(args.global_seed)

    # Model parameters
    num_balls: int = args.num_balls
    n = 2 * num_balls
    degree: int = args.graph_degree
    box_size = 0.1
    intervention_order = rng.permutation(n)
    fill_rate = n * degree / ((n * (n - 1)) / 2)

    n_samples: int = args.num_samples

    width: int = args.image_size
    height = width
    ball_radius = int(width * 0.08)

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Models
    scm = BoxSCM(n, fill_rate, box_size=box_size, np_rng=rng)
    renderer = Renderer(num_balls, width, height, ball_radius, rng=rng)

    # Save the configuration
    with open(
        os.path.join(data_dir, "generate_data_cfg.pkl"),
        "wb", pickle.HIGHEST_PROTOCOL
    ) as f:
        pickle.dump({
            "num_balls": num_balls,
            "n": n,
            "degree": degree,
            "box_size": box_size,
            "intervention_order": intervention_order,
            "n_samples": n_samples,
            "width": width,
            "height": height,
            "ball_radius": ball_radius,
            "data_dir": data_dir,
            "scm": scm,
        }, f)

    # Sample latent variables
    zs_obs = scm.sample((n_samples,))[..., 0]
    zs_hard_1 = np.stack([scm.sample((n_samples,), [i], 'hard int', 'scale', 0.25)[..., 0] for i in intervention_order])
    zs_hard_2 = np.stack([scm.sample((n_samples,), [i], 'hard int', 'scale', 4.00)[..., 0] for i in intervention_order])

    assert zs_obs.shape == (n_samples, n)
    assert np.all(np.abs(zs_obs) <= box_size)

    # Re-scale to [0, 1] interval
    zs_obs = 0.5 * (1.0 + zs_obs / box_size)
    zs_hard_1 = 0.5 * (1.0 + zs_hard_1 / box_size)
    zs_hard_2 = 0.5 * (1.0 + zs_hard_2 / box_size)

    # "Decode" the latents into images
    xs_obs = renderer.render_n_balls(zs_obs)
    xs_hard_1 = renderer.render_n_balls(zs_hard_1)
    xs_hard_2 = renderer.render_n_balls(zs_hard_2)

    # save the data
    np.savez_compressed(
        os.path.join(data_dir, "z_and_x.npz"),
        zs_obs=zs_obs,
        zs_hard_1=zs_hard_1,
        zs_hard_2=zs_hard_2,
        xs_obs=xs_obs,
        xs_hard_1=xs_hard_1,
        xs_hard_2=xs_hard_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("num_balls", type=int, help="Number of balls to draw. Latent dimenion is twice this number.")
    parser.add_argument("num_samples", type=int, help="Number of samples to draw from the latent graph and transform to images.")
    parser.add_argument("data_dir", type=str, metavar="DIR", help="Directory to store data and logs")
    parser.add_argument("--graph-degree", type=int, default=2, metavar="D", help="degree d in ER(n, d) ")
    parser.add_argument("--image-size", type=int, default=64, metavar="SIZE", help="Size of the rendered RGB images. Aspect ratio is always 1.")
    parser.add_argument("--global-seed", type=int, default=9724, metavar="SEED", help="Seed for controlling randomness")
    args = parser.parse_args()

    main(args)
