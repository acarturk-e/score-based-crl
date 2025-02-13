from collections import OrderedDict
from torch import nn


class DenseAutoencoder2(nn.Sequential):
    """Autoencoder from `d` input shape to `n` latent dimension"""
    def __init__(self, n: int, d: int, widths: list[int] = [256]):
        self.n = n
        self.d = d
        assert len(widths) > 0, "there is no such thing as MLP w/o hidden layers"
        self.widths = widths
        # Construct the encoder
        encoder = nn.Sequential()
        encoder.append(nn.Linear(d, widths[0]))
        encoder.append(nn.ReLU(inplace=False))
        encoder.append(nn.LayerNorm(widths[0]))
        for i in range(1, len(widths)):
            encoder.append(nn.Linear(widths[i-1], widths[i]))
            encoder.append(nn.ReLU(inplace=False))
            encoder.append(nn.LayerNorm(widths[i]))
        encoder.append(nn.Linear(widths[-1], n))
        # Construct the decoder using widths in reverse order
        decoder = nn.Sequential()
        decoder.append(nn.Linear(n, widths[-1]))
        decoder.append(nn.ReLU(inplace=False))
        decoder.append(nn.LayerNorm(widths[-1]))
        for i in reversed(range(0, len(widths)-1)):
            decoder.append(nn.Linear(widths[i+1], widths[i]))
            decoder.append(nn.ReLU(inplace=False))
            decoder.append(nn.LayerNorm(widths[i]))
        decoder.append(nn.Linear(widths[0], d))
        super(DenseAutoencoder2, self).__init__(OrderedDict((
            ("encoder", encoder),
            ("decoder", decoder),
        )))


class DenseAutoencoder(nn.Sequential):
    """Autoencoder for 64x64 RGB images with `n` latent dimensions"""
    def __init__(self, n: int):
        self.n = n
        od = OrderedDict()
        od["encoder"] = nn.Sequential(
            # 3 x 64^2
            nn.Flatten(1, -1),
            # 3 * 64^2
            nn.Linear(3 * 64 * 64, 256),
            nn.ReLU(inplace=False),
            # nn.Dropout1d(p=0.1, inplace=False),
            nn.LayerNorm(256),
            # 256
            nn.Linear(256, n),
            # n
        )
        od["decoder"] = nn.Sequential(
            # n
            nn.Linear(n, 256),
            nn.ReLU(False),
            # nn.Dropout1d(p=0.1, inplace=False),
            nn.LayerNorm(256),
            # 256
            nn.Linear(256, 3 * 64 * 64),
            # 3 * 64^2
            nn.Unflatten(-1, (3, 64, 64)),
            nn.Sigmoid()
            # 3 x 64^2
        )
        super(DenseAutoencoder, self).__init__(od)

# CNN-based autoencoder model with dropout
class CnnAutoencoder(nn.Sequential):
    """Autoencoder for 64x64 RGB images with `n` latent dimensions"""
    def __init__(self, n: int):
        self.n = n
        od = OrderedDict()
        od["encoder"] = nn.Sequential(
            # 3 x 64^2
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(False),
            nn.MaxPool2d(2, 2),
            # 64 x 32^2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(False),
            nn.MaxPool2d(2, 2),
            # 128 x 16^2
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(False),
            nn.MaxPool2d(2, 2),
            # 256 x 8^2
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, track_running_stats=False),
            nn.ReLU(False),
            nn.MaxPool2d(2, 2),
            # 512 x 4^2
            nn.Flatten(1, -1),
            # 2 ^ 13
            nn.Linear(512 * 4 * 4, n)
            # n
        )
        od["decoder"] = nn.Sequential(
            # n
            nn.Linear(n, 512 * 4 * 4),
            # 2 ^ 13
            nn.Unflatten(-1, (512, 4, 4)),
            # 512 x 4^2
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(False),
            # 256 x 8^2
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(False),
            # 128 x 16^2
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(False),
            # 64 x 32^2
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            # 3 x 64^2
        )
        super(CnnAutoencoder, self).__init__(od)

class LinearAutoencoder(nn.Sequential):
    """Linear autoencoder model with `n` latent dimensions"""
    def __init__(self, n: int, d: int):
        self.n = n
        od = OrderedDict()
        od["encoder"] = nn.Linear(d, n)
        od["decoder"] = nn.Linear(n, d)
        super(LinearAutoencoder, self).__init__(od)
