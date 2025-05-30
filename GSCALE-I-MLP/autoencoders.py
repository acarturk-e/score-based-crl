"""NOTE: Dropout introduces randomness to the forward pass,
which is a problem during Jacobian computation. Therefore, it is not
included in these models."""

from collections import OrderedDict
from torch import nn


class DenseAutoencoder(nn.Sequential):
    """Autoencoder with dense layers"""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 256, n_layers: int = 3
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        od = OrderedDict()
        encoder_layers = [
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.LayerNorm(hidden_dim),
        ]
        for _ in range(n_layers - 1):
            encoder_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=False),
                    nn.LayerNorm(hidden_dim),
                ]
            )
        encoder_layers.append(nn.Linear(hidden_dim, input_dim))
        od["encoder"] = nn.Sequential(*encoder_layers)
        decoder_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.LayerNorm(hidden_dim),
        ]
        for _ in range(n_layers - 1):
            decoder_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=False),
                    nn.LayerNorm(hidden_dim),
                ]
            )
        decoder_layers.append(nn.Linear(hidden_dim, output_dim))
        od["decoder"] = nn.Sequential(*decoder_layers)
        super(DenseAutoencoder, self).__init__(od)


class DenseAutoencoderTanh(nn.Sequential):
    """Autoencoder with dense layers and final (encoder) layer with tanh activation"""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 256, n_layers: int = 3
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        od = OrderedDict()
        encoder_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.LayerNorm(hidden_dim),
        ]
        for _ in range(n_layers - 1):
            encoder_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=False),
                    nn.LayerNorm(hidden_dim),
                ]
            )
        encoder_layers.append(nn.Linear(hidden_dim, output_dim))
        encoder_layers.append(nn.Softmax(dim=-1))
        od["encoder"] = nn.Sequential(*encoder_layers)
        decoder_layers = [
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.LayerNorm(hidden_dim),
        ]
        for _ in range(n_layers - 1):
            decoder_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=False),
                    nn.LayerNorm(hidden_dim),
                ]
            )
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        od["decoder"] = nn.Sequential(*decoder_layers)
        super(DenseAutoencoderTanh, self).__init__(od)
