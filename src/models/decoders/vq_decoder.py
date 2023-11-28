import torch.nn as nn
from models.layers.common import ResidualLayer

class VQDecoder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, hidden_dims = [256, 128], n_residual_layers = 6):
        super(VQDecoder, self).__init__()
        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[0],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(n_residual_layers):
            modules.append(ResidualLayer(hidden_dims[0], hidden_dims[0]))

        modules.append(nn.LeakyReLU())

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=in_channels,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.decoder(x)