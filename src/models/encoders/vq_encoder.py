import torch
import torch.nn as nn
from models.layers.common import ResidualLayer

class VQEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim , hidden_dims = [128, 256], n_residual_layers = 6):
        super(VQEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim

        modules = []
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
        )

        for _ in range(n_residual_layers):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU()
            )
        )

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)