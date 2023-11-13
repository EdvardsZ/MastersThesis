import torch
import torch.nn as nn
from models.layers.common import ResidualLayer

class SimpleVQEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super(SimpleVQEncoder, self).__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, 1, padding=0)
        )

    def forward(self, x):
        return self.encoder(x)