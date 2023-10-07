import torch
import torch.nn as nn
from .MaskedCNN import MaskedCNN


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        divided = out_channels/2

        self.res = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=1),
            # nn.ReLU(),
            # MaskedCNN('B', out_channels, (out_channels / 2), kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d((out_channels / 2), out_channels, kernel_size=1),
            # nn.ReLU()

            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            MaskedCNN('B',128,64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),

        )

    def forward(self, x):
        result = self.res(x)
        return result
