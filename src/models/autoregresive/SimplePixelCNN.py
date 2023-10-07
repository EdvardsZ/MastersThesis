import torch
import torch.nn as nn
from .MaskedCNN import MaskedCNN
from .ResidualBlock import ResidualBlock

class SimplePixelCNN(nn.Module):
    def __init__(self):
        super(SimplePixelCNN, self).__init__()
        self.res = nn.Sequential(
            MaskedCNN('A', 1, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            ResidualBlock(in_channels=128, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128),
            MaskedCNN('B', 128, 128, kernel_size=1),
            nn.ReLU(),
            MaskedCNN('B', 128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1)
        )

    def forward(self, x):
        x = self.res(x)
        return x