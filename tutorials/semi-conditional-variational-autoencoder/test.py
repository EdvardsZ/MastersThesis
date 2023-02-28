import torch

import torch.nn as nn

inverter = nn.Sequential(
    nn.ConvTranspose2d(2, hidden_dims[0], kernel_size=3, stride=2, padding=1, output_padding=1)
)