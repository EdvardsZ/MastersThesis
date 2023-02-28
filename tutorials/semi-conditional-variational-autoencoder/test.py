import torch

import torch.nn as nn

image = torch.randn(128, 256, 7, 7)

inverter = nn.Sequential(
    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.Conv2d(128, out_channels= 1, kernel_size= 3, padding= 1),
)



print(inverter(image).shape)