import torch.nn as nn
from models.helpers import get_encoder_stride_sizes
from typing import List, Tuple


class Encoder(nn.Module):
    def __init__(self, image_size : Tuple[int, int, int], hidden_dims : List[int] =[32, 64, 128, 256]):
        super(Encoder, self).__init__()

        self.image_size = image_size
        self.hidden_dims = hidden_dims
        
        in_channels = self.image_size[0]

        modules = []


        stride_sizes = get_encoder_stride_sizes(image_size[1], len(hidden_dims))

        for i, h_dim in enumerate(hidden_dims):
            stride = stride_sizes[i]

            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim


        self.encoder = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.encoder(inputs)
        