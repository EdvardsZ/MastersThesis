import torch.nn as nn
import torch
from models.helpers import sampling

class Encoder(nn.Module):
    def __init__(self, image_size=(1, 28, 28), hidden_dims=[32, 64, 128, 256], latent_dim=2):
        super(Encoder, self).__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        in_channels = self.image_size[0]

        modules = []

        # Build Encoder
        feature_map_size = self.image_size[1:]
        for h_dim in hidden_dims:
            stride = 2 if any(map(lambda x: x % 2 == 0, feature_map_size)) else 1

            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            if stride == 2:
                feature_map_size = tuple(map(lambda x: (x + 1) // 2, feature_map_size))


        self.encoder = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.encoder(inputs)
        