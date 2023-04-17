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
        print(hidden_dims)
        feature_map_size = self.image_size[1:]
        for i, h_dim in enumerate(hidden_dims):
            print(feature_map_size)
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

        modules.append(nn.Flatten())

        resulting_size = in_channels * feature_map_size[0] * feature_map_size[1]

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(resulting_size, latent_dim)
        self.fc_var = nn.Linear(resulting_size, latent_dim)

    def forward(self, inputs):
        x = self.encoder(inputs)
        z_mean = self.fc_mu(x)
        z_log_var = self.fc_var(x)
        z = sampling(z_mean, z_log_var)
        
        return z_mean, z_log_var, z