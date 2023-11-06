import torch
import torch.nn as nn
from models.encoders import Encoder
from models.layers.common import LatentLayer
from typing import Tuple
from typing import List

class EncoderWithLatentLayer(nn.Module):
    def __init__(self, latent_dim: int, image_size: Tuple[int, int, int] = (1, 28, 28), hidden_dims: List[int] = [32, 64, 128, 256]):
        super(EncoderWithLatentLayer, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = Encoder(image_size, hidden_dims)
        self.latent = LatentLayer(latent_dim)

    def forward(self, x):
        feature_map = self.encoder(x)
        z, z_mean, z_log_var = self.latent(feature_map)
        return z, z_mean, z_log_var
