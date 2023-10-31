import torch.nn as nn
from models.encoders import VQEncoder
from models.layers.common.latent_layer import LatentLayer

class EncoderWithLatentLayer(nn.Module):
    def __init__(self, latent_dim):
        super(EncoderWithLatentLayer, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = VQEncoder(in_channels=1, embedding_dim=256, n_residual_layers=1)
        self.latent = LatentLayer(latent_dim)

    def forward(self, x):
        feature_map = self.encoder(x)
        z, z_mean, z_log_var = self.latent(feature_map)
        return z, z_mean, z_log_var
