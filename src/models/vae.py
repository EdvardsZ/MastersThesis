import torch
import torch.nn as nn
from models.decoders import Decoder
from models.encoders import Encoder
import torch.nn.functional as F
from loss import VAELoss, SoftAdaptVAELoss
from models.helpers import sampling
from models.layers import LatentLayer, LatentToFeatureMap


class VAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim)

        self.latent = LatentLayer(latent_dim)
        self.latentToFeatureMap = LatentToFeatureMap(latent_dim=latent_dim, feature_map_size=7, num_channels=256)

        self.decoder = Decoder(latent_dim=latent_dim)

        #self.loss = SoftAdaptVAELoss(n = 100, variant=["Normalized", "Loss Weighted"])
        self.loss = VAELoss(weight_kl=1.0)
        
    def forward(self, x, x_cond, y):

        feature_map = self.encoder(x)

        z, z_mean, z_log_var = self.latent(feature_map)

        feature_map = self.latentToFeatureMap(z)

        output = self.decoder(feature_map)
        
        return output, z_mean, z_log_var, z
    
    def decode(self, z):
        decoder_input = self.decoder_input(z)
        return self.decoder(decoder_input)