import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import VAELoss, SoftAdaptVAELoss
from models.helpers import sampling
from models.layers import EncoderWithLatentLayer, DecoderWithLatentLayer


class VAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = EncoderWithLatentLayer(latent_dim=latent_dim)

        self.decoder = DecoderWithLatentLayer()

        #self.loss = SoftAdaptVAELoss(n = 100, variant=["Normalized", "Loss Weighted"])
        self.loss = VAELoss(weight_kl=1.0)
        
    def forward(self, x, x_cond, y):

        z, z_mean, z_log_var = self.encoder(x)

        output = self.decoder(z)

        return output, z_mean, z_log_var, z
    
    def decode(self, z):
        return self.decoder(z)