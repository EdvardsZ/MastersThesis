import torch
import torch.nn as nn
from models.decoders import Decoder
from models.encoders import Encoder
import torch.nn.functional as F
from loss import VAELoss

# Conventional VAE
class VAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(hidden_dims = hidden_dims, latent_dim = latent_dim)
        self.decoder = Decoder(hidden_dims= [256, 128], latent_dim = latent_dim)
        self.latent_dim = latent_dim

        self.weight_kl = 0.00025
        self.loss = VAELoss(weight_kl=self.weight_kl)

        
    def forward(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        output = self.decoder(z)
        return output, z_mean, z_log_var, z
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decoder(z)
        return samples
    