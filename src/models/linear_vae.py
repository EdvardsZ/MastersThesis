import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.linear_encoder import LinearEncoder
from models.decoders.linear_decoder import LinearDecoder


class LinearVAE(nn.Module):
    def __init__(self, image_size = (1, 28, 28), n_hidden=500, latent_dim=2, keep_prob=0.99):
        super(LinearVAE, self).__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.n_hidden = n_hidden
        self.keep_prob = keep_prob
        self.encoder = LinearEncoder(image_size = image_size, n_hidden = n_hidden, latent_dim = latent_dim, keep_prob = keep_prob)
        self.decoder = LinearDecoder(image_size = image_size, n_hidden = n_hidden, latent_dim = latent_dim, keep_prob = keep_prob)

        # TODO try to learn weight for KL loss through backprop
        self.weight_kl = 1
        self.weight_recon = 1

    def forward(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        output = self.decoder(z)
        return output, z_mean, z_log_var, z
    
    def recon_loss(self, inputs, outputs):
        return F.binary_cross_entropy(outputs, inputs, reduction='sum')
    
    def kl_loss(self, z_mean, z_log_var):
        return -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    
    def loss(self, inputs, outputs, z_mean, z_log_var):
        recon_loss = self.recon_loss(inputs, outputs)
        kl_loss = self.kl_loss(z_mean, z_log_var)
        return recon_loss, kl_loss, self.weight_recon * recon_loss + self.weight_kl * kl_loss
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decoder(z)
        return samples
    



    