import torch
import torch.nn as nn
from decoders import ConditionalDecoder
from encoders import Encoder

# CONDITIONAL VAE
class ConditionalVAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(ConditionalVAE, self).__init__()
        self.encoder = Encoder(hidden_dims = hidden_dims, latent_dim = latent_dim)
        self.decoder = ConditionalDecoder(hidden_dims= [256, 128], latent_dim = latent_dim)
        self.latent_dim = latent_dim

        # learn weight for KL loss through backprop
        self.weight_kl = 1
        self.weight_recon = 1

        
    def forward(self, inputs, cond_input):
        z_mean, z_log_var, z = self.encoder(inputs)
        output = self.decoder(z, cond_input)
        return output, z_mean, z_log_var, z
    
    def recon_loss(self, inputs, outputs):
        return F.mse_loss(inputs, outputs)
    
    def kl_loss(self, z_mean, z_log_var):
        return -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    
    def loss(self, inputs, outputs, z_mean, z_log_var):
        recon_loss = self.recon_loss(inputs, outputs)
        kl_loss = self.kl_loss(z_mean, z_log_var)
        return recon_loss, kl_loss, self.weight_recon * recon_loss + self.weight_kl * kl_loss
    
    def sample(self, num_samples, cond_input):
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decoder(z, cond_input)
        return samples
    