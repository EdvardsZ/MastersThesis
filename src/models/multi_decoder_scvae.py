import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decoders import ConditionalDecoder
from models.decoders import Decoder
from models.encoders import Encoder

# Multi decoder conditional VAE
class MultiDecoderConditionalVAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(MultiDecoderConditionalVAE, self).__init__()
        self.encoder = Encoder(hidden_dims = hidden_dims, latent_dim = latent_dim)
        self.conditional_decoder = ConditionalDecoder(hidden_dims= [256, 128], latent_dim = latent_dim)
        self.decoder = Decoder(hidden_dims= [256, 128], latent_dim = latent_dim)
        self.latent_dim = latent_dim

        # TODO try to learn weight for KL loss through backprop
        self.weight_kl = 1
        self.weight_recon = 1

        
    def forward(self, inputs, cond_input):
        z_mean, z_log_var, z = self.encoder(inputs)
        output = self.conditional_decoder(z, cond_input)
        output_2 = self.decoder(z)
        return output, output_2, z_mean, z_log_var, z
    
    def recon_loss(self, inputs, outputs):
        return F.mse_loss(inputs, outputs)
    
    def kl_loss(self, z_mean, z_log_var):
        return -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    
    def loss(self, inputs, outputs, output_2, z_mean, z_log_var):
        recon_loss = self.recon_loss(inputs, outputs)
        recon_loss_2 = self.recon_loss(inputs, output_2)
        kl_loss = self.kl_loss(z_mean, z_log_var)
        return recon_loss, recon_loss_2, kl_loss, self.weight_recon * recon_loss + self.weight_recon * recon_loss_2 + self.weight_kl * kl_loss
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decoder(z)
        return samples
    