import torch
import torch.nn as nn
from models.decoders import Decoder, PixelConditionedDecoder
from models.encoders import Encoder

from loss.vae_loss import VAELoss

class PixelConditionedVAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(PixelConditionedVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = PixelConditionedDecoder()
        self.latent_dim = latent_dim

        self.loss = VAELoss()
        
    def forward(self, inputs, cond_input):
        z_mean, z_log_var, z = self.encoder(inputs)
        output = self.decoder(z, cond_input)
        return output, z_mean, z_log_var, z