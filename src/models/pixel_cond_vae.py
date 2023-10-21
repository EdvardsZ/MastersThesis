import torch
import torch.nn as nn
from models.decoders import Decoder, PixelConditionedDecoder
from models.encoders import Encoder
from models.layers import EncoderWithLatentLayer, DecoderWithLatentLayer
from models.helpers import concat_latent_with_cond

from loss.vae_loss import VAELoss

class PixelConditionedVAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(PixelConditionedVAE, self).__init__()
        image_size_flattented = 28 * 28
        self.encoder = EncoderWithLatentLayer(latent_dim)
        self.decoder = DecoderWithLatentLayer(latent_dim, input_size=latent_dim + image_size_flattented)
        self.latent_dim = latent_dim

        self.loss = VAELoss()
        
    def forward(self, x, x_cond, y):
        z_mean, z_log_var, z = self.encoder(x)

        x_cat = concat_latent_with_cond(z, x_cond)

        output = self.decoder(x_cat)
        return output, z_mean, z_log_var, z