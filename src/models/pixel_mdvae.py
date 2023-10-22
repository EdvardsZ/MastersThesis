import torch
import torch.nn as nn
from loss.vae_loss import VAELoss
from models.layers import EncoderWithLatentLayer, DecoderWithLatentLayer
from models.helpers import concat_latent_with_cond

class PixelMDVAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(PixelMDVAE, self).__init__()
        image_size_flattented = 28 * 28
        self.encoder = EncoderWithLatentLayer(latent_dim)

        self.decoder = DecoderWithLatentLayer()

        self.pixel_decoder = DecoderWithLatentLayer()

        self.latent_dim = latent_dim

        self.loss = VAELoss(loss_type='double')

    def forward(self, x, x_cond, y):
        z, z_mean, z_log_var = self.encoder(x)

        x_cat = concat_latent_with_cond(z, x_cond)

        output_0 = self.decoder(z)
        output_1 = self.pixel_decoder(x_cat)

        return [output_0, output_1], z_mean, z_log_var, z
    
    def decode(self, z):
        return self.decoder(z)
