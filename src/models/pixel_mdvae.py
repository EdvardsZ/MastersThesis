import torch
import torch.nn as nn
from models.decoders import Decoder, PixelConditionedDecoder
from models.encoders import Encoder
from loss.vae_loss import VAELoss
from models.layers import EncoderWithLatentLayer, DecoderWithLatentLayer

class PixelMDVAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(PixelMDVAE, self).__init__()
        image_size_flattented = 28 * 28
        self.encoder = EncoderWithLatentLayer(latent_dim)

        self.decoder = DecoderWithLatentLayer(latent_dim)

        self.pixel_decoder = DecoderWithLatentLayer(latent_dim, input_size=latent_dim + image_size_flattented)

        self.latent_dim = latent_dim

        self.loss = VAELoss(loss_type='double')

    def forward(self, x, x_cond, y):
        z_mean, z_log_var, z = self.encoder(x)

        x_cond = torch.flatten(x_cond, start_dim=1)
        x_cat = torch.cat((z, x_cond), dim=1)

        output_0 = self.decoder(z)
        output_1 = self.pixel_decoder(x_cat)
        return [output_0, output_1], z_mean, z_log_var, z
    def decode(self, z):
        return self.decoder(z)
