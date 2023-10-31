import torch.nn as nn
import torch
from loss import VAELoss
from models.layers import EncoderWithLatentLayer, DecoderWithLatentLayer
from models.helpers import concat_latent_with_cond

class SCVAE1D(nn.Module):
    def __init__(self, latent_dim=2):
        super(SCVAE1D, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = EncoderWithLatentLayer(latent_dim=latent_dim)
        self.pixel_decoder = DecoderWithLatentLayer()

        self.loss = VAELoss(weight_kl=1.0)

    def forward(self, x, x_cond, y):
        z, z_mean, z_log_var = self.encoder(x)

        x_cat = concat_latent_with_cond(z, x_cond)

        output = self.pixel_decoder(x_cat)

        return output, z, z_mean, z_log_var

    def decode(self, z):
        input_shape = self.pixel_decoder.get_input_shape()
        z_shape = z.shape
        zeros = torch.zeros((z_shape[0], input_shape[1] - z_shape[1]), device=z.device)
        x_cat = concat_latent_with_cond(z, zeros)
        return self.pixel_decoder(x_cat)