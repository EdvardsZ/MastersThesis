import torch.nn as nn
import torch
from loss import VAELoss
from models.layers import EncoderWithLatentLayer, DecoderWithLatentLayer
from models.helpers import concat_latent_with_cond
from .base_vae import BaseVAE
from models.outputs import VAEModelOutput

class SCVAE1D(BaseVAE):
    def __init__(self, latent_dim=2, image_shape=(1, 28, 28)):
        super(SCVAE1D, self).__init__(latent_dim, image_shape)
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.encoder = EncoderWithLatentLayer(latent_dim=latent_dim, image_size=image_shape)
        self.pixel_decoder = DecoderWithLatentLayer(image_size=image_shape)

        self.loss = VAELoss(weight_kl=1.0)

    def forward(self, x, x_cond, y) -> VAEModelOutput:
        z, z_mean, z_log_var = self.encoder(x)

        x_cat = concat_latent_with_cond(z, x_cond)

        output = self.pixel_decoder(x_cat)

        with torch.no_grad():
            self.pixel_decoder.eval()
            x_cat_masked = torch.zeros_like(x_cat, requires_grad=False)
            output_masked = self.pixel_decoder(x_cat_masked)
            self.pixel_decoder.train()

        return [output], [output_masked], z, z_mean, z_log_var

    def decode(self, z):
        input_shape = self.pixel_decoder.get_input_shape()
        z_shape = z.shape
        zeros = torch.zeros((z_shape[0], input_shape[1] - z_shape[1]), device=z.device)
        x_cat = concat_latent_with_cond(z, zeros)
        return self.pixel_decoder(x_cat)