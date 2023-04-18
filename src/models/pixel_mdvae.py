import torch
import torch.nn as nn
from models.decoders import Decoder, PixelConditionedDecoder
from models.encoders import Encoder
from loss.vae_loss import VAELoss

class PixelMDVAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(PixelMDVAE, self).__init__()
        self.encoder = Encoder()

        self.pixel_decoder = PixelConditionedDecoder()
        self.decoder = Decoder()

        self.latent_dim = latent_dim

        self.loss = VAELoss(loss_type='double')

    def forward(self, inputs, cond_input):
        z_mean, z_log_var, z = self.encoder(inputs)
        output_0 = self.decoder(z)
        output_1 = self.pixel_decoder(z, cond_input)
        return [output_0, output_1], z_mean, z_log_var, z
    def decode(self, z):
        return self.decoder(z)
