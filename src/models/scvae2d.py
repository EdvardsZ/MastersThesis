from email.mime import image
import torch.nn as nn
from loss import VAELoss
from models.layers import EncoderWithLatentLayer, DecoderWithLatentLayer
from models.helpers import concat_latent_with_cond

class SCVAE2D(nn.Module):
    def __init__(self, latent_dim=2, image_shape=(1, 28, 28)):
        super(SCVAE2D, self).__init__()
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.encoder = EncoderWithLatentLayer(latent_dim=latent_dim, image_size=image_shape)
        self.decoder = DecoderWithLatentLayer(image_size=image_shape)

        self.pixel_decoder = DecoderWithLatentLayer(image_size=image_shape)

        self.loss = VAELoss(weight_kl=1.0)

    def forward(self, x, x_cond, y):
        print("TEST")
        z, z_mean, z_log_var = self.encoder(x)

        print("z shape: ", z.shape)


        output_1 = self.decoder(z)

        print("output 1 shape: ", output_1.shape)

        x_cat = concat_latent_with_cond(z, x_cond)

        output_2 = self.pixel_decoder(x_cat)

        print("output 2 shape: ", output_2.shape)

        return [output_1, output_2], z, z_mean, z_log_var

    def decode(self, z):
        return self.decoder(z)