import torch
import torch.nn as nn
from models.decoders import Decoder, LabelConditionedDecoder
from models.encoders import Encoder
from loss.vae_loss import VAELoss

class LabelMDVAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(LabelMDVAE, self).__init__()
        self.encoder = Encoder()

        self.label_decoder = LabelConditionedDecoder()
        self.decoder = Decoder()

        self.latent_dim = latent_dim

        self.loss = VAELoss(weight_kl=1.0)

    def forward(self, inputs, label):
        z_mean, z_log_var, z = self.encoder(inputs)
        output_0 = self.decoder(z)
        output_1 = self.label_decoder(z, label)
        return [output_0, output_1], z_mean, z_log_var, z



