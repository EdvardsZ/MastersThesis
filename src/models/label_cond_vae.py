import torch
import torch.nn as nn
from models.decoders import Decoder, LabelConditionedDecoder
from models.encoders import Encoder

from loss.vae_loss import VAELoss

class LabelConditionedVAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(LabelConditionedVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = LabelConditionedDecoder()
        self.latent_dim = latent_dim

        self.loss = VAELoss()
        
    def forward(self, inputs, label):
        z_mean, z_log_var, z = self.encoder(inputs)
        output = self.decoder(z, label)
        return output, z_mean, z_log_var, z