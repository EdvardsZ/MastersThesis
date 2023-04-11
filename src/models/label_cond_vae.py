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

        self.loss = VAELoss(weight_kl=1.0)
        
    def forward(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        output = self.decoder(z)
        return output, z_mean, z_log_var, z