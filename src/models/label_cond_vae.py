import torch
import torch.nn as nn
from models.layers import EncoderWithLatentLayer, DecoderWithLatentLayer
from models.helpers import concat_latent_with_cond

from loss.vae_loss import VAELoss

class LabelConditionedVAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(LabelConditionedVAE, self).__init__()
        self.classes_count = 10
        self.encoder = EncoderWithLatentLayer(latent_dim)
        self.decoder = DecoderWithLatentLayer()
        self.latent_dim = latent_dim

        self.loss = VAELoss()
        
    def forward(self, x, x_cond, y):
        z_mean, z_log_var, z = self.encoder(x)

        x_cat = concat_latent_with_cond(z, y)

        output = self.decoder(x_cat)
        return output, z_mean, z_log_var, z