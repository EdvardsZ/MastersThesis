import torch
import torch.nn as nn
from loss.vae_loss import VAELoss
from models.layers import EncoderWithLatentLayer, DecoderWithLatentLayer, ClassificationLayer

class LabelMTVAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(LabelMTVAE, self).__init__()
        self.encoder = EncoderWithLatentLayer(latent_dim=latent_dim)
        self.decoder = DecoderWithLatentLayer()

        self.classification = ClassificationLayer(num_classes=10)
        self.latent_dim = latent_dim

        self.loss = VAELoss(loss_type='double')

    def forward(self, x, x_cond, y):
        z, z_mean, z_log_var = self.encoder(x)

        output = self.decoder(z)

        classification = self.classification(z)

        return output, z_mean, z_log_var, z, classification
    def decode(self, z):
        return self.decoder(z)
