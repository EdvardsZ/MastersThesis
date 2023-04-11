from models.encoders import LinearEncoder
from models.decoders import LinearDecoder
from loss.vae_loss import VAELoss
import torch.nn as nn

class LinearVAE(nn.Module):
    def __init__(self, hidden_dims = [512, 256], latent_dim = 2):
        super(LinearVAE, self).__init__()
    
        image_size = (1, 28, 28)

        self.encoder = LinearEncoder(image_size= image_size, hidden_dims = hidden_dims, latent_dim = latent_dim)
        self.decoder = LinearDecoder(image_size= image_size, hidden_dims = [256, 512], latent_dim = latent_dim)

        self.loss = VAELoss(weight_kl=1)
    
    def forward(self, x):
        mu, log_var, z = self.encoder(x)
        return self.decoder(z), mu, log_var, z