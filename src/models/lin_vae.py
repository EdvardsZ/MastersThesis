from models.encoders import LinearEncoder
from models.decoders import LinearDecoder
from models.helpers import reparameterize
import torch.nn as nn

class LinearVAE(nn.Module):
    def __init__(self, hidden_dims = [512, 256], latent_dim = 2):
        super(LinearVAE, self).__init__()
    
        image_size = (1, 28, 28)

        self.encoder = LinearEncoder(image_size= image_size, hidden_dims = hidden_dims, latent_dim = latent_dim)
        self.decoder = LinearDecoder(image_size= image_size, hidden_dims = [256, 512], latent_dim = latent_dim)
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var