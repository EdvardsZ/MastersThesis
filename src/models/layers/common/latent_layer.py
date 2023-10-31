import torch.nn as nn
from models.helpers import sampling

class LatentLayer(nn.Module):
    def __init__(self, latent_dim):
        super(LatentLayer, self).__init__()
        self.latent_dim = latent_dim
        self.fc_mu = nn.LazyLinear(latent_dim)
        self.fc_var = nn.LazyLinear(latent_dim)
    
    def forward(self, x):
        flattened = nn.Flatten()(x)
        z_mean = self.fc_mu(flattened)
        z_log_var = self.fc_var(flattened)
        z = sampling(z_mean, z_log_var)

        return z, z_mean, z_log_var 
