import torch.nn as nn
from loss import VAELoss
from models.layers import EncoderWithLatentLayer, DecoderWithLatentLayer


class VAE(nn.Module):
    def __init__(self, latent_dim=2, image_shape=(1, 28, 28)):
        super(VAE, self).__init__()
        self.image_shape = image_shape
        hidden_dims = [32, 64, 128, 256] #for now lets keep this hyperparam a constant
        self.latent_dim = latent_dim
        
        self.encoder = EncoderWithLatentLayer(latent_dim=latent_dim, image_size=image_shape, hidden_dims=hidden_dims)
        hidden_dims.reverse()
        self.decoder = DecoderWithLatentLayer(image_size=image_shape, hidden_dims=hidden_dims)

        self.loss = VAELoss(weight_kl=1.0)
        
    def forward(self, x, x_cond, y):

        z, z_mean, z_log_var = self.encoder(x)

        output = self.decoder(z)

        return output, z, z_mean, z_log_var
    
    def decode(self, z):
        return self.decoder(z)