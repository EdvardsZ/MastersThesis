import torch
import torch.nn as nn

class LinearDecoder(nn.Module):
    def __init__(self, image_size = (1, 28, 28), hidden_dims = [256, 512], latent_dim = 2):
        super(LinearDecoder, self).__init__()

        output_dim = image_size[0] * image_size[1] * image_size[2]

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_hat = self.decoder(z)
        return x_hat