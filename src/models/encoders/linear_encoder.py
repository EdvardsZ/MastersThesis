import torch
import torch.nn as nn

class LinearEncoder(nn.Module):
    def __init__(self, image_size = (1, 28, 28), hidden_dims = [512, 256], latent_dim = 2):
        super(LinearEncoder, self).__init__()

        input_dim = image_size[0] * image_size[1] * image_size[2]

 
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.z_mean = nn.Linear(hidden_dims[1], latent_dim)
        self.z_log_var = nn.Linear(hidden_dims[1], latent_dim)

    def forward(self, x):

        x = self.encoder(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        
        return z_mean, z_log_var
        