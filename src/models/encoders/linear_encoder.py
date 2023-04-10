import torch
import torch.nn as nn

class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(LinearEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.latent_dim = latent_dim
 
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
        )
        self.z_mean = nn.Linear(hidden_dim2, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim2, latent_dim)

    def forward(self, x):

        x = self.encoder(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        
        return z_mean, z_log_var
        