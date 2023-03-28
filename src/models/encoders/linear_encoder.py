import torch
import torch.nn as nn

class LinearEncoder(nn.Module):
    def __init__(self, image_size = (1, 28, 28), n_hidden=500, latent_dim=2, keep_prob=0.99):
        super(LinearEncoder, self).__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim
        
        in_channels = self.image_size[0]

        flat_size = image_size[1] * image_size[2]

        self.encoder =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, n_hidden),
            nn.ELU(inplace=True),
            nn.Dropout(1-keep_prob),

            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(1-keep_prob),
        )

        self.fc_mu = nn.Linear(n_hidden, latent_dim)

        self.fc_var = nn.Linear(n_hidden, latent_dim)

    def forward(self, inputs):
        x = self.encoder(inputs)
        z_mean = self.fc_mu(x)
        z_log_var = self.fc_var(x)
        z = self.sampling(z_mean, z_log_var)
        
        return z_mean, z_log_var, z
    
    def sampling(self, z_mean, z_log_var):
        eps = torch.randn_like(z_log_var)
        z = z_mean + torch.exp(z_log_var / 2) * eps
        return z
    





