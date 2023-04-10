import torch
import torch.nn as nn

class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim):
        super(LinearEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.latent_dim = latent_dim

        # encoder part
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc31 = nn.Linear(hidden_dim2, latent_dim)
        self.fc32 = nn.Linear(hidden_dim2, latent_dim)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))

        x = self.relu(self.fc2(x))

        z_mean = self.fc31(x)
        z_log_var = self.fc32(x)
        
        return z_mean, z_log_var
        