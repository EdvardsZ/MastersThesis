import torch
import torch.nn as nn

class SimpleVQEncoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(SimpleVQEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, 1, padding=0)
        )
        
    def forward(self, x):
        return self.encoder(x)