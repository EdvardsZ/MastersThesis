import torch
import torch.nn as nn

class SimpleVQDecoder(nn.Module):
    def __init__(self, in_channels, embedding_dim,):
        super(SimpleVQDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)