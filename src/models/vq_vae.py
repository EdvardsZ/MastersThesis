import torch
import torch.nn as nn
from models.encoders import VQEncoder
from models.decoders import VQDecoder

class VQVAE(nn.Module):
    def __init__(self, in_channels, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = VQEncoder(in_channels, embedding_dim)

        self.codebook = VectorQuantizer(num_embeddings, embedding_dim)

        self.decoder = VQDecoder(in_channels, embedding_dim)


    def forward(self, x):
        # Input: (B, C, H, W)
        z = self.encoder(x)

        z = z.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        z, embedding_indices, loss = self.codebook(z)
        z = z.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)

        x_hat = self.decoder(z)

        return x_hat, z, embedding_indices, loss