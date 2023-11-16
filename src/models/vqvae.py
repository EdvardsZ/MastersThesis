from email.mime import image
import torch
import torch.nn as nn
from models.encoders import SimpleVQEncoder
from models.decoders import SimpleVQDecoder
from models.layers import SimpleVectorQuantizer, NewVectorQuantizer, VectorQuantizer
from loss import VQLoss

from typing import Tuple

class VQVAE(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, image_shape: Tuple[int, int, int]):
        super(VQVAE, self).__init__()
        
        self.image_shape = image_shape
        self.in_channels = image_shape[0]
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = SimpleVQEncoder(self.in_channels, embedding_dim)

        self.codebook = NewVectorQuantizer(num_embeddings, embedding_dim)

        self.decoder = SimpleVQDecoder(self.in_channels, embedding_dim)

        self.loss = VQLoss()


    def forward(self, x, x_cond, y):
        # Input: (B, C, H, W)
        latent = self.encoder(x)

        quantized_with_grad, quantized, embedding_indices = self.codebook(latent)

        x_hat = self.decoder(quantized_with_grad)

        return x_hat, quantized, latent, embedding_indices
    
    def reconstruct_from_indices(self, indices, batch_size):
        quantized = self.codebook.quantize_from_indices(indices, batch_size)
        x_hat = self.decoder(quantized)
        return x_hat