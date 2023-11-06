from email.mime import image
import torch
import torch.nn as nn
from models.encoders import VQEncoder
from models.decoders import VQDecoder
from models.layers import SimpleVectorQuantizer, NewVectorQuantizer, new_vector_quantizer
from loss import VQLoss

from typing import Tuple

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, image_shape: Tuple[int, int, int]):
        super(VQVAE, self).__init__()
        
        self.in_channels = image_shape[0]
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = VQEncoder(self.in_channels, embedding_dim)

        self.codebook = NewVectorQuantizer(num_embeddings, embedding_dim)

        self.decoder = VQDecoder(self.in_channels, embedding_dim)

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