import torch
import torch.nn as nn
from models.encoders import VQEncoder, SimpleVQEncoder
from models.decoders import VQDecoder, SimpleVQDecoder
from models.layers import VectorQuantizer, SimpleVectorQuantizer, NewVectorQuantizer
from loss import VQLoss

class VQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, in_channels = 1):
        super(VQVAE, self).__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = SimpleVQEncoder(in_channels, embedding_dim)

        self.codebook = NewVectorQuantizer(num_embeddings, embedding_dim)

        self.decoder = SimpleVQDecoder(in_channels, embedding_dim)

        self.loss = VQLoss()


    def forward(self, x):
        # Input: (B, C, H, W)
        latent = self.encoder(x)

        quantized_with_grad, quantized, embedding_indices = self.codebook(latent)

        x_hat = self.decoder(quantized_with_grad)

        return x_hat, quantized, latent, embedding_indices
    
    def reconstruct_from_indices(self, indices, batch_size):
        quantized = self.codebook.quantize_from_indices(indices, batch_size)
        x_hat = self.decoder(quantized)
        return x_hat