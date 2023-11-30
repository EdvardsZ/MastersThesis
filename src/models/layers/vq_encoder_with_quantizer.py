import torch.nn as nn
from models.encoders import VQEncoder, SimpleVQEncoder
from models.layers import NewVectorQuantizer, VectorQuantizer, SimpleVectorQuantizer
from typing import List

class VQEncoderWithQuantizer(nn.Module):
    def __init__(self, in_channels: int, num_embeddings: int, embedding_dim: int, hidden_dims: List[int], n_residual_layers: int):
        super(VQEncoderWithQuantizer, self).__init__()

        self.encoder = VQEncoder(in_channels, embedding_dim, hidden_dims, n_residual_layers)
        self.codebook = NewVectorQuantizer(num_embeddings, embedding_dim)

    def forward(self, x):
        latent = self.encoder(x)
        quantized_with_grad, quantized, embedding_indices = self.codebook(latent)
        return latent, quantized_with_grad, quantized, embedding_indices