import torch
import torch.nn as nn
from models.encoders import VQEncoder
from models.decoders import VQDecoder
from models.layers import VectorQuantizer, SimpleVectorQuantizer, NewVectorQuantizer
from models.layers.common import ToFeatureMap
from loss import VQLoss
from models.helpers import concat_latent_with_cond

from typing import Tuple

class SCVQVAE2D(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, image_shape: Tuple[int, int, int]):
        super(SCVQVAE2D, self).__init__()
        self.in_channels = image_shape[0]
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = VQEncoder(self.in_channels, embedding_dim)
        self.codebook = NewVectorQuantizer(num_embeddings, embedding_dim)

        self.decoder_input = ToFeatureMap(feature_map_size=image_shape[1] // 4, num_channels=embedding_dim)

        self.pixel_decoder = VQDecoder(self.in_channels, embedding_dim)
        self.decoder = VQDecoder(self.in_channels, embedding_dim)

        self.loss = VQLoss(loss_type='double')

    def forward(self, x, x_cond, y):
        # Input: (B, C, H, W)
        latent = self.encoder(x)

        quantized_with_grad, quantized, embedding_indices = self.codebook(latent)

        quantized_with_grad_concat = nn.Flatten()(quantized_with_grad) # TODO: make a module for this

        quantized_with_grad_concat = concat_latent_with_cond(quantized_with_grad_concat, x_cond)

        quantized_with_grad_concat = self.decoder_input(quantized_with_grad_concat)

        output_1 = self.pixel_decoder(quantized_with_grad)
        output_2 = self.pixel_decoder(quantized_with_grad_concat)

        return [output_1, output_2], quantized, latent, embedding_indices
    
    def reconstruct_from_indices(self, indices, batch_size):
        quantized = self.codebook.quantize_from_indices(indices, batch_size)
        x_hat = self.decoder(quantized)
        return x_hat