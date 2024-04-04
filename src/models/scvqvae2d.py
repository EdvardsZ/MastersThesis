from unittest.mock import Base
import torch
import torch.nn as nn
from models.encoders import VQEncoder
from models.decoders import VQDecoder
from models.layers import VectorQuantizer, SimpleVectorQuantizer, NewVectorQuantizer
from models.layers.common import ToFeatureMap
from models.layers import VQEncoderWithQuantizer, VQDecoderWithConcat
from loss import VQLoss
from models.helpers import concat_latent_with_cond
from .base_vqvae import BaseVQVAE
from models.outputs import VAEModelOutput

from typing import Tuple, List


class SCVQVAE2D(BaseVQVAE):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        hidden_dims: List[int],
        n_residual_layers: int,
        image_shape: Tuple[int, int, int],
        adaptive_mode=None,
        soft_adapt_beta=None,
    ):
        super(SCVQVAE2D, self).__init__(num_embeddings, embedding_dim, image_shape)
        self.image_shape = image_shape

        self.in_channels = image_shape[0]
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoderWithQuantizer = VQEncoderWithQuantizer(
            self.in_channels,
            num_embeddings,
            embedding_dim,
            hidden_dims,
            n_residual_layers,
        )

        hidden_dims.reverse()

        self.decoder_with_concat = VQDecoderWithConcat(
            self.in_channels, embedding_dim, hidden_dims, n_residual_layers, image_shape
        )

        self.decoder = VQDecoderWithConcat(
            self.in_channels, embedding_dim, hidden_dims, n_residual_layers, image_shape
        )

        self.loss = VQLoss(adaptive_mode, soft_adapt_beta)

    def forward(self, x, x_cond, y) -> VAEModelOutput:

        latent, quantized_with_grad, quantized, embedding_indices = (
            self.encoderWithQuantizer(x)
        )

        output_1 = self.decoder_with_concat(quantized_with_grad)

        output_2 = self.decoder_with_concat(quantized_with_grad, x_cond)

        with torch.no_grad():
            self.decoder_with_concat.eval()
            x_cond_masked = torch.zeros_like(x_cond, requires_grad=False)
            output_2_masked = self.decoder_with_concat(
                quantized_with_grad, x_cond_masked
            )

        return (
            [output_1, output_2],
            [None, output_2_masked],
            quantized,
            latent,
            embedding_indices,
        )

    def reconstruct_from_indices(self, indices, batch_size):
        quantized = self.codebook.quantize_from_indices(indices, batch_size)
        x_hat = self.decoder(quantized)
        return x_hat
