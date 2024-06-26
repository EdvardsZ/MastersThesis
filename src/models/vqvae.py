from email.mime import image
import torch
import torch.nn as nn
from models.encoders import SimpleVQEncoder, VQEncoder
from models.decoders import SimpleVQDecoder, VQDecoder
from models.layers import SimpleVectorQuantizer, NewVectorQuantizer, VectorQuantizer
from models.layers import VQEncoderWithQuantizer, VQDecoderWithConcat
from loss import VQLoss
from .base_vqvae import BaseVQVAE
from models.outputs import VAEModelOutput

from typing import Tuple, List


class VQVAE(BaseVQVAE):
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
        super(VQVAE, self).__init__(num_embeddings, embedding_dim, image_shape)

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
        self.decoderWithConcat = VQDecoderWithConcat(
            self.in_channels, embedding_dim, hidden_dims, n_residual_layers, image_shape
        )

        self.loss = VQLoss(adaptive_mode, soft_adapt_beta)

    def forward(self, x, x_cond, y) -> VAEModelOutput:

        latent, quantized_with_grad, quantized, embedding_indices = (
            self.encoderWithQuantizer(x)
        )

        x_hat = self.decoderWithConcat(quantized_with_grad)

        return [x_hat], [], quantized, latent, embedding_indices

    def reconstruct_from_indices(self, indices, batch_size):
        quantized = self.codebook.quantize_from_indices(indices, batch_size)
        x_hat = self.decoder(quantized)
        return x_hat
