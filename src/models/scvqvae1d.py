import torch.nn as nn
from models.encoders import VQEncoder
from models.decoders import VQDecoder
from models.layers import VectorQuantizer, SimpleVectorQuantizer, NewVectorQuantizer
from models.layers import VQEncoderWithQuantizer, VQDecoderWithConcat
from models.layers.common import ToFeatureMap
from loss import VQLoss
from models.helpers import concat_latent_with_cond
from .base_vqvae import BaseVQVAE
from models.outputs import VAEModelOutput

from typing import Tuple

class SCVQVAE1D(BaseVQVAE):
    def __init__(self, num_embeddings: int, embedding_dim: int, image_shape: Tuple[int, int, int]):
        super(SCVQVAE1D, self).__init__(num_embeddings, embedding_dim, image_shape)
        self.image_shape = image_shape
        
        self.in_channels = image_shape[0]
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoderWithQuantizer = VQEncoderWithQuantizer(self.in_channels, num_embeddings, embedding_dim)

        self.decoder_with_concat = VQDecoderWithConcat(self.in_channels, embedding_dim, image_shape)

        self.loss = VQLoss()

    def forward(self, x, x_cond, y) -> VAEModelOutput:
        # Input: (B, C, H, W)
        latent, quantized_with_grad, quantized, embedding_indices = self.encoderWithQuantizer(x)

        x_hat = self.decoder_with_concat(quantized_with_grad, x_cond)

        return [x_hat], [], quantized, latent, embedding_indices

    