import torch
import torch.nn as nn
from models.encoders import VQEncoder, SimpleVQEncoder
from models.decoders import VQDecoder, SimpleVQDecoder
from models.layers import VectorQuantizer, SimpleVectorQuantizer, NewVectorQuantizer, ToFeatureMap
from loss import VQLoss
from models.helpers import concat_latent_with_cond

class PixelConditionedVQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, in_channels = 1):
        super(PixelConditionedVQVAE, self).__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = SimpleVQEncoder(in_channels, embedding_dim)

        self.codebook = NewVectorQuantizer(num_embeddings, embedding_dim)

        self.decoder_input = ToFeatureMap(feature_map_size=7, num_channels=embedding_dim)

        self.decoder = SimpleVQDecoder(in_channels, embedding_dim)

        self.loss = VQLoss()

    def forward(self, x, x_cond, y):
        # Input: (B, C, H, W)
        latent = self.encoder(x)

        quantized_with_grad, quantized, embedding_indices = self.codebook(latent)

        quantized_with_grad = nn.Flatten()(quantized_with_grad)

        quantized_with_grad = concat_latent_with_cond(quantized_with_grad, x_cond)

        quantized_with_grad = self.decoder_input(quantized_with_grad)

        x_hat = self.decoder(quantized_with_grad)

        return x_hat, quantized, latent, embedding_indices

    