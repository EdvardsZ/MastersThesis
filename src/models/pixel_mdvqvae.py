import torch
import torch.nn as nn
from models.encoders import VQEncoder, SimpleVQEncoder
from models.decoders import VQDecoder, SimpleVQDecoder
from models.layers import VectorQuantizer, SimpleVectorQuantizer, NewVectorQuantizer, ToFeatureMap
from loss import VQLoss
from models.helpers import concat_latent_with_cond

class PixelMDVQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, in_channels = 1):
        super(PixelMDVQVAE, self).__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = VQEncoder(in_channels, embedding_dim)

        self.codebook = NewVectorQuantizer(num_embeddings, embedding_dim)

        self.decoder_input = ToFeatureMap(feature_map_size=7, num_channels=embedding_dim)

        self.pixel_decoder = VQDecoder(in_channels, embedding_dim)

        self.decoder = VQDecoder(in_channels, embedding_dim)

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