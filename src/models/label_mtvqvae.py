import torch
import torch.nn as nn
from models.encoders import VQEncoder, SimpleVQEncoder
from models.decoders import VQDecoder, SimpleVQDecoder
from models.layers import VectorQuantizer, SimpleVectorQuantizer, NewVectorQuantizer, ClassificationLayer
from loss import VQLoss

class LabelMTVQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, in_channels = 1):
        super(LabelMTVQVAE, self).__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_classes = 10

        self.encoder = VQEncoder(in_channels, embedding_dim)

        self.codebook = NewVectorQuantizer(num_embeddings, embedding_dim)

        self.classification = ClassificationLayer(self.num_classes)

        self.decoder = VQDecoder(in_channels, embedding_dim)

        self.loss = VQLoss()


    def forward(self, x, x_cond, y):
        # Input: (B, C, H, W)
        latent = self.encoder(x)

        quantized_with_grad, quantized, embedding_indices = self.codebook(latent)

        classification = self.classification(quantized_with_grad)

        x_hat = self.decoder(quantized_with_grad)

        return x_hat, quantized, latent, embedding_indices, classification
    
    def reconstruct_from_indices(self, indices, batch_size):
        quantized = self.codebook.quantize_from_indices(indices, batch_size)
        x_hat = self.decoder(quantized)
        return x_hat