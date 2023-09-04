import torch 
import torch.nn as nn
from models.encoders import SimpleVQEncoder
from models.decoders import SimpleVQDecoder
from models.layers import SimpleVectorQuantizer
from loss import VQLoss
import torch.nn.functional as F

class SimpleVQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(SimpleVQVAE, self).__init__()
        self.encoder = SimpleVQEncoder(embedding_dim)
        self.decoder = SimpleVQDecoder(embedding_dim)

        self.vector_quantizer = SimpleVectorQuantizer(num_embeddings, embedding_dim, beta=beta)

        self.loss = VQLoss(beta)

    def forward(self, x):
        latent = self.encoder(x)
        quantized_with_grad, quantized, encoding_indices = self.vector_quantizer(latent)
        x_hat = self.decoder(quantized_with_grad)

        return x_hat, quantized, latent, encoding_indices
    


        