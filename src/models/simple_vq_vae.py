import torch 
import torch.nn as nn
from models.encoders import SimpleVQEncoder
from models.decoders import SimpleVQDecoder
from models.layers import SimpleVectorQuantizer
import torch.nn.functional as F

class SimpleVQVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(SimpleVQVAE, self).__init__()
        self.encoder = SimpleVQEncoder(embedding_dim)
        self.decoder = SimpleVQDecoder(embedding_dim)

        self.vector_quantizer = SimpleVectorQuantizer(num_embeddings, embedding_dim, beta=beta)

    def forward(self, x):
        z = self.encoder(x)
        quantized_with_grad, quantized, encoding_indices = self.vector_quantizer(z)
        x_hat = self.decoder(quantized_with_grad)

        commitment_loss = torch.mean((quantized.detach() - z) ** 2)
        codebook_loss = torch.mean((quantized - z.detach()) ** 2)
        vq_loss = 0.25 * commitment_loss + codebook_loss

        vq_loss = vq_loss + F.mse_loss(x_hat, x)
        return x_hat, vq_loss, encoding_indices
    


        