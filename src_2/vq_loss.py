import torch
import torch.nn as nn

class VQLoss(nn.Module):
    def __init__(self, beta = 0.25):
        super(VQLoss, self).__init__()
        self.beta = beta

    def forward(self, quantized, encoding_indices, x, reconstructions):
        # Calculate the MSE loss between the quantized latent vectors and the input vectors.
        recon_loss = nn.MSELoss()(quantized, x)

        # Calculate the MSE loss between the reconstructions and the input vectors.
        vq_loss = nn.MSELoss()(reconstructions, x)

        # Calculate the commitment loss between the quantized latent vectors and the input vectors.
        commitment_loss = nn.MSELoss()(quantized.detach(), x)

        # Calculate the loss for the encoder.
        loss = recon_loss + self.beta * vq_loss + self.beta * commitment_loss

        # Calculate the perplexity.
        perplexity = torch.exp(vq_loss)

        return loss, recon_loss, commitment_loss, perplexity