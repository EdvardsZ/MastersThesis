import torch
import torch.nn as nn
import torch.nn.functional as F

class VQLoss(nn.Module):
    def __init__(self, beta = 0.5):
        super(VQLoss, self).__init__()
        self.beta = beta

    def forward(self, inputs, outputs):
        x, x_cond, y = inputs
        reconstructions, quantized, latent, embedding_indices = outputs

        # Calculate the MSE loss between the quantized latent vectors and the input vectors.
        recon_loss = F.mse_loss(reconstructions, x) / 0.09493041879725218 

        embeddding_loss = F.mse_loss(quantized, latent.detach())
        commitment_loss = F.mse_loss(quantized.detach(), latent)

        vq_loss = commitment_loss * self.beta + embeddding_loss

        loss = recon_loss + vq_loss
        
        return { "loss": loss, "recon_loss": recon_loss, "vq_loss": vq_loss, "commitment_loss": commitment_loss, "embeddding_loss": embeddding_loss }