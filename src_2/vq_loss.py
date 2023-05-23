import torch
import torch.nn as nn
import torch.nn.functional as F

class VQLoss(nn.Module):
    def __init__(self, beta = 0.25):
        super(VQLoss, self).__init__()
        self.beta = beta

    def forward(self, vq_loss, reconstructions, x):
        # Calculate the MSE loss between the quantized latent vectors and the input vectors.
        recon_loss = F.mse_loss(reconstructions, x)

        loss = recon_loss + vq_loss
        
        return loss