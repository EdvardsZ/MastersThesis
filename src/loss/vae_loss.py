import torch
import torch.nn as nn
from loss.functional.helpers import kl_loss, recon_loss


class VAELoss(nn.Module):
    def __init__(self, weight_kl=0.00025):
        super(VAELoss, self).__init__()
        self.weight_kl = weight_kl

    def forward(self, inputs, outputs, z_mean, z_log_var):
        recon = recon_loss(inputs, outputs)
        kl = kl_loss(z_mean, z_log_var)
        return { 'recon_loss': recon, 'kl_loss': kl, 'loss': recon + self.weight_kl * kl }