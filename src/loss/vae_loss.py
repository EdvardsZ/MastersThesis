import torch
import torch.nn as nn
from loss.functional.helpers import kl_loss, recon_loss


class VAELoss(nn.Module):
    def __init__(self, weight_kl=1, loss_type = 'single'):
        super(VAELoss, self).__init__()
        self.weight_kl = weight_kl
        self.loss_type = loss_type

    def forward(self, inputs, outputs, z_mean, z_log_var):
        if self.loss_type == 'single':
            recon = recon_loss(inputs, outputs)
            kl = kl_loss(z_mean, z_log_var)
            return { 'recon_loss': recon, 'kl_loss': kl, 'loss': recon + self.weight_kl * kl }
        elif self.loss_type == 'double':
            recon_loss_1 = recon_loss(inputs, outputs[0])
            recon_loss_2 = recon_loss(inputs, outputs[1])
            kl = kl_loss(z_mean, z_log_var)
            return { 'recon_loss': recon_loss_1, 'recon_loss_2': recon_loss_2, 'kl_loss': kl, 'loss': recon_loss_1 + recon_loss_2 + self.weight_kl * kl }