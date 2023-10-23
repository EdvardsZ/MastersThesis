import torch
import torch.nn as nn
from loss.functional.helpers import kl_loss, recon_loss


class VAELoss(nn.Module):
    def __init__(self, weight_kl=1.0, loss_type = 'single'):
        super(VAELoss, self).__init__()
        self.weight_kl = weight_kl
        self.loss_type = loss_type

    def forward(self, inputs, outputs):
        reconstructions, mu, z_log_var, z_mean = outputs
        x, x_cond, y = inputs

        if self.loss_type == 'single':
            recon = recon_loss(x, reconstructions)
            kl = kl_loss(z_mean, z_log_var)
            return { 'recon_loss': recon, 'kl_loss': kl, 'loss': recon + self.weight_kl * kl }
        if self.loss_type == 'double':
            recon_loss_1 = recon_loss(x, reconstructions[0])
            recon_loss_2 = recon_loss(x, reconstructions[1])
            kl = kl_loss(z_mean, z_log_var)
            return { 'recon_loss': recon_loss_1, 'recon_loss_2': recon_loss_2, 'kl_loss': kl, 'loss': recon_loss_1 + recon_loss_2 + self.weight_kl * kl }
        
        raise Exception('Invalid loss type')