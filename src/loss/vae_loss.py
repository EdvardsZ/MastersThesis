import torch
import torch.nn as nn
import torch.nn.functional as F
from models.outputs import VAEModelOutput
from typing import Tuple

class VAELoss(nn.Module):
    def __init__(self, weight_kl=1.0, loss_type = 'single'):
        super(VAELoss, self).__init__()
        self.weight_kl = weight_kl
        self.loss_type = loss_type

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], outputs: VAEModelOutput):
        reconstructions_unmasked, reconstructions_masked,  z, z_mean, z_log_var = outputs

        loss_dict = {}

        ### adding reconstructions_masked to loss_dict for logging purposes
        for i, recon in enumerate(reconstructions_masked):
            if recon is not None:
                loss_dict[f'recon_loss_{i}(MASKED)'] = recon

        loss = 0

        x, x_cond, y = inputs

        for i, recon in enumerate(reconstructions_unmasked):
            recon = recon_loss(x, recon)
            loss_dict[f'recon_loss_{i}'] = recon
            loss += recon
        
        kl = kl_loss(z_mean, z_log_var)
        loss_dict['kl_loss'] = kl
        loss += kl

        loss_dict['loss'] = loss

        return loss_dict
    
def kl_loss(z_mean, z_log_var):
        return -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

def recon_loss(inputs, outputs):
    return F.mse_loss(inputs, outputs, reduction='sum')