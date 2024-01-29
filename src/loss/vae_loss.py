import torch
import torch.nn as nn
import torch.nn.functional as F
from models.outputs import VAEModelOutput
from typing import Tuple
from .soft_adapt import SoftAdaptModule
from loss import soft_adapt

class VAELoss(nn.Module):
    def __init__(self, beta : int | None = None):
        super(VAELoss, self).__init__()
        self.beta = beta
        if beta is not None:
            self.soft_adapt = SoftAdaptModule(beta = beta)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], outputs: VAEModelOutput, training = False):
        reconstructions_unmasked, reconstructions_masked,  z, z_mean, z_log_var = outputs

        loss_dict = {}

        ### adding reconstructions_masked to loss_dict for logging purposes
        for i, recon in enumerate(reconstructions_masked):
            if recon is not None:
                recon = recon_loss(inputs[0], recon)
                loss_dict[f'recon_loss_{i}(MASKED)'] = recon

        losses = []

        for i, recon in enumerate(reconstructions_unmasked):
            loss_dict[f'recon_loss_{i}'] = recon_loss(inputs[0], recon)
            losses.append(loss_dict[f'recon_loss_{i}'])
        
        kl = kl_loss(z_mean, z_log_var)
        loss_dict['kl_loss'] = kl_loss(z_mean, z_log_var)
        losses.append(kl)

        loss_dict['loss_sum'] = sum(losses)
        loss_dict['loss'] = self.adaptive_sum(losses, training)
        return loss_dict


    def adaptive_sum(self, losses, training):
         if self.beta is None:
            return sum(losses)
         else:
            return self.soft_adapt(losses, training)
         
    
def kl_loss(z_mean, z_log_var):
        return -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

def recon_loss(inputs, outputs):
    return F.mse_loss(inputs, outputs, reduction='sum')