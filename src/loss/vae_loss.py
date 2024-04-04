import torch
import torch.nn as nn
import torch.nn.functional as F
from models.outputs import VAEModelOutput
from typing import Tuple
from loss.adapt import Adapt, AdaptiveMode 

class VAELoss(nn.Module):
    def __init__(self, adaptive_mode: str | None, beta_soft_adapt : float | None = None):
        super(VAELoss, self).__init__()
        
        mode = AdaptiveMode(adaptive_mode) if adaptive_mode is not None else None
        beta_soft_adapt = beta_soft_adapt if beta_soft_adapt is not None else 0.0001

        self.adapt = Adapt(mode = mode, soft_adapt_beta = beta_soft_adapt)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], outputs: VAEModelOutput, training = False):
        reconstructions_unmasked, reconstructions_masked,  z, z_mean, z_log_var = outputs

        loss_dict = {}

        ### adding reconstructions_masked to loss_dict for logging purposes
        for i, recon in enumerate(reconstructions_masked):
            if recon is not None:
                recon = recon_loss(inputs[0], recon)
                loss_dict[f'recon_loss_{i}(MASKED)'] = recon

        recon_losses = []

        for i, recon in enumerate(reconstructions_unmasked):
            loss_dict[f'recon_loss_{i}'] = recon_loss(inputs[0], recon)
            recon_losses.append(loss_dict[f'recon_loss_{i}'])
        
        kl = kl_loss(z_mean, z_log_var)
        loss_dict['kl_loss'] = kl_loss(z_mean, z_log_var)

        one_recon_loss = loss_dict['recon_loss_0(MASKED)'] if 'recon_loss_0(MASKED)' in loss_dict else loss_dict['recon_loss_0']
        one_recon_loss = one_recon_loss + kl
        name = 'recon_0(MASKED) + kl_loss' if 'recon_loss_0(MASKED)' in loss_dict else 'recon_0 + kl_loss'
        loss_dict[name] = one_recon_loss


        loss_dict['loss_sum'] = sum(recon_losses) + kl
        loss_dict['loss'] = self.adapt(recon_losses, kl, training)
        return loss_dict
    
def kl_loss(z_mean, z_log_var):
        return -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

def recon_loss(inputs, outputs):
    return F.mse_loss(inputs, outputs, reduction='sum')