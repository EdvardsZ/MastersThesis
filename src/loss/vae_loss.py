import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, weight_kl=1.0, loss_type = 'single'):
        super(VAELoss, self).__init__()
        self.weight_kl = weight_kl
        self.loss_type = loss_type

    def forward(self, inputs, outputs):
        reconstructions, z, z_mean, z_log_var = outputs

        loss_dict = {}
        loss = 0

        x, x_cond, y = inputs

        if isinstance(reconstructions, list):
            recon_0 = recon_loss(x, reconstructions[0])
            loss_dict['recon_loss'] = recon_0
            loss += recon_0
            recon_1 = recon_loss(x, reconstructions[1])
            loss_dict['recon_loss_2'] = recon_1
            loss += recon_1
        else:
            recon_0 = recon_loss(x, reconstructions)
            loss_dict['recon_loss'] = recon_0
            loss += recon_0
        
        kl = kl_loss(z_mean, z_log_var)
        loss_dict['kl_loss'] = kl
        loss += kl

        loss_dict['loss'] = loss

        return loss_dict
    
def kl_loss(z_mean, z_log_var):
        return -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

def recon_loss(inputs, outputs):
    return F.mse_loss(inputs, outputs, reduction='sum')