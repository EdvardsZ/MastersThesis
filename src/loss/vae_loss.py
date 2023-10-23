import torch
import torch.nn as nn
from loss.functional.helpers import kl_loss, recon_loss


class VAELoss(nn.Module):
    def __init__(self, weight_kl=1.0, loss_type = 'single'):
        super(VAELoss, self).__init__()
        self.weight_kl = weight_kl
        self.loss_type = loss_type

    def forward(self, inputs, outputs):
        classification = None
        try:
            reconstructions, mu, z_log_var, z_mean, classification = outputs
        except:
            reconstructions, mu, z_log_var, z_mean = outputs

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

        if classification is not None:
            classification = nn.CrossEntropyLoss(reduction='sum')(classification, y)
            loss_dict['classification_loss'] = classification
            loss += classification

        loss_dict['loss'] = loss

        return loss_dict