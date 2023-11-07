import torch
import torch.nn as nn
import torch.nn.functional as F

class VQLoss(nn.Module):
    def __init__(self, beta = 0.25 , loss_type = 'single'):
        super(VQLoss, self).__init__()
        self.beta = beta
        self.loss_type = loss_type

    def forward(self, inputs, outputs):
        x, x_cond, y = inputs

        reconstructions, quantized, latent, embedding_indices = outputs


        loss = 0
        loss_dict = {}

        if isinstance(reconstructions, list):
            recon_0 = F.mse_loss(reconstructions[0], x)
            loss_dict['recon_loss'] = recon_0
            loss += recon_0
            recon_1 = F.mse_loss(reconstructions[1], x)
            loss_dict['recon_loss_2'] = recon_1
            loss += recon_1
        else:
            recon_0 = F.mse_loss(reconstructions, x)
            loss_dict['recon_loss'] = recon_0
            loss += recon_0

        embeddding_loss = F.mse_loss(quantized, latent.detach())
        loss_dict['embeddding_loss'] = embeddding_loss

        commitment_loss = F.mse_loss(quantized.detach(), latent) * self.beta
        loss_dict['commitment_loss'] = commitment_loss 

        vq_loss = commitment_loss + embeddding_loss

        loss_dict['vq_loss'] = vq_loss

        loss += vq_loss

        loss_dict['loss'] = loss

        return loss_dict
        