import torch
import torch.nn as nn
import torch.nn.functional as F

class VQLoss(nn.Module):
    def __init__(self, beta = 0.25 , loss_type = 'single'):
        super(VQLoss, self).__init__()
        self.beta = beta
        self.loss_type = loss_type

    def forward(self, inputs, outputs, traininig = False ):
        x, x_cond, y = inputs

        reconstructions, reconstructions_masked, quantized, latent, embedding_indices = outputs

        loss = 0
        loss_dict = {}

        for i, recon in enumerate(reconstructions_masked):
            if recon is not None:
                recon = F.mse_loss(x, recon)
                loss_dict[f'recon_loss_{i}(MASKED)'] = recon

        for i, recon in enumerate(reconstructions):
            recon = F.mse_loss(x, recon)
            loss_dict[f'recon_loss_{i}'] = recon
            loss += recon

        embeddding_loss = F.mse_loss(quantized, latent.detach())
        loss_dict['embeddding_loss'] = embeddding_loss

        commitment_loss = F.mse_loss(quantized.detach(), latent) * self.beta
        loss_dict['commitment_loss'] = commitment_loss 

        vq_loss = commitment_loss + embeddding_loss

        loss_dict['vq_loss'] = vq_loss

        loss += vq_loss

        loss_dict['loss'] = loss

        return loss_dict
        