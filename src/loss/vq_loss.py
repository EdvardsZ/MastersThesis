import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.adapt import Adapt, AdaptiveMode 


class VQLoss(nn.Module):
    def __init__(self, adaptive_mode: AdaptiveMode | None = None, beta_soft_adapt : int | None = None):
        super(VQLoss, self).__init__()

        self.adapt = Adapt(mode = adaptive_mode, soft_adapt_beta = beta_soft_adapt)

    def forward(self, inputs, outputs, training = False ):
        x, x_cond, y = inputs

        reconstructions, reconstructions_masked, quantized, latent, embedding_indices = outputs

        loss_dict = {}

        recon_losses = []

        for i, recon in enumerate(reconstructions_masked):
            if recon is not None:
                recon = F.mse_loss(x, recon)
                loss_dict[f'recon_loss_{i}(MASKED)'] = recon

        for i, recon in enumerate(reconstructions):
            recon = F.mse_loss(x, recon)
            loss_dict[f'recon_loss_{i}'] = recon
            recon_losses += recon

        embeddding_loss = F.mse_loss(quantized, latent.detach())
        loss_dict['embeddding_loss'] = embeddding_loss

        commitment_loss = F.mse_loss(quantized.detach(), latent) * 0.25
        loss_dict['commitment_loss'] = commitment_loss 

        vq_loss = commitment_loss + embeddding_loss

        loss_dict['vq_loss'] = vq_loss

        one_recon_loss = loss_dict['recon_loss_0(MASKED)'] if 'recon_loss_0(MASKED)' in loss_dict else loss_dict['recon_loss_0']

        one_recon_loss = one_recon_loss + vq_loss

        name = 'recon_0(MASKED) + vq_loss' if 'recon_loss_0(MASKED)' in loss_dict else 'recon_0 + vq_loss'

        loss_dict[name] = one_recon_loss

        loss_dict['loss_sum'] = sum(recon_losses) + vq_loss
        loss_dict['loss'] = self.adapt(recon_losses, vq_loss, training)

        return loss_dict
        