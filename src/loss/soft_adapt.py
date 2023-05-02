import torch
import torch.nn as nn
from loss.functional.helpers import kl_loss, recon_loss

class SoftAdaptVAELoss(nn.Module):
    def __init__(self, n, variant=[]):
        super(SoftAdaptVAELoss, self).__init__()
        self.n = n
        self.variant = variant
        self.loss_buffer = []

    def forward(self, inputs, outputs, z_mean, z_log_var):
        recon = recon_loss(inputs, outputs)
        kl = kl_loss(z_mean, z_log_var)

        self.update_losses(recon + kl)

        weighted_loss = self.weighted_loss([recon, kl])

        return { 'recon_loss': recon, 'kl_loss': kl, 'loss': weighted_loss, "loss(no_weights)" : recon + kl }

    def update_losses(self, loss):
        self.loss_buffer.append(loss)
        if len(self.loss_buffer) > self.n:
            self.loss_buffer.pop(0)

    def weighted_loss(self, losses):
        # Compute the rate of change and average of previous losses
        delta_l = []
        for i in range(len(losses)):
            if len(self.loss_buffer) > i:
                delta_l.append((losses[i] - self.loss_buffer[-i-1]) / self.loss_buffer[-i-1])
            else:
                delta_l.append(0.0)
        avg_l = sum(self.loss_buffer) / len(self.loss_buffer)

        # Compute the normalization factor
        if "Normalized" in self.variant:
            norm_factor = sum([abs(dl) for dl in delta_l]) + 1e-8
        else:
            norm_factor = 1.0
        
        # Compute the weighting factors
        beta = 0.1
        alpha = []
        for i in range(len(losses)):
            if "Normalized" in self.variant:
                ns_i = delta_l[i] / norm_factor
                exp_sum = sum([torch.exp(beta * (ns_j - max(delta_l))) for j, ns_j in enumerate(delta_l)]) + 1e-8
                alpha_i = torch.exp(beta * (ns_i - max(delta_l))) / exp_sum
            else:
                si = delta_l[i]
                exp_sum = sum([torch.exp(beta * (sj - max(delta_l))) for j, sj in enumerate(delta_l)]) + 1e-8
                alpha_i = torch.exp(beta * (si - max(delta_l))) / exp_sum
            if "Loss Weighted" in self.variant:
                alpha_i *= self.loss_buffer[i] / avg_l
            alpha.append(alpha_i)
        
        # Compute the weighted loss
        weighted_loss = sum([alpha[i] * losses[i] for i in range(len(losses))])
        
        return weighted_loss

        