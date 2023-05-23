import torch
import torch.nn as nn
from loss.functional.helpers import kl_loss, recon_loss

class SoftAdaptVAELoss(nn.Module):
    def __init__(self, n, variant=[], beta=0.1):
        super(SoftAdaptVAELoss, self).__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.n = n
        self.variant = variant
        self.loss_buffer = []
        self.beta = beta

    def forward(self, inputs, outputs, z_mean, z_log_var):
        recon = recon_loss(inputs, outputs)
        kl = kl_loss(z_mean, z_log_var)

        self.update_losses(recon + kl)

        weighted_loss = self.weighted_loss([recon + kl])

        return {'recon_loss': recon, 'kl_loss': kl, 'loss': weighted_loss, "loss(no_weights)": recon + kl}

    def update_losses(self, loss):
        self.loss_buffer.append(loss)
        if len(self.loss_buffer) > self.n:
            self.loss_buffer.pop(0)

    def weighted_loss(self, losses):
        # Compute the rate of change and average of previous losses
        if len(self.loss_buffer) < self.n:
            return torch.sum(torch.stack(losses))

        delta_l = []
        for i in range(1, len(self.loss_buffer)):
            delta_l.append(self.loss_buffer[i] - self.loss_buffer[i-1])
        si = torch.stack(delta_l)
        fi = torch.mean(torch.stack(self.loss_buffer))

        # Compute the SoftAdapt weights
        beta = self.beta
        if "Normalized" in self.variant:
            nsi = si / torch.abs(si).sum() + 1e-8
            exp_sum = torch.sum(torch.exp(beta * (nsi - torch.max(nsi)))) + 1e-8
            alpha = [torch.exp(beta * (ns_i - torch.max(nsi))) / exp_sum for ns_i in nsi]
        else:
            exp_sum = torch.sum(torch.exp(beta * (si - torch.max(si)))) + 1e-8
            alpha = [torch.exp(beta * (s_i - torch.max(si))) / exp_sum for s_i in si]

        if "Loss Weighted" in self.variant:
            avg_l = torch.mean(torch.stack(self.loss_buffer))
            alpha = [(alpha_i * self.loss_buffer[i]) / avg_l for i, alpha_i in enumerate(alpha)]

        # Compute the weighted loss
        weighted_loss = torch.sum(torch.stack([alpha[i] * losses[i] for i in range(len(losses))]))
        return weighted_loss