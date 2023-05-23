import torch
import torch.nn as nn
from loss.functional.helpers import kl_loss, recon_loss
import numpy as np

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

        self.update_losses(recon, kl)

        weighted_loss = self.weighted_loss(recon, kl)

        return {'recon_loss': recon, 'kl_loss': kl, 'loss': weighted_loss, "loss(no_weights)": recon + kl}

    def update_losses(self, loss_1, loss_2):
        loss_1 = loss_1.detach().cpu().numpy()
        loss_2 = loss_2.detach().cpu().numpy()
        losses = np.array([loss_1, loss_2])
        self.loss_buffer.append(losses)
        if len(self.loss_buffer) > self.n:
            self.loss_buffer.pop(0)

    def weighted_loss(self, loss_1, loss_2):
        if len(self.loss_buffer) < self.n:
            return loss_1 + loss_2
        # Calculate the rates of change and averages of previous losses
        si = np.gradient(self.loss_buffer, axis=0)[-1]
        fi = np.mean(self.loss_buffer, axis=0)

        # Calculate the normalized rates of change
        nsi = si / (np.sum(np.abs(si)) + 1e-8)

        # Calculate the max values of rates of change
        max_si = np.max(si)
        max_nsi = np.max(nsi)

        # Calculate the exponentiated values of rates of change
        exp_si = np.exp(self.beta * (nsi - max_nsi))
        exp_fi = np.exp(self.beta * (fi - max_si))

        # Calculate the sum of exponentiated values
        sum_exp_si = np.sum(exp_si)
        sum_exp_fi = np.sum(exp_fi)

        # Calculate the weights based on the variant
        if "Normalized" in self.variant:
            alpha_i = exp_si / (sum_exp_si + 1e-8)
        else:
            alpha_i = exp_fi / (sum_exp_fi + 1e-8)

        # Calculate the weighted loss
        weighted_loss = loss_1 * alpha_i[0] + loss_2 * alpha_i[1]

        return weighted_loss
        