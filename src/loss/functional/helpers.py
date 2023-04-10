import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_loss(z_mean, z_log_var):
    return -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

def recon_loss(inputs, outputs):
    return F.mse_loss(inputs, outputs, reduction='sum')