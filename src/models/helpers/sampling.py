import torch
    
def sampling(z_mean, z_log_var):
    eps = torch.randn_like(z_log_var)
    z = z_mean + torch.exp(z_log_var / 2) * eps
    return z