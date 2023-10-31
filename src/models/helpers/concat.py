import torch

def concat_latent_with_cond(z, x_cond):
    x_cond = torch.flatten(x_cond, start_dim=1)
    x_cat = torch.cat((z, x_cond), dim=1)
    return x_cat