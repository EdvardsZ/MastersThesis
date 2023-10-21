import torch
    
def sampling(z_mean, z_log_var):
    eps = torch.randn_like(z_log_var)
    z = z_mean + torch.exp(z_log_var / 2) * eps
    return z

def count_div_by_2(num):
    count = 0
    while num % 2 == 0 and num > 0:
        count += 1
        num //= 2
    return count

def concat_latent_with_cond(z, x_cond):
    x_cond = torch.flatten(x_cond, start_dim=1)
    x_cat = torch.cat((z, x_cond), dim=1)
    return x_cat