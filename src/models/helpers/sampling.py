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