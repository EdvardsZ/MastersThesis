import torch
from .pixel_sampler import PixelSampler

class ExactPixelSampler(PixelSampler):
    def __init__(self, add_mask = False):
        self.add_mask = add_mask

    def sample(self, image: torch.Tensor) -> torch.Tensor:
    
        return image