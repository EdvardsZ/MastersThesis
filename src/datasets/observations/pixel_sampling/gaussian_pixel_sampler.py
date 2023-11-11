import torch
from .pixel_sampler import PixelSampler

class GaussianPixelSampler(PixelSampler):
    def __init__(self, add_mask: bool = False):
        self.add_mask = add_mask

    def sample(self, image: torch.Tensor, pixel_count : int) -> torch.Tensor:
        
        return image