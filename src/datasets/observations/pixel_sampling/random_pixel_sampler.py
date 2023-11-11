import torch
from .pixel_sampler import PixelSampler

class RandomPixelSampler(PixelSampler):
    def __init__(self, pixel_count):
        self.pixel_count = pixel_count

    def sample(self, image: torch.Tensor, pixel_count = None) -> torch.Tensor:
        
        return image