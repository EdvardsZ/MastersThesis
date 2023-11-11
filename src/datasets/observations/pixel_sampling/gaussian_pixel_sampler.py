from random import sample
from numpy import add
import torch
from .pixel_sampler import PixelSampler

class GaussianPixelSampler(PixelSampler):
    def __init__(self, add_mask: bool = False):
        self.add_mask = add_mask

    def sample(self, image: torch.Tensor, pixel_count : int):
            # Get image dimensions
        _, height, width = image.size()

        mean = height / 2
        std = height / 6


        indices = torch.normal(mean, std, size = (pixel_count, 2))
        indices = torch.clamp(indices, 0, height - 1).long()

        sampled_indices = torch.zeros_like(image)
        sampled_indices[:, indices[:, 0], indices[:, 1]] = image[:, indices[:, 0], indices[:, 1]]
        if self.add_mask:
            mask = torch.zeros_like(image[0:1, :, :]) 
            mask[:, indices[:, 0], indices[:, 1]] = 1
            sampled_indices = torch.cat([sampled_indices, mask], dim=0)

        # lets remember that this means that the same pixel can be sampled twice

        return sampled_indices


        