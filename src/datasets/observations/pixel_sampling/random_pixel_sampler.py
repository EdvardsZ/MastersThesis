import torch
from .pixel_sampler import PixelSampler

class RandomPixelSampler(PixelSampler):
    def __init__(self, add_mask: bool = False):
        self.add_mask = add_mask

    def sample(self, image: torch.Tensor, pixel_count: int) -> torch.Tensor:
        zeros = torch.zeros((pixel_count, 2))
        indices = torch.randint_like(zeros, 0, image.shape[1], dtype=torch.long) # hmm this is not quite right because the same pixel can be sampled twice

        sampled_pixels = torch.zeros_like(image)

        sampled_pixels[:, indices[:, 0], indices[:, 1]] = image[:, indices[:, 0], indices[:, 1]]
        
        if self.add_mask:
            mask = torch.zeros_like(image[0:1, :, :]) 
            mask[:, indices[:, 0], indices[:, 1]] = 1
            sampled_pixels = torch.cat([sampled_pixels, mask], dim=0)


        return sampled_pixels