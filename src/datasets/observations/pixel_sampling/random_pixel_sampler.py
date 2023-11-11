import torch
from .pixel_sampler import PixelSampler

class RandomPixelSampler(PixelSampler):
    def __init__(self, add_mask = False):
        self.add_mask = add_mask

    def sample(self, image: torch.Tensor, pixel_count: int) -> torch.Tensor:
        zeros = torch.zeros((pixel_count))
        indices = torch.randint_like(zeros, 0, image.shape[1] * image.shape[2], dtype=torch.long)

        sampled_pixels = torch.zeros_like(image)
        sampled_pixels.view(-1)[indices] = image.view(-1)[indices]
        
        if self.add_mask:
            mask = torch.zeros_like(image[0:1, :, :]) 
            mask.view(-1)[indices] = 1
            sampled_pixels = torch.cat([sampled_pixels, mask], dim=0)


        return sampled_pixels