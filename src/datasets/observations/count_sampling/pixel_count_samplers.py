import numpy as np
import torch
from .count_sampler import CountSampler

class ExponentialPixelCountSampler(CountSampler):
    def __init__(self):
        return
    def get_pixel_count(self, image: torch.Tensor) -> int:
        total_pixels = image.shape[1] * image.shape[2]
        pixel_count = int(np.random.exponential(total_pixels / 8))

        if pixel_count > total_pixels:
            pixel_count = total_pixels
        return pixel_count

class PowerLawPixelCountSampler(CountSampler):
    def __init__(self):
        self.exponent = 10.0
        return
    def get_pixel_count(self, image: torch.Tensor) -> int:
        total_pixels = image.shape[1] * image.shape[2]
        return int(total_pixels - np.random.power(self.exponent) * total_pixels)

class VariablePixelCountSampler(CountSampler):
    def __init__(self, rate_of_pixels: float = 0.05, zero_pixels_rate: float = 0.5):
        self.rate_of_pixels = rate_of_pixels
        self.zero_pixels_rate = zero_pixels_rate
        return
    def get_pixel_count(self, image: torch.Tensor) -> int:
        pixel_count = int(image.shape[1] * image.shape[2] * self.rate_of_pixels)

        if np.random.rand() < self.zero_pixels_rate:
            return 0
        else:
            return pixel_count
        
class ExactPixelCountSampler(CountSampler):
    def __init__(self, rate_of_pixels: float = 0.05):
        self.rate_of_pixels = rate_of_pixels
        return
    def get_pixel_count(self, image: torch.Tensor) -> int:
        pixel_count = int(image.shape[1] * image.shape[2] * self.rate_of_pixels)
        return pixel_count