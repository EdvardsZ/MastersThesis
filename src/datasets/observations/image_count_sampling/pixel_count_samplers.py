import numpy as np
from .image_count_sampler import ImageCountSampler
from typing import Tuple

class ExponentialPixelCountSampler(ImageCountSampler):
    def __init__(self, rate: float):
        self.rate = rate
    
    def get_pixel_count(self) -> int:
        return int(np.random.exponential(self.rate))
    

class UniformPixelCountSampler(ImageCountSampler):
    def __init__(self, low, high):
        self.low = low
        self.high = high
    
    def get_pixel_count(self):
        return np.random.randint(self.low, self.high + 1)