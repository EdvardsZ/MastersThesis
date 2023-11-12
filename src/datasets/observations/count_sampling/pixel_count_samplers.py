import numpy as np
from .count_sampler import CountSampler
from typing import Tuple

class ExponentialPixelCountSampler(CountSampler):
    def __init__(self, rate: float):
        self.rate = rate
    
    def get_pixel_count(self) -> int:
        return int(np.random.exponential(self.rate))
    

class HalfExactPixelCountSampler(CountSampler):
    # Half-exact sampling: 50% of the time exact, 50% of the time 0
    def __init__(self, pixel_count: int):
        self.low = low
        self.high = high
    
    def get_pixel_count(self):
        return np.random.randint(self.low, self.high + 1)