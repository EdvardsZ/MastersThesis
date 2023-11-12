from gc import get_count
import math
import torch
from typing import Tuple, List
from PIL import Image
import torchvision.transforms as transforms
from .enums import CountSamplingMethod, PixelSamplingMethod
from datasets.observations.count_sampling import VariablePixelCountSampler, ExactPixelCountSampler, ExponentialPixelCountSampler
from datasets.observations.pixel_sampling import ExactPixelSampler, RandomPixelSampler, GaussianPixelSampler

class PartialObservation:
    def __init__(self, 
                count_sampling: CountSamplingMethod = CountSamplingMethod.EXACT,
                pixel_sampling: PixelSamplingMethod = PixelSamplingMethod.EXACT,
                add_mask: bool = True):
        
        self.add_mask = add_mask
        self.count_sampler = self.get_count_sampler(count_sampling)
        self.pixel_sampler = self.get_pixel_sampler(pixel_sampling)

    def get_partial_observation(self, image: torch.Tensor) -> torch.Tensor:
        pixel_count = self.count_sampler.get_pixel_count(image)
        partial_observation = self.pixel_sampler.sample(image, pixel_count)
        return partial_observation
        
    def get_count_sampler(self, count_sampling: CountSamplingMethod):
        if count_sampling == CountSamplingMethod.EXACT:
            return ExactPixelCountSampler()
        elif count_sampling == CountSamplingMethod.VARIABLE:
            return VariablePixelCountSampler()
        elif count_sampling == CountSamplingMethod.EXPONENTIAL:
            return ExponentialPixelCountSampler()
        else:
            raise ValueError("Unknown count sampling method")
        
    def get_pixel_sampler(self, pixel_sampling: PixelSamplingMethod):
        if pixel_sampling == PixelSamplingMethod.EXACT:
            return ExactPixelSampler(self.add_mask)
        elif pixel_sampling == PixelSamplingMethod.RANDOM:
            return RandomPixelSampler(self.add_mask)
        elif pixel_sampling == PixelSamplingMethod.GAUSSIAN:
            return GaussianPixelSampler(self.add_mask)
        else:
            raise ValueError("Unknown pixel sampling method")

    


