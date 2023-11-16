from gc import get_count
import math
import torch
from typing import Tuple, List
from PIL import Image
import torchvision.transforms as transforms
from .enums import CountSamplingMethod, PixelSamplingMethod
from datasets.observations.count_sampling import VariablePixelCountSampler, ExactPixelCountSampler, ExponentialPixelCountSampler
from datasets.observations.pixel_sampling import ExactPixelSampler, UniformPixelSampler, GaussianPixelSampler

class PartialObservation:
    def __init__(self, 
                count_sampling: CountSamplingMethod = CountSamplingMethod.EXACT,
                pixel_sampling: PixelSamplingMethod = PixelSamplingMethod.EXACT,
                add_mask: bool = True):
        self.add_mask = add_mask
        self.count_sampler = count_sampling.get_sampler()
        self.pixel_sampler = pixel_sampling.get_sampler(add_mask)

    def get_partial_observation(self, image: torch.Tensor) -> torch.Tensor:
        pixel_count = self.count_sampler.get_pixel_count(image)
        partial_observation = self.pixel_sampler.sample(image, pixel_count)
        return partial_observation
    


