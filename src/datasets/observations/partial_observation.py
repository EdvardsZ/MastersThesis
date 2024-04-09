from gc import get_count
import math
import torch
from typing import Tuple, List
from PIL import Image
import torchvision.transforms as transforms
from .enums import CountSamplingMethod, PixelSamplingMethod
from datasets.observations.count_sampling import VariablePixelCountSampler, ExactPixelCountSampler, ExponentialPixelCountSampler, CountSampler
from datasets.observations.pixel_sampling import ExactPixelSampler, UniformPixelSampler, GaussianPixelSampler, PixelSampler, pixel_sampler

class PartialObservation:
    def __init__(self, 
                data_config: dict | None
                ):
        
        
        self.count_sampler = self.get_count_sampler(data_config)
        self.pixel_sampler = self.get_pixel_sampler(data_config)
        
        
    def get_count_sampler(self, data_config: dict | None) -> CountSampler:
        if data_config is None:
            return CountSamplingMethod.EXACT.get_sampler()
        
        count_sampler = CountSamplingMethod(data_config.get('count_sampling', CountSamplingMethod.EXACT))
        
        if count_sampler == CountSamplingMethod.POWER_LAW:
            exponent = data_config.get('exponent', 40)
            return CountSamplingMethod.POWER_LAW.get_sampler(exponent)
        
        return count_sampler.get_sampler()
    
    def get_pixel_sampler(self, data_config: dict | None) -> PixelSampler:
        if data_config is None:
            return PixelSamplingMethod.EXACT.get_sampler()
        pixel_sampler = PixelSamplingMethod(data_config.get('pixel_sampling', PixelSamplingMethod.EXACT))
        return pixel_sampler.get_sampler()
    
    def get_partial_observation(self, image: torch.Tensor) -> torch.Tensor:
        pixel_count = self.count_sampler.get_pixel_count(image)
        partial_observation = self.pixel_sampler.sample(image, pixel_count)
        return partial_observation
    


