from enum import Enum
from datasets.observations.count_sampling import VariablePixelCountSampler, ExactPixelCountSampler, ExponentialPixelCountSampler
from datasets.observations.pixel_sampling import ExactPixelSampler, UniformPixelSampler, GaussianPixelSampler

class PixelSamplingMethod(Enum):
    EXACT = "EXACT"
    UNIFORM = "UNIFORM"
    GAUSSIAN = "GAUSSIAN"

    def get_sampler(self, add_mask: bool = True):
        if self == PixelSamplingMethod.EXACT:
            return ExactPixelSampler(add_mask)
        elif self == PixelSamplingMethod.UNIFORM:
            return UniformPixelSampler(add_mask)
        elif self == PixelSamplingMethod.GAUSSIAN:
            return GaussianPixelSampler(add_mask)
        else:
            raise ValueError("Unknown pixel sampling method")

class CountSamplingMethod(Enum):
    EXACT = "EXACT"
    VARIABLE = "VARIABLE"
    EXPONENTIAL = "EXPONENTIAL"

    def get_sampler(self):
        if self == CountSamplingMethod.EXACT:
            return ExactPixelCountSampler()
        elif self == CountSamplingMethod.VARIABLE:
            return VariablePixelCountSampler()
        elif self == CountSamplingMethod.EXPONENTIAL:
            return ExponentialPixelCountSampler()
        else:
            raise ValueError("Unknown count sampling method")

