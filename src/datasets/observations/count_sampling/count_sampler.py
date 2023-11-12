from abc import ABC, abstractmethod
from enum import Enum

class CountSamplingMethod(Enum):
    EXACT = 1
    HALFEXACT= 2
    EXPONENTIAL = 3

class CountSampler(ABC):
    @abstractmethod
    def get_pixel_count(self) -> int:
        pass
