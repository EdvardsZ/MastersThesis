import torch
from abc import ABC, abstractmethod

class PixelSampler(ABC):
    @abstractmethod
    def __init__(self, add_mask: bool = False):
        pass
    @abstractmethod
    def sample(self, image: torch.Tensor, pixel_count : int) -> torch.Tensor:
        pass
