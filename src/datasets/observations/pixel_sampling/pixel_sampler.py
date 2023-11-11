import torch
from abc import ABC, abstractmethod

class PixelSampler(ABC):
    @abstractmethod
    def sample(self, image: torch.Tensor, pixel_count = None) -> torch.Tensor:
        pass
