from abc import ABC, abstractmethod
import torch
class CountSampler(ABC):
    @abstractmethod
    def get_pixel_count(self, image: torch.Tensor) -> int:
        pass
