from abc import ABC, abstractmethod

class ImageCountSampler(ABC):
    @abstractmethod
    def get_pixel_count(self) -> int:
        pass
