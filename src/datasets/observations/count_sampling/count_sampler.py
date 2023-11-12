from abc import ABC, abstractmethod

class CountSampler(ABC):
    @abstractmethod
    def get_pixel_count(self) -> int:
        pass
