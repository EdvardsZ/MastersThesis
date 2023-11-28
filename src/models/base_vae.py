from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
import torch.nn as nn
from models.outputs import VAEModelOutput


class BaseVAE(ABC, nn.Module):
    @abstractmethod
    def __init__(self, latent_dim: int, image_shape: Tuple[int, int, int]):
        super(BaseVAE, self).__init__()

    @abstractmethod
    def forward(self, x, x_cond, y) -> VAEModelOutput:
        pass