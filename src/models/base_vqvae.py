from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
import torch.nn as nn
from models.outputs import VAEModelOutput


class BaseVQVAE(ABC, nn.Module):
    @abstractmethod
    def __init__(self, num_embeddings: int, embedding_dim: int, image_shape: Tuple[int, int, int]):
        super(BaseVQVAE, self).__init__()

    @abstractmethod
    def forward(self, x, x_cond, y) -> VAEModelOutput:
        pass