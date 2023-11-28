import torch
import torch.nn as nn
from typing import Tuple
from models.decoders import VQDecoder
from torch import Tensor
from .concat_layer import ConcatLayer

class VQDecoderWithConcat(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int):
        super(VQDecoderWithConcat, self).__init__()
        self.embedding_dim = embedding_dim

        hidden_dims = [32, 64]
        hidden_dims.reverse()
        n_residual_layers = 0

        self.decoder = VQDecoder(in_channels, embedding_dim, hidden_dims, n_residual_layers)
        self.concat = ConcatLayer()

    def forward(self, quantized: Tensor, x_cond: Tensor | None = None):
        if x_cond is None:
            return self.decoder(quantized)
        
        quantized = self.concat(quantized, x_cond)

        return self.decoder(quantized)