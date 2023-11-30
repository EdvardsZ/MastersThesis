import torch
import torch.nn as nn
from typing import Tuple, List
from models.decoders import VQDecoder, SimpleVQDecoder
from torch import Tensor
from .concat_layer import ConcatLayer
from models.layers.common import ToFeatureMap, to_feature_map

class VQDecoderWithConcat(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, hidden_dims: List[int], n_residual_layers: int, image_shape: Tuple[int, int, int]):
        super(VQDecoderWithConcat, self).__init__()
        self.embedding_dim = embedding_dim

        self.concat = ConcatLayer()
        self.decoder_input = ToFeatureMap(feature_map_size= image_shape[1] // 4, num_channels=embedding_dim)

        self.decoder = VQDecoder(in_channels, embedding_dim, hidden_dims, n_residual_layers)

    def forward(self, quantized: Tensor, x_cond: Tensor | None = None):
        if x_cond is None:
            return self.decoder(quantized)
        
        quantized = self.concat(quantized, x_cond)
        to_feature_map = self.decoder_input(quantized)

        return self.decoder(to_feature_map)