from numpy import imag
import torch
import torch.nn as nn
from models.decoders import Decoder
from models.layers.common import ToFeatureMap
from models.helpers import get_feature_map_size
from typing import Tuple
class DecoderWithLatentLayer(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int], hidden_dims=[256, 128, 64, 32]):
        super(DecoderWithLatentLayer, self).__init__()

        feature_map_size = get_feature_map_size(image_size[1], len(hidden_dims))
        self.latentToFeatureMap = ToFeatureMap(feature_map_size=feature_map_size, num_channels=hidden_dims[0])
        self.decoder = Decoder(image_size=image_size, hidden_dims=hidden_dims)

    def forward(self, z):
        feature_map = self.latentToFeatureMap(z)
        output = self.decoder(feature_map)
        return output
    
    def get_input_shape(self):
        return self.latentToFeatureMap.get_input_shape()