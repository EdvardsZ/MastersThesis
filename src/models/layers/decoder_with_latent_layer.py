import torch
import torch.nn as nn
from models.decoders import VQDecoder
from models.layers.common import ToFeatureMap

class DecoderWithLatentLayer(nn.Module):
    def __init__(self):
        super(DecoderWithLatentLayer, self).__init__()
        
        self.latentToFeatureMap = ToFeatureMap(feature_map_size=7, num_channels=256)
        self.decoder = VQDecoder(in_channels=1, embedding_dim=256, n_residual_layers=1)

    def forward(self, z):
        feature_map = self.latentToFeatureMap(z)
        output = self.decoder(feature_map)
        return output
    
    def get_input_shape(self):
        return self.latentToFeatureMap.get_input_shape()