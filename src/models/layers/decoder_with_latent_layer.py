import torch
import torch.nn as nn
from models.decoders import Decoder
from .to_feature_map import ToFeatureMap

class DecoderWithLatentLayer(nn.Module):
    def __init__(self):
        super(DecoderWithLatentLayer, self).__init__()
        
        self.latentToFeatureMap = ToFeatureMap(feature_map_size=7, num_channels=256)
        self.decoder = Decoder()

    def forward(self, z):
        feature_map = self.latentToFeatureMap(z)
        output = self.decoder(feature_map)
        return output