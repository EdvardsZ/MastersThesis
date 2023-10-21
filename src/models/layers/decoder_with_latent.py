import torch
import torch.nn as nn
from models.decoders import Decoder
from .latent_to_feature_map import LatentToFeatureMap
class DecoderWithLatentLayer(nn.Module):
    def __init__(self, latent_dim, input_size = None):
        super(DecoderWithLatentLayer, self).__init__()
        
        self.latentToFeatureMap = LatentToFeatureMap(latent_dim=latent_dim, feature_map_size=7, num_channels=256, input_size=input_size)
        self.decoder = Decoder()

    def forward(self, z):
        feature_map = self.latentToFeatureMap(z)
        output = self.decoder(feature_map)
        return output