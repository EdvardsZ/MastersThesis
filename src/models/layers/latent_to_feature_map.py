import torch.nn as nn

class LatentToFeatureMap(nn.Module):
    def __init__(self, latent_dim, feature_map_size, num_channels):
        super(LatentToFeatureMap, self).__init__()

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, num_channels * feature_map_size * feature_map_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_channels * feature_map_size * feature_map_size),
            nn.Unflatten(1, (num_channels, feature_map_size, feature_map_size))
        )

    def forward(self, z):
        return self.decoder_input(z)
    
