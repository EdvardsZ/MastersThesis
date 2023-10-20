import torch.nn as nn

class LatentToFeatureMap(nn.Module):
    def __init__(self, latent_dim, feature_map_size, num_channels, input_size=None):
        super(LatentToFeatureMap, self).__init__()
        if input_size is None:
            input_size = latent_dim

        self.decoder_input = nn.Sequential(
            nn.Linear(input_size, num_channels * feature_map_size * feature_map_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_channels * feature_map_size * feature_map_size),
            nn.Unflatten(1, (num_channels, feature_map_size, feature_map_size))
        )

    def forward(self, z):
        return self.decoder_input(z)
    
