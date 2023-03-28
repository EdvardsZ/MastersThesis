import torch
import torch.nn as nn

class LinearDecoder(nn.Module):
    def __init__(self, image_size = (1, 28, 28), n_hidden=500, latent_dim=2, keep_prob=0.99):
        super(LinearDecoder, self).__init__()
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        in_channels = self.image_size[0]
        
        flat_size = image_size[1] * image_size[2]
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_hidden),
            nn.ELU(inplace=True),
            nn.Dropout(1-keep_prob),
            
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(1-keep_prob),
            
            nn.Linear(n_hidden, flat_size),
            nn.Sigmoid(),
        )
        
    def forward(self, inputs):
        x = self.decoder(inputs)
        #reshape to image size
        x = x.view(-1, *self.image_size)
        return x