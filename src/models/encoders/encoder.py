import torch.nn as nn

# ENCODER
class Encoder(nn.Module):
    def __init__(self, image_size = (1, 28, 28), hidden_dims = [128, 256], latent_dim=2):
        super(Encoder, self).__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        in_channels = self.image_size[0]

        modules = []
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(nn.Flatten())

        resulting_size = (image_size[1] // 2**len(hidden_dims))**2 * hidden_dims[-1]

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(resulting_size, latent_dim)
        self.fc_var = nn.Linear(resulting_size, latent_dim)