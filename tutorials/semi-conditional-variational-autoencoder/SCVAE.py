import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, image_size, kernel_size=2, filters=32, latent_dim=2):
        super(Encoder, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.image_shape = (1, image_size, image_size)
        self.cond_shape = (1, image_size, image_size)

        # Encoder input layer
        self.inputs = nn.Sequential(
            nn.Conv2d(1, self.filters, kernel_size=self.kernel_size, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.filters, self.filters*2, kernel_size=self.kernel_size, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Latent vector layer
        self.z_mean = nn.Linear(16, latent_dim)
        self.z_log_var = nn.Linear(16, latent_dim)
        self.z = nn.Sequential(nn.Linear(latent_dim, latent_dim))

    def forward(self, inputs):
        x = self.inputs(inputs)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.z(z_mean)
        return z_mean, z_log_var, z


encoder = Encoder(28)

from mnist import load_mnist
train_loader, test_loader, val_loader  = load_mnist(BATCH_SIZE=128)

example = next(iter(train_loader))[0]

res = encoder(example)

print(res[0].shape)

print(res[1].shape)

print(res[2].shape)

print(res[0])

print(res[1])

print(res[2])