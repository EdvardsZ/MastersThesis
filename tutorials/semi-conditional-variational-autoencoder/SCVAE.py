import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,kernel_size=2, filters=128, latent_dim=2):
        super(Encoder, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        # Encoder input layer
        # padding = "same" and stride = 2 not supported in PyTorch
        # https://discuss.pytorch.org/t/conv2d-padding-same-and-stride-2/4691/2
        self.inputs = nn.Sequential(
            nn.Conv2d(1, self.filters, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.filters, self.filters*2, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Latent vector layer
        self.z_mean = nn.Linear(16, latent_dim, bias=True)
        self.z_log_var = nn.Linear(16, latent_dim, bias=True)
        self.z = nn.Sequential(nn.Linear(latent_dim, latent_dim))

    def forward_cnn(self, inputs):
        x = self.inputs(inputs)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.z(z_mean)
        return z_mean, z_log_var, z


encoder = Encoder(28)


from ConditionalMNIST import load_mnist
train_loader, test_loader, val_loader  = load_mnist(BATCH_SIZE=128)

example = next(iter(train_loader))[0]

res = encoder.forward_cnn(example)

print(res[0].shape)

print(res[1].shape)

print(res[2].shape)

print(res[0])

print(res[1])

print(res[2])