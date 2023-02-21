import torch 
import torch.nn as nn


# ENCODER

class Encoder(nn.Module):
    def __init__(self,kernel_size=3, hidden_dims = [ 128, 256], latent_dim=2):
        super(Encoder, self).__init__()

        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        in_channels = 1
        modules = []
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=kernel_size, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(nn.Flatten())

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(12544, latent_dim) # To do calculate dimension size dynamicyally
        self.fc_var = nn.Linear(12544, latent_dim)


    def forward(self, inputs):
        x = self.encoder(inputs)
        z_mean = self.fc_mu(x)
        z_log_var = self.fc_var(x)
        z = self.sampling(z_mean, z_log_var)
        
        return x, z_mean, z_log_var, z
    
    def sampling(self, z_mean, z_log_var):
        eps = torch.randn_like(z_log_var)
        z = z_mean + torch.exp(z_log_var / 2) * eps
        return z
    
# CONDITONAL DECODER

class ConditionalDecoder(nn.Module):
    def __init__(self):
        super(ConditionalDecoder, self).__init__()
        return
    
    def forward(self, inputs):
        return inputs


decoder = ConditionalDecoder()


from ConditionalMNIST import load_mnist
train_loader, test_loader, val_loader  = load_mnist(BATCH_SIZE=128)

example = next(iter(train_loader))[0]

print(example.shape)

output = decoder(example)

print(output.shape)
