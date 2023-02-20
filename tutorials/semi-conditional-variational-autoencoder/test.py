import torch
import torch.nn as nn

class GundNet(nn.Module):
    def __init__(self, image_size, kernel_size=2, filters=32):
        super(GundNet, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.image_size = image_size
        
        # Conditional input layer
        self.cond_shape = (1, image_size, image_size)
        self.cond_input = nn.Sequential(
            nn.Conv2d(1, self.filters, kernel_size=self.kernel_size, stride=2, padding='same'),
            nn.ReLU(),
            nn.Conv2d(self.filters, self.filters*2, kernel_size=self.kernel_size, stride=2, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16, latent_dim)
        )

        # Encoder input layer
        self.inputs = nn.Sequential(
            nn.Conv2d(1, self.filters, kernel_size=self.kernel_size, stride=2, padding='same'),
            nn.ReLU(),
            nn.Conv2d(self.filters, self.filters*2, kernel_size=self.kernel_size, stride=2, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Latent vector layer
        self.z_mean = nn.Linear(16, latent_dim)
        self.z_log_var = nn.Linear(16, latent_dim)
        self.z = nn.Sequential(nn.Linear(latent_dim, latent_dim))

        # Decoder input layer
        self.latent_inputs = nn.Sequential(
            nn.Linear(latent_dim, self.image_size*self.image_size*self.filters*2),
            nn.ReLU(),
            nn.Unflatten(1, (self.filters*2, self.image_size, self.image_size))
        )
        self.decoder_inputs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.cond_shape[1]*self.cond_shape[2], self.image_size*self.image_size*self.filters*2),
            nn.ReLU(),
            nn.Unflatten(1, (self.filters*2, self.image_size, self.image_size))
        )
        
        # Decoder output layer
        self.decoder_output = nn.Sequential(
            nn.ConvTranspose2d(self.filters*2, self.filters, kernel_size=self.kernel_size, stride=2, padding='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(self.filters, 1, kernel_size=self.kernel_size, stride=2, padding='same'),
            nn.Sigmoid(),
        )

    def encode(self, inputs, cond_input):
        x = self.inputs(inputs)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.z(z_mean, z_log_var)
        return z_mean, z_log_var, z
    
    def decode(self, z, cond_input):
        x = self.latent_inputs(z)
        y = self.decoder_inputs(cond_input)
        x = torch.cat((x, y), dim=1)
        x = self.decoder_output(x)
        return x

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, inputs, cond_input):
        z_mean, z_log_var, z = self.encode(inputs, cond_input)
        z = self.reparameterize(z_mean, z_log_var)
        recon = self.decode(z, cond_input)
        return recon, z_mean, z_log_var, z

# Create the model and the loss function
model = GundNet(image_size=image_size)
optimizer = torch.optim.Adam(model
