import torch 
import torch.nn as nn


# ENCODER

class Encoder(nn.Module):
    def __init__(self, image_size = (1, 28, 28), hidden_dims = [128, 256], latent_dim=2):
        super(Encoder, self).__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        

        print("Image size: ", image_size)
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


    def forward(self, inputs):
        x = self.encoder(inputs)
        z_mean = self.fc_mu(x)
        z_log_var = self.fc_var(x)
        z = self.sampling(z_mean, z_log_var)
        
        return z_mean, z_log_var, z
    
    def sampling(self, z_mean, z_log_var):
        eps = torch.randn_like(z_log_var)
        z = z_mean + torch.exp(z_log_var / 2) * eps
        return z
    
# CONDITONAL DECODER

class ConditionalDecoder(nn.Module):
    def __init__(self, image_size = (1, 28, 28), hidden_dims = [256, 128], latent_dim=2):
        super(ConditionalDecoder, self).__init__()
        self.image_size = image_size

        resulting_size = (image_size[1] // 2**len(hidden_dims))**2 * hidden_dims[0]
        in_channels = image_size[0]



        

        self.decoder_input = nn.Sequential(
            nn.Linear(self.image_size[1] * self.image_size[1] + latent_dim, resulting_size),
            nn.ReLU(),
            nn.BatchNorm1d(resulting_size),
            nn.Unflatten(1, (hidden_dims[0], int(image_size[1]/(2**len(hidden_dims))), int(image_size[2]/(2**len(hidden_dims)))))
        )

        modules = []

        modules.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        ))

        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU())
            )

        modules.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], in_channels, kernel_size=3,padding=1),
            nn.Sigmoid()
        ))

        self.decoder_output = nn.Sequential(*modules)
        
        return
    
    def forward(self, z, cond_input):
        x_cond = torch.flatten(cond_input, start_dim=1)
        x_cat = torch.cat((z, x_cond), dim=1)

        input = self.decoder_input(x_cat)

        output = self.decoder_output(input)

        return output
    

# CONDITIONAL VAE
class ConditionalVAE(nn.Module):
    def __init__(self, kernel_size=3, hidden_dims = [128, 256], latent_dim=2):
        super(ConditionalVAE, self).__init__()
        self.encoder = Encoder(hidden_dims = hidden_dims, latent_dim = latent_dim)
        self.decoder = ConditionalDecoder(hidden_dims= [256, 128], latent_dim = latent_dim)

        # learn weight for KL loss through backprop
        self.weight_kl = 1
        self.weight_recon = 1

        
    def forward(self, inputs, cond_input):
        z_mean, z_log_var, z = self.encoder(inputs)
        output = self.decoder(z, cond_input)
        return output, z_mean, z_log_var, z
    
    def recon_loss(self, inputs, outputs):
        return nn.functional.binary_cross_entropy(outputs, inputs, reduction='sum')
    
    def kl_loss(self, z_mean, z_log_var):
        return -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    
    def loss(self, inputs, outputs, z_mean, z_log_var):
        recon_loss = self.recon_loss(inputs, outputs)
        kl_loss = self.kl_loss(z_mean, z_log_var)
        return recon_loss, kl_loss, self.weight_recon * recon_loss + self.weight_kl * kl_loss

# from ConditionalMNIST import load_mnist
# train_loader, test_loader, val_loader  = load_mnist(BATCH_SIZE=128)

# example = next(iter(train_loader))
# image = example[0]
# cond_image = example[1]
# print("Example shape image: ", image.shape)
# print("Example shape cond_image: ", cond_image.shape)
# encoder = Encoder()
# z_mean, z_log_var, z = encoder(image)

# decoder = ConditionalDecoder()

# print("Example shape z_mean: ", z_mean.shape)
# print("Example shape z_log_var: ", z_log_var.shape)
# print("Example shape z: ", z.shape)

# output = decoder(z, cond_image)

# print(output.shape)

# cond_vae = ConditionalVAE()
# output, z_mean, z_log_var, z = cond_vae(image, cond_image)

# print("Example shape output: ", output.shape)
# print("Example shape z_mean: ", z_mean.shape)
# print("Example shape z_log_var: ", z_log_var.shape)


# loss = cond_vae.loss(image, output, z_mean, z_log_var)

