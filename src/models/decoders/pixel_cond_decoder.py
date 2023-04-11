#Conditional decoder
import torch
import torch.nn as nn

class PixelConditionedDecoder(nn.Module):
    def __init__(self, image_size = (1, 28, 28), hidden_dims = [256, 128], latent_dim=2):
        super(PixelConditionedDecoder, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        resulting_size = (image_size[1] // 2**len(hidden_dims))**2 * hidden_dims[0]
        in_channels = image_size[0]

        self.decoder_input = nn.Sequential(
            nn.Linear(self.image_size[1] * self.image_size[1] + latent_dim, resulting_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(resulting_size),
            nn.Unflatten(1, (hidden_dims[0], int(image_size[1]/(2**len(hidden_dims))), int(image_size[2]/(2**len(hidden_dims)))))
        )

        modules = []

        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 
                               kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1, stride=1),
            )
        )

        self.decoder_output = nn.Sequential(*modules)
        
        return
    
    def forward(self, z, cond_input):
        x_cond = torch.flatten(cond_input, start_dim=1)
        x_cat = torch.cat((z, x_cond), dim=1)

        input = self.decoder_input(x_cat)

        output = self.decoder_output(input)

        return output