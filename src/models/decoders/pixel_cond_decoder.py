#Conditional decoder
import torch
import torch.nn as nn
from models.helpers.sampling import count_div_by_2

class PixelConditionedDecoder(nn.Module):
    def __init__(self, image_size = (1, 28, 28), hidden_dims = [256, 128, 64, 32], latent_dim=2):
        super(PixelConditionedDecoder, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        count_div = count_div_by_2(image_size[1])
        feature_size = len(hidden_dims) - count_div


        res_dim = (image_size[1] // 2**feature_size)
        resulting_size = res_dim**2 * hidden_dims[0]

        self.decoder_input = nn.Sequential(
            nn.Linear(self.image_size[1] * self.image_size[1] + latent_dim, resulting_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(resulting_size),
            nn.Unflatten(1, (hidden_dims[0], res_dim, res_dim))
        )

        modules = []

        for i in range(len(hidden_dims)-1):
            stride = 1 if feature_size > 0 else 2
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],
                                       kernel_size=3, stride=stride, padding=1, output_padding=stride - 1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU())
            )
            feature_size -= 1

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