import torch
import torch.nn as nn
from models.helpers.sampling import count_div_by_2
    
# Conventional Decoder
class Decoder(nn.Module):
    def __init__(self, image_size = (1, 28, 28), hidden_dims = [256, 128, 64, 32]):
        super(Decoder, self).__init__()
        self.image_size = image_size


        count_div = count_div_by_2(image_size[1])
        feature_size = len(hidden_dims) - count_div


        res_dim = (image_size[1] // 2**feature_size)
        resulting_size = res_dim**2 * hidden_dims[0]

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
    
    def forward(self, z):
        output = self.decoder_output(z)

        return output