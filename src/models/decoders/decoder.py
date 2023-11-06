import torch.nn as nn
from models.helpers import get_decoder_stride_sizes, stride_size
    
# Conventional Decoder
class Decoder(nn.Module):
    def __init__(self, image_size = (1, 28, 28), hidden_dims = [256, 128, 64, 32]):
        super(Decoder, self).__init__()

        modules = []

        stride_sizes = get_decoder_stride_sizes(image_size[1], len(hidden_dims))

        for i in range(len(hidden_dims)-1):
            stride = stride_sizes[i]
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],
                                       kernel_size=3, stride=stride, padding=1, output_padding=stride - 1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 
                               kernel_size=3, stride = stride_sizes[-1], padding=1, output_padding=1),
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
    