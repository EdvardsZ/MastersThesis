#Conditional decoder
import torch
import torch.nn as nn

class LabelConditionalDecoder(nn.Module):
    def __init__(self, image_size = (1, 28, 28), hidden_dims = [256, 128], latent_dim=2, classes_count = 10):
        super(LabelConditionalDecoder, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.classes = classes_count
        self.hidden_dims = hidden_dims

        resulting_size = (image_size[1] // 2**len(hidden_dims))**2 * hidden_dims[0]
        
        self.decoder_input = nn.Sequential(
            nn.Linear(self.image_size[1] * self.image_size[1] + classes_count, resulting_size),
            nn.ReLU(),
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
    
    def forward(self, z, label):
        label = torch.nn.functional.one_hot(label, self.classes).float()
        x = torch.cat((z, label), 1)
        x = self.decoder_input(x)
        x = self.decoder_output(x)
        return x

    
