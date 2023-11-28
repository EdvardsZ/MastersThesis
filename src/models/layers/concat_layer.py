import torch
import torch.nn as nn
class ConcatLayer(nn.Module):
    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, x, y):
        x = torch.flatten(x, start_dim=1)
        y = torch.flatten(y, start_dim=1)

        return torch.cat((x, y), dim=1)