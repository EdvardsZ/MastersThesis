import torch
import torch.nn as nn

class ClassificationLayer(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationLayer, self).__init__()


        self.linear = nn.LazyLinear(num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return x