import torch
import torch.nn as nn

class ClassificationLayer(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationLayer, self).__init__()

        self.clasification = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        return self.clasification(x)