import torch
import torch.nn as nn
from enum import Enum
from .soft_adapt import SoftAdaptModule
from typing import List


class AdaptiveMode(Enum):
    NONE = "NONE",
    SOFT = "SoftAdapt",
    SCALED = "Scaled",


class Adapt(nn.Module):
    def __init__(self, mode: AdaptiveMode | None , soft_adapt_beta: float | None = None):
        super(Adapt, self, ).__init__()
        self.mode = mode
        self.beta = soft_adapt_beta

        if mode == AdaptiveMode.SOFT:
            if soft_adapt_beta is None:
                raise ValueError("SoftAdapt requires a beta value")
            self.soft_adapt = SoftAdaptModule(beta = soft_adapt_beta)


    def forward(self, reconstructions: List[torch.Tensor] , latent_loss: torch.Tensor , training = False):
        
        if self.mode == AdaptiveMode.NONE or self.mode is None:
            return sum(reconstructions) + latent_loss
        elif self.mode == AdaptiveMode.SOFT:
            losses = reconstructions + [latent_loss]
            return self.soft_adapt(losses, training)
        elif self.mode == AdaptiveMode.SCALED:
            n_reconstructions = len(reconstructions)
            return sum(reconstructions) + n_reconstructions * latent_loss

        



    

