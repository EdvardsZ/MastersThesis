from ast import Dict
import torch
import torch.nn as nn
from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
from typing import List

class SoftAdaptModule(nn.Module):
    def __init__(self, beta: float):
        super(SoftAdaptModule, self).__init__()

        self.adapt_weights : torch.Tensor | None = None
        self.values_of_components = {}
        self.beta = beta

        self.softadapt_object = LossWeightedSoftAdapt(beta=beta)

    def forward(self, losses: List[torch.Tensor], training: bool) -> torch.Tensor:      
        if self.adapt_weights is None:
            self.adapt_weights = torch.ones(len(losses), dtype=torch.float64)
        
        if training:
            for i, loss in enumerate(losses):
                if self.values_of_components.get(i) is None:
                    self.values_of_components[i] = [ loss ]
                else:
                    self.values_of_components[i].append(loss)

            if len(self.values_of_components[0]) > 100:
                self.update_weights()

        loss = sum([self.adapt_weights[i] * loss for i, loss in enumerate(losses)])

        return loss
    
    def update_weights(self):
        values = tuple(self.values_of_components.values())
        values = tuple(torch.tensor(x, dtype=torch.float64) for x in values)
        self.adapt_weights = self.softadapt_object.get_component_weights(*values, verbose=False)
        self.values_of_components = {}
