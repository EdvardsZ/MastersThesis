from ast import Dict
import torch
import torch.nn as nn
from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
from typing import List

class SoftAdaptModule(nn.Module):
    def __init__(self):
        super(SoftAdaptModule, self).__init__()

        self.adapt_weights = torch.tensor([1,1])
        self.values_of_components = {}

        self.softadapt_object = LossWeightedSoftAdapt(beta=0.001)

    def forward(self, losses: List[torch.Tensor], training: bool) -> torch.Tensor:
        
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
        print("UPDATING WEIGHTS")
        print(self.adapt_weights)
        values = tuple(self.values_of_components.values())
        values = tuple(torch.tensor(x, dtype=torch.float64) for x in values)
        self.adapt_weights = self.softadapt_object.get_component_weights(*values, verbose=False)
        self.values_of_components = {}
