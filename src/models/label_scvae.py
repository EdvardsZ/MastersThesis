import torch
import torch.nn as nn
class LabelConditionalVAE(nn.Module):
    def __init__(self) -> None:
        super(LabelConditionalVAE, self).__init__()
        