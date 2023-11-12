import math
import torch
from typing import Tuple, List
from PIL import Image
import torchvision.transforms as transforms

class PartialObservation:
    def __init__(self, conditioning_mode: str, add_mask: bool = True):
        self.conditioning_mode = conditioning_mode

        self.add_mask = add_mask

        if add_mask is None and conditioning_mode == "random":
            self.add_mask = True

    


