import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST
from .observations import PartialObservation

from typing import Tuple

class ConditionalDataset(Dataset):
    def __init__(self, root: str ='data', train: bool =True, transform=None, target_transform=None, download: bool=False, dataset: str = "MNIST", conditioning_mode: str = "exact"):
        if dataset == "MNIST":
            self.dataset = MNIST(root, train, transform, target_transform, download)
        else:
            if dataset == "FashionMNIST":
                self.dataset = FashionMNIST(root, train, transform, target_transform, download)
            else:
                raise ValueError("dataset must be MNIST or FashionMNIST")
        self.classes_count = len(self.dataset.classes)
        self.partial_observation = PartialObservation(conditioning_mode)


    def __getitem__(self, index):
        x, y = self.dataset[index]

        x_cond = self.partial_observation.get_partial_observation(x)

        return x, x_cond, y
    
    def __len__(self):
        return len(self.dataset)