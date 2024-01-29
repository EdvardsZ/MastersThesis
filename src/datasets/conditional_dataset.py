import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR100, CIFAR10, CelebA
from .observations import PartialObservation
from torchvision import transforms
from datasets.observations import CountSamplingMethod, PixelSamplingMethod

from typing import Tuple


class ConditionalDataset(Dataset):
    def __init__(self, 
                 root: str ='data', 
                 train: bool =True, 
                 transform=None, 
                 target_transform=None, 
                 download: bool=False, 
                 dataset: str = "MNIST", 
                 count_sampling: CountSamplingMethod = CountSamplingMethod.EXACT, 
                 pixel_sampling: PixelSamplingMethod = PixelSamplingMethod.EXACT):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        self.dataset = self.get_dataset(dataset, root, train, transform, target_transform, download)
        self.partial_observation = PartialObservation(count_sampling, pixel_sampling, add_mask = True)


    def get_dataset(self, dataset: str, root: str, train: bool, transform, target_transform, download: bool):
        if dataset == "MNIST":
            return MNIST(root, train, transform, target_transform, download)
        if dataset == "FashionMNIST":
            return FashionMNIST(root, train, transform, target_transform, download)
        if dataset == "CIFAR100":
            return CIFAR100(root, train, transform, target_transform, download)
        if dataset == "CIFAR10":
            return CIFAR10(root, train, transform, target_transform, download)
        if dataset == "CelebA":
            # Adding the extra transformation
            transform.transforms.insert(0, transforms.Resize((64, 64)))
            return CelebA(root, split='train' if train else 'test', transform=transform, download=download)
        else:
            raise ValueError("dataset must be MNIST or FashionMNIST or CIFAR100 or CIFAR10 or CelebA")


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = self.dataset[index]

        x_cond = self.partial_observation.get_partial_observation(x)

        return x, x_cond, y
    
    def __len__(self) -> int:
        return len(self.dataset)

    def get_image_shape(self) -> Tuple[int, int, int]:
        return self.dataset[0][0].shape