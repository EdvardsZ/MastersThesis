import torch
from torch.utils.data import TensorDataset, Dataset

from torchvision.datasets import MNIST, FashionMNIST, CIFAR100, CIFAR10, CelebA, VisionDataset
from .observations import PartialObservation
from torchvision import transforms
from datasets.observations import CountSamplingMethod, PixelSamplingMethod
import os
import numpy as np


from typing import Tuple



class CelebACached(VisionDataset):
    def __init__(self, root: str, train: bool, transform=None, target_transform=None, download: bool = False, load_in_memory: bool = True):
        
        self.root = root
        self.train = train
        
        self.path = os.path.join(root, "celeba")
        
        self.images, self.labels = self.load_celeba_transformed_cached(self.path, train, transform)
        
        if load_in_memory:
            self.images = self.images[:]
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.images[index])
        y = self.labels[index]
        
        return x, y
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def get_image_shape(self) -> Tuple[int, int, int]:
        return self.images[0].shape
    
    def load_celeba_transformed_cached(self, path, train, transform) -> Tuple[np.memmap, list]:
        # check if the transformed dataset is already cached
        cache_file_images = os.path.join(path, "images_celeba_transformed" + ("_train" if train else "_test") + ".dat")
        cache_file_labels = os.path.join(path, "labels_celeba_transformed" + ("_train" if train else "_test") + ".pt")
        
        if os.path.exists(cache_file_images) and os.path.exists(cache_file_labels):
            return np.memmap(cache_file_images, dtype='float32', mode='r', shape=(len(torch.load(cache_file_labels)), 3, 64, 64)), torch.load(cache_file_labels)
        
        transform.transforms.insert(0, transforms.Resize((64, 64)))
        dataset = CelebA(self.root, split='train' if train else 'test', transform=transform, download=False)
        # lets create np.memmap for the images
        images = np.memmap(cache_file_images, dtype='float32', mode='w+', shape=(len(dataset), 3, 64, 64))
        labels = []
        for i in range(len(dataset)):
            if i % 1000 == 0:
                print(f"Processing image {i}/{len(dataset)}")
            images[i] = dataset[i][0].numpy()
            labels.append(dataset[i][1])
            
        torch.save(labels, cache_file_labels)
        # save the images
        images.flush()
        del images
        
        images = np.memmap(cache_file_images, dtype='float32', mode='r', shape=(len(labels), 3, 64, 64))
        return images, labels
        
        
            
    
    
class ConditionalDataset(Dataset):
    def __init__(self, 
                 root: str ='data', 
                 train: bool =True, 
                 transform=None, 
                 target_transform=None, 
                 download: bool=False, 
                 dataset: str = "MNIST", 
                 data_config: dict | None = None):
        
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            
        self.root = root
        
        self.dataset = self.get_dataset(dataset, root, train, transform, target_transform, download)
        self.partial_observation = PartialObservation(data_config)


    def get_dataset(self, dataset: str, root: str, train: bool, transform, target_transform, download: bool) -> VisionDataset:
        if dataset == "MNIST":
            return MNIST(root, train, transform, target_transform, download)
        if dataset == "FashionMNIST":
            return FashionMNIST(root, train, transform, target_transform, download)
        if dataset == "CIFAR100":
            return CIFAR100(root, train, transform, target_transform, download)
        if dataset == "CIFAR10":
            return CIFAR10(root, train, transform, target_transform, download)
        if dataset == "CelebA":
            return CelebACached(root, train, transform, target_transform, download)
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