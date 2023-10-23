import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from .observations import get_observation_pixels, get_random_observation_pixels


def get_observation(mode):
    if mode == "exact":
        return get_observation_pixels
    elif mode == "random":
        return get_random_observation_pixels
    else:
        raise ValueError("mode must be exact or random")

class ConditionalMNIST(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, dataset = "MNIST", conditioning_mode = "exact"):
        if dataset == "MNIST":
            self.mnist = MNIST(root, train, transform, target_transform, download)
        else:
            if dataset == "FashionMNIST":
                self.mnist = FashionMNIST(root, train, transform, target_transform, download)
            else:
                raise ValueError("dataset must be MNIST or FashionMNIST")
        self.data = self.mnist.data
        self.classes_count = len(self.mnist.classes)
        self.conditioning_mode = conditioning_mode

    def __getitem__(self, index):
        x, y = self.mnist[index]

        x_cond = self.condition(x)
        classes_count = len(self.mnist.classes)
        y_one_hot = torch.zeros(classes_count)
        y_one_hot[y] = 1
        return x, x_cond, y_one_hot
    
    def __len__(self):
        return len(self.mnist)

    def condition(self, data):
        obs_x, obs_y = get_observation(self.conditioning_mode)(data.shape)
        if self.conditioning_mode == "exact":
            cond_data = torch.zeros_like(data)
            cond_data[:, obs_x, obs_y] = data[:, obs_x, obs_y]
            return cond_data
        if self.conditioning_mode == "random": 
            cond_data = torch.zeros((data.shape[0] + 1, data.shape[1], data.shape[2]))
            cond_data[1:, obs_x, obs_y] = data[:, obs_x, obs_y]
            cond_data[0, obs_x, obs_y] = 1

            return cond_data
        raise ValueError("conditioning_mode must be exact or random!!!")

def load_dataset(data_config):
    BATCH_SIZE = data_config['batch_size']
    dataset = data_config['dataset']
    conditioning_mode = data_config.get("conditioning_mode", "exact") # optional, default is exact


    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x - 0.5)
    ])

    train_set = ConditionalMNIST(root='data', train=True, download=True, transform=preprocess, dataset = dataset, conditioning_mode = conditioning_mode)
    test_val_set = ConditionalMNIST(root='data', train=False, download=True, transform=preprocess, dataset = dataset, conditioning_mode = conditioning_mode)
    # split test and validation set
    test_size = 0.5
    test_set_size = int(len(test_val_set) * test_size)
    val_set_size = len(test_val_set) - test_set_size

    test_set, val_set = torch.utils.data.random_split(test_val_set, [test_set_size, val_set_size])


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

    return train_loader, test_loader, val_loader