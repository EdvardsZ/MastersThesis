import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms

class ConditionalMNIST(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, dataset = "MNIST"):
        if dataset == "MNIST":
            self.mnist = MNIST(root, train, transform, target_transform, download)
        else:
            if dataset == "FashionMNIST":
                self.mnist = FashionMNIST(root, train, transform, target_transform, download)
            else:
                raise ValueError("dataset must be MNIST or FashionMNIST")
        self.data = self.mnist.data
        self.classes_count = len(self.mnist.classes)

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
        obs_x, obs_y = get_observation_pixels()
        cond_data = torch.zeros_like(data)
        cond_data[:, obs_x, obs_y] = data[:, obs_x, obs_y]
        return cond_data



def get_observation_pixels():
        start=2
        stop=26
        obs_x_n=6
        obs_y_n=6

        obs_x=[]
        obs_y=[]
        for i in range(start,stop,obs_x_n):
            for j in range(start,stop,obs_y_n):
                obs_x.append(i)
                obs_y.append(j)
        return obs_x, obs_y

def load_dataset(data_config):
    BATCH_SIZE = data_config['batch_size']
    dataset = data_config['dataset']

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x - 0.5)
    ])

    train_set = ConditionalMNIST(root='data', train=True, download=True, transform=preprocess, dataset = dataset)
    test_val_set = ConditionalMNIST(root='data', train=False, download=True, transform=preprocess, dataset = dataset)
    # split test and validation set
    test_size = 0.5
    test_set_size = int(len(test_val_set) * test_size)
    val_set_size = len(test_val_set) - test_set_size

    test_set, val_set = torch.utils.data.random_split(test_val_set, [test_set_size, val_set_size])


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

    return train_loader, test_loader, val_loader