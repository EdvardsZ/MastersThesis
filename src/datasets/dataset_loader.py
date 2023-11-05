import torch
from torchvision import transforms
from .conditional_dataset import ConditionalDataset

from typing import Tuple

def load_dataset(data_config: dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, Tuple[int, int, int]]:
    BATCH_SIZE = data_config['batch_size']
    dataset = data_config['dataset']
    conditioning_mode = data_config.get("conditioning_mode", "exact") # optional, default is exact


    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = ConditionalDataset(train=True, download=True, transform=preprocess, dataset = dataset, conditioning_mode = conditioning_mode)
    test_val_set = ConditionalDataset(train=False, download=True, transform=preprocess, dataset = dataset, conditioning_mode = conditioning_mode)
    # split test and validation set
    test_size = 0.5
    test_set_size = int(len(test_val_set) * test_size)
    val_set_size = len(test_val_set) - test_set_size

    test_set, val_set = torch.utils.data.random_split(test_val_set, [test_set_size, val_set_size])


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

    return train_loader, val_loader, test_loader, train_set.get_image_shape()