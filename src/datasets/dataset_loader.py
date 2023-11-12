import torch
from torchvision import transforms
from .conditional_dataset import ConditionalDataset
from datasets.observations import CountSamplingMethod, PixelSamplingMethod

from typing import Tuple

def load_dataset(data_config: dict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, Tuple[int, int, int]]:
    BATCH_SIZE = data_config['batch_size']
    dataset = data_config['dataset']
    count_sampling = data_config['count_sampling']
    pixel_sampling = data_config['pixel_sampling']

    count_sampling = CountSamplingMethod(count_sampling)
    pixel_sampling = PixelSamplingMethod(pixel_sampling)


    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = ConditionalDataset(train=True, transform=preprocess, dataset = dataset, count_sampling = count_sampling, pixel_sampling = pixel_sampling)
    test_val_set = ConditionalDataset(train=False, transform=preprocess, dataset = dataset, count_sampling = count_sampling, pixel_sampling = pixel_sampling)
    # split test and validation set
    test_size = 0.5
    test_set_size = int(len(test_val_set) * test_size)
    val_set_size = len(test_val_set) - test_set_size

    test_set, val_set = torch.utils.data.random_split(test_val_set, [test_set_size, val_set_size])


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

    return train_loader, val_loader, test_loader, train_set.get_image_shape()