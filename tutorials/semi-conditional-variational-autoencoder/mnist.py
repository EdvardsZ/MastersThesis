import torch
from torchvision import datasets
from torchvision import transforms

def load_mnist(BATCH_SIZE):
    train_set = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    test_val_set = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    # split test and validation set
    test_size = 0.5
    test_set_size = int(len(test_val_set) * test_size)
    val_set_size = len(test_val_set) - test_set_size
    test_set, val_set = torch.utils.data.random_split(test_val_set, [test_set_size, val_set_size])

    mean = train_set.data.float().mean() / 255
    std = train_set.data.float().std() / 255

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    train_set.transform = preprocess
    test_set.transform = preprocess
    val_set.transform = preprocess

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, val_loader


    