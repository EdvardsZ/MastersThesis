import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

class ConditionalMNIST(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.mnist = MNIST(root, train, transform, target_transform, download)
        self.data = self.mnist.data

    def __getitem__(self, index):
        x, y = self.mnist[index]
        x_cond = self.condition(x)
        return x, x_cond, y
    
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

def load_mnist(BATCH_SIZE):
    train_set = ConditionalMNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    test_val_set = ConditionalMNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    # split test and validation set
    test_size = 0.5
    test_set_size = int(len(test_val_set) * test_size)
    val_set_size = len(test_val_set) - test_set_size
    test_set, val_set = torch.utils.data.random_split(test_val_set, [test_set_size, val_set_size])

    # mean = train_set.data.float().mean() / 255
    # std = train_set.data.float().std() / 255

    # preprocess = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((mean,), (std,))
    # ])

    # train_set.transform = preprocess
    # test_set.transform = preprocess
    # val_set.transform = preprocess

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

    return train_loader, test_loader, val_loader