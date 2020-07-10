import config
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize


class Dataset:
    def __init__(self):
        self.data = MNIST(root='data',
                          train=True,
                          download=True,
                          transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))

    def get_loader(self):
        return DataLoader(self.data, config.BATCH_SIZE, shuffle=True)
