from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFAR10Wrapper(Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),  # Ensure size
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ])
        )

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        return img, 0  # label not used

    def __len__(self):
        return len(self.dataset)
