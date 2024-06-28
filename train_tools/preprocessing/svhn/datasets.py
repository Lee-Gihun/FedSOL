import torch
import torch.utils.data as data
from torchvision.datasets import SVHN
import os, sys


class SVHN_truncated(data.Dataset):
    def __init__(
        self, root, split="train", dataidxs=None, transform=None, download=True,
    ):
        self.root = root
        self.split = split
        self.dataidxs = dataidxs
        self.transform = transform
        self.download = download
        self.num_classes = 10

        self.data, self.targets = self._build_truncated_dataset()

    def _build_truncated_dataset(self):
        sys.stdout = open(os.devnull, "w")
        base_dataset = SVHN(
            os.path.join(self.root, "svhn"),
            self.split,
            self.transform,
            None,
            self.download,
        )
        sys.stdout = sys.__stdout__

        data = torch.FloatTensor(base_dataset.data)
        targets = torch.LongTensor(base_dataset.labels)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return len(self.data)
