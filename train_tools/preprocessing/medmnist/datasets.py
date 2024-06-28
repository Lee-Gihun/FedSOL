import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class MedMNIST(Dataset):
    def __init__(
        self, root, train=True, dataidxs=None, transform=None, modality="tissue"
    ):
        self.root = root
        self.train = train
        self.dataidxs = dataidxs
        self.transform = transform
        self.modality = modality
        self.num_classes = None
        self.data, self.targets = self._build_truncated_dataset()

    def _build_truncated_dataset(self):
        npz_file = np.load(
            os.path.join(self.root, "medmnist", self.modality + "mnist.npz")
        )

        if self.train:
            data = np.concatenate(
                [npz_file["train_images"], npz_file["val_images"]], axis=0
            )
            targets = np.concatenate(
                [npz_file["train_labels"], npz_file["val_labels"]], axis=0
            ).ravel()

        else:
            data = npz_file["test_images"]
            targets = npz_file["test_labels"].ravel()

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        targets = torch.LongTensor(targets)
        self.num_classes = len(np.unique(npz_file["test_labels"]))

        return data, targets

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, targets

    def __len__(self):
        return len(self.data)
