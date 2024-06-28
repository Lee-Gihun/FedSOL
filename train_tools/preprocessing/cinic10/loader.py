import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import os

from .datasets import CINIC10


def _data_transforms_cinic10():
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    # Transformer for train set: random crops and horizontal flip
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: F.pad(
                    x.unsqueeze(0), (4, 4, 4, 4), mode="reflect"
                ).data.squeeze()
            ),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std),
        ]
    )

    # Transformer for test set
    valid_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=cinic_mean, std=cinic_std),]
    )
    return train_transform, valid_transform


def get_all_targets_cinic10(root, train=True):
    if train:
        root = os.path.join(root, "cinic10", "train")
    else:
        root = os.path.join(root, "cinic10", "test")

    dataset = CINIC10(root)
    all_targets = dataset.get_train_labels

    return all_targets


def get_dataloader_cinic10(
    root, train=True, batch_size=50, dataidxs=None, biasidxs=None
):
    train_transform, valid_transform = _data_transforms_cinic10()

    if train:
        root = os.path.join(root, "cinic10", "train")
        dataset = CINIC10(root, dataidxs, transform=train_transform)
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
        )

    else:
        root = os.path.join(root, "cinic10", "test")
        dataset = CINIC10(root, dataidxs, transform=valid_transform)
        dataloader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=5
        )

    return dataloader
