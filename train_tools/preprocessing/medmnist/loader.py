import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from .datasets import MedMNIST


def _data_transforms_medmnist():
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    valid_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )
    return train_transforms, valid_transforms


def get_all_targets_tissuemnist(root, train=True):
    dataset = MedMNIST(root=root, train=train, modality="tissue")
    all_targets = dataset.targets
    return all_targets


def get_all_targets_pathmnist(root, train=True):
    dataset = MedMNIST(root=root, train=train, modality="path")
    all_targets = dataset.targets
    return all_targets


def get_dataloader_pathmnist(root, train=True, batch_size=50, dataidxs=None):
    train_transforms, valid_transforms = _data_transforms_medmnist()
    dataset = MedMNIST(
        root,
        train=train,
        dataidxs=dataidxs,
        transform=train_transforms if train else valid_transforms,
        modality="path",
    )
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True if train else False, num_workers=5
    )

    return dataloader


def get_dataloader_tissuemnist(root, train=True, batch_size=50, dataidxs=None):
    train_transforms, valid_transforms = _data_transforms_medmnist()
    dataset = MedMNIST(
        root,
        train=train,
        dataidxs=dataidxs,
        transform=train_transforms if train else valid_transforms,
        modality="tissue",
    )
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True if train else False, num_workers=5
    )

    return dataloader
