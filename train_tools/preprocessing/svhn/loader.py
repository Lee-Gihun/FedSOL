import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from .datasets import SVHN_truncated


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def _data_transforms_svhn():
    SVHN_MEAN = [0.4376821, 0.4437697, 0.47280442]
    SVHN_STD = [0.19803012, 0.20101562, 0.19703614]

    train_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )

    valid_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )

    return train_transforms, valid_transforms


def get_all_targets_svhn(root, train=True):
    split = "train" if train else "test"
    dataset = SVHN_truncated(root=root, split=split)
    all_targets = dataset.targets
    return all_targets


def get_dataloader_svhn(root, train=True, batch_size=50, dataidxs=None):
    split = "train" if train else "test"
    train_transforms, valid_transforms = _data_transforms_svhn()
    dataset = SVHN_truncated(
        root,
        split=split,
        dataidxs=dataidxs,
        transform=train_transforms if train else valid_transforms,
    )
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True if train else False, num_workers=5
    )

    return dataloader
