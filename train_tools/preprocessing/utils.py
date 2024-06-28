import numpy as np

from .mnist.loader import get_all_targets_mnist, get_dataloader_mnist
from .cifar10.loader import get_all_targets_cifar10, get_dataloader_cifar10
from .svhn.loader import get_all_targets_svhn, get_dataloader_svhn
from .cinic10.loader import get_all_targets_cinic10, get_dataloader_cinic10
from .medmnist.loader import get_all_targets_tissuemnist, get_dataloader_tissuemnist
from .medmnist.loader import get_all_targets_pathmnist, get_dataloader_pathmnist

__all__ = ["get_local_loader_funcs", "get_global_loader"]


def get_local_loader_funcs(root, dataset_name):
    """
    get local loader function for local data.
    """

    all_targets = INSTANCE_FUNCS[dataset_name](root, train=True)
    all_targets_test = INSTANCE_FUNCS[dataset_name](root, train=False)

    return all_targets, all_targets_test, LOADER_FUNCS[dataset_name]


def get_global_loader(root, dataset_name, batch_size):
    """
    Set global testloader by the given modes.
    """

    global_loader = LOADER_FUNCS[dataset_name](root, False, batch_size)

    return global_loader


# For Local Loaders
INSTANCE_FUNCS = {
    "mnist": get_all_targets_mnist,
    "cifar10": get_all_targets_cifar10,
    "svhn": get_all_targets_svhn,
    "cinic10": get_all_targets_cinic10,
    "tissuemnist": get_all_targets_tissuemnist,
    "pathmnist": get_all_targets_pathmnist,
}

LOADER_FUNCS = {
    "mnist": get_dataloader_mnist,
    "cifar10": get_dataloader_cifar10,
    "svhn": get_dataloader_svhn,
    "cinic10": get_dataloader_cinic10,
    "tissuemnist": get_dataloader_tissuemnist,
    "pathmnist": get_dataloader_pathmnist,
}
