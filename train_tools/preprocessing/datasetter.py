import numpy as np

from .partition_strategy import *
from .utils import *

__all__ = ["data_distributer"]


def data_distributer(root, dataset_name, batch_size, n_clients, partition):
    """
    Distribute dataloaders for server and locals by the given partition method.
    """

    assert dataset_name in [
        "mnist",
        "cifar10",
        "svhn",
        "cinic10",
        "pathmnist",
        "tissuemnist",
    ]

    all_targets, all_targets_test, local_loader_func = get_local_loader_funcs(
        root, dataset_name
    )
    num_classes = len(np.unique(all_targets))

    # Apply Label-skewness
    partition_map = apply_partition_strategy(all_targets, n_clients, partition)
    test_partition_map = test_data_partitioning(
        all_targets_test, all_targets, partition_map
    )

    print(">>> Distributing client train data...")

    # Set local train & test loaders
    local_loaders = {}
    local_sizes = {}

    for client_idx in range(n_clients):
        if (client_idx + 1) % 10 == 0:
            print(">>> %d th Client is ready..." % (client_idx + 1))

        dataidxs = partition_map[client_idx]
        local_trainloader = local_loader_func(root, True, batch_size, dataidxs)

        local_test_dataidxs = test_partition_map[client_idx]
        local_align_testloader = local_loader_func(
            root, False, batch_size, local_test_dataidxs,
        )

        local_loaders[client_idx] = {
            "train": local_trainloader,
            "test_align": local_align_testloader,
        }
        local_sizes[client_idx] = len(dataidxs)

    print(">>> Building Global evaluation data...")

    # Set global test loaders
    global_loaders = dict()
    global_loaders["clean"] = get_global_loader(root, dataset_name, batch_size)

    for client_idx in range(n_clients):
        local_loaders[client_idx]["test_clean_global"] = global_loaders["clean"]

    data_distributed = {
        "local": local_loaders,
        "global": global_loaders,
        "local_sizes": local_sizes,
        "partition_map": partition_map,
        "test_partition_map": test_partition_map,
        "all_targets": all_targets,
        "all_targets_test": all_targets_test,
        "num_classes": num_classes,
    }

    return data_distributed
