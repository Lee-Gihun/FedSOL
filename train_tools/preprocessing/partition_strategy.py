import torch, copy
import numpy as np
from collections import Counter

__all__ = [
    "apply_partition_strategy",
    "test_data_partitioning",
    "get_label_distribution",
]


def apply_partition_strategy(all_targets, n_clients, partition):
    """
    Apply Label-skewness to the clients
    """
    if partition.method == "centralized":
        partition_map = _centralized_partition(all_targets)

    elif partition.method == "iid":
        partition_map = _iid_partition(all_targets, n_clients)

    elif partition.method == "sharding":
        partition_map = _sharding_partition(
            all_targets, n_clients, partition.shard_per_client
        )

    elif partition.method == "lda":
        partition_map = _lda_partition(all_targets, n_clients, partition.alpha)

    else:
        raise NotImplementedError

    return partition_map


def test_data_partitioning(
    all_targets_test, all_targets, partition_map, amount_data=1000, label_dist_map=None
):
    """
    Apply pre-defined Label-skewness to the clients using test samples
    """
    # Bulid label dictionary. (key): label (item): dataidxs corresponding to the label
    test_idxs_dict = {}

    for i in range(len(all_targets_test)):
        label = torch.tensor(all_targets_test[i]).item()
        if label not in test_idxs_dict.keys():
            test_idxs_dict[label] = []
        test_idxs_dict[label].append(i)

    # Build label distribution for each client
    if label_dist_map is None:
        label_dist_map = get_label_distribution(partition_map, all_targets)

    # Distribute test samples according to the label distributions
    test_partition_map = {}

    for client_idx, label_dist in label_dist_map.items():
        assign_idxs = []

        candidates = copy.deepcopy(test_idxs_dict)

        if amount_data == -1:
            base_amount = int(100 / max(label_dist))
        else:
            base_amount = amount_data

        for class_idx, prob in enumerate(label_dist):
            amount = min(len(candidates[class_idx]), int(base_amount * prob))
            selected = np.random.choice(candidates[class_idx], amount, replace=False)
            assign_idxs.append(selected)

        assign_idxs = np.concatenate(assign_idxs)
        test_partition_map[client_idx] = np.array(assign_idxs).astype("int")

    return test_partition_map


def get_label_distribution(partition_map, all_targets):
    """
    Calculate label distribution (sum to 1) for each client
    """
    num_classes = len(np.unique(all_targets))
    label_dist_map = {}

    for client_idx, dataidxs in partition_map.items():
        label_dist = torch.zeros(num_classes)
        counter = Counter(all_targets[dataidxs].numpy())

        for class_idx, count in counter.items():
            label_dist[class_idx] = count

        label_dist /= len(dataidxs)
        label_dist_map[client_idx] = label_dist

    return label_dist_map


def _centralized_partition(all_targets):
    """
    Assign all data indices to a single client
    """
    tot_idxs = np.arange(len(all_targets))
    partition_map = {}

    # Shuffle and assign all idices to a single client
    tot_idxs = np.array(tot_idxs)
    np.random.shuffle(tot_idxs)
    partition_map[0] = tot_idxs

    return partition_map


def _iid_partition(all_targets, n_partitions):
    """
    Randomly assign sampled data indices to each client (e.g., i.i.d. sampling)
    """
    length = int(len(all_targets) / n_partitions)
    tot_idxs = np.arange(len(all_targets))
    partition_map = {}

    for i in range(n_partitions):
        np.random.shuffle(tot_idxs)
        data_idxs = tot_idxs[:length]
        tot_idxs = tot_idxs[length:]
        partition_map[i] = np.array(data_idxs)

    return partition_map


def _sharding_partition(all_targets, n_partitions=10, shard_per_partition=2):
    """
    Partition the dataidxs using the Sharding strategy.
    """
    partition_map = {i: np.array([], dtype="int64") for i in range(n_partitions)}
    idxs_dict = {}

    # Bulid label dictionary. (key): label (item): dataidxs corresponding to the label
    for i in range(len(all_targets)):
        label = torch.tensor(all_targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(all_targets))
    shard_per_class = int(shard_per_partition * n_partitions / num_classes)

    assert shard_per_class > 0

    for label, indices in idxs_dict.items():
        num_leftover = len(indices) % shard_per_class
        leftover = indices[-num_leftover:] if num_leftover > 0 else []
        indices = indices[:-num_leftover] if num_leftover > 0 else indices
        indices = np.array(indices).reshape((shard_per_class, -1))
        indices = [list(shard) for shard in indices]

        for i, idx in enumerate(leftover):
            indices[i].append(idx)
        idxs_dict[label] = indices

    # Build random meta-indice set to divide shards
    rand_set_all = np.repeat(np.arange(num_classes), shard_per_class)
    np.random.shuffle(rand_set_all)

    temp_idx = 0
    while (rand_set_all.shape[0] % n_partitions) != 0:
        temp_idx += 1
        temp = rand_set_all
        temp = np.concatenate((rand_set_all, rand_set_all[temp_idx : temp_idx + 1]))
        rand_set_all = temp

    rand_set_all = rand_set_all.reshape((n_partitions, -1))

    # Assign shards to partitions using random indices
    for i in range(n_partitions):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            try:
                selected = np.random.choice(len(idxs_dict[label]), replace=False)
                elem = idxs_dict[label].pop(selected)
                rand_set.append(elem)
            except:
                pass

        try:
            partition_map[i] = np.concatenate(rand_set)
        except:
            partition_map[i] = partition_map[i - 1]

    return partition_map


def _lda_partition(all_targets, n_partitions, alpha=0.1):
    """
    Partition the dataidxs using the Latent Dirichlet Allocation (LDA) strategy.
    """
    length = int(len(all_targets) / n_partitions)
    partition_map, idxs_dict = {}, {}

    # Bulid label dictionary. (key): label (item): dataidxs corresponding to the label
    for i in range(len(all_targets)):
        label = torch.tensor(all_targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_samples, num_classes = len(all_targets), len(np.unique(all_targets))

    min_size = 0

    while min_size < 10:
        idx_batch = [[] for _ in range(n_partitions)]

        for label in range(num_classes):
            # Get a list of batch indices which belongs to class k
            idx_k = idxs_dict[label]
            idx_batch, min_size = _partition_class_samples_with_dirichlet_distribution(
                num_samples, alpha, n_partitions, idx_batch, idx_k
            )
        for i in range(n_partitions):
            np.random.shuffle(idx_batch[i])
            partition_map[i] = np.array(idx_batch[i]).astype("int")

    return partition_map


def _partition_class_samples_with_dirichlet_distribution(
    num_samples, alpha, n_partitions, idx_batch, idx_k
):
    np.random.shuffle(idx_k)

    sizes = np.array([len(idx_j) for idx_j in idx_batch])
    min_size = num_samples // n_partitions
    proportions = np.random.dirichlet(np.repeat(alpha, n_partitions))
    proportions *= sizes < min_size
    proportions /= proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    new_idx_batch = [
        np.concatenate((idx_j, idx_k[start:end])).astype("int64").tolist()
        for idx_j, start, end in zip(
            idx_batch, [0] + proportions.tolist(), proportions.tolist() + [None]
        )
    ]
    min_size = min([len(idx_j) for idx_j in new_idx_batch])

    return new_idx_batch, min_size
