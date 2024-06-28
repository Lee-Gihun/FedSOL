import numpy as np
import random
from .models import *

__all__ = ["random_seeder", "create_models"]


MODELS = {
    "fedavg_mnist": fedavgnet.FedAvgNetMNIST,
    "fedavg_cifar": fedavgnet.FedAvgNetCIFAR,
    "vgg11": vgg.vgg11,
    "res10": resnet.resnet10,
    "res18": resnet.resnet18,
}

NUM_CLASSES = {
    "mnist": 10,
    "cifar10": 10,
    "svhn": 10,
    "cinic10": 10,
    "tissuemnist": 8,
    "pathmnist": 9,
}

IMG_SIZES = {
    "mnist": 28,
    "cifar10": 32,
    "svhn": 32,
    "cinic10": 32,
    "tissuemnist": 28,
    "pathmnist": 28,
}


def create_models(model_name, dataset_name):
    """Create a network model"""

    num_classes = NUM_CLASSES[dataset_name]

    if model_name == "vit":
        image_size = IMG_SIZES[dataset_name]
        patch_size = min(IMG_SIZES[dataset_name] // 4, 16)
        model = small_vit.ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=192,
        )

    else:
        model = MODELS[model_name](num_classes=num_classes)

    return model


def random_seeder(seed):
    """Fix randomness"""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
