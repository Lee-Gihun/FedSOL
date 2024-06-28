import torch
import torch.nn as nn
import os
import sys
import random

# import faiss
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """

        self.capacity = self.algo_params.capacity
        self.dimension = self.algo_params.dimension
        self.weight = self.algo_params.weight
        self.scale = self.algo_params.scale
        self.k = self.algo_params.k

        self.datastore = DataStore(self.capacity, self.dimension)

    def finetune(self):
        self.model.train()
        self.model.to(self.device)

        tuning_criterion = nn.CrossEntropyLoss()

        for _ in range(self.ft_epochs):
            for data, targets in self.local_loaders["train"]:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)
                loss = tuning_criterion(output, targets)

                # backward pass
                loss.backward()
                self.optimizer.step()

        finetune_results = self.evaluate_knn_model_on_loaders()

        return finetune_results

    @torch.no_grad()
    def evaluate_knn_model_on_loaders(self):
        self.model.eval()
        self.model.to(self.device)
        results = {}
        prefix = "Local_Finetune"

        train_logits, train_features, train_targets = [], [], []

        for data, targets in self.local_loaders["train"]:
            data, targets = data.to(self.device), targets.to(self.device)
            logits, features = self.model(data, get_features=True)
            train_logits.append(logits.cpu())
            train_features.append(features.cpu())
            train_targets.append(targets.cpu())

        train_logits = torch.cat(train_logits, dim=0)
        train_features = torch.cat(train_features, dim=0)
        train_targets = torch.cat(train_targets, dim=0)

        train_pred = torch.max(train_logits, dim=1)[1]
        train_accuracy = (train_pred == train_targets).sum().item() / len(train_targets)

        self.datastore.clear()
        self.datastore.build(train_features, train_targets)

        for loader_key, dataloader in self.local_loaders.items():
            if dataloader is not None:
                if loader_key == "train":
                    results[f"{prefix}_{loader_key}"] = round(train_accuracy, 4)
                else:
                    results[f"{prefix}_{loader_key}"] = self._evaluate_knn_model(
                        dataloader
                    )

        return results

    def _evaluate_knn_model(self, dataloader):
        model_logits, test_features, test_targets = [], [], []

        for data, targets in dataloader:
            data, targets = data.to(self.device), targets.to(self.device)
            logits, features = self.model(data, get_features=True)
            model_logits.append(logits.cpu())
            test_features.append(features.cpu())
            test_targets.append(targets.cpu())

        model_logits = torch.cat(model_logits, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_targets = torch.cat(test_targets, dim=0)

        knn_logits = self._get_knn_logits(test_features)

        pred_logits = self.weight * knn_logits + (1 - self.weight) * model_logits
        pred = torch.max(pred_logits, dim=1)[1]
        correct = (pred == test_targets).sum().item()

        accuracy = round(correct / len(test_targets), 4)

        return accuracy

    def _get_knn_logits(self, features):
        distances, indices = self.datastore.index.search(features.cpu().numpy(), self.k)
        similarities = np.exp(-distances / (features.shape[-1] * self.scale))
        neighbors_targets = self.datastore.targets[indices]

        masks = np.zeros((self.num_classes,) + similarities.shape)

        for class_idx in range(self.num_classes):
            masks[class_idx] = neighbors_targets == class_idx

        knn_logits = (similarities * masks).sum(axis=2) / similarities.sum(axis=1)

        return torch.Tensor(knn_logits.T)


class DataStore:
    def __init__(self, capacity, dimension):
        self.capacity = capacity
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.targets = None

    def build(self, features, targets):
        num_samples = len(features)
        features_ = features.numpy()
        targets_ = targets.numpy()
        if num_samples <= self.capacity:
            self.index.add(features_)
            self.targets = targets_
        else:
            indices = random.sample(list(range(num_samples)), self.capacity)
            self.index.add(features_[indices])
            self.targets = targets_[indices]

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.targets = None
