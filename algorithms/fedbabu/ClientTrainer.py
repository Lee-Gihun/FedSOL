import os
import sys
import torch
import torch.nn as nn
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer
from algorithms.BaseClientTrainer import evaluate_model_on_loaders

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """

    def finetune(self):
        self.model.train()
        self.model.to(self.device)

        for params in self.model.parameters():
            params.requires_grad = True

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

        finetune_results = evaluate_model_on_loaders(
            self.model, self.local_loaders, self.device, "Local_Finetune"
        )

        return finetune_results

    def download_global(self, server_weights, server_optimizer):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)

        classifier_name = list(self.model.named_children())[-1][0]

        # Freeze Classifier Head
        for name, params in self.model.named_parameters():
            if classifier_name in name:
                params.requires_grad = False
