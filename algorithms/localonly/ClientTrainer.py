import os
import sys
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer
from algorithms.measures import *

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """

    def finetune(self):
        """Local training"""
        tuning_criterion = nn.CrossEntropyLoss()

        self.model.train()
        self.model.to(self.device)

        ft_epochs = self.local_epochs

        for _ in range(ft_epochs):
            for data, targets in self.local_loaders["train"]:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)
                loss = tuning_criterion(output, targets)

                # backward pass
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

        finetune_results = evaluate_model_on_loaders(
            self.model, self.local_loaders, self.device, "Local_Finetune"
        )

        return finetune_results

    def download_global(self, server_weights, server_optimizer):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, gamma=0.99, step_size=1)
