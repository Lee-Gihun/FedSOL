import torch
import os
import sys
import math
import copy

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

        self.beta = self.algo_params.beta

    def train(self):
        """Local training"""

        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize
        iter_loader = iter(self.local_loaders["train"])
        num_iter = math.ceil(len(self.local_loaders["train"]) / self.local_epochs)

        for _ in range(self.local_epochs):
            for _ in range(num_iter):
                temp_net = copy.deepcopy(list(self.model.parameters()))

                # Step-1
                for params in self.optimizer.param_groups:
                    params["lr"] = self.lr

                try:
                    data, targets = next(iter_loader)
                except:
                    iter_loader = iter(self.local_loaders["train"])
                    data, targets = next(iter_loader)

                data, targets = data.to(self.device), targets.to(self.device)

                self.model.zero_grad()

                output = self.model(data)
                loss = self.criterion(output, targets)
                loss.backward()
                self.optimizer.step()

                # Step-2
                for params in self.optimizer.param_groups:
                    params["lr"] = self.beta

                try:
                    data, targets = next(iter_loader)
                except:
                    iter_loader = iter(self.local_loaders["train"])
                    data, targets = next(iter_loader)

                data, targets = data.to(self.device), targets.to(self.device)

                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, targets)
                loss.backward()

                # restore the model parameters to the one before first update
                for old_p, new_p in zip(self.model.parameters(), temp_net):
                    old_p.data = new_p.data.clone()

                self.optimizer.step()

        local_results = self._get_local_stats()

        return local_results, local_size

    def download_global(self, server_weights, server_optimizer):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)
        optim_params = self.optimizer.state_dict()
        self.lr = optim_params["param_groups"][0]["lr"]

    def upload_local(self):
        """Uploads local model's parameters"""
        local_weights = copy.deepcopy(self.model.state_dict())

        return local_weights

    def reset(self):
        """Clean existing setups"""
        self.datasize = None
        self.local_loaders = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)

    def finetune(self):
        """Local training"""

        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize
        iter_loader = iter(self.local_loaders["train"])
        num_iter = math.ceil(len(self.local_loaders["train"]) / self.local_epochs)

        for _ in range(self.ft_epochs):
            for batch_idx in range(num_iter):
                temp_net = copy.deepcopy(list(self.model.parameters()))

                # Step-1
                for params in self.optimizer.param_groups:
                    params["lr"] = self.lr

                try:
                    data, targets = next(iter_loader)
                except:
                    iter_loader = iter(self.local_loaders["train"])
                    data, targets = next(iter_loader)

                data, targets = data.to(self.device), targets.to(self.device)

                self.model.zero_grad()

                output = self.model(data)
                loss = self.criterion(output, targets)
                loss.backward()
                self.optimizer.step()

                # Step-2
                for params in self.optimizer.param_groups:
                    params["lr"] = self.beta

                try:
                    data, targets = next(iter_loader)
                except:
                    iter_loader = iter(self.local_loaders["train"])
                    data, targets = next(iter_loader)

                data, targets = data.to(self.device), targets.to(self.device)

                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, targets)
                loss.backward()

                # restore the model parameters to the one before first update
                for old_p, new_p in zip(self.model.parameters(), temp_net):
                    old_p.data = new_p.data.clone()

                self.optimizer.step()

        finetune_results = evaluate_model_on_loaders(
            self.model, self.local_loaders, self.device, "Local_Finetune"
        )

        return finetune_results
