import torch
import torch.nn as nn
import numpy as np
import copy
import time
import os
import sys
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.feddyn.ClientTrainer import ClientTrainer
from algorithms.feddyn.utils import *
from algorithms.measures import *


__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler=None, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )

        # Dictionaries for local weights and local grads
        self.local_weights = {}
        self.local_grads = {}

        # FedDyn global dependency
        self.dyn_alpha = self.algo_params.dyn_alpha
        self.h_t = None

        self.client = ClientTrainer(
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

        print("\n>>> FedDyn Server initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        # Initialize updated_local_weights and grads
        self._feddyn_init()

        for round_idx in range(self.n_rounds):
            # Initial Model Statistics
            if round_idx == 0:
                global_results = evaluate_model_on_loaders(
                    self.server_model, self.global_loaders, self.device
                )
                self.server_results = self._results_updater(
                    self.server_results, global_results
                )

            start_time = time.time()

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            history_dict = {"client_history": sampled_clients}
            self.server_results = self._results_updater(
                self.server_results, history_dict
            )

            # (Distributed) global weights
            dg_weights = detached_statedict(self.server_model.state_dict())

            # Client training stage to upload weights & stats
            client_sizes, round_local_results = self._clients_training(sampled_clients)

            # Get aggregated weights & update global
            ag_weights = self._aggregation(dg_weights, client_sizes)
            self.server_model.load_state_dict(ag_weights)

            # Evaluate server statistics
            global_results = evaluate_model_on_loaders(
                self.server_model, self.global_loaders, self.device, "Global"
            )
            self.server_results = self._results_updater(
                self.server_results, global_results
            )

            # Print results and logging
            print(
                f"\n[Round {round_idx+1}/{self.n_rounds}] (Elapsed {round(time.time()-start_time, 1)}s)"
            )
            self._print_stats_with_logging(global_results, round_idx)
            self._print_stats_with_logging(round_local_results, round_idx)

            # Change learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # if (round_idx + 1) % 10 == 0:
            #     if round_idx == self.n_rounds - 1:
            #         sampled_clients = np.arange(self.n_clients)

            #     _, round_local_results = self._clients_training(
            #         sampled_clients, finetune=True
            #     )
            #     self._print_stats_with_logging(round_local_results, round_idx + 1)

    def _feddyn_init(self):
        """
        Initialize updated_local_params & updated_local_params
        For consistency, we send model to the cuda device at the first.
        """
        self.server_model.to(self.device)

        init_weights = self.server_model.state_dict()
        init_grads = detached_statedict(init_weights)
        for k in init_grads.keys():
            init_grads[k] = torch.zeros_like(init_grads[k])

        for idx in range(self.n_clients):
            self.local_weights[idx] = detached_statedict(init_weights)
            self.local_grads[idx] = detached_statedict(init_grads)

        self.h_t = detached_statedict(init_grads)

    def _clients_training(self, sampled_clients, finetune=False):
        """
        Conduct local training and get trained local models' weights
        """
        client_sizes = {}
        round_results = {}

        server_weights = self.server_model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(
                server_weights, server_optimizer, self.local_grads[client_idx]
            )

            if finetune:
                local_results = self.client.finetune()

            else:
                # Local training
                local_results, local_size = self.client.train()

                # Upload locals
                (
                    self.local_weights[client_idx],
                    self.local_grads[client_idx],
                ) = self.client.upload_local(server_weights)

                client_sizes[client_idx] = local_size

            # Update results
            round_results = self._results_updater(round_results, local_results)

            # Reset local model
            self.client.reset()

        return client_sizes, round_results

    def _aggregation(self, dg_weights, ns):
        """
        Average locally trained model parameters
        For FedDyn algorithm, we also update h_t here.
        ns is dictionary with key-value pair:
         {client_idx sampled at this round : data size}
        """
        # Calculate the proportions of each client
        total_size = torch.sum(torch.tensor(list(ns.values()), dtype=torch.float))
        prop = {}
        for idx in ns.keys():
            prop[idx] = ns[idx] / total_size

        with torch.no_grad():
            # Update the server state; h_t
            for k in self.h_t.keys():
                for idx in ns.keys():
                    self.h_t[k] -= (self.dyn_alpha / self.n_clients) * (
                        self.local_weights[idx][k] - dg_weights[k]
                    )
            # Calculate w_avg from -h_t/dyn_alpha
            w_avg = detached_statedict(self.h_t)
            for k in w_avg.keys():
                w_avg[k] *= -(1.0 / self.dyn_alpha)
                for idx in ns.keys():
                    w_avg[k] += self.local_weights[idx][k] * prop[idx]

        return detached_statedict(w_avg)
