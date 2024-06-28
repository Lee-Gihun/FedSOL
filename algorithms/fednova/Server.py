import torch
import numpy as np
import time
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.measures import *
from algorithms.BaseServer import BaseServer
from algorithms.fednova.ClientTrainer import ClientTrainer
from algorithms.optim_utils import *

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        """
        Server class controls the overall experiment.
        """
        self.client = ClientTrainer(
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

        self.local_a = self._init_nova_dependencies()
        self.weighted_d, self.local_d = self._init_control_variates()

        print("\n>>> FedNova Server initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()

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
            dg_weights = copy.deepcopy(self.server_model.state_dict())

            # Client training stage to upload weights & stats
            (
                updated_local_weights,
                client_sizes,
                round_local_results,
            ) = self._clients_training(sampled_clients)

            # Get aggregated weights & update global
            ag_weights = self._nova_aggregation(
                dg_weights, updated_local_weights, client_sizes
            )
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

            # if ((round_idx + 1) % 10 == 0):
            #     if round_idx == self.n_rounds - 1:
            #         sampled_clients = np.arange(self.n_clients)

            #     _, _, round_local_results = self._clients_training(
            #         sampled_clients, finetune=True
            #     )
            #     self._print_stats_with_logging(round_local_results, round_idx + 1)

    def _clients_training(self, sampled_clients, finetune=False):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, client_sizes = {}, {}
        round_results = {}

        server_weights = self.server_model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:

            # Fetch client datasets
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(
                server_weights,
                server_optimizer,
                self.weighted_d,
                self.local_d[client_idx],
            )

            if finetune:
                local_results = self.client.finetune()

            else:
                # Local training
                local_results, local_size, aidi = self.client.train()

                # Upload locals
                updated_local_weights[client_idx] = self.client.upload_local()
                self.local_d[client_idx] = aidi / self.local_a[client_idx]
                client_sizes[client_idx] = local_size

            # Update results
            round_results = self._results_updater(round_results, local_results)

            # Reset local model
            self.client.reset()

        return updated_local_weights, client_sizes, round_results

    @torch.no_grad()
    def _nova_aggregation(self, dg_weights, local_weights, client_sizes):
        """Average locally trained model parameters"""
        sampled_datasize = sum(list(client_sizes.values()))
        p_i = copy.deepcopy(client_sizes)

        for idx in local_weights.keys():
            p_i[idx] /= sampled_datasize

        c_1, c_2 = 0.0, 0.0

        for idx in local_weights.keys():
            c_1 += p_i[idx] * self.local_a[idx]
            c_2 += p_i[idx] / self.local_a[idx]

        c = 1.0 - c_1 * c_2

        ag_weights = copy.deepcopy(dg_weights)

        for k in ag_weights.keys():
            ag_weights[k] = ag_weights[k] * c

        for k in ag_weights.keys():
            for idx in local_weights.keys():
                ag_weights[k] += (
                    local_weights[idx][k] * c_1 * p_i[idx] / self.local_a[idx]
                )

        return ag_weights

    @torch.no_grad()
    def _update_weighted_d(self, sampled_clients):

        sampled_size = 0
        total_size = 0

        for i in range(self.n_clients):
            dsize = self.data_distributed["local"][i]["datasize"]
            if i in sampled_clients:
                sampled_size += dsize
            total_size += dsize

        p = sampled_size * 1.0 / total_size
        self.weighted_d *= 1.0 - p

        for i in sampled_clients:
            dsize = self.data_distributed["local"][i]["datasize"]
            self.weighted_d += (dsize / total_size) * self.local_d[i]

    def __get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def __get_momentum(self):
        for param_group in self.optimizer.param_groups:
            return param_group["momentum"]

    def _init_nova_dependencies(self):
        tau_i, a_i = {}, {}
        rho = self.__get_momentum()
        for i in range(self.n_clients):
            tau_i[i] = self.local_epochs * len(
                list(self.data_distributed["local"][i]["train"])
            )
            a_i[i] = (tau_i[i] - rho * (1.0 - pow(rho, tau_i[i])) / (1.0 - rho)) / (
                1.0 - rho
            )

        return a_i

    def _init_control_variates(self):
        weighted_d = flatten_weights(self.server_model)
        weighted_d = torch.from_numpy(weighted_d).fill_(0)
        local_d = {}

        for i in range(self.n_clients):
            local_d[i] = copy.deepcopy(weighted_d)

        return weighted_d, local_d
