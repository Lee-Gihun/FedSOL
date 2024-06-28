import torch
import copy
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseServer import BaseServer
from algorithms.scaffold.ClientTrainer import ClientTrainer
from algorithms.optim_utils import *
from algorithms.measures import *

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

        self.c, self.local_c = self._init_control_variates()

        print("\n>>> Scaffold Server initialized...\n")

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

            # Client training stage to upload weights & stats
            (
                updated_local_weights,
                updated_c_amounts,
                client_sizes,
                round_local_results,
            ) = self._clients_training(sampled_clients)

            # Get aggregated weights & update global
            ag_weights = self._aggregation(updated_local_weights, client_sizes)
            self.server_model.load_state_dict(ag_weights)

            # Get aggregated control variates
            with torch.no_grad():
                updated_c_amounts = torch.stack(updated_c_amounts).mean(dim=0)
                self.c += self.sample_ratio * updated_c_amounts

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

            #     _, _, _, round_local_results = self._clients_training(
            #         sampled_clients, finetune=True
            #     )
            #     self._print_stats_with_logging(round_local_results, round_idx + 1)

    def _clients_training(self, sampled_clients, finetune=False):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, updated_c_amounts, client_sizes = [], [], []
        round_results = {}

        server_weights = self.server_model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(
                server_weights, server_optimizer, self.c, self.local_c[client_idx]
            )

            # Local training
            if finetune:
                local_results = self.client.finetune()

            else:
                (
                    local_results,
                    local_size,
                    c_i_plus,
                    c_update_amount,
                ) = self.client.train()

                # Upload locals
                updated_local_weights.append(self.client.upload_local())
                updated_c_amounts.append(c_update_amount)
                client_sizes.append(local_size)
                self.local_c[client_idx] = c_i_plus

            # Update results
            round_results = self._results_updater(round_results, local_results)

            # Reset local model
            self.client.reset()

        return updated_local_weights, updated_c_amounts, client_sizes, round_results

    def _init_control_variates(self):
        c = flatten_weights(self.server_model)
        c = torch.from_numpy(c).fill_(0)

        local_c = []

        for _ in range(self.n_clients):
            local_c.append(copy.deepcopy(c))

        return c, local_c
