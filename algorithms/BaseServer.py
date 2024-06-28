import torch
import torch.nn as nn
import numpy as np
import copy
import time
import wandb
import os

from .measures import *

__all__ = ["BaseServer"]


class BaseServer:
    def __init__(
        self,
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler,
        n_rounds=200,
        sample_ratio=0.1,
        local_epochs=5,
        device="cuda:0",
    ):
        """
        Server class controls the overall experiment.
        """
        self.algo_params = algo_params
        self.num_classes = data_distributed["num_classes"]
        self.server_model = model
        self.global_loaders = data_distributed["global"]
        self.criterion = nn.CrossEntropyLoss()
        self.data_distributed = data_distributed
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sample_ratio = sample_ratio
        self.n_rounds = n_rounds
        self.device = device
        self.n_clients = len(data_distributed["partition_map"].keys())
        self.local_epochs = local_epochs
        self.server_results = {}

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
                client_sizes,
                round_local_results,
            ) = self._clients_training(sampled_clients)

            # Get aggregated weights & update global
            ag_weights = self._aggregation(updated_local_weights, client_sizes)
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

            # Measure personalized performance
            # if ((round_idx + 1) % 10 == 0):
            #     if round_idx == self.n_rounds - 1:
            #         sampled_clients = np.arange(self.n_clients)

            #     _, _, round_local_results = self._clients_training(
            #         sampled_clients, finetune=True
            #     )
            #     self._print_stats_with_logging(round_local_results, round_idx + 1)

    def _clients_training(self, sampled_clients, finetune=False):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, client_sizes = [], []
        round_local_results = {}

        server_weights = self.server_model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:

            # Fetch client datasets
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(server_weights, server_optimizer)

            # Local training
            if finetune:
                local_results = self.client.finetune()

            else:
                local_results, local_size = self.client.train()

                # Upload locals
                updated_local_weights.append(self.client.upload_local())
                client_sizes.append(local_size)

            # Update results
            round_local_results = self._results_updater(
                round_local_results, local_results
            )

            # Reset local model
            self.client.reset()

        return updated_local_weights, client_sizes, round_local_results

    def _client_sampling(self, round_idx):
        """Sample clients by given sampling ratio"""

        # make sure for same client sampling for fair comparison
        np.random.seed(round_idx)
        clients_per_round = max(int(self.n_clients * self.sample_ratio), 1)
        sampled_clients = np.random.choice(
            self.n_clients, clients_per_round, replace=False
        )

        return sampled_clients

    def _set_client_data(self, client_idx):
        """Assign local client datasets."""
        self.client.datasize = self.data_distributed["local_sizes"][client_idx]
        self.client.local_loaders = self.data_distributed["local"][client_idx]

    def _aggregation(self, w, ns):
        """Average locally trained model parameters"""
        prop = torch.tensor(ns, dtype=torch.float)
        prop /= torch.sum(prop)
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * prop[0]

        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * prop[i]

        return copy.deepcopy(w_avg)

    def _print_stats_with_logging(self, result_dict, round_idx):
        for key, item in result_dict.items():
            if ("Local" in key) or type(item) == list:
                wandb.log({key: np.mean(item)}, step=round_idx)
                wandb.log({key + "_std": np.std(item)}, step=round_idx)
                print(
                    f"[{key}]: {item}, Avg - {np.mean(item):2.4f} (std {np.std(item):2.4f})"
                )
            else:
                wandb.log({key: item}, step=round_idx)
                print(f"[{key}]: - {item:2.4f}")

    def _results_updater(self, result_dict, result_dict_elem):
        """Combine multiple results as clean format"""

        for key, item in result_dict_elem.items():
            if key not in result_dict.keys():
                result_dict[key] = [item]
            else:
                result_dict[key].append(item)

        return result_dict

    def _save_server_model(self, round_idx):
        weights = self.server_model.state_dict()
        save_path = os.path.join(wandb.run.dir, f"model_{round_idx+1}.pth")
        torch.save(weights, save_path)
        print(f"model_{round_idx+1}.pth Saved!")

    def _print_start(self):
        """Print initial log for experiment"""

        if self.device == "cpu":
            return "cpu"

        if isinstance(self.device, str):
            device_idx = int(self.device[-1])
        elif isinstance(self.device, torch._device):
            device_idx = self.device.index

        device_name = torch.cuda.get_device_name(device_idx)

        print("\n" + "=" * 50)
        print(f"Train start on device: {device_name}")
        print("=" * 50)
