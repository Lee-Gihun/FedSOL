import copy
import time
import os, sys


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.fedsol_fixed.ClientTrainer import ClientTrainer
from algorithms.BaseServer import BaseServer
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

        self.rho = self.algo_params["rho"]

        print("\n>>> FedSOL (Fixed Rho) Server initialized...\n")

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

            # if (round_idx + 1) % 10 == 0:
            # if True:
            #     if round_idx == self.n_rounds - 1:
            #         sampled_clients = np.arange(self.n_clients)

            #     _, _, round_local_results = self._clients_training(
            #         sampled_clients, finetune=True
            #     )
            #     self._print_stats_with_logging(round_local_results, round_idx + 1)

            if (round_idx + 1) % 100 == 0:
                self._save_server_model(round_idx)

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
