import copy
import os, time
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.localonly.ClientTrainer import ClientTrainer
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

        print("\n>>> Local Only initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        start_time = time.time()

        all_clients = np.arange(self.n_clients)
        _, _, round_local_results = self._clients_training(all_clients, finetune=True)

        # Print results and logging
        print(f"\n(Elapsed {round(time.time()-start_time, 1)}s)")
        self._print_stats_with_logging(round_local_results, 1)
