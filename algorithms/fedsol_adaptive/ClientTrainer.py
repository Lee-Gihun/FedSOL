import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer
from algorithms.optim_utils import *

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        self.sam_optimizer = None
        self.perturb_head = self.algo_params.perturb_head
        self.perturb_body = self.algo_params.perturb_body
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")

    def train(self):
        """Local training"""

        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize

        for _ in range(self.local_epochs):
            for data, targets in self.local_loaders["train"]:
                data, targets = data.to(self.device), targets.to(self.device)

                # first forward-backward pass
                enable_running_stats(self.model)

                if not self.perturb_head:
                    freeze_head(self.model)

                if not self.perturb_body:
                    freeze_body(self.model)

                data, targets = data.to(self.device), targets.to(self.device)
                logits, dg_logits = self.model(data), self._get_dg_logits(data)

                with torch.no_grad():
                    dg_probs = torch.softmax(dg_logits / 3, dim=1)
                pred_probs = F.log_softmax(logits / 3, dim=1)

                loss = self.KLDiv(
                    pred_probs, dg_probs
                )  # use this loss for any training statistics
                loss.backward()

                if not self.perturb_head:
                    zerograd_head(self.model)

                if not self.perturb_body:
                    zerograd_body(self.model)

                self.sam_optimizer.first_step(zero_grad=True)

                unfreeze(self.model)

                # second forward-backward pass
                disable_running_stats(self.model)
                self.criterion(
                    self.model(data), targets
                ).backward()  # make sure to do a full forward pass
                self.sam_optimizer.second_step(zero_grad=True)

        local_results = self._get_local_stats()

        return local_results, local_size

    def download_global(self, server_weights, server_optimizer, rho):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)
        self._keep_global()
        self.sam_optimizer = self._get_sam_optimizer(self.optimizer, rho)

    def _get_sam_optimizer(self, base_optimizer, rho):
        optim_params = base_optimizer.state_dict()
        lr = optim_params["param_groups"][0]["lr"]
        momentum = optim_params["param_groups"][0]["momentum"]
        weight_decay = optim_params["param_groups"][0]["weight_decay"]
        sam_optimizer = ExpSAM(
            self.model.parameters(),
            self.dg_model.parameters(),
            base_optimizer=torch.optim.SGD,
            rho=rho,
            adaptive=False,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        return sam_optimizer

    @torch.no_grad()
    def _get_dg_logits(self, data):
        dg_logits = self.dg_model(data)

        return dg_logits


class ExpSAM(torch.optim.Optimizer):
    def __init__(
        self, params, ref_params, base_optimizer, rho=0.05, adaptive=False, **kwargs
    ):
        # assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(ExpSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        self.ref_param_groups = []
        ref_param_groups = list(ref_params)

        if not isinstance(ref_param_groups[0], dict):
            ref_param_groups = [{"params": ref_param_groups}]

        for ref_param_group in ref_param_groups:
            self.add_ref_param_group(ref_param_group)

        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group, ref_group in zip(self.param_groups, self.ref_param_groups):
            scale = group["rho"] / (grad_norm + 1e-12)

            for p, ref_p in zip(group["params"], ref_group["params"]):
                if p.grad is None:
                    try:
                        self.state[p]["old_p"] = p.data.clone()
                    except:
                        pass

                    continue

                # avg_mag = torch.abs(p - ref_p).mean()

                self.state[p]["old_p"] = p.data.clone()
                e_w = F.normalize((p - ref_p).abs(), 2, dim=0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    (1.0 * p.grad).norm(p=2).to(shared_device)
                    for group, ref_group in zip(
                        self.param_groups, self.ref_param_groups
                    )
                    for p, ref_p in zip(group["params"], ref_group["params"])
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def add_ref_param_group(self, param_group):
        params = param_group["params"]

        if isinstance(params, torch.Tensor):
            param_group["params"] = [params]
        else:
            param_group["params"] = list(params)

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        params = param_group["params"]

        param_set = set()
        for group in self.ref_param_groups:
            param_set.update(set(group["params"]))

        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.ref_param_groups.append(param_group)
