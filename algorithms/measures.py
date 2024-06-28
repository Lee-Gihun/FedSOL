import torch
import torch.nn.functional as F
from .utils import *

__all__ = [
    "evaluate_model_on_loaders",
    "evaluate_model",
    "evaluate_model_classwise",
]


@torch.no_grad()
def evaluate_model_on_loaders(model, dataloaders, device="cuda:0", prefix="Global"):
    results = {}

    for loader_key, dataloader in dataloaders.items():
        if dataloader is not None:
            key = f"{prefix}_{loader_key}"
            results[key] = evaluate_model(model, dataloader, device)

    return results


@torch.no_grad()
def evaluate_model(model, dataloader, device="cuda:0", return_loss=False):
    """Evaluate model accuracy for the given dataloader"""
    model.eval()
    model.to(device)

    running_count = 0
    running_correct = 0
    running_loss = 0.0

    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)

        logits = model(data)
        pred = logits.max(dim=1)[1]
        loss = F.cross_entropy(logits, targets, reduction="none")

        running_correct += (targets == pred).sum().item()
        running_loss += loss.sum().item()
        running_count += data.size(0)

    accuracy = round(running_correct / running_count, 4)
    loss = round(running_loss / running_count, 4)

    if return_loss:
        return accuracy, loss

    else:
        return accuracy


@torch.no_grad()
def evaluate_model_classwise(
    model, dataloader, num_classes=10, device="cuda:0",
):
    """Evaluate class-wise accuracy for the given dataloader."""

    model.eval()
    model.to(device)

    classwise_count = torch.Tensor([0 for _ in range(num_classes)]).to(device)
    classwise_correct = torch.Tensor([0 for _ in range(num_classes)]).to(device)

    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)

        logits = model(data)
        preds = logits.max(dim=1)[1]

        for class_idx in range(num_classes):
            class_elem = targets == class_idx
            classwise_count[class_idx] += class_elem.sum().item()
            classwise_correct[class_idx] += (targets == preds)[class_elem].sum().item()

    classwise_accuracy = (classwise_correct / classwise_count).cpu()

    return classwise_accuracy
