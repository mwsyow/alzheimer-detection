"""Task definitions shared by training, evaluation, and artifact export."""

import torch
from torch import nn


TASKS = ("ad_classification", "age_regression")


def task_name(config: dict) -> str:
    name = config.get("task", {}).get("name", "ad_classification")
    if name not in TASKS:
        raise ValueError(f"Unsupported task {name!r}; expected one of {TASKS}")
    return name


def is_regression(config: dict) -> bool:
    return task_name(config) == "age_regression"


class StandardizedHuberLoss(nn.Module):
    """Huber loss in train-fold-standardized units, with reversible scaling."""

    task_name = "age_regression"

    def __init__(self, mean: float, std: float, delta: float = 1.0):
        super().__init__()
        if std <= 0:
            raise ValueError("Age standard deviation must be positive")
        self.mean = float(mean)
        self.std = float(std)
        self.delta = float(delta)
        self.loss = nn.HuberLoss(delta=self.delta)

    def standardize(self, target: torch.Tensor) -> torch.Tensor:
        return (target.float() - self.mean) / self.std

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction.float() * self.std + self.mean

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(output.reshape(-1), self.standardize(target).reshape(-1))

    def metadata(self) -> dict:
        return {"mean": self.mean, "std": self.std, "delta": self.delta}


def adaptation_mode(config: dict) -> str:
    pretrained = config.get("model", {}).get("pretrained", {})
    if not pretrained.get("enabled", pretrained.get("enable", False)):
        return "scratch"
    source = pretrained.get("source") or "local"
    if not pretrained.get("freeze_backbone", True):
        return f"{source}:full_finetune"
    stages = int(pretrained.get("unfreeze_last_stages", 0))
    return f"{source}:unfreeze_last_{stages}" if stages else f"{source}:head_only"
