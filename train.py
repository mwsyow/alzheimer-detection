import argparse
import copy
import json
import math
from pathlib import Path

import torch
from dotenv import load_dotenv
from monai.data import DataLoader
from torch import nn

import wandb
from datasets import build_dataset_items, build_split_indices, build_train_val_loaders
from metrics import (
    TRAINING_LOG_METRICS,
    VALIDATION_LOG_METRICS,
    collect_predictions,
    compute_metrics,
    select_threshold,
    summarize_predictions,
    to_wandb_logs,
)
from models import build_model

load_dotenv(override=True)

# checkpoint.monitor value -> key in the metrics dict returned by compute_metrics.
MONITOR_METRIC_KEYS = {
    "val_loss": "loss",
    "val_auc": "roc_auc",
    "val_roc_auc": "roc_auc",
    "val_balanced_accuracy": "balanced_accuracy",
    "val_accuracy": "accuracy",
    "val_f1": "f1",
}

# define_metric(summary="max") stores a nested {"max": ...} dict rather than a scalar,
# which a sweep's metric.name cannot read directly. Write the best value to a flat
# summary key as well, and point the sweep at that.
MONITOR_SUMMARY_KEYS = {
    "val_loss": "Best Validation Loss",
    "val_auc": "Best Validation AUC",
    "val_roc_auc": "Best Validation AUC",
    "val_balanced_accuracy": "Best Validation Balanced Accuracy",
    "val_accuracy": "Best Validation Accuracy",
    "val_f1": "Best Validation F1",
}


def deep_update(base: dict, updates: dict):
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: Path):
    with config_path.open() as f:
        return json.load(f)


def set_nested(config: dict, dotted_key: str, value):
    current = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def apply_sweep_overrides(config: dict, sweep_config) -> dict:
    config = deep_update({}, config)
    for key, value in dict(sweep_config).items():
        if key.startswith("_"):
            continue
        if "." not in key and key in config and not isinstance(value, dict):
            config[key] = value

    for key, value in dict(sweep_config).items():
        if "." in key:
            set_nested(config, key, value)
    return config


def get_checkpoint_dir(run: wandb.Run, config) -> Path:
    checkpoint_dir = Path(config["checkpoint"]["dir"]) / run.id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_metadata(
    checkpoint_dir: Path,
    run: wandb.Run,
    config,
    split_indices: dict[str, list[int]],
):
    torch.save(
        {
            "run_id": run.id,
            "sweep_id": run.sweep_id,
            "config": dict(config),
            "split": split_indices,
        },
        checkpoint_dir / "metadata.pth",
    )


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
    best_monitor_value: float,
    early_stopping_counter: int,
    threshold: float,
    monitor_name: str,
    monitor_value: float,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_monitor_value": best_monitor_value,
            "early_stopping_counter": early_stopping_counter,
            # The operating point tuned on validation; evaluate.py applies it at test time.
            "threshold": threshold,
            "monitor_name": monitor_name,
            "monitor_value": monitor_value,
        },
        checkpoint_path,
    )


def resolve_monitor_value(metrics: dict, monitor_name: str):
    if monitor_name not in MONITOR_METRIC_KEYS:
        raise ValueError(
            f"Unsupported checkpoint.monitor: {monitor_name!r}. "
            f"Expected one of {sorted(MONITOR_METRIC_KEYS)}."
        )
    return metrics.get(MONITOR_METRIC_KEYS[monitor_name])


def is_improvement(value, best_value, mode: str, min_delta: float) -> bool:
    # roc_auc is None when a split happens to be single-class; never treat that as progress.
    if value is None:
        return False
    if mode == "max":
        return value > best_value + min_delta
    return value < best_value - min_delta


def worst_monitor_value(mode: str) -> float:
    return float("-inf") if mode == "max" else float("inf")


def train(
    epochs: int | range,
    model: nn.Module,
    optim: torch.optim.AdamW,
    loss: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: wandb.Run,
    checkpoint_dir: Path = None,
    best_monitor_value: float = None,
    early_stopping_counter: int = 0,
    checkpoint_config: dict = None,
    early_stopping_config: dict = None,
    device: torch.device = None,
):
    if isinstance(epochs, int):
        epochs = range(epochs)

    checkpoint_config = checkpoint_config or {}
    early_stopping_config = early_stopping_config or {}
    min_delta = checkpoint_config.get("min_delta", 0.0)
    monitor_name = checkpoint_config.get("monitor", "val_loss")
    mode = checkpoint_config.get("mode", "min")
    if best_monitor_value is None:
        best_monitor_value = worst_monitor_value(mode)

    for ep in epochs:
        model.train()
        train_logits = []
        train_labels = []
        for images, labels in train_loader:
            _, outputs = train_step(
                model=model,
                optim=optim,
                loss=loss,
                images=images,
                labels=labels,
                device=device,
            )
            train_logits.append(outputs.detach().cpu())
            train_labels.append(labels.detach().cpu())

        # Training predictions come from the parameters as they were mid-epoch and with
        # augmentation applied, so these are running estimates, not a clean eval pass.
        train_results = summarize_predictions(
            logits=torch.cat(train_logits),
            labels=torch.cat(train_labels),
            loss_fn=loss,
        )
        val_results = collect_predictions(
            model=model,
            loader=val_loader,
            loss_fn=loss,
            device=device,
        )

        threshold, _ = select_threshold(val_results["y_true"], val_results["y_prob"])
        val_metrics = compute_metrics(
            y_true=val_results["y_true"],
            y_prob=val_results["y_prob"],
            threshold=threshold,
            loss=val_results["loss"],
        )
        # Same threshold for both splits, otherwise the train/val gap compares two
        # different operating points.
        train_metrics = compute_metrics(
            y_true=train_results["y_true"],
            y_prob=train_results["y_prob"],
            threshold=threshold,
            loss=train_results["loss"],
        )

        val_loss = val_metrics["loss"]
        monitor_value = resolve_monitor_value(val_metrics, monitor_name)
        improved = is_improvement(monitor_value, best_monitor_value, mode, min_delta)

        if improved:
            best_monitor_value = monitor_value
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        checkpoint_kwargs = {
            "epoch": ep,
            "model": model,
            "optimizer": optim,
            "val_loss": val_loss,
            "best_monitor_value": best_monitor_value,
            "early_stopping_counter": early_stopping_counter,
            "threshold": threshold,
            "monitor_name": monitor_name,
            "monitor_value": monitor_value,
        }

        if (
            checkpoint_dir is not None
            and checkpoint_config.get("save_best", True)
            and improved
        ):
            best_filename = checkpoint_config.get(
                "best_filename", "best_epoch_{epoch:03d}.pth"
            ).format(epoch=ep)
            save_checkpoint(
                checkpoint_path=checkpoint_dir / best_filename,
                **checkpoint_kwargs,
            )

        if checkpoint_dir is not None and checkpoint_config.get("save_last", True):
            save_checkpoint(
                checkpoint_path=checkpoint_dir
                / checkpoint_config.get("last_filename", "last.pth"),
                **checkpoint_kwargs,
            )

        early_stopped = early_stopping_config.get(
            "enabled", False
        ) and early_stopping_counter >= early_stopping_config.get("patience", 5)
        logs = {"Epoch": ep}
        logs.update(to_wandb_logs(train_metrics, "Training", TRAINING_LOG_METRICS))
        logs.update(to_wandb_logs(val_metrics, "Validation", VALIDATION_LOG_METRICS))
        logger.log(logs)
        if early_stopped:
            break

    if math.isfinite(best_monitor_value):
        logger.summary[MONITOR_SUMMARY_KEYS[monitor_name]] = best_monitor_value


def train_step(
    model: nn.Module,
    optim: torch.optim.AdamW,
    loss: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device = None,
):
    model.train()
    if device is not None:
        images = images.to(device)
        labels = labels.to(device)
    optim.zero_grad()  # Clear gradients before each step
    output = model(images)
    train_loss = loss(output, labels)
    train_loss.backward()
    optim.step()

    return train_loss, output


def build_loss(config):
    loss_config = config["loss"]
    if loss_config["name"] != "CrossEntropyLoss":
        raise ValueError(f"Unsupported loss: {loss_config['name']}")
    return nn.CrossEntropyLoss(**loss_config.get("params", {}))


def build_optimizer(config, model):
    optimizer_config = config["optimizer"]
    if optimizer_config["name"] != "AdamW":
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
    return torch.optim.AdamW(model.parameters(), **optimizer_config.get("params", {}))


def runner(
    run: wandb.Run,
    config,
    resume_checkpoint: dict | None = None,
    metadata: dict | None = None,
):
    is_resume = resume_checkpoint is not None

    requested_device = config["device"]
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    config.update({"device": str(device)}, allow_val_change=True)

    dataset_items = build_dataset_items(config)

    checkpoint_dir = get_checkpoint_dir(run, config)
    if is_resume:
        split_indices = metadata["split"]
    else:
        split_indices = build_split_indices(dataset_items, config)
        save_metadata(
            checkpoint_dir=checkpoint_dir,
            run=run,
            config=config,
            split_indices=split_indices,
        )

    train_loader, val_loader = build_train_val_loaders(
        dataset_items=dataset_items,
        split_indices=split_indices,
        config=config,
    )

    loss = build_loss(config)
    model = build_model(config)
    model.to(device)
    optim = build_optimizer(config, model)
    if is_resume:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optim.load_state_dict(resume_checkpoint["optimizer_state_dict"])
    epochs = config["epochs"]
    monitor_name = config["checkpoint"].get("monitor", "val_loss")
    mode = config["checkpoint"].get("mode", "min")
    best_monitor_value = worst_monitor_value(mode)
    early_stopping_counter = 0
    if is_resume:
        epochs = range(resume_checkpoint["epoch"] + 1, config["epochs"])
        # Only carry the stored best forward when it was measured on the same quantity.
        # Checkpoints written before threshold tuning only have "best_val_loss", which
        # would be meaningless as a starting point for e.g. an AUC monitor.
        stored_monitor = resume_checkpoint.get(
            "monitor_name", "val_loss" if "best_val_loss" in resume_checkpoint else None
        )
        if stored_monitor == monitor_name:
            best_monitor_value = resume_checkpoint.get(
                "best_monitor_value", resume_checkpoint.get("best_val_loss")
            )
        early_stopping_counter = resume_checkpoint.get("early_stopping_counter", 0)

    train(
        epochs=epochs,
        model=model,
        optim=optim,
        loss=loss,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=run,
        checkpoint_dir=checkpoint_dir,
        best_monitor_value=best_monitor_value,
        early_stopping_counter=early_stopping_counter,
        checkpoint_config=config["checkpoint"],
        early_stopping_config=config["early_stopping"],
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a 3D CNN on OASIS MRI data.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint file to resume from.",
    )
    args = parser.parse_args()

    resume_checkpoint = None
    metadata = None
    wandb_init_kwargs = {}
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        metadata_path = args.resume.parent / "metadata.pth"
        metadata = torch.load(metadata_path, map_location="cpu")
        config = metadata["config"]
        wandb_init_kwargs = {
            "id": metadata["run_id"],
            "resume": "must",
        }
    else:
        config = load_config(args.config)

    wandb_run = wandb.init(
        entity=config["wandb_entity"],
        project=config["wandb_project"],
        name=config.get("wandb_name"),
        mode=config["wandb_mode"],
        config=config,
        **wandb_init_kwargs,
    )
    # Without these the run summary holds the *last* epoch's value, so sweeps rank runs
    # by an arbitrary point on the curve rather than by their best epoch.
    wandb_run.define_metric("Validation AUC", summary="max")
    wandb_run.define_metric("Validation Balanced Accuracy", summary="max")
    wandb_run.define_metric("Validation F1", summary="max")
    wandb_run.define_metric("Validation Loss", summary="min")

    if args.resume is None:
        config = apply_sweep_overrides(config, wandb_run.config)
        wandb_run.config.update(config, allow_val_change=True)

    runner(
        run=wandb_run,
        config=config,
        resume_checkpoint=resume_checkpoint,
        metadata=metadata,
    )
    wandb_run.finish()


if __name__ == "__main__":
    main()
