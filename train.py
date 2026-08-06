import argparse
import copy
import json
from pathlib import Path

import torch
from dotenv import load_dotenv
from monai.data import DataLoader
from torch import nn

import wandb
from datasets import build_dataset_items, build_split_indices, build_train_val_loaders
from models import build_model

load_dotenv(override=True)


DEFAULT_CONFIG = {
    "epochs": 1,
    "device": "auto",
    "image_glob": "data/T88_111_masked/*masked_gfc.img",
    "label_path": "data/oasis_cross-sectional_cdr_cleaned.xlsx",
    "wandb_entity": "wmarcellius123",
    "wandb_project": "alzheimer-detection",
    "wandb_name": None,
    "wandb_mode": "online",
    "model": {
        "name": "Simple3DCNN",
        "params": {
            "num_classes": 2,
            "in_channels": 1,
            "channels": [16, 32, 64],
            "kernel_size": 3,
            "padding": 1,
            "pool_kernel_size": 2,
            "use_batch_norm": True,
            "dropout": 0.0,
        },
    },
    "loss": {
        "name": "CrossEntropyLoss",
        "params": {},
    },
    "optimizer": {
        "name": "AdamW",
        "params": {
            "lr": 1e-3,
        },
    },
    "transforms": {
        "resize": True,
        "spatial_size": [96, 128, 96],
        "resize_mode": "trilinear",
        "scale_intensity": False,
        "normalize_intensity": True,
        "normalize_nonzero": True,
        "normalize_channel_wise": True,
        "rand_rotate90": False,
        "rand_rotate90_prob": 0.5,
        "rand_rotate90_spatial_axes": [0, 2],
    },
    "split": {
        "train_size": 0.70,
        "val_size": 0.15,
        "test_size": 0.15,
        "random_seed": None,
    },
    "dataloader": {
        "batch_size": 2,
        "num_workers": 0,
    },
    "checkpoint": {
        "dir": "checkpoints",
        "save_last": True,
        "save_best": True,
        "best_filename": "best_epoch_{epoch:03d}.pth",
        "last_filename": "last.pth",
        "monitor": "val_loss",
        "mode": "min",
        "min_delta": 0.0,
    },
    "early_stopping": {
        "enabled": False,
        "patience": 5,
    },
}


def deep_update(base: dict, updates: dict):
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: Path | None):
    if config_path is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    with config_path.open() as f:
        user_config = json.load(f)

    return deep_update(DEFAULT_CONFIG, user_config)


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
    best_val_loss: float,
    early_stopping_counter: int,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "early_stopping_counter": early_stopping_counter,
        },
        checkpoint_path,
    )


def train(
    epochs: int | range,
    model: nn.Module,
    optim: torch.optim.AdamW,
    loss: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: wandb.Run,
    metrics: dict = None,
    checkpoint_dir: Path = None,
    best_val_loss: float = float("inf"),
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

    for ep in epochs:
        model.train()
        train_losses = []
        for images, labels in train_loader:
            train_loss = train_step(
                model=model,
                optim=optim,
                loss=loss,
                images=images,
                labels=labels,
                device=device,
            )
            train_losses.append(train_loss.item())

        model.eval()
        val_losses = []
        val_metrics = {}
        for images, labels in val_loader:
            val_loss, batch_metrics = val_step(
                model=model,
                loss=loss,
                images=images,
                labels=labels,
                metrics=metrics,
                device=device,
            )
            val_losses.append(val_loss.item())
            for name, value in batch_metrics.items():
                val_metrics.setdefault(name, []).append(value.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        is_improvement = val_loss < best_val_loss - min_delta

        if is_improvement:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if (
            checkpoint_dir is not None
            and checkpoint_config.get("save_best", True)
            and is_improvement
        ):
            best_filename = checkpoint_config.get(
                "best_filename", "best_epoch_{epoch:03d}.pth"
            ).format(epoch=ep)
            save_checkpoint(
                checkpoint_path=checkpoint_dir / best_filename,
                epoch=ep,
                model=model,
                optimizer=optim,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                early_stopping_counter=early_stopping_counter,
            )

        if checkpoint_dir is not None and checkpoint_config.get("save_last", True):
            save_checkpoint(
                checkpoint_path=checkpoint_dir
                / checkpoint_config.get("last_filename", "last.pth"),
                epoch=ep,
                model=model,
                optimizer=optim,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                early_stopping_counter=early_stopping_counter,
            )

        early_stopped = early_stopping_config.get(
            "enabled", False
        ) and early_stopping_counter >= early_stopping_config.get("patience", 5)
        logs = {
            "Epoch": ep,
            "Training Loss": train_loss,
            "Validation Loss": val_loss,
        }
        logs.update(
            {
                f"Validation {name}": sum(values) / len(values)
                for name, values in val_metrics.items()
            }
        )
        logger.log(logs)
        if early_stopped:
            break


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

    return train_loss


def val_step(
    model: nn.Module,
    loss: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    metrics: dict = None,
    device: torch.device = None,
):
    model.eval()
    results = {}
    with torch.no_grad():
        if device is not None:
            images = images.to(device)
            labels = labels.to(device)
        val_output = model(images)
        val_loss = loss(val_output, labels)
        metrics = metrics or {}
        for name, metric in metrics.items():
            results[name] = metric(val_output, labels)

    return val_loss, results


def accuracy(output, labels):
    return (output.argmax(dim=1) == labels).float().mean()


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
    metrics = {"accuracy": accuracy}
    epochs = config["epochs"]
    best_val_loss = float("inf")
    early_stopping_counter = 0
    if is_resume:
        epochs = range(resume_checkpoint["epoch"] + 1, config["epochs"])
        best_val_loss = resume_checkpoint.get(
            "best_val_loss", resume_checkpoint["val_loss"]
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
        metrics=metrics,
        checkpoint_dir=checkpoint_dir,
        best_val_loss=best_val_loss,
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
        default=None,
        help="Path to a JSON config file. Defaults are used when omitted.",
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
        name=config["wandb_name"],
        mode=config["wandb_mode"],
        config=config,
        **wandb_init_kwargs,
    )
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
