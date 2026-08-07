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
from datasets import (
    build_cv_split_indices,
    build_dataset_items,
    build_fold_loaders,
    build_split_indices,
    build_train_val_loaders,
    is_cv_enabled,
)
from metrics import (
    TRAINING_LOG_METRICS,
    VALIDATION_LOG_METRICS,
    WANDB_METRIC_LABELS,
    aggregate_fold_metrics,
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

# Cross-validation layout: checkpoints/<run_id>/split_<k>/best_model.pth
FOLD_DIR_TEMPLATE = "split_{fold}"
CV_BEST_FILENAME = "best_model.pth"
# Deliberately not one of MONITOR_SUMMARY_KEYS: only the parent writes the sweep
# objective, so fold runs can never win sweep.best_run().
FOLD_SUMMARY_KEY = "Fold Best Validation Metric"


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


def find_metadata_path(checkpoint_path: Path) -> Path:
    """Locate metadata.pth for a checkpoint.

    Single-split runs keep it beside the checkpoint; CV runs keep one at the run root,
    a level above checkpoints/<run_id>/split_<k>/.
    """
    for directory in (checkpoint_path.parent, checkpoint_path.parent.parent):
        candidate = directory / "metadata.pth"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No metadata.pth beside or above {checkpoint_path}. Expected it in "
        f"{checkpoint_path.parent} or {checkpoint_path.parent.parent}."
    )


def load_metadata(metadata_path: Path) -> dict:
    """Load a metadata.pth we wrote ourselves.

    It holds bookkeeping rather than tensors -- config, split indices, per-fold results
    -- so torch's weights_only guard, which rejects arbitrary scalar types, is disabled.
    """
    return torch.load(metadata_path, map_location="cpu", weights_only=False)


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
    fold: int = None,
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
            # None for single-split runs; the 1-based fold index under CV, so a
            # checkpoint is self-describing even away from its metadata.
            "fold": fold,
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
    fold: int = None,
    summary_key: str = None,
    on_epoch_end=None,
    best_result: dict = None,
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
    if summary_key is None:
        summary_key = MONITOR_SUMMARY_KEYS[monitor_name]
    # Carried across a resume so an interrupted fold keeps the best epoch it already found.
    best_result = dict(best_result) if best_result else None

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
            best_result = {
                "epoch": ep,
                "threshold": threshold,
                "monitor_name": monitor_name,
                "monitor_value": monitor_value,
                "metrics": val_metrics,
            }
            if fold is not None:
                best_result["fold"] = fold
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
            "fold": fold,
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

        # Persist resume state only after the epoch's checkpoints are on disk, so
        # metadata never points at an epoch whose weights were not written.
        if on_epoch_end is not None:
            on_epoch_end(
                epoch=ep,
                fold=fold,
                best_monitor_value=best_monitor_value,
                best_result=best_result,
                early_stopping_counter=early_stopping_counter,
            )

        if early_stopped:
            break

    if math.isfinite(best_monitor_value):
        logger.summary[summary_key] = best_monitor_value

    return {
        "fold": fold,
        "best_monitor_value": best_monitor_value if math.isfinite(best_monitor_value) else None,
        "best_result": best_result,
        "early_stopping_counter": early_stopping_counter,
    }


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


def resolve_device(config) -> torch.device:
    requested_device = config["device"]
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    config.update({"device": str(device)}, allow_val_change=True)
    return device


def resume_monitor_state(resume_checkpoint: dict, monitor_name: str, mode: str):
    """Best-so-far and early-stopping counter to carry into a resumed run.

    The stored best is only reused when it was measured on the same quantity: a
    checkpoint written before threshold tuning holds a loss under "best_val_loss",
    which would be meaningless as a starting point for an AUC monitor.
    """
    best_monitor_value = worst_monitor_value(mode)
    stored_monitor = resume_checkpoint.get(
        "monitor_name", "val_loss" if "best_val_loss" in resume_checkpoint else None
    )
    if stored_monitor == monitor_name:
        best_monitor_value = resume_checkpoint.get(
            "best_monitor_value", resume_checkpoint.get("best_val_loss")
        )
    return best_monitor_value, resume_checkpoint.get("early_stopping_counter", 0)


def start_fold_run(
    parent: wandb.Run,
    config,
    cv_state: dict,
    fold_number: int,
    fold_checkpoint: dict | None,
):
    """A grouped child run for one fold, resuming the previous one if interrupted."""
    stored_id = cv_state.get("fold_run_ids", {}).get(fold_number)
    resume_kwargs = {}
    if fold_checkpoint is not None and stored_id:
        resume_kwargs = {"id": stored_id, "resume": "must"}

    return wandb.init(
        entity=config["wandb_entity"],
        project=config["wandb_project"],
        name=f"{parent.name}-fold{fold_number}",
        group=cv_state["group_id"],
        job_type="fold",
        mode=config["wandb_mode"],
        config={**dict(config), "fold": fold_number},
        reinit="create_new",
        **resume_kwargs,
    )


def log_fold_summary(parent: wandb.Run, fold_number: int, result: dict):
    best = result.get("best_result") or {}
    metrics = best.get("metrics", {})
    parent.log(
        {
            "Fold": fold_number,
            **{
                f"Fold {WANDB_METRIC_LABELS[key]}": metrics[key]
                for key in VALIDATION_LOG_METRICS
                if metrics.get(key) is not None
            },
            "Fold Best Epoch": best.get("epoch"),
        }
    )


def log_cv_aggregate(parent: wandb.Run, cv_state: dict, monitor_name: str):
    """Mean +/- std across folds, including the flat scalar the sweep optimises."""
    fold_results = cv_state.get("fold_results", {})
    ordered = [fold_results[k] for k in sorted(fold_results) if fold_results[k]]
    if not ordered:
        return

    aggregate = aggregate_fold_metrics([r["metrics"] for r in ordered])
    for key, label in WANDB_METRIC_LABELS.items():
        if key in aggregate["mean"]:
            parent.summary[f"CV Mean Validation {label}"] = aggregate["mean"][key]
            parent.summary[f"CV Std Validation {label}"] = aggregate["std"][key]

    monitor_key = MONITOR_METRIC_KEYS[monitor_name]
    if monitor_key in aggregate["mean"]:
        # Same flat scalar name the single-split path writes, so sweep configs are
        # identical for CV and non-CV runs.
        parent.summary[MONITOR_SUMMARY_KEYS[monitor_name]] = aggregate["mean"][monitor_key]
        parent.summary[
            MONITOR_SUMMARY_KEYS[monitor_name].replace("Best ", "Std ")
        ] = aggregate["std"][monitor_key]

    parent.summary["CV Completed Folds"] = len(ordered)
    parent.summary["CV Fold Best Epochs"] = [r.get("epoch") for r in ordered]


def fold_dir(checkpoint_dir: Path, fold: int) -> Path:
    return checkpoint_dir / FOLD_DIR_TEMPLATE.format(fold=fold)


def save_cv_metadata(checkpoint_dir: Path, run: wandb.Run, config, cv_state: dict):
    torch.save(
        {
            "run_id": run.id,
            "sweep_id": run.sweep_id,
            "config": dict(config),
            "cv": cv_state,
        },
        checkpoint_dir / "metadata.pth",
    )


def run_cross_validation(
    run: wandb.Run,
    config,
    metadata: dict | None = None,
    is_resume: bool = False,
):
    device = resolve_device(config)
    dataset_items = build_dataset_items(config)
    checkpoint_dir = get_checkpoint_dir(run, config)

    checkpoint_config = dict(config["checkpoint"])
    # One file per fold, overwritten on improvement: what makes evaluate.py's directory
    # mode a simple glob rather than an epoch-number search.
    checkpoint_config["best_filename"] = CV_BEST_FILENAME
    monitor_name = checkpoint_config.get("monitor", "val_loss")
    mode = checkpoint_config.get("mode", "min")

    if is_resume:
        cv_state = metadata["cv"]
    else:
        splits = build_cv_split_indices(dataset_items, config)
        cv_state = {
            "n_splits": config["cv"]["n_splits"],
            "group_id": wandb.util.generate_id(),
            "test_idx": splits["test_idx"],
            "folds": splits["folds"],
            "fold_run_ids": {},
            "completed_folds": [],
            "current_fold": 1,
            "current_epoch": None,
            "fold_results": {},
        }
        save_cv_metadata(checkpoint_dir, run, config, cv_state)

    run.summary["CV Folds"] = len(cv_state["folds"])

    for fold_number, fold in enumerate(cv_state["folds"], start=1):
        if fold_number in cv_state["completed_folds"]:
            continue

        current_dir = fold_dir(checkpoint_dir, fold_number)
        current_dir.mkdir(parents=True, exist_ok=True)

        # Only the fold that was interrupted has state to pick up; later folds start clean.
        fold_checkpoint = None
        resuming_fold = is_resume and fold_number == cv_state["current_fold"]
        if resuming_fold:
            last_path = current_dir / checkpoint_config.get("last_filename", "last.pth")
            if last_path.exists():
                fold_checkpoint = torch.load(last_path, map_location="cpu")

        child = start_fold_run(run, config, cv_state, fold_number, fold_checkpoint)
        cv_state["fold_run_ids"][fold_number] = child.id
        cv_state["current_fold"] = fold_number

        train_loader, val_loader = build_fold_loaders(dataset_items, fold, config)
        loss = build_loss(config)
        model = build_model(config).to(device)
        optim = build_optimizer(config, model)

        epochs = config["epochs"]
        best_monitor_value = worst_monitor_value(mode)
        early_stopping_counter = 0
        best_result = None
        if fold_checkpoint is not None:
            model.load_state_dict(fold_checkpoint["model_state_dict"])
            optim.load_state_dict(fold_checkpoint["optimizer_state_dict"])
            epochs = range(fold_checkpoint["epoch"] + 1, config["epochs"])
            best_monitor_value, early_stopping_counter = resume_monitor_state(
                fold_checkpoint, monitor_name, mode
            )
            best_result = cv_state.get("fold_results", {}).get(fold_number)

        def on_epoch_end(epoch, fold, best_result, **_):
            cv_state["current_epoch"] = epoch
            if best_result is not None:
                cv_state["fold_results"][fold] = best_result
            save_cv_metadata(checkpoint_dir, run, config, cv_state)

        result = train(
            epochs=epochs,
            model=model,
            optim=optim,
            loss=loss,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=child,
            checkpoint_dir=current_dir,
            best_monitor_value=best_monitor_value,
            early_stopping_counter=early_stopping_counter,
            checkpoint_config=checkpoint_config,
            early_stopping_config=config["early_stopping"],
            device=device,
            fold=fold_number,
            # Children must not write the sweep objective, or they would compete with
            # the parent in the sweep ranking.
            summary_key=FOLD_SUMMARY_KEY,
            on_epoch_end=on_epoch_end,
            best_result=best_result,
        )
        child.finish()

        if result["best_result"] is not None:
            cv_state["fold_results"][fold_number] = result["best_result"]
        cv_state["completed_folds"] = sorted(
            set(cv_state["completed_folds"]) | {fold_number}
        )
        cv_state["current_epoch"] = None
        save_cv_metadata(checkpoint_dir, run, config, cv_state)

        log_fold_summary(run, fold_number, result)

    log_cv_aggregate(run, cv_state, monitor_name)
    return cv_state


def runner(
    run: wandb.Run,
    config,
    resume_checkpoint: dict | None = None,
    metadata: dict | None = None,
):
    if is_cv_enabled(config):
        return run_cross_validation(
            run=run, config=config, metadata=metadata, is_resume=metadata is not None
        )

    is_resume = resume_checkpoint is not None
    device = resolve_device(config)

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
        best_monitor_value, early_stopping_counter = resume_monitor_state(
            resume_checkpoint, monitor_name, mode
        )

    return train(
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
        help=(
            "Checkpoint file to resume from, or a run directory "
            "(checkpoints/<run_id>) to resume a cross-validation run at the fold and "
            "epoch it stopped at."
        ),
    )
    args = parser.parse_args()

    resume_checkpoint = None
    metadata = None
    wandb_init_kwargs = {}
    if args.resume is not None:
        if args.resume.is_dir():
            # CV resume: the fold and epoch to pick up from live in metadata, and the
            # per-fold weights are found from there.
            metadata = load_metadata(args.resume / "metadata.pth")
        else:
            resume_checkpoint = torch.load(args.resume, map_location="cpu")
            metadata = load_metadata(find_metadata_path(args.resume))
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
