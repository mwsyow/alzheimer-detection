import argparse
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from monai.data import DataLoader
from torch import nn

import wandb
from datasets import (
    build_cv_split_indices,
    build_dataset_source,
    build_refit_split_indices,
    build_split_indices,
    resolve_split_mode,
)
from metrics import (
    CV_THRESHOLD_STRATEGIES,
    DEFAULT_CV_THRESHOLD_STRATEGY,
    DEFAULT_NEW_CV_THRESHOLD_STRATEGY,
    DEFAULT_NUM_THRESHOLDS,
    DEFAULT_THRESHOLD_OBJECTIVE,
    DEFAULT_THRESHOLD_TIE_BREAK,
    EPOCH_LOG_METRICS,
    THRESHOLD_OBJECTIVES,
    THRESHOLD_TIE_BREAKS,
    WANDB_METRIC_LABELS,
    aggregate_fold_metrics,
    collect_predictions,
    pack_predictions,
    ranking_metrics,
    select_cv_thresholds,
    summarize_predictions,
    to_wandb_logs,
    unpack_prediction_bundle,
)
from models import build_model

load_dotenv(override=True)

# checkpoint.monitor value -> key in the metrics dict returned by ranking_metrics.
#
# Threshold-free only, because that is all training computes. A monitor naming a
# thresholded metric used to be accepted and then quietly do nothing: the key was absent,
# resolve_monitor_value returned None, is_improvement read None as "no improvement", and
# the run trained to completion without ever checkpointing. Better to reject the name.
MONITOR_METRIC_KEYS = {
    "val_loss": "loss",
    "val_auc": "roc_auc",
    "val_roc_auc": "roc_auc",
    "val_average_precision": "average_precision",
}

# define_metric(summary="max") stores a nested {"max": ...} dict rather than a scalar,
# which a sweep's metric.name cannot read directly. Write the best value to a flat
# summary key as well, and point the sweep at that.
MONITOR_SUMMARY_KEYS = {
    "val_loss": "Best Validation Loss",
    "val_auc": "Best Validation AUC",
    "val_roc_auc": "Best Validation AUC",
    "val_average_precision": "Best Validation Average Precision",
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


def normalize_threshold_config(
    config: dict, legacy_missing_strategy: bool = False
) -> dict:
    """Return a copy with canonical threshold keys and validated legacy aliases.

    Fresh files default to the common OOF sweep. Metadata with no strategy at all keeps
    the old per-fold-Youden fallback, so historical runs remain reproducible.
    """
    config = deep_update({}, config)
    block = dict(config.get("threshold", {}))
    canonical = block.get("strategy")
    legacy = block.get("cv_strategy")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise ValueError(
            "Conflicting threshold.strategy and threshold.cv_strategy values: "
            f"{canonical!r} != {legacy!r}"
        )
    strategy = canonical or legacy
    if strategy is None:
        strategy = (
            DEFAULT_CV_THRESHOLD_STRATEGY
            if legacy_missing_strategy
            else DEFAULT_NEW_CV_THRESHOLD_STRATEGY
        )
    if strategy not in CV_THRESHOLD_STRATEGIES:
        raise ValueError(
            f"Unsupported threshold strategy: {strategy!r}. "
            f"Expected one of {sorted(CV_THRESHOLD_STRATEGIES)}."
        )

    block["strategy"] = strategy
    if strategy == "cv_common_threshold":
        legacy_grid = block.get("threshold_grid")
        canonical_grid = block.get("num_thresholds")
        if (
            canonical_grid is not None
            and legacy_grid not in (None, 0)
            and canonical_grid != legacy_grid
        ):
            raise ValueError(
                "Conflicting threshold.num_thresholds and threshold.threshold_grid "
                f"values: {canonical_grid!r} != {legacy_grid!r}"
            )
        block["objective"] = block.get("objective", DEFAULT_THRESHOLD_OBJECTIVE)
        block["num_thresholds"] = (
            canonical_grid
            if canonical_grid is not None
            else legacy_grid
            if legacy_grid not in (None, 0)
            else DEFAULT_NUM_THRESHOLDS
        )
        block["tie_break"] = block.get(
            "tie_break", DEFAULT_THRESHOLD_TIE_BREAK
        )
        if block["objective"] not in THRESHOLD_OBJECTIVES:
            raise ValueError(
                f"Unsupported threshold.objective: {block['objective']!r}"
            )
        if block["tie_break"] not in THRESHOLD_TIE_BREAKS:
            raise ValueError(
                f"Unsupported threshold.tie_break: {block['tie_break']!r}"
            )
        if (
            not isinstance(block["num_thresholds"], int)
            or block["num_thresholds"] < 2
        ):
            raise ValueError("threshold.num_thresholds must be an integer >= 2")
    elif strategy == "fixed":
        value = block.get("value")
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                "threshold.strategy='fixed' requires threshold.value in [0, 1]"
            )

    config["threshold"] = block
    return config


def load_config(
    config_path: Path,
    legacy_missing_strategy: bool = False,
    normalize_threshold: bool = True,
):
    with config_path.open() as f:
        config = json.load(f)
    if not normalize_threshold:
        return config
    return normalize_threshold_config(
        config, legacy_missing_strategy=legacy_missing_strategy
    )


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
    best_monitor_value: float,
    early_stopping_counter: int,
    monitor_name: str,
    val_loss: float = None,
    monitor_value: float = None,
    fold: int = None,
    val_predictions: dict = None,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # All three are None for a refit run, which has no validation split to
            # measure them on. Nothing is substituted: a fabricated value here would be
            # indistinguishable downstream from one that was actually measured.
            "val_loss": val_loss,
            "best_monitor_value": best_monitor_value,
            "early_stopping_counter": early_stopping_counter,
            "monitor_name": monitor_name,
            "monitor_value": monitor_value,
            # None for single-split runs; the 1-based fold index under CV, so a
            # checkpoint is self-describing even away from its metadata.
            "fold": fold,
            # This epoch's validation predictions, packed as tensors. Since training
            # selects no threshold, these are the only route to an operating point at
            # all: evaluate.py chooses one across the folds from exactly these arrays.
            # A checkpoint without them can only be evaluated at a pinned threshold.
            "val_predictions": val_predictions,
        },
        checkpoint_path,
    )


def validate_monitor(monitor_name: str) -> None:
    """Reject an unsupported checkpoint.monitor before a single epoch runs.

    Checked eagerly because MONITOR_SUMMARY_KEYS is indexed while train() is still
    setting up, which would otherwise raise a bare KeyError naming neither the config
    key at fault nor the accepted values.
    """
    if monitor_name not in MONITOR_METRIC_KEYS:
        raise ValueError(
            f"Unsupported checkpoint.monitor: {monitor_name!r}. "
            f"Expected one of {sorted(MONITOR_METRIC_KEYS)}. Training computes only "
            "threshold-free metrics, so accuracy, balanced accuracy and F1 are no "
            "longer monitorable -- they need an operating point, which evaluate.py "
            "chooses after training from the stored validation predictions."
        )


def resolve_monitor_value(metrics: dict, monitor_name: str):
    validate_monitor(monitor_name)
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
    val_indices=None,
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
    validate_monitor(monitor_name)
    if best_monitor_value is None:
        best_monitor_value = worst_monitor_value(mode)
    if summary_key is None:
        summary_key = MONITOR_SUMMARY_KEYS[monitor_name]

    # A refit run pools train and val, so there is no held-out split to score. Every
    # decision that reads one -- the monitor, save_best, early stopping -- is skipped
    # rather than fed a substitute, and the last epoch's weights are the result.
    has_validation = val_loader is not None
    if not has_validation and not checkpoint_config.get("save_last", True):
        raise ValueError(
            "Training without a validation split selects no best epoch, so "
            "checkpoint.save_last must be true or the run would write no checkpoint "
            "at all."
        )
    final_train_metrics = None
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
        train_metrics = ranking_metrics(
            train_results["y_true"], train_results["y_prob"], loss=train_results["loss"]
        )
        final_train_metrics = train_metrics

        val_metrics = None
        val_loss = None
        monitor_value = None
        improved = False
        if has_validation:
            val_results = collect_predictions(
                model=model,
                loader=val_loader,
                loss_fn=loss,
                device=device,
            )
            val_metrics = ranking_metrics(
                val_results["y_true"], val_results["y_prob"], loss=val_results["loss"]
            )
            val_loss = val_metrics["loss"]
            monitor_value = resolve_monitor_value(val_metrics, monitor_name)
            improved = is_improvement(
                monitor_value, best_monitor_value, mode, min_delta
            )

            if improved:
                best_monitor_value = monitor_value
                early_stopping_counter = 0
                best_result = {
                    "epoch": ep,
                    "monitor_name": monitor_name,
                    "monitor_value": monitor_value,
                    "metrics": val_metrics,
                }
                if fold is not None:
                    best_result["fold"] = fold
            else:
                # Only counted while there is a validation signal to stall against.
                # Incrementing without one would early-stop a refit run at `patience`.
                early_stopping_counter += 1

        checkpoint_kwargs = {
            "epoch": ep,
            "model": model,
            "optimizer": optim,
            "val_loss": val_loss,
            "best_monitor_value": best_monitor_value,
            "early_stopping_counter": early_stopping_counter,
            "monitor_name": monitor_name,
            "monitor_value": monitor_value,
            "fold": fold,
            "val_predictions": (
                pack_predictions(
                    val_results["y_true"],
                    val_results["y_prob"],
                    indices=val_indices,
                )
                if has_validation
                else None
            ),
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

        early_stopped = (
            has_validation
            and early_stopping_config.get("enabled", False)
            and early_stopping_counter >= early_stopping_config.get("patience", 5)
        )
        logs = {"Epoch": ep}
        logs.update(to_wandb_logs(train_metrics, "Training"))
        if has_validation:
            logs.update(to_wandb_logs(val_metrics, "Validation"))
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

    # Without a validation split best_monitor_value never leaves its infinite sentinel,
    # so no "Best Validation ..." key is written. That is deliberate: a refit run has no
    # held-out number, and an absent key is honest where any written value would not be.
    if math.isfinite(best_monitor_value):
        logger.summary[summary_key] = best_monitor_value

    return {
        "fold": fold,
        "best_monitor_value": best_monitor_value if math.isfinite(best_monitor_value) else None,
        "best_result": best_result,
        "early_stopping_counter": early_stopping_counter,
        # The last epoch's training metrics. For a refit run this is the only read on
        # how the model ended up, since there is no validation curve.
        "final_train_metrics": final_train_metrics,
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
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW(
        trainable_parameters, **optimizer_config.get("params", {})
    )


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
    if fold_checkpoint is not None and stored_id:
        resume_kwargs = {"id": stored_id, "resume": "must"}
    else:
        resume_kwargs = {"id": wandb.util.generate_id()}

    # While wandb's global settings carry a sweep_id, init strips project, entity
    # and run_id from every call -- "Ignoring run_id ... when running a sweep" --
    # so the child falls back to the trial's own id and dies with "run ID <parent>
    # is in use". A fold is not a sweep trial in its own right, and the parent is
    # already registered with the sweep, so clear the flag just long enough to
    # create the child.
    library = wandb.setup()
    sweep_id = library.settings.sweep_id
    library.settings.sweep_id = None
    try:
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
    finally:
        library.settings.sweep_id = sweep_id


def log_fold_summary(parent: wandb.Run, fold_number: int, result: dict):
    best = result.get("best_result") or {}
    metrics = best.get("metrics", {})
    parent.log(
        {
            "Fold": fold_number,
            **{
                f"Fold {WANDB_METRIC_LABELS[key]}": metrics[key]
                for key in EPOCH_LOG_METRICS
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


def load_verified_oof_predictions(
    checkpoint_dir: Path,
    cv_state: dict,
    dataset_items: list[dict],
    require_indices: bool = True,
) -> dict:
    """Load selected fold predictions and prove that they are out-of-fold."""
    test = set(cv_state["test_idx"])
    expected_development = set(range(len(dataset_items))) - test
    tiled_validation = set()
    predictions = {}

    for fold_number, fold in enumerate(cv_state["folds"], start=1):
        path = fold_dir(checkpoint_dir, fold_number) / CV_BEST_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"Selected checkpoint missing for fold {fold_number}: {path}"
            )
        checkpoint = torch.load(path, map_location="cpu")
        packed = checkpoint.get("val_predictions")
        if packed is None:
            raise ValueError(f"{path} stores no validation predictions")
        y_true, y_prob, saved_indices = unpack_prediction_bundle(packed)

        train_indices = list(fold["train_idx"])
        val_indices = list(fold["val_idx"])
        train_set, val_set = set(train_indices), set(val_indices)
        if train_set & val_set:
            raise ValueError(
                f"fold {fold_number} has overlapping train and validation indices"
            )
        if (train_set | val_set) & test:
            raise ValueError(
                f"fold {fold_number} train/validation indices overlap the test set"
            )
        if tiled_validation & val_set:
            raise ValueError("fold validation sets are not disjoint")
        tiled_validation.update(val_set)

        if saved_indices is None:
            if require_indices:
                raise ValueError(
                    f"{path} has no validation prediction indices; refusing a new "
                    "common-threshold search because OOF provenance cannot be verified"
                )
        else:
            saved = [int(index) for index in np.asarray(saved_indices).tolist()]
            if saved != val_indices:
                raise ValueError(
                    f"fold {fold_number} saved prediction indices do not exactly "
                    "match its ordered validation indices"
                )
            expected_labels = np.asarray(
                [dataset_items[index]["label"] for index in saved], dtype=int
            )
            if not np.array_equal(np.asarray(y_true, dtype=int), expected_labels):
                raise ValueError(
                    f"fold {fold_number} prediction labels do not match dataset labels"
                )

        if len(y_true) != len(val_indices) or len(y_prob) != len(val_indices):
            raise ValueError(
                f"fold {fold_number} prediction count does not match validation count"
            )
        predictions[fold_number] = (y_true, y_prob)

    if tiled_validation != expected_development:
        missing = sorted(expected_development - tiled_validation)
        extra = sorted(tiled_validation - expected_development)
        raise ValueError(
            "fold validation sets do not tile the development pool "
            f"(missing={missing}, extra={extra})"
        )
    return predictions


def _json_threshold_selection(selection: dict) -> dict:
    """Drop the full curve and add the stable scalar name used by donors."""
    summary = {key: value for key, value in selection.items() if key != "curve"}
    if selection.get("shared_threshold") is not None:
        summary["threshold"] = float(selection["shared_threshold"])
    return summary


def select_and_store_cv_threshold(
    checkpoint_dir: Path,
    run: wandb.Run,
    config: dict,
    cv_state: dict,
    dataset_items: list[dict],
) -> dict | None:
    """Run post-CV threshold selection once and persist its reproducible artifacts."""
    block = normalize_threshold_config(config)["threshold"]
    strategy = block["strategy"]
    if strategy == "per_fold_youden":
        # Explicit old-checkpoint compatibility. New checkpoints deliberately contain
        # no independently tuned fold threshold, so there is nothing to recompute here.
        return None

    predictions = load_verified_oof_predictions(
        checkpoint_dir,
        cv_state,
        dataset_items,
        require_indices=(strategy == "cv_common_threshold"),
    )
    selection = select_cv_thresholds(
        strategy=strategy,
        fold_predictions=predictions,
        fpr_rounding=block.get("fpr_rounding", "at_least"),
        fpr_grid=block.get("fpr_grid", 101),
        threshold_grid=block.get("threshold_grid", 0),
        objective=block.get("objective", DEFAULT_THRESHOLD_OBJECTIVE),
        num_thresholds=block.get("num_thresholds", DEFAULT_NUM_THRESHOLDS),
        tie_break=block.get("tie_break", DEFAULT_THRESHOLD_TIE_BREAK),
        fixed_value=block.get("value"),
    )
    summary = _json_threshold_selection(selection)
    curve = selection.get("curve", [])
    selection_path = checkpoint_dir / "threshold_selection.json"
    curve_path = checkpoint_dir / "threshold_curve.csv"

    with selection_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    if curve:
        with curve_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)

    artifacts = {
        "threshold_selection": str(selection_path),
        "threshold_curve": str(curve_path) if curve else None,
    }
    cv_state["threshold_selection"] = {**summary, "artifacts": artifacts}

    if strategy == "cv_common_threshold":
        run.log(
            {
                "CV Threshold Sweep": wandb.Table(
                    columns=list(curve[0]),
                    data=[[row[key] for key in curve[0]] for row in curve],
                )
            }
        )
        run.summary["cv_best_threshold"] = summary["threshold"]
        run.summary["cv_threshold_objective"] = summary["objective"]
        run.summary["cv_best_mean_objective"] = summary["mean_objective"]
        run.summary["cv_best_std_objective"] = summary["std_objective"]
        run.summary["cv_best_mean_balanced_accuracy"] = summary[
            "mean_balanced_accuracy"
        ]
        run.summary["cv_best_std_balanced_accuracy"] = summary[
            "std_balanced_accuracy"
        ]
    elif summary.get("threshold") is not None:
        run.summary["cv_best_threshold"] = summary["threshold"]
        run.summary["cv_threshold_objective"] = strategy
    return selection


def run_cross_validation(
    run: wandb.Run,
    config,
    metadata: dict | None = None,
    is_resume: bool = False,
):
    device = resolve_device(config)
    source = build_dataset_source(config)
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
        splits = build_cv_split_indices(source.items, config)
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
    # The "Fold ..." series is a chart across folds, not a run-level result. Without this
    # wandb would summarise each of them with its last logged value, putting the final
    # fold's numbers in the summary where they read as the run's -- next to the CV Mean
    # keys, which are what the run actually scored.
    run.define_metric("Fold*", summary="none")

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

        train_loader, val_loader = source.fold_loaders(fold)
        loss = build_loss(config)
        model = build_model(
            config, initialize_pretrained=fold_checkpoint is None
        ).to(device)
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
            val_indices=fold["val_idx"],
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

    if len(cv_state["completed_folds"]) == len(cv_state["folds"]):
        # Deliberately recomputed on completed resumes: the selected checkpoints are
        # the source of truth, and both artifacts are deterministic replacements.
        select_and_store_cv_threshold(
            checkpoint_dir, run, config, cv_state, source.items
        )
        save_cv_metadata(checkpoint_dir, run, config, cv_state)

    log_cv_aggregate(run, cv_state, monitor_name)
    return cv_state


def runner(
    run: wandb.Run,
    config,
    resume_checkpoint: dict | None = None,
    metadata: dict | None = None,
):
    split_mode = resolve_split_mode(config)
    if split_mode == "cv":
        return run_cross_validation(
            run=run, config=config, metadata=metadata, is_resume=metadata is not None
        )

    is_resume = resume_checkpoint is not None
    device = resolve_device(config)

    source = build_dataset_source(config)

    checkpoint_dir = get_checkpoint_dir(run, config)
    if is_resume:
        split_indices = metadata["split"]
    else:
        build_indices = (
            build_refit_split_indices if split_mode == "refit" else build_split_indices
        )
        split_indices = build_indices(source.items, config)
        save_metadata(
            checkpoint_dir=checkpoint_dir,
            run=run,
            config=config,
            split_indices=split_indices,
        )

    train_loader, val_loader = source.train_val_loaders(split_indices)
    if split_mode != "refit" and val_loader is None:
        raise ValueError(
            "The split produced an empty validation set, so there would be nothing to "
            "monitor, checkpoint on, or select an operating point from. Enable the "
            "refit block if training on everything but the test set was intended."
        )

    loss = build_loss(config)
    model = build_model(config, initialize_pretrained=not is_resume)
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

    checkpoint_config = config["checkpoint"]
    early_stopping_config = config["early_stopping"]
    if split_mode == "refit":
        # A refit config is its cross-validation sibling plus one block, so it carries
        # these settings verbatim and they are ignored rather than rejected -- rejecting
        # would force per-arm edits to the very fields the benchmark protocol requires to
        # be identical across arms. Announced so the omission is visible, not assumed.
        checkpoint_config = {**checkpoint_config, "save_best": False}
        early_stopping_config = {}
        ignored = ["checkpoint.save_best", "checkpoint.monitor", "checkpoint.mode",
                   "checkpoint.min_delta"]
        if config["early_stopping"].get("enabled", False):
            ignored.append("early_stopping")
        print(
            f"refit: training on {len(split_indices['train_idx'])} volumes with no "
            f"validation split for {config['epochs']} epochs; "
            f"{checkpoint_config.get('last_filename', 'last.pth')} is the model. "
            f"Ignoring {', '.join(ignored)} -- nothing is held out to select on."
        )
        run.summary["Refit Pool N"] = len(split_indices["train_idx"])
        run.summary["Refit Test N"] = len(split_indices["test_idx"])
        run.summary["Refit Epochs"] = config["epochs"]
        run.summary["Refit Ignored Settings"] = ", ".join(ignored)

    result = train(
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
        checkpoint_config=checkpoint_config,
        early_stopping_config=early_stopping_config,
        device=device,
    )

    if split_mode == "refit":
        # No held-out number exists, so the final training metrics are the only read on
        # how the model ended up -- whether it collapsed, or memorised the pool.
        final = result.get("final_train_metrics") or {}
        for key in EPOCH_LOG_METRICS:
            if final.get(key) is not None:
                run.summary[f"Refit Final Training {WANDB_METRIC_LABELS[key]}"] = final[key]

    return result


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
        config = normalize_threshold_config(
            metadata["config"], legacy_missing_strategy=True
        )
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
    wandb_run.define_metric("Validation Average Precision", summary="max")
    wandb_run.define_metric("Validation Loss", summary="min")

    if args.resume is None:
        config = apply_sweep_overrides(config, wandb_run.config)
        config = normalize_threshold_config(config)
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
