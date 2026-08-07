import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb

from datasets import build_test_loader
from metrics import (
    DEFAULT_THRESHOLD,
    aggregate_fold_metrics,
    collect_predictions,
    compute_metrics,
    save_confusion_matrix_plot,
)
from models import build_model
from train import CV_BEST_FILENAME, build_loss, find_metadata_path, load_metadata


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def load_checkpoint_and_metadata(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = load_metadata(find_metadata_path(checkpoint_path))
    return checkpoint, metadata


def resolve_test_idx(metadata: dict) -> list[int]:
    """The held-out test set, which CV shares across every fold."""
    if "cv" in metadata:
        return metadata["cv"]["test_idx"]
    return metadata["split"]["test_idx"]


def detect_fold(checkpoint: dict, checkpoint_path: Path):
    """Which CV fold produced a checkpoint.

    Prefer the value stored in the checkpoint; fall back to the split_<k> directory
    name so checkpoints written before the fold field still resolve.
    """
    fold = checkpoint.get("fold")
    if fold is not None:
        return fold
    match = re.fullmatch(r"split_(\d+)", checkpoint_path.parent.name)
    return int(match.group(1)) if match else None


def discover_fold_checkpoints(run_dir: Path) -> dict[int, Path]:
    """Map fold number -> best_model.pth for every fold present in a run directory."""
    found = {}
    for path in sorted(run_dir.glob(f"split_*/{CV_BEST_FILENAME}")):
        match = re.fullmatch(r"split_(\d+)", path.parent.name)
        if match:
            found[int(match.group(1))] = path
    return found


def evaluate_checkpoint(
    checkpoint: dict,
    config: dict,
    test_loader,
    device: torch.device,
    threshold_override: float = None,
):
    """Run one model over the test set and score it at its stored threshold."""
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    loss = build_loss(config)

    threshold = (
        threshold_override
        if threshold_override is not None
        else checkpoint.get("threshold", DEFAULT_THRESHOLD)
    )

    results = collect_predictions(model, test_loader, loss, device)
    metrics = compute_metrics(
        y_true=results["y_true"],
        y_prob=results["y_prob"],
        threshold=threshold,
        loss=results["loss"],
    )
    # Historical field name kept so old and new test_metrics.json stay comparable.
    metrics["test_loss"] = metrics.pop("loss")
    return metrics, results, threshold


def build_predictions_frame(dataset_items, test_idx, results, threshold):
    return pd.DataFrame(
        {
            "image": [dataset_items[idx]["image"] for idx in test_idx],
            "label": results["y_true"],
            "pred": (results["y_prob"] >= threshold).astype(int),
            "prob_class_1": results["y_prob"],
            "threshold": threshold,
        }
    )


def save_outputs(output_dir: Path, metrics: dict, predictions: pd.DataFrame):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)


def log_to_wandb(metadata, config, metrics: dict):
    run = wandb.init(
        id=metadata["run_id"],
        resume="must",
        entity=config["wandb_entity"],
        project=config["wandb_project"],
        mode=config["wandb_mode"],
    )
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        run.summary[f"Test {key}"] = value

    plot_path = Path("evaluations") / metadata["run_id"] / "confusion_matrix.png"
    run.log({"Test Confusion Matrix": wandb.Image(str(plot_path))})
    run.finish()


def evaluate_cv_run(run_dir: Path, args):
    """Every fold's best model on the shared test set, plus mean/std and an ensemble."""
    metadata = load_metadata(run_dir / "metadata.pth")
    config = metadata["config"]
    device = resolve_device(args.device or config.get("device", "auto"))
    config["device"] = str(device)

    fold_paths = discover_fold_checkpoints(run_dir)
    if not fold_paths:
        raise FileNotFoundError(
            f"No split_*/{CV_BEST_FILENAME} under {run_dir}. Has any fold finished?"
        )

    test_idx = resolve_test_idx(metadata)
    test_loader, dataset_items = build_test_loader(config, test_idx)
    output_root = Path("evaluations") / metadata["run_id"]

    fold_metrics = []
    fold_probs = []
    thresholds = []
    y_true = None
    for fold_number, path in sorted(fold_paths.items()):
        checkpoint = torch.load(path, map_location="cpu")
        metrics, results, threshold = evaluate_checkpoint(
            checkpoint, config, test_loader, device, args.threshold
        )
        metrics["fold"] = fold_number
        metrics["checkpoint"] = str(path)

        fold_dir = output_root / f"fold_{fold_number}"
        save_outputs(
            fold_dir,
            metrics,
            build_predictions_frame(dataset_items, test_idx, results, threshold),
        )
        save_confusion_matrix_plot(
            confusion_matrix_values=metrics["confusion_matrix"],
            output_path=fold_dir / "confusion_matrix.png",
            title=f"Test Confusion Matrix (fold {fold_number})",
        )

        fold_metrics.append(metrics)
        fold_probs.append(results["y_prob"])
        thresholds.append(threshold)
        # Same loader and ordering for every fold, so the labels are shared.
        y_true = results["y_true"]
        print(
            f"fold {fold_number}: auc={metrics['roc_auc']:.3f} "
            f"balanced_acc={metrics['balanced_accuracy']:.3f} "
            f"acc={metrics['accuracy']:.3f} threshold={threshold:.4f}"
        )

    ensemble_prob = np.mean(fold_probs, axis=0)
    # The per-fold thresholds were each tuned on their own validation fold; averaging
    # them keeps the ensemble's operating point off the test set.
    ensemble_threshold = float(np.mean(thresholds))
    ensemble_metrics = compute_metrics(y_true, ensemble_prob, ensemble_threshold)
    ensemble_metrics["threshold_note"] = (
        "Mean of per-fold validation-tuned thresholds; not re-tuned on test."
    )
    ensemble_dir = output_root / "ensemble"
    save_outputs(
        ensemble_dir,
        ensemble_metrics,
        pd.DataFrame(
            {
                "image": [dataset_items[idx]["image"] for idx in test_idx],
                "label": y_true,
                "pred": (ensemble_prob >= ensemble_threshold).astype(int),
                "prob_class_1": ensemble_prob,
                "threshold": ensemble_threshold,
            }
        ),
    )
    save_confusion_matrix_plot(
        confusion_matrix_values=ensemble_metrics["confusion_matrix"],
        output_path=ensemble_dir / "confusion_matrix.png",
        title="Test Confusion Matrix (fold ensemble)",
    )

    summary = {
        "run_id": metadata["run_id"],
        "folds_evaluated": sorted(fold_paths),
        "test_n": len(test_idx),
        "aggregate": aggregate_fold_metrics(fold_metrics),
        "ensemble": ensemble_metrics,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    mean, std = summary["aggregate"]["mean"], summary["aggregate"]["std"]
    print(f"\n{len(fold_metrics)}-fold mean +/- std on {len(test_idx)} test samples:")
    for key in ("roc_auc", "balanced_accuracy", "accuracy", "f1", "sensitivity", "specificity"):
        if key in mean:
            print(f"  {key:20s} {mean[key]:.3f} +/- {std[key]:.3f}")
    print(
        f"  {'ensemble roc_auc':20s} {ensemble_metrics['roc_auc']:.3f}"
        f"   (balanced_acc {ensemble_metrics['balanced_accuracy']:.3f})"
    )
    print(f"\nSaved evaluation outputs to {output_root}")

    if args.log_wandb:
        log_cv_to_wandb(metadata, config, summary)
    return summary


def log_cv_to_wandb(metadata, config, summary: dict):
    run = wandb.init(
        id=metadata["run_id"],
        resume="must",
        entity=config["wandb_entity"],
        project=config["wandb_project"],
        mode=config["wandb_mode"],
    )
    for key, value in summary["aggregate"]["mean"].items():
        run.summary[f"Test Mean {key}"] = value
    for key, value in summary["aggregate"]["std"].items():
        run.summary[f"Test Std {key}"] = value
    for key, value in summary["ensemble"].items():
        if isinstance(value, (int, float)):
            run.summary[f"Test Ensemble {key}"] = value
    run.finish()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MRI checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "A checkpoint file, or a run directory (checkpoints/<run_id>) to evaluate "
            "every cross-validation fold and report mean +/- std plus an ensemble."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override the decision threshold. Defaults to the value tuned on validation "
            f"and stored in the checkpoint, or {DEFAULT_THRESHOLD} if absent."
        ),
    )
    args = parser.parse_args()

    # A run directory means cross-validation: evaluate every fold on the shared test set.
    if args.checkpoint.is_dir():
        evaluate_cv_run(args.checkpoint, args)
        return

    checkpoint, metadata = load_checkpoint_and_metadata(args.checkpoint)
    config = metadata["config"]
    requested_device = args.device or config.get("device", "auto")
    device = resolve_device(requested_device)
    config["device"] = str(device)

    test_idx = resolve_test_idx(metadata)
    test_loader, dataset_items = build_test_loader(config, test_idx)

    metrics, results, threshold = evaluate_checkpoint(
        checkpoint, config, test_loader, device, args.threshold
    )

    fold = detect_fold(checkpoint, args.checkpoint)
    output_dir = Path("evaluations") / metadata["run_id"]
    if fold is not None:
        metrics["fold"] = fold
        output_dir = output_dir / f"fold_{fold}"

    predictions = build_predictions_frame(dataset_items, test_idx, results, threshold)
    save_outputs(output_dir, metrics, predictions)
    save_confusion_matrix_plot(
        confusion_matrix_values=metrics["confusion_matrix"],
        output_path=output_dir / "confusion_matrix.png",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation outputs to {output_dir}")

    if args.log_wandb:
        log_to_wandb(
            metadata=metadata,
            config=config,
            metrics=metrics,
        )


if __name__ == "__main__":
    main()
