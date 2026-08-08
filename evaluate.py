import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb

from datasets import build_dataset_source
from metrics import (
    CV_THRESHOLD_STRATEGIES,
    DEFAULT_CV_THRESHOLD_STRATEGY,
    DEFAULT_FPR_ROUNDING,
    DEFAULT_THRESHOLD,
    FPR_ROUNDING_POLICIES,
    WANDB_METRIC_LABELS,
    aggregate_fold_metrics,
    collect_predictions,
    compute_metrics,
    save_confusion_matrix_plot,
    select_cv_thresholds,
    unpack_predictions,
)
from models import build_model
from train import (
    CV_BEST_FILENAME,
    build_loss,
    deep_update,
    find_metadata_path,
    load_config,
    load_metadata,
)

# Evaluation may override only settings that cannot change what the model computes.
# Everything else describes how the model was built and how its inputs were preprocessed,
# so it has to match the checkpoint: swapping transforms or model params at evaluation
# time would silently invalidate every number reported, which is exactly how a whole
# sweep was once spent training from scratch while its config claimed otherwise.
EVAL_OVERRIDABLE_KEYS = frozenset(
    {
        "threshold",
        "evaluation",
        "device",
        "wandb_entity",
        "wandb_project",
        "wandb_mode",
        "wandb_name",
        # Safe because evaluation runs under model.eval(): BatchNorm uses its running
        # statistics, so batch size cannot move a prediction.
        "dataloader",
    }
)

# Bookkeeping that aggregate_fold_metrics averages because it happens to be numeric.
# "Test Mean fold" is 3.0 for any 5-fold run, and the operating_fpr pair records how the
# shared cut was chosen rather than how the model scored. All three stay in summary.json
# and the per-fold files; none of them belongs in a run's headline summary.
NON_RESULT_METRICS = frozenset(
    {"fold", "operating_fpr_target", "operating_fpr_realised_on_val"}
)

# Rows of the comparison table, in reporting order. Ranking metrics first: they are the
# only ones that need no operating point.
COMPARISON_METRICS = (
    "roc_auc",
    "average_precision",
    "balanced_accuracy",
    "accuracy",
    "f1",
    "precision",
    "sensitivity",
    "specificity",
    "loss",
    "threshold",
)
COMPARISON_LABELS = {
    **WANDB_METRIC_LABELS,
    "average_precision": "Average Precision",
}
COMPARISON_COLUMNS = ("Metric", "CV Validation", "Test Ensemble", "Test Mean")


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


def resolve_eval_config(metadata: dict, config_path: Path = None) -> tuple[dict, dict]:
    """The run's own config, with evaluation-time settings layered over it.

    Returns (config, report). Only EVAL_OVERRIDABLE_KEYS are honoured; the rest are
    properties of the trained run. They are ignored rather than rejected because the
    expected usage is passing the very config that trained the run -- but any whose value
    differs from the run's own is named, since a silent mismatch there is the one that
    invalidates results.
    """
    config = deep_update({}, metadata["config"])
    report = {"config_path": None if config_path is None else str(config_path),
              "overridden": [], "ignored_differing": []}
    if config_path is None:
        return config, report

    supplied = load_config(config_path)
    overrides = {}
    for key, value in supplied.items():
        if key in EVAL_OVERRIDABLE_KEYS:
            if config.get(key) != value:
                report["overridden"].append(key)
            overrides[key] = value
        elif config.get(key) != value:
            report["ignored_differing"].append(key)

    if report["ignored_differing"]:
        print(
            f"Using the run's own values for {', '.join(sorted(report['ignored_differing']))} "
            f"-- {config_path} differs but these describe the trained model, not its "
            "evaluation."
        )
    return deep_update(config, overrides), report


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


def build_predictions_frame(source, test_idx, results, threshold):
    return pd.DataFrame(
        {
            "image": [source.item_id(idx) for idx in test_idx],
            "label": results["y_true"],
            "pred": (results["y_prob"] >= threshold).astype(int),
            "prob_class_1": results["y_prob"],
            "threshold": threshold,
        }
    )


def save_outputs(
    output_dir: Path,
    metrics: dict,
    predictions: pd.DataFrame,
    save_predictions: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    if save_predictions:
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

    plot_path = (
        Path(config.get("evaluation", {}).get("output_dir", "evaluations"))
        / metadata["run_id"]
        / "confusion_matrix.png"
    )
    run.log({"Test Confusion Matrix": wandb.Image(str(plot_path))})
    run.finish()


def resolve_evaluation_config(config: dict, args) -> dict:
    """Where outputs go and what gets written: CLI over config over defaults."""
    block = config.get("evaluation", {})
    return {
        "output_dir": Path(block.get("output_dir", "evaluations")),
        # store_true cannot express "off", so the flag can only turn logging on.
        "log_wandb": bool(args.log_wandb or block.get("log_wandb", False)),
        "save_predictions": bool(block.get("save_predictions", True)),
    }


def resolve_threshold_config(config: dict, args) -> dict:
    """Threshold-selection settings: CLI over config over module defaults."""
    block = config.get("threshold", {})
    return {
        "cv_strategy": (
            args.threshold_strategy
            or block.get("cv_strategy", DEFAULT_CV_THRESHOLD_STRATEGY)
        ),
        "fpr_rounding": (
            args.fpr_rounding or block.get("fpr_rounding", DEFAULT_FPR_ROUNDING)
        ),
        "fpr_grid": block.get("fpr_grid", 101),
        "threshold_grid": block.get("threshold_grid", 0),
    }


def load_fold_validation_predictions(fold_paths: dict) -> tuple[dict, dict]:
    """Every fold's stored validation predictions and its own tuned threshold.

    Deliberately two-pass: this reads each checkpoint, keeps only the small prediction
    tensors, and drops the reference before the caller reloads the weights it needs. One
    extra read per fold buys not holding five DenseNet121 state dicts in memory at once.

    Raises when any checkpoint predates prediction storage, listing every offender --
    a resumed run can mix folds from before and after, so failing on the first would hide
    the rest. Never degrades silently to a per-fold threshold: the averaging strategies
    are not defined without these.
    """
    predictions, stored_thresholds, missing = {}, {}, []
    for fold_number, path in sorted(fold_paths.items()):
        checkpoint = torch.load(path, map_location="cpu")
        packed = checkpoint.get("val_predictions")
        if packed is None:
            missing.append(str(path))
        else:
            predictions[fold_number] = unpack_predictions(packed)
        stored_thresholds[fold_number] = checkpoint.get("threshold", DEFAULT_THRESHOLD)
        del checkpoint

    if missing:
        listed = "\n  ".join(missing)
        raise ValueError(
            "These checkpoints store no validation predictions, so a shared operating "
            f"point cannot be chosen from them:\n  {listed}\n"
            "They were written before predictions were stored. Either retrain, or "
            "evaluate with --threshold-strategy per_fold_youden to use each fold's own "
            "stored threshold, or pin one with --threshold."
        )
    return predictions, stored_thresholds


def select_fold_thresholds(fold_paths: dict, config: dict, args) -> dict:
    """One operating point per fold, plus a record of how it was chosen."""
    if args.threshold is not None:
        return {
            "strategy": "cli_override",
            "shared_threshold": args.threshold,
            "fold_thresholds": {fold: args.threshold for fold in fold_paths},
            "skipped_folds": [],
        }

    threshold_config = resolve_threshold_config(config, args)
    strategy = threshold_config["cv_strategy"]

    if strategy == "per_fold_youden":
        # Reads the thresholds already in the checkpoints, so runs trained before
        # predictions were stored stay evaluable exactly as they were.
        stored = load_stored_thresholds(fold_paths)
        return select_cv_thresholds(strategy=strategy, fold_stored_thresholds=stored)

    predictions, _ = load_fold_validation_predictions(fold_paths)
    return select_cv_thresholds(
        strategy=strategy,
        fold_predictions=predictions,
        fpr_rounding=threshold_config["fpr_rounding"],
        fpr_grid=threshold_config["fpr_grid"],
        threshold_grid=threshold_config["threshold_grid"],
    )


def load_stored_thresholds(fold_paths: dict) -> dict:
    """Just the tuned threshold each fold wrote, with no requirement on predictions."""
    stored = {}
    for fold_number, path in sorted(fold_paths.items()):
        checkpoint = torch.load(path, map_location="cpu")
        stored[fold_number] = checkpoint.get("threshold", DEFAULT_THRESHOLD)
        del checkpoint
    return stored


def describe_selection(selection: dict) -> str:
    """One line naming the operating point, for the console."""
    strategy = selection["strategy"]
    if strategy == "vertical_average":
        return (
            f"threshold selection: {strategy} -> target FPR "
            f"{selection['target_fpr']:.3f} (specificity "
            f"{selection['target_specificity']:.3f}), mean TPR "
            f"{selection['mean_tpr_at_target']:.3f}"
        )
    if selection["shared_threshold"] is not None:
        return f"threshold selection: {strategy} -> threshold {selection['shared_threshold']:.4f}"
    cuts = ", ".join(f"{k}:{v:.4f}" for k, v in sorted(selection["fold_thresholds"].items()))
    return f"threshold selection: {strategy} -> per-fold {cuts}"


def ensemble_operating_point(
    selection: dict, fold_thresholds: dict
) -> tuple[float, str]:
    """The cut the ensemble inherits from the folds, and a phrase naming where it came from.

    A shared threshold when the strategy produced one; otherwise the mean of the per-fold
    cuts, which is what `vertical_average` leaves behind since it equalises false positive
    rate rather than probability.
    """
    shared = selection.get("shared_threshold")
    if shared is not None:
        return float(shared), f"{selection['strategy']} shared threshold"
    return (
        float(np.mean(list(fold_thresholds.values()))),
        f"mean of the per-fold {selection['strategy']} thresholds",
    )


def cv_validation_aggregate(metadata: dict) -> dict:
    """Mean +/- std of the folds' *validation* metrics, as recorded during training.

    Read from the run's own metadata rather than from wandb, so evaluation stays usable
    offline. Each fold's numbers come from its best epoch at the threshold that epoch
    tuned on the same validation split it scores, so they are optimistic -- they are here
    as the reference the test columns are read against, not as a result.
    """
    fold_results = metadata.get("cv", {}).get("fold_results", {})
    ordered = [
        fold_results[key].get("metrics", {})
        for key in sorted(fold_results)
        if fold_results.get(key)
    ]
    return aggregate_fold_metrics([m for m in ordered if m])


def build_comparison_rows(
    validation_mean: dict, ensemble: dict, test_mean: dict
) -> list:
    """One row per metric, one column per column of COMPARISON_COLUMNS.

    Kept as plain lists so it lands in summary.json unchanged and only becomes a
    wandb.Table at logging time.
    """
    rows = []
    for key in COMPARISON_METRICS:
        # Per-fold metrics rename loss -> test_loss, so that the field name in
        # test_metrics.json stays what it has always been.
        test_key = "test_loss" if key == "loss" else key
        values = [
            validation_mean.get(key),
            ensemble.get(key),
            test_mean.get(test_key),
        ]
        if all(value is None for value in values):
            continue
        rows.append([COMPARISON_LABELS.get(key, key), *values])
    return rows


def print_comparison(summary: dict, n_folds: int, test_n: int):
    std = summary["aggregate"]["std"]
    label_to_key = {label: key for key, label in COMPARISON_LABELS.items()}
    print(
        f"\n{n_folds} folds on {test_n} test samples "
        f"(validation / test ensemble / test mean +/- std):"
    )

    def cell(value):
        return "     -" if value is None else f"{value:6.3f}"

    for label, validation, ensemble, test in summary["comparison"]:
        key = label_to_key.get(label)
        key = "test_loss" if key == "loss" else key
        spread = f" +/- {std[key]:.3f}" if key in std else ""
        print(
            f"  {label:20s} {cell(validation)} / {cell(ensemble)} / {cell(test)}{spread}"
        )


def evaluate_cv_run(run_dir: Path, args):
    """Every fold's best model on the shared test set, plus mean/std and an ensemble."""
    metadata = load_metadata(run_dir / "metadata.pth")
    config, config_report = resolve_eval_config(metadata, args.config)
    device = resolve_device(args.device or config.get("device", "auto"))
    config["device"] = str(device)

    fold_paths = discover_fold_checkpoints(run_dir)
    if not fold_paths:
        raise FileNotFoundError(
            f"No split_*/{CV_BEST_FILENAME} under {run_dir}. Has any fold finished?"
        )

    evaluation = resolve_evaluation_config(config, args)
    selection = select_fold_thresholds(fold_paths, config, args)
    fold_thresholds = selection["fold_thresholds"]
    print(describe_selection(selection))

    test_idx = resolve_test_idx(metadata)
    source = build_dataset_source(config)
    test_loader = source.test_loader(test_idx)
    output_root = evaluation["output_dir"] / metadata["run_id"]

    fold_metrics = []
    fold_probs = []
    y_true = None
    for fold_number, path in sorted(fold_paths.items()):
        checkpoint = torch.load(path, map_location="cpu")
        metrics, results, threshold = evaluate_checkpoint(
            checkpoint, config, test_loader, device, fold_thresholds[fold_number]
        )
        metrics["fold"] = fold_number
        metrics["checkpoint"] = str(path)
        # What the shared operating point asked for, and what this fold could actually
        # realise on its own validation split. They differ because a fold's achievable
        # false positive rates are quantised by its negative count.
        if "target_fpr" in selection:
            metrics["operating_fpr_target"] = selection["target_fpr"]
            metrics["operating_fpr_realised_on_val"] = selection[
                "fold_realised_fpr"
            ].get(fold_number)

        fold_dir = output_root / f"fold_{fold_number}"
        save_outputs(
            fold_dir,
            metrics,
            build_predictions_frame(source, test_idx, results, threshold),
            save_predictions=evaluation["save_predictions"],
        )
        save_confusion_matrix_plot(
            confusion_matrix_values=metrics["confusion_matrix"],
            output_path=fold_dir / "confusion_matrix.png",
            title=f"Test Confusion Matrix (fold {fold_number})",
        )

        fold_metrics.append(metrics)
        fold_probs.append(results["y_prob"])
        # Same loader and ordering for every fold, so the labels are shared.
        y_true = results["y_true"]
        print(
            f"fold {fold_number}: auc={metrics['roc_auc']:.3f} "
            f"balanced_acc={metrics['balanced_accuracy']:.3f} "
            f"acc={metrics['accuracy']:.3f} threshold={threshold:.4f}"
        )

    ensemble_prob = np.mean(fold_probs, axis=0)
    ensemble_threshold, threshold_source = ensemble_operating_point(
        selection, fold_thresholds
    )
    ensemble_metrics = compute_metrics(
        y_true, ensemble_prob, threshold=ensemble_threshold
    )
    ensemble_metrics["n"] = int(len(y_true))
    ensemble_metrics["threshold_source"] = threshold_source
    # roc_auc and average_precision are the honest read on the ensemble; the thresholded
    # metrics below them are comparability, not a validated operating point. Averaging
    # the folds' probabilities does not average their calibration, so the inherited cut
    # sits at an unknown place on the ensemble's own score scale -- and every sample in
    # the CV pool was trained on by all but one of the models, so no held-out data is
    # left on which a better cut could be chosen.
    ensemble_metrics["note"] = (
        f"Thresholded at {ensemble_threshold:.4f} ({threshold_source}), which is "
        "inherited rather than validated: the ensemble has no held-out data of its own "
        "on which to choose an operating point, and averaging the folds' probabilities "
        "does not average their calibration. roc_auc and average_precision need no "
        "threshold and are unaffected."
    )
    ensemble_dir = output_root / "ensemble"
    save_outputs(
        ensemble_dir,
        ensemble_metrics,
        pd.DataFrame(
            {
                "image": [source.item_id(idx) for idx in test_idx],
                "label": y_true,
                "pred": (ensemble_prob >= ensemble_threshold).astype(int),
                "prob_class_1": ensemble_prob,
                "threshold": ensemble_threshold,
            }
        ),
        save_predictions=evaluation["save_predictions"],
    )
    save_confusion_matrix_plot(
        confusion_matrix_values=ensemble_metrics["confusion_matrix"],
        output_path=ensemble_dir / "confusion_matrix.png",
        title="Test Confusion Matrix (ensemble)",
    )

    aggregate = aggregate_fold_metrics(fold_metrics)
    validation = cv_validation_aggregate(metadata)
    summary = {
        "run_id": metadata["run_id"],
        "folds_evaluated": sorted(fold_paths),
        "test_n": len(test_idx),
        "threshold_selection": selection,
        "eval_config": config_report,
        "aggregate": aggregate,
        "ensemble": ensemble_metrics,
        "cv_validation": validation,
        "comparison": build_comparison_rows(
            validation_mean=validation["mean"],
            ensemble=ensemble_metrics,
            test_mean=aggregate["mean"],
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print_comparison(summary, n_folds=len(fold_metrics), test_n=len(test_idx))
    print(f"\nSaved evaluation outputs to {output_root}")

    if evaluation["log_wandb"]:
        log_cv_to_wandb(metadata, config, summary)
    return summary


def prune_stale_summary_keys(run) -> list[str]:
    """Drop summary entries that are no longer written, on a run that already has them.

    A wandb summary persists: a key this script stops writing does not disappear, it just
    sits at whatever it last held. Removing them here means re-evaluating an older run
    cleans up its summary instead of leaving `Test Mean operating_fpr_target` and the
    per-fold `Fold ...` values next to the current numbers.
    """
    retired = {
        f"Test {statistic} {metric}"
        for statistic in ("Mean", "Std")
        for metric in NON_RESULT_METRICS
    }
    stale = [
        key
        for key in list(run.summary.keys())
        if key in retired or key == "Fold" or key.startswith("Fold ")
    ]
    for key in stale:
        del run.summary[key]
    return stale


def log_cv_to_wandb(metadata, config, summary: dict):
    run = wandb.init(
        id=metadata["run_id"],
        resume="must",
        entity=config["wandb_entity"],
        project=config["wandb_project"],
        mode=config["wandb_mode"],
    )
    prune_stale_summary_keys(run)

    for key, value in summary["aggregate"]["mean"].items():
        if key not in NON_RESULT_METRICS:
            run.summary[f"Test Mean {key}"] = value
    for key, value in summary["aggregate"]["std"].items():
        if key not in NON_RESULT_METRICS:
            run.summary[f"Test Std {key}"] = value
    for key, value in summary["ensemble"].items():
        if isinstance(value, (int, float)) and key not in NON_RESULT_METRICS:
            run.summary[f"Test Ensemble {key}"] = value

    run.log(
        {
            "Metric Comparison": wandb.Table(
                columns=list(COMPARISON_COLUMNS), data=summary["comparison"]
            )
        }
    )
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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "JSON config supplying evaluation-time settings, the same file format "
            "train.py takes. Only the threshold, evaluation, device, wandb and "
            "dataloader blocks are read; everything else comes from the run's own "
            "metadata, since it describes the trained model rather than its evaluation."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Pin the decision threshold for every fold, bypassing threshold selection "
            "entirely. Defaults to whatever the configured strategy chooses, or "
            f"{DEFAULT_THRESHOLD} when a checkpoint carries no threshold at all."
        ),
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=CV_THRESHOLD_STRATEGIES,
        default=None,
        help="Override threshold.cv_strategy for a cross-validation run.",
    )
    parser.add_argument(
        "--fpr-rounding",
        choices=FPR_ROUNDING_POLICIES,
        default=None,
        help=(
            "Override threshold.fpr_rounding. How a fold resolves a target false "
            "positive rate it cannot hit exactly."
        ),
    )
    args = parser.parse_args()

    # A run directory means cross-validation: evaluate every fold on the shared test set.
    if args.checkpoint.is_dir():
        evaluate_cv_run(args.checkpoint, args)
        return

    checkpoint, metadata = load_checkpoint_and_metadata(args.checkpoint)
    config, _ = resolve_eval_config(metadata, args.config)
    evaluation = resolve_evaluation_config(config, args)
    requested_device = args.device or config.get("device", "auto")
    device = resolve_device(requested_device)
    config["device"] = str(device)

    test_idx = resolve_test_idx(metadata)
    source = build_dataset_source(config)
    test_loader = source.test_loader(test_idx)

    # A single checkpoint has one validation split, so there are no curves to average
    # across; it keeps the threshold it tuned during training unless one is pinned.
    metrics, results, threshold = evaluate_checkpoint(
        checkpoint, config, test_loader, device, args.threshold
    )

    fold = detect_fold(checkpoint, args.checkpoint)
    output_dir = evaluation["output_dir"] / metadata["run_id"]
    if fold is not None:
        metrics["fold"] = fold
        output_dir = output_dir / f"fold_{fold}"

    predictions = build_predictions_frame(source, test_idx, results, threshold)
    save_outputs(
        output_dir, metrics, predictions, save_predictions=evaluation["save_predictions"]
    )
    save_confusion_matrix_plot(
        confusion_matrix_values=metrics["confusion_matrix"],
        output_path=output_dir / "confusion_matrix.png",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation outputs to {output_dir}")

    if evaluation["log_wandb"]:
        log_to_wandb(
            metadata=metadata,
            config=config,
            metrics=metrics,
        )


if __name__ == "__main__":
    main()
